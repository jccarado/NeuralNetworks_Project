import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import ZINB_Loss, KLD_Loss, MeanAct, DispAct
import numpy as np
from sklearn.cluster import KMeans
import math, os
from sklearn import metrics

def buildNetwork(layers):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.ReLU())
    return nn.Sequential(*net)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, z_dim, encoder_layers=[], decoder_layers=[], device="cuda"):
        super(Autoencoder, self).__init__()

        torch.set_default_dtype(torch.float64)
        self.device = device
        
        # Encoder
        self.encoder = buildNetwork([input_dim] + encoder_layers)

        # Latent
        self.z_dim = z_dim
        self.latent_space = nn.Linear(encoder_layers[-1], z_dim)
        
        # Decoder
        self.decoder = buildNetwork([z_dim] + decoder_layers)

        

        # output layers
        self.out_mean = nn.Sequential(nn.Linear(decoder_layers[-1], input_dim), MeanAct())
        self.out_disp = nn.Sequential(nn.Linear(decoder_layers[-1], input_dim), DispAct())
        self.out_dropout = nn.Sequential(nn.Linear(decoder_layers[-1], input_dim), nn.Sigmoid())

        self.alpha = 1.
        self.sigma = 2.5
        self.gamma = 1.
        self.zinb_loss = ZINB_Loss().to(self.device)
        self.cluster_loss = KLD_Loss().to(self.device)
        self.to(device)



    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q  


    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()   

    def encodeBatch(self, X, batch_size=256):
        self.eval()
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch).to(self.device)
            z, _, _, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded.to(self.device)           


    def forward(self, x, cluster=False):
        h = self.encoder(x + torch.randn_like(x))
        z = self.latent_space(h)
        output = self.decoder(z)

        mean = self.out_mean(output)
        disp = self.out_disp(output)
        dropout = self.out_dropout(output)

        h0 = self.encoder(x)
        z0 = self.latent_space(h0)
        q = 0
        if cluster:
            q = self.soft_assign(z0)

        return z0, q, mean, disp, dropout
    
    def pretrain(self, X, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        self.train()
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(size_factor.values))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            loss_val = 0
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).to(self.device)
                x_raw_tensor = Variable(x_raw_batch).to(self.device)
                sf_tensor = Variable(sf_batch).to(self.device)
                _, _, mean_tensor, disp_tensor, pi_tensor = self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.detach().item() * len(x_batch)
            print('Pretrain epoch %3d, ZINB loss: %.8f' % (epoch+1, loss_val/X.shape[0]))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)            




    def fit(self, X, X_raw, size_factor, n_clusters, init_centroid=None, y=None, y_pred_init=None, lr=1., batch_size=256, 
            num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        '''X: tensor data'''
        self.train()
        print("Clustering stage")
        X = torch.tensor(X, dtype=torch.float64)
        X_raw = torch.tensor(X_raw, dtype=torch.float64)
        size_factor = torch.tensor(size_factor, dtype=torch.float64)
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        if init_centroid is None:
            kmeans = KMeans(n_clusters, n_init=20)
            data = self.encodeBatch(X)
            self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            self.y_pred_last = self.y_pred
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float64))
        else:
            self.mu.data.copy_(torch.tensor(init_centroid, dtype=torch.float64))
            self.y_pred = y_pred_init
            self.y_pred_last = self.y_pred
        if y is not None:
#            acc = np.round(cluster_acc(y, self.y_pred), 5)
            ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: AMI= %.4f, ARI= %.4f' % (ami, ari))

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))

        final_ami, final_ari, final_epoch = 0, 0, 0


        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X.to(self.device))
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                if y is not None:
#                    final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_ami = ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
                    final_epoch = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    print('Clustering   %d: AMI= %.4f, ARI= %.4f' % (epoch+1, ami, ari))

                # # save current model
                # if (epoch>0 and delta_label < tol) or epoch%10 == 0:
                #     self.save_checkpoint({'epoch': epoch+1,
                #             'state_dict': self.state_dict(),
                #             'mu': self.mu,
                #             'y_pred': self.y_pred,
                #             'y_pred_last': self.y_pred_last,
                #             'y': y
                #             }, epoch+1, filename=save_dir)

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch>0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break


            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = size_factor[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch).to(self.device)
                rawinputs = Variable(xrawbatch).to(self.device)
                sfinputs = Variable(sfbatch).to(self.device)
                target = Variable(pbatch).to(self.device)

                zbatch, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs, cluster=True)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)

                loss = cluster_loss*self.gamma + recon_loss
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.item() * len(inputs)
                recon_loss_val += recon_loss.item() * len(inputs)
                train_loss += loss.item() * len(inputs)

            print("Epoch %3d: Total: %.8f Clustering Loss: %.8f ZINB Loss: %.8f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))

        return self.y_pred, final_ami, final_ari, final_epoch