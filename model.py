import torch
import torch.nn as nn

class Autoencoder(nn.module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),   
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),   
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)                        
        )

    def forward(self, x):
        latent_space = self.encoder(x)
        output = self.decoder(latent_space)

        return output