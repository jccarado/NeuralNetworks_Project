import torch
import torch.nn as nn
import torch.nn.functional as F

class ZINB_Loss(nn.Module):
    def __init__(self):
        super(ZINB_Loss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        # if ridge_lambda > 0:
        #     ridge = ridge_lambda*torch.square(pi)
        #     result += ridge
        
        result = torch.mean(result)
        return result
    

class KLD_Loss(nn.Module):
    def __init__(self):
        super(KLD_Loss, self).__init__()

    def forward(self, target, pred):
        return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
    

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)