import torch
import torch.nn.functional as F

def reconstruction_loss(x, x_hat):
    return F.mse_loss(x_hat, x, reduction='mean')

