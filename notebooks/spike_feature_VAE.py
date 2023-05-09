# %%
import numpy as np
import os
from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# %%
class Encoder_cnn(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder_cnn, self).__init__()
        self.K = latent_dims
        self.conv1 = nn.Conv2d(1, 50, 5)
        self.bn1 = nn.BatchNorm2d(50, eps=1e-4)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(50, 12, 5)
        
        self.bn2 = nn.BatchNorm2d(12, eps=1e-4)
        self.conv3 = nn.Conv2d(1, 12, 5)
        
        self.fc1 = nn.Linear(12 * (121-8) * (40-8), 120)
        self.fc2 = nn.Linear(120, 84)
        
        self.linear_log_var = nn.Linear(84, self.K)
        self.linear_mean = nn.Linear(84, self.K)



    def forward(self, x):
        # skip = self.conv3(x)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))  
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) 
        
        # x = torch.flatten(x + skip, 1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.linear_mean(x)
        log_var = self.linear_log_var(x)

        return mean, log_var


# %%
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dims, 84)
        self.fc2 = nn.Linear(84, 120)
        self.fc3 = nn.Linear(120, 12 * (121-8) * (40-8))
        
        self.conv2dtrans1 = nn.ConvTranspose2d(12, 50, 5)
        self.conv2dtrans2 = nn.ConvTranspose2d(50, 1, 5)
        

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = torch.reshape(z, (-1, 12, 121-8, 40-8))
        z = F.relu(self.conv2dtrans1(z))
        
        return F.relu(self.conv2dtrans2(z))


# %%
class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder_cnn(latent_dims)
        self.decoder = Decoder(latent_dims)

    def _encoding(self, input_x):
        mu_z, log_var_z = self.encoder.forward(input_x)
        return mu_z, log_var_z

    def _z_sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def _decoding(self, t):
        reconstruct_x = self.decoder.forward(t)
        return reconstruct_x

    def forward(self, input_x):
        mu, log_var = self._encoding(input_x)
        t = self._z_sample(mu, log_var)
        reconstruct_x = self._decoding(t)
        return reconstruct_x, mu, log_var, t


# %%

# %%
def build_loss(x_reconstruct, input_x, mu, log_var):
    std = torch.exp(0.5*log_var)
    recon_loss = F.mse_loss(x_reconstruct, input_x, reduction='sum')
    #recon_loss = torch.mean(x_hat.log_prob(input_x))
    divergence = 0.5*torch.sum(1 + torch.log(std) - mu**2 - std)

    elbo = recon_loss + divergence 

    return elbo
