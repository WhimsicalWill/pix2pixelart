import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import cv2

from src.utils import weights_init
from collections import OrderedDict
from itertools import combinations

class ResidualBlock(nn.Module):
    def __init__(self, in_feat, num_feat, reduction=16, attention=False):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_feat, num_feat, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_feat))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_feat, in_feat, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(in_feat))
        self.main = nn.Sequential(*layers)
        self.is_attention = attention
        if attention :
            attentionlayer = [] 
            attentionlayer.append(nn.AdaptiveAvgPool2d(1))
            attentionlayer.append(nn.Conv2d(in_feat, in_feat//reduction, kernel_size=1))
            attentionlayer.append(nn.ReLU(inplace=True))
            attentionlayer.append(nn.Conv2d(in_feat//reduction, in_feat, kernel_size=1))
            attentionlayer.append(nn.Sigmoid())
            self.attention = nn.Sequential(*attentionlayer)

    def forward(self, x):
        out = self.main(x)
        if self.is_attention:
            x = x + out * self.attention(out)
        else:
            x = x + out 
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, num_feat, num_res):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, num_feat, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.ReLU(inplace=True))

        curr_dim = num_feat
        for _ in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_dim*2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        
        for _ in range(num_res):
            layers.append(ResidualBlock(curr_dim, curr_dim))


        for _ in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_dim//2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(curr_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(curr_dim, in_channels, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        
        self.main = nn.Sequential(*layers)
        self.apply(weights_init())

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, in_channels, device, num_feat=64, num_repeat=6):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, num_feat, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))

        curr_dim = num_feat
        for _ in range(1, num_repeat):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self.apply(weights_init())

    def forward(self, x):
        # inject gaussian noise into discriminator model for robustness
        noise_sigma = 0.1 # guess for good noise level
        x =  x + torch.randn(x.shape).to(self.device) * noise_sigma
        h = self.main(x)
        out = self.sigmoid(self.conv1(h))
        return out

    def calc_dis_loss(self, x_real, x_fake):
        real_pred = self.forward(x_real)
        fake_pred = self.forward(x_fake)
        # discriminator output is tanh and loss uses 0.9 and 0.1 smooth labels
        soft_fake, soft_real = 0.1, 0.9
        # TODO: experiment with hard and soft labels (or maybe something inbetween)
        loss = torch.mean((real_pred - soft_real)**2) + torch.mean((fake_pred - soft_fake)**2)
        return loss
    
    # TODO: add auxilary loss for generator to enforce realistic colors
    def calc_gen_loss(self, gen_x):
        pred = self.forward(gen_x)
        return torch.mean((pred - 1)**2) # hard label at 1 for generator

    def calc_color_loss(self, x, gen_x):
        # print(x.shape)
        for i in range(x.shape[0]):
            blurred_img = cv2.medianBlur(np.asarray(x[i].permute(1, 2, 0).cpu(), dtype=np.uint8), 65) # median blur w/ cv2
            x[i] = torch.from_numpy(blurred_img).permute(2, 0, 1) # reshape dimensions
        return torch.mean((x - gen_x)**2) # float and int op will return float
        # TODO: find the general norm of this loss
        # TODO: add intermediate step to GPU device for speed
        # 2^7 * 2^7 = 2^14 = 16384
        # 16384 * 256^2 = 2^14 * 2^16 = 2^30
        # # we can scale by max possible deviation, 2^30 (calc general case)

class SiameseNet(nn.Module):
    def __init__(self, image_size, in_channels, num_feat=64, num_repeat=5, gamma=10):
        super().__init__()
        layers = []
        self.gamma = gamma 
        layers.append(nn.Conv2d(in_channels, num_feat, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))

        curr_dim = num_feat
        for _ in range(1, num_repeat):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            curr_dim = curr_dim * 2

        in_feat = image_size // 2**(num_repeat)

        self.main = nn.Sequential(*layers)
        self.linear = nn.Linear(curr_dim*in_feat**2, 1024)
        self.apply(weights_init())

    def _forward(self, x1, x2):
        latent1 = self.main(x1)
        latent2 = self.main(x2)
        latent1 = self.linear(latent1.flatten(1))
        latent2 = self.linear(latent2.flatten(1))
        return latent1, latent2
        
    def calc_loss(self, x1, x2):
        pairs = np.asarray(list(combinations(list(range(x1.size(0))), 2)))
        latent1, latent2 = self._forward(x1, x2)
        v1 = latent1[pairs[:,0]] - latent1[pairs[:,1]]
        v2 = latent2[pairs[:,0]] - latent2[pairs[:,1]]
        distance = F.mse_loss(v1, v2) - torch.mean(F.cosine_similarity(v1, v2))
        margin = self.margin_loss(v1)
        return distance + margin

    def margin_loss(self, v1):
        return torch.mean(F.relu(10 - torch.norm(v1, dim=1)))
