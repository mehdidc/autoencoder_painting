from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def norm_min_max(x):
    xp = x.view(x.size(0), -1)
    xmin, _ = xp.min(dim=1, keepdim=True)
    xmax, _ = xp.max(dim=1, keepdim=True)
    xp = (xp - xmin) / (xmax - xmin + 1e-7)
    return xp.view(x.size())


class AE(nn.Module):

    def __init__(self, nc=1, ndf=64, latent_size=None, w=64, act='sigmoid', objective='vae', nb_iter_generation=1):
        super().__init__()
        self.act = act
        self.latent_size = latent_size
        self.ndf = ndf
        self.objective = objective
        self.nb_iter_generation = 1
        self.w = w
        self.nc = nc
        nb_blocks = int(np.log(w)/np.log(2)) - 3
        nf = ndf
        layers = [
            nn.Conv2d(nc, nf, 4, 2, 1, bias=True),
            nn.ReLU(True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=True),
                #nn.BatchNorm2d(nf * 2),
                nn.ReLU(True),
            ])
            nf = nf * 2

        self.encoder = nn.Sequential(*layers)
        nf = ndf 
        layers = [
            nn.Conv2d(nc, nf, 4, 2, 1, bias=True),
            nn.ReLU(True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=True),
                #nn.BatchNorm2d(nf * 2),
                nn.ReLU(True),
            ])
            nf = nf * 2

        self.loss = nn.Sequential(*layers)
        wl = w // 2**(nb_blocks+1)
        self.latent = nn.Sequential(
            nn.Linear(nf * wl * wl, latent_size * 2),
        )
        self.post_latent = nn.Sequential(
            nn.Linear(latent_size, nf * wl * wl)
        )
        self.post_latent_shape = (nf, wl, wl)
        layers = []
        for _ in range(nb_blocks):
            layers.extend([
                nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1, bias=True),
                #nn.BatchNorm2d(nf // 2),
                nn.ReLU(True),
            ])
            nf = nf // 2
        layers.append(
            nn.ConvTranspose2d(nf,  nc, 4, 2, 1, bias=True)
        )
        self.decoder = nn.Sequential(*layers)
        self.apply(weights_init)
    
    def parameters(self):
        return chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.latent.parameters(),
            self.post_latent.parameters(),
        )
    def sample(self, nb_examples=1):
        device = next(self.parameters()).device
        if self.objective == 'vae':
            z = torch.randn(nb_examples, self.latent_size).to(device) 
            x = self.decode(z)
            return x
        else:
            x = torch.rand(nb_examples, self.nc, self.w, self.w).to(device) 
            for _ in range(self.nb_iter_generation):
                x, _, _ = self(x)
            return x

    def decode(self, h):
        x = self.post_latent(h)
        x = x.view((x.size(0),) + self.post_latent_shape)
        xrec = self.decoder(x)
        if self.act == 'sigmoid':
            return nn.Sigmoid()(xrec)
        return xrec
    
    def forward(self, input):
        x = self.encoder(input)
        x = x.view(x.size(0), -1)
        h = self.latent(x)
        mu, logvar = h[:, 0:self.latent_size], h[:, self.latent_size:]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        if self.objective == 'vae':
            h = mu + eps * std 
        else:
            h = mu
        xrec = self.decode(h)
        return xrec, mu, logvar
    
    def rec(self, input):
        xrec, mu, logvar = self(input)
        return xrec

    def loss_function(self, x, xrec, mu, logvar):
        #percep = ((self.loss(xrec) - self.loss(x))**2).sum()
        percep = 0
        x = x.view(x.size(0), -1)
        xrec = xrec.view(xrec.size(0), -1)
        rec = ((xrec - x) ** 2).sum(1).mean()
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        return rec + percep, kld

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname == 'Linear':
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)


