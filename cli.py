import numpy as np
import os
from clize import run
import shutil
from skimage.io import imsave

import torch
import torch.optim as optim

from model import AE

from viz import grid_of_images_default
from data import load_dataset, PatchDataset


def train(*,
          folder=None,
          dataset='mnist',
          patch_size=8,
          resume=False,
          log_interval=1,
          device='cpu',
          objective='vae',
          batch_size=64,
          nz=100,
          lr=0.001,
          num_workers=1,
          nb_filters=64,
          nb_draw_layers=1):
    if folder is None:
        folder = f'results/{dataset}/{patch_size}x{patch_size}'
    try:
        os.makedirs(folder)
    except Exception:
        pass
    act = 'sigmoid'
    nb_epochs = 3000
    dataset = load_dataset(dataset, split='train')
    if patch_size is not None:
        patch_size = int(patch_size)
        dataset = PatchDataset(dataset, patch_size)
    x0, _ = dataset[0]
    nc = x0.size(0)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    if resume:
        net = torch.load('{}/net.th'.format(folder))
    else:
        net = AE(
            latent_size=nz, 
            nc=nc, 
            w=patch_size,
            ndf=nb_filters, 
            act=act,
            objective=objective,
        )
    opt = optim.Adam(net.parameters(), lr=lr)
    net = net.to(device)
    niter = 0
    for epoch in range(nb_epochs):
        for i, (X, _), in enumerate(dataloader):
            net.zero_grad()
            X = X.to(device)
            Xrec, mu, logvar = net(X)
            rec, kld = net.loss_function(X, Xrec, mu, logvar)
            loss = rec + kld
            loss.backward()
            opt.step()
            if niter % log_interval == 0:
                print(f'Epoch: {epoch:05d}/{nb_epochs:05d} iter: {niter:05d} loss: {loss.item():.2f} rec: {rec.item():.2f} kld:{kld.item():.2f}')
            if niter % 100 == 0:
                Xsamples = net.sample(nb_examples=100)
                X = 0.5 * (X + 1) if act == 'tanh' else X
                Xrecs = 0.5 * (Xrec + 1) if act == 'tanh' else Xrec
                Xsamples = 0.5 * (Xsamples + 1) if act == 'tanh' else Xsamples
                X = X.detach().to('cpu').numpy()
                Xrecs = Xrecs.detach().to('cpu').numpy()
                Xsamples = Xsamples.detach().to('cpu').numpy()
                imsave(f'{folder}/real_samples.png', grid_of_images_default(X))
                imsave(f'{folder}/rec_samples.png', grid_of_images_default(Xrecs))
                imsave(f'{folder}/fake_samples.png', grid_of_images_default(Xsamples))
                torch.save(net, '{}/net.th'.format(folder))
            niter += 1

def paint(models, *, output_size=256, nb_iter=200, seed=42, device='cpu'):
    rng = np.random.RandomState(seed)
    models = [torch.load(model) for model in models.split(',')]
    w = max(map(lambda m:m.w, models))
    res = torch.rand((1, 1, output_size + 2 * w, output_size + 2 * w)).to(device)
    for it in range(nb_iter):
        y = rng.randint(0, output_size + 1)
        x = rng.randint(0, output_size + 1)
        m = rng.randint(0, len(models))
        ae = models[m]
        w = ae.w
        v = res[:, :, y:y+w, x:x+w]
        res[:, :, y:y+w, x:x+w] =ae.rec(v) 
        if it % 100 == 0:
            im = res[0,0, w:-w, w:-w].to('cpu').detach().numpy()
            imsave('out.png', im)



if __name__ == '__main__':
    run([train, paint])
