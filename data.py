import os
import numpy as np

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

#from torchsample.datasets import TensorDataset

from utils import Invert
from utils import Gray


DATA_PATH = 'data'
def data_path(folder):
    return os.path.join(DATA_PATH, folder)


class PatchDataset:

    def __init__(self, dataset, patch_size):
        self.dataset = dataset
        self.patch_size = patch_size
        self.nc, self.h, self.w = dataset[0][0].size()
        self.nb = len(dataset)

    def __getitem__(self, i):
        im, y = self.dataset[i]
        #y = np.random.randint(0, self.h // self.patch_size) * self.patch_size
        #x = np.random.randint(0, self.w // self.patch_size) * self.patch_size
        y = np.random.randint(0, self.h - self.patch_size + 1)
        x = np.random.randint(0, self.w - self.patch_size + 1)
        patch = im[:, x:x + self.patch_size, y:y + self.patch_size]
        return patch, y

    def __len__(self):
        return len(self.dataset)


class H5Dataset:

    def __init__(self, X, y, transform=lambda x:x):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(torch.from_numpy(self.X[index])), self.y[index]

    def __len__(self):
        return len(self.X)

def load_dataset(dataset_name, split='full'):
    if dataset_name == 'mnist':
        dataset = dset.MNIST(
            root=data_path('mnist'), 
            download=True,
            transform=transforms.Compose([
                transforms.Scale(32),
                transforms.ToTensor(),
            ])
        )
        return dataset
    elif dataset_name == 'celeba_h5':
        dataset = _load_h5('data/celeba64_align.h5')
        return dataset
    elif dataset_name == 'coco':
        dataset = dset.ImageFolder(root=data_path('coco'),
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'coco_256':
        dataset = dset.ImageFolder(root=data_path('coco'),
            transform=transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'footwear':
        dataset = dset.ImageFolder(root=data_path('shoes/ut-zap50k-images'),
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'footwear_32':
        dataset = dset.ImageFolder(root=data_path('shoes/ut-zap50k-images'),
            transform=transforms.Compose([
            transforms.Scale(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'celeba':
        dataset = dset.ImageFolder(root=data_path('celeba'),
            transform=transforms.Compose([
            transforms.Scale(78),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'celeba_128':
        dataset = dset.ImageFolder(root=data_path('celeba'),
            transform=transforms.Compose([
            transforms.Scale(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'celeba_256':
        dataset = dset.ImageFolder(root=data_path('celeba'),
            transform=transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
 
    elif dataset_name == 'birds':
        dataset = dset.ImageFolder(root=data_path('birdsfull'),
            transform=transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]))
        return dataset
    elif dataset_name == 'fonts':
        dataset = dset.ImageFolder(root=data_path('fonts/full'),
            transform=transforms.Compose([
            transforms.ToTensor(),
            Invert(),
            Gray(),
         ]))
        return dataset
    else:
        dataset = dset.ImageFolder(root=data_path(dataset_name),
            transform=transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]))
        return dataset
def _load_npy(filename):
    data = np.load(filename)
    X = torch.from_numpy(data['X']).float()
    if 'y' in data:
        y  = torch.from_numpy(data['y'])
    else:
        y = torch.zeros(len(X))
    X /= X.max()
    X = X * 2 - 1
    print(X.min(), X.max())
    dataset = dset.TensorDataset(
        inputs=X, 
        targets=y,
    )
    return dataset


def _load_h5(filename):
    import h5py
    data = h5py.File(filename, 'r')
    X = data['X']
    if 'y' in data:
        y  = (data['y'])
    else:
        y = np.zeros(len(X))
    dataset = H5Dataset(X, y, transform=lambda u:2*(u.float()/255.)-1)
    return dataset
