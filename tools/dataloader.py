import os

import imageio
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

data_location = '/data'


class CustomDataset(Dataset):
    def __init__(self, data, celeba=False):
        self.data = data
        self.celeba = celeba

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, 0.


def get_loaders(imgstr, resolution=178, batch_size=1, use_train_set=False):
    """
    Load dataloaders for an image dataset, center-cropped to a resolution.
    """
    if imgstr == 'celeba':
        T_base = T.Compose([T.CenterCrop(resolution), T.ToTensor()])
        trainset = datasets.CelebA(data_location, split='train', target_type='attr', transform=T_base)
        testset = datasets.CelebA(data_location, split='test', target_type='attr', transform=T_base)

    elif imgstr == 'sdf':
        assert resolution == 178
        sdf = np.load(f'{data_location}/data_2d_sdf.npz')

        # numpy and torch images have different channel axis
        sdf_train = np.transpose(sdf['train_data.npy'], (0, 3, 1, 2)).astype(np.float32) / 255.
        sdf_test = np.transpose(sdf['test_data.npy'], (0, 3, 1, 2)).astype(np.float32) / 255.

        trainset = CustomDataset(torch.from_numpy(sdf_train).float())
        testset = CustomDataset(torch.from_numpy(sdf_test).float())

    elif imgstr == 'imagenette':
        assert resolution == 178

        T_base = T.Compose([T.Resize(200), T.CenterCrop(resolution), T.ToTensor()])
        train_dir = os.path.join(data_location, 'imagenette', 'train')
        trainset = datasets.ImageFolder(train_dir, transform=T_base)
        test_dir = os.path.join(data_location, 'imagenette', 'val')
        testset = datasets.ImageFolder(test_dir, transform=T_base)

    else:
        raise NotImplementedError()

    shuffle = False if use_train_set else True
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return trainloader, testloader
