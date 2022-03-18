import torch
import torch.nn.init as init
import torchvision
import os
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
from os.path import isfile, join

def get_DataLoader_fromDataset(dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )
    return train_loader

def get_DataLoader_fromFolder(path, batch_size, transform=None):
    train_dataset = ImageFolder(
        root=path,
        transform= transforms.Compose([transforms.ToTensor(), 
                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]))
    return get_DataLoader_fromDataset(train_dataset, batch_size)

def get_indices(dataset,class_name):
    indices =  []
    try : 
        labels = dataset.labels
    except :
        labels = dataset.targets
    for i in range(len(labels)):
        if labels[i] == class_name:
            indices.append(i)
    return indices

def weights_init(init_type='kaiming'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun