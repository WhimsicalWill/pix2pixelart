import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.data as Data 

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from utils import get_DataLoader_fromFolder
from config.parser import parser
from src.travelgan import TravelGan
from logs.logger import Logger

parser = parser('config/config.ini')
config = parser.to_dict()
logger = Logger(config['logfile'], config['enable_wandb'])

print("Creating Dataloaders")
# create torch dataloaders for source and target domains
source_loader = get_DataLoader_fromFolder(source_path, config['batch_size'])
target_loader = get_DataLoader_fromFolder(target_path, config['batch_size'])

# NOTE: source and target domains are not in any correspondence to one another
# NOTE: forgot to normalize source domain images

print("Creating model")
model = TravelGan(config, logger)

# print("Loading saved model")
# model.load()

print("The model is being trained!")
model.train(source_loader, target_loader)
