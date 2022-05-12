import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.data as Data 

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from src.utils import get_DataLoader_fromFolder
from src.travelgan import TravelGan
from config.parser import parser
from logs.logger import Logger

parser = parser('config/config.ini')
config = parser.to_dict()
logger = Logger(config['logfile'], config['enable_wandb'])

# paths to source and target domains
source_path = '../../final_source128'
target_path = '../../final_target128'

print("Creating Dataloaders")
# create torch dataloaders for source and target domains
source_loader = get_DataLoader_fromFolder(source_path, config['batch_size'])
target_loader = get_DataLoader_fromFolder(target_path, config['batch_size'])

print("Creating model")
model = TravelGan(config, logger)

# print("Loading saved model")
model.load()

print("The model is being trained!")
model.train(source_loader, target_loader)
