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

source_path = './final_source128'
target_path = './final_target128'

# create torch dataloaders for source and target domains
source_loader = get_DataLoader_fromFolder(source_path, config['batch_size'])
target_loader = get_DataLoader_fromFolder(target_path, config['batch_size'])

# NOTE: source and target domains are not in any correspondence to one another
# NOTE: forgot to normalize source domain images

model = TravelGan(config, logger)
model.load()
model.train(source_loader, target_loader)
