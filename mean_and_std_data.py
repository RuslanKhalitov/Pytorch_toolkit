# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import random
import numpy as np


def deterministic(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cuda
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


deterministic(42)

# Load data
train_dataset = datasets.MNIST(root='dataset/', train=True,
                                 transform=transforms.ToTensor(),
                                 download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


def get_mean_std(loader):
    # VAR[x] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])  # not doing for channels
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


mean, std = get_mean_std(train_loader)
print(mean)
print(std)
