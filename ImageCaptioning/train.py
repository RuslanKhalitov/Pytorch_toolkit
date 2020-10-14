# Idea
# First: Pretrained CNN -> Embedding
# Second: Embedding -> LSTM -> output

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from loader import get_loader
from model import CNNtoRNN

