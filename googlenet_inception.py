import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Is it possible to use BERT or ROBERTA for time-series classification
# XOR problem (there is a add problem)?

# research question
# How do you mask input? XLNET
# LogSparse test for simple task (filling gaps task)
# L*2 L(LogL2), using much fewer attention links
# the most elegant — within approximation

