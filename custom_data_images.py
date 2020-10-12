# Imports
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from skimage import io
import torchvision
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 32
num_epochs = 1


# Load Data
class C_n_D_dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)  # 25000

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(self.annotations.iloc[index, 0])

        if self.transform:
            image = self.transform(image)

        return image, y_label


dataset = C_n_D_dataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resize',
                        transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)


def save_checkpoint(state, filname="my_checkpoint.pth.tar"):
    print('saving checkpoint')
    torch.save(state, filname)


def load_checkpoint(checkpoint):
    print('loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    if epoch % 3 == 1:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_ix, (data, targets) in enumerate(train_loader):
        print('Yo')
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Grad descent/Adam step
        optimizer.step()

    print(f'Epoch {epoch} has been finished')


# Check accuracy on training and test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            # 64x10, 0

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with '
              f'accuracy {float(num_correct) / float(num_samples) * 100:.2f}'
              )

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
