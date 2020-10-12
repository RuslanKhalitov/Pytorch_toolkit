# Imports
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Load pretrained model and modify it
model = torchvision.models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 10))

model.to(device)


def save_checkpoint(state, filname="my_checkpoint.pth.tar"):
    print('saving checkpoint')
    torch.save(state, filname)


def load_checkpoint(checkpoint):
    print('loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Load Data
train_dataset = datasets.CIFAR10(root='dataset/',
                                 train=True,
                                 transform=transforms.ToTensor(),
                                 download=True
                                 )
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True
                          )

test_dataset = datasets.CIFAR10(root='dataset/',
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True
                                )
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True
                         )



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
