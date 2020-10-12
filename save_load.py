# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10
load_model = True


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=8,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1)
                               )  # SAME convolution

        self.pool = nn.MaxPool2d(kernel_size=(2, 2),
                                 stride=(2, 2)
                                 )

        self.conv2 = nn.Conv2d(in_channels=8,
                               out_channels=16,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1)
                               )  # SAME convolution

        # 16 for filters, 28x28 -> 7x7
        self.fc1 = nn.Linear(16 * 7 * 7,
                             num_classes)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)

        # Now its a 4-dimensional tensor with batches
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


def save_checkpoint(state, filname="my_checkpoint.pth.tar"):
    print('saving checkpoint')
    torch.save(state, filname)


def load_checkpoint(checkpoint):
    print('loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Load Data
train_dataset = datasets.MNIST(root='dataset/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True
                               )
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True
                          )

test_dataset = datasets.MNIST(root='dataset/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True
                              )
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True
                         )

# Initialize network
model = CNN(in_channels=in_channels,
            num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate)
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

# Train Network
for epoch in range(num_epochs):
    losses = []

    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_ix, (data, targets) in enumerate(train_loader):
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
