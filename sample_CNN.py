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


# Create FC Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 150)
        self.fc2 = nn.Linear(150, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


my_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])


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

        self.initialize_weights()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)

        # Now its a 4-dimensional tensor with batches
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

    # For initialization
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(root='dataset/',
                               train=True,
                               transform=my_transforms,
                               download=True
                               )
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True
                          )

test_dataset = datasets.MNIST(root='dataset/',
                              train=False,
                              transform=my_transforms,
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
# Train Network
for epoch in range(num_epochs):
    for batch_ix, (data, targets) in tqdm(enumerate(train_loader),
                                          total=len(train_loader),
                                          leave=False):
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
