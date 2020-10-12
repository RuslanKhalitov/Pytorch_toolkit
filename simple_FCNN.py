#Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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

model = NN(784, 10)
x = torch.randn(64, 784)
print(model(x).shape)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

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
model = NN(input_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate)
# Train Network
for epoch in range(num_epochs):
    for batch_ix, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        # Get the correct shape
        data = data.reshape(data.shape[0], -1)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        #Grad descent/Adam step
        optimizer.step()


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
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            # 64x10, 0

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with '
              f'accuracy {float(num_correct)/float(num_samples)*100:.2f}'
              )

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

