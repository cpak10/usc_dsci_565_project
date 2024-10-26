from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2) # conv + pooling reduces from 256 -> 128
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2) # conv + pooling reduces from 128 -> 64
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # conv + pooling reduces from 64 -> 32
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) # conv + pooling reduces from 32 -> 16
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 16 * 16, 256) # 256 channels, 16x16 after 4 conv + pool
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten to vector for fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# create dataset
data_path = 'data/tensors.pt'
batch_size = 8
all_data = DataLoader(torch.load(data_path), batch_size=batch_size, shuffle=True)

# define simple CNN
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# perform training
for epoch in range(3): 
    running_loss = 0.0
    for i, data in enumerate(all_data, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics 10 times per epoch
        running_loss += loss.item()
        j = 10000 / batch_size // 10
        if i % j == j - 1:    
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / j:.3f}')
            running_loss = 0.0

print('Finished Training')