import torch.nn as nn


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.max2x2 = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(in_features=7 * 7 * 32, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        act1 = self.relu(self.conv1(x))
        act2 = self.max2x2(act1)
        act3 = self.relu(self.conv2(act2))
        act4 = self.max2x2(act3)
        flatten = act4.view(-1, 7 * 7 * 32)
        act5 = self.relu(self.fc1(flatten))
        act6 = self.relu(self.fc2(act5))
        out = self.out(act6)

        return out
