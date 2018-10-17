import torch
import torch.nn as nn
import torch.nn.functional as F  

class FCN(nn.Module):
    def __init__(self, width):
        super(FCN, self).__init__()
        self.fc0 = nn.Linear(9, 512)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 4096)
        self.conv0 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = F.relu
        self.sigmoid = F.sigmoid
        self.upsample = nn.Upsample(scale_factor=2)
        self.init()

    def init(self):
        nn.init.kaiming_uniform_(self.fc0.weight)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.conv0.weight)
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.kaiming_uniform_(self.conv3.weight)
        
    def forward(self, x):
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = x.view(-1, 16, 16, 16)
        x = self.upsample(x)
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.upsample(x)
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x.reshape(-1, 64, 64)
