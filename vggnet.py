import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)

def cfg(depth):
    depth_lst = [11, 13, 16, 19]
    assert (depth in depth_lst), "Error : VGGnet depth should be either 11, 13, 16, 19"
    cf_dict = {
        '11': [
            64, 'mp',
            128, 'mp',
            256, 256, 'mp',
            512, 512, 'mp',
            512, 512, 'mp'],
        '13': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 'mp',
            512, 512, 'mp',
            512, 512, 'mp'
            ],
        '16': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 256, 'mp',
            512, 512, 512, 'mp',
            512, 512, 512, 'mp'
            ],
        '19': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 256, 256, 'mp',
            512, 512, 512, 512, 'mp',
            512, 512, 512, 512, 'mp'
            ],
    }

    return cf_dict[str(depth)]

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class VGG(nn.Module):
    def __init__(self, depth, num_outputs):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg(depth))
        self.fc0 = nn.Linear(2048, 256)
        self.fc1 = nn.Linear(256, num_outputs)
        self.fc2 = nn.Linear(num_outputs, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        y = F.relu(self.fc2(x))
        y = self.fc3(y)
        return x, y

    def _make_layers(self, cfg):
        layers = []
        in_planes = 3

        for x in cfg:
            if x == 'mp':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [conv3x3(in_planes, x), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_planes = x

        # After cfg convolution
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

if __name__ == "__main__":
    net = VGG(16, 10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
