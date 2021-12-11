import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init

class Conv2DNet(nn.Module):
    def __init__(self, feature_type):
        super(Conv2DNet, self).__init__()
        self.n_classes = 2
        self.stride = (4, 2) if feature_type == 'spec' else 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=self.stride, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, stride=self.stride, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=5, stride=self.stride, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU())

        self.fc1 = nn.Linear(1200, 256)
        self.fc2 = nn.Linear(256, self.n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class Conv1DNet(nn.Module):
    def __init__(self):
        super(Conv1DNet, self).__init__()
        self.n_classes = 2
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 10, kernel_size=25, stride=3),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(10, 20, kernel_size=25, stride=3),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(20, 40, kernel_size=25, stride=3),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.MaxPool1d(3)
        )

        self.fc = nn.Linear(1040, self.n_classes)

    def forward(self, x):
        x = F.pad(x, [12, 13], mode='constant')
        out = self.layer1(x)
        out = F.pad(out, [14, 15], mode='constant')
        out = self.layer2(out)
        out = F.pad(out, [16, 17], mode='constant')
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out







if __name__=='__main__':
    import numpy as np
    
    model = Conv1DNet()
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print(params)
    
    model = Conv2DNet('mel_spec')
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print(params)