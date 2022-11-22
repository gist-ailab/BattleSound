import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init

# CNN 2D
class Conv2DNet(nn.Module):
    def __init__(self, feature_type, duration):
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

        if duration == 0.5:
            in_channel = 1440
        elif duration == 2.0:
            in_channel = 5040
        elif duration == 4.0:
            in_channel = 9840
        elif duration == 8.0:
            in_channel = 19440
        else:
            raise('Error!')
        
        self.fc1 = nn.Linear(in_channel, 256)
        self.fc2 = nn.Linear(256, self.n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# CNN 1D
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


# CRNN
class CRNN_2D(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_mels=41, num_classes=2):
        """
        Args:
          n_mels (float): number of mel bin
          n_class (int): number of class
        """
        super(CRNN_2D, self).__init__()
        # Spectrogram
        self.stride = 2
        self.layer1 = nn.Sequential(
            nn.Conv1d(num_mels, hidden_dim, kernel_size=5, stride=self.stride, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=self.stride, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=self.stride, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU())

        # Predict tag using the aggregated features.
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
            

    def forward(self, x):
        B, C, M, T = x.shape
        x = x.reshape(B, C*M, T) # for 1D conv
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.permute(0, 2, 1).contiguous()
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# Bi-LSTM
class CRNN_1D(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, num_classes=2):
        super(CRNN_1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=25, stride=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=25, stride=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=25, stride=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(3)
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = F.pad(x, [12, 13], mode='constant')
        x = self.layer1(x)
        x = F.pad(x, [14, 15], mode='constant')
        x = self.layer2(x)
        x = F.pad(x, [16, 17], mode='constant')
        x = self.layer3(x)
        x = x.permute(0, 2, 1).contiguous() # Batch x Time x Feature

        # Forward propagate LSTM
        out, _ = self.lstm(x)  # shape = (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out



if __name__=='__main__':
    import numpy as np
    
    # Input
    mel_spec = torch.ones([2, 1, 41,41]).float()
    signal = torch.ones([2, 8000]).float()

    # Model
    model = Conv1DNet(hidden_dim=64)
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print(params)

    model = CRNN_1D(hidden_dim=64, num_layers=2)
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print(params)
    
    model = Conv2DNet('mel_spec', duration=0.5)
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print(params)

    model = CRNN_2D(hidden_dim=64, num_layers=2, num_classes=2)
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print(params)


