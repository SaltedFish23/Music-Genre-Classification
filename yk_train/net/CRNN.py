from torch import nn

import torch
from torch import nn

# Assume input size is 96x1366

class Net(nn.Module):
    def __init__(self, include_top=True):
        super(Net, self).__init__()
        self.pad = nn.ConstantPad2d((0, 37, 0, 0), 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 64, (3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.elu = nn.ELU()
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(64, 128, (3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((3, 3), stride=(3, 3))
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(128, 128, (3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((4, 4), stride=(4, 4))
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(128, 128, (3, 3), padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d((4, 4), stride=(4, 4))
        self.dropout4 = nn.Dropout(0.1)
        self.gru1 = nn.GRU(128, 32, batch_first=True)
        self.gru2 = nn.GRU(32, 32, batch_first=True)
        self.dropout5 = nn.Dropout(0.3)
        if include_top:
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(448, 10)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.elu(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), -1, 128)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.dropout5(x)
        if hasattr(self, 'fc'):
            x = self.flatten(x)
            # print(x.shape)
            x = self.fc(x)
            x = self.sigmoid(x)
        return x


class sim_net(nn.Module):
    def __init__(self):
            super(sim_net, self).__init__()
            self.pad = nn.ConstantPad2d((0, 37, 0, 0), 0)
            self.bn0 = nn.BatchNorm2d(1)
            self.conv1 = nn.Conv2d(1, 64, (3, 3), padding='same')
            self.bn1 = nn.BatchNorm2d(64)
            self.elu = nn.ELU()
            self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
            self.dropout1 = nn.Dropout(0.1)
            self.conv2 = nn.Conv2d(64, 128, (3, 3), padding='same')
            self.bn2 = nn.BatchNorm2d(128)
            self.pool2 = nn.MaxPool2d((3, 3), stride=(3, 3))
            self.dropout2 = nn.Dropout(0.1)
            self.conv3 = nn.Conv2d(128, 128, (3, 3), padding='same')
            self.bn3 = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d((4, 4), stride=(4, 4))
            self.dropout3 = nn.Dropout(0.1)
            self.gru1 = nn.GRU(128, 32, batch_first=True)
            self.gru2 = nn.GRU(32, 32, batch_first=True)
            self.dropout5 = nn.Dropout(0.3)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(7424, 10)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), -1, 128)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.dropout5(x)
        if hasattr(self, 'fc'):
            x = self.flatten(x)
            # print(x.shape)
            x = self.fc(x)
            x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    from torchsummary import summary

    model = sim_net()
    summary(model, (1, 96, 1366))
    print(model(torch.randn(1, 1, 96, 1366)).shape)
    print(model(torch.randn(1, 1, 96, 1366)))