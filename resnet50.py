import torch.nn as nn
import torch.optim as optim
from torch.optim.of_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, indim, outdim, is_first_resblock=False):
        super().__init__()

        dim_inter = outdim // 4
        self.conv1 = nn.Conv2d(indim, dim_inter, kernel_size=(1,1))
        self.bn1 = nn.BatchNorm2d(dim_inter)
        self.conv2 = nn.Conv2d(dim_inter, dim_inter, kernel_size=(3,3), stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(dim_inter)
        self.conv3 = nn.Conv2d(dim_inter, outdim, kernel_size=(1,1))
        self.bn3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()

        self.shortcut = self._shortcut(indim, outdim)

        self.processes = [self.conv1, self.bn1, self.relu, self.conv2, self.bn2, self.relu, self.conv3, self.bn3]

    def forward(self, x):
        shortcut = self.shortcut(x)
        for process in self.processes:
            x = process(x)
        
        x += shortcut
        out = self.relu(x)

        return out

    def _shortcut(self, indim, outdim):
        if indim != outdim:
            return nn.Conv2d(indim, outdim, kernel_size=(1,1), padding=0)
        else:
            return lambda x: x

class ResNet50(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)
        self.block0 = self._building_block(256, indim=64)
        self.block1 = nn.ModuleList([
            self._building_block(256) for _ in range(2)
        ])
        self.conv2 = nn.Conv2d(256, 512, kernel_size=(1,1), stride=(2,2))
        self.block2 = nn.ModuleList([
            self._building_block(512) for _ in range(4)
        ])
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1,1), stride=(2,2))
        self.block3 = nn.ModuleList([
            self._building_block(1024) for _ in range(6)
        ])
        self.conv4 = nn.Conv2d(1024, 2048, kerenel_size=(1,1), stride=(2,2))
        self.block4 = nn.ModuleList([
            self._building_block(2048) for _ in range(3)
        ])
        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Linear(2048, 1000)
        self.out = nn.Linear(1000, output_dim)

    def _building_block(self, outdim, indim=None):
        if indim is None:
            indim = outdim
        return Block(indim, outdim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.block0(out)
        for block in self.block1:
            out = block(out)
        out = self.conv2(out)
        for block in self.block2:
            out = block(out)
        out = self.conv3(out)
        for block in self.block3:
            out = block(out)
        out = self.conv4(out)
        for block in self.block4:
            out = block(out)
        out = self.avg_pool(out)
        out = self.fc(out)
        out = self.relu(out)
        out = self.out(out)
        
        return out

class GlobalAvgPool2d(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))
