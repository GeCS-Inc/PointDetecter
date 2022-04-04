import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointDetecter(pl.LightningModule):

    def __init__(self, num_points):
        super().__init__()
        #self.save_hyperparameters()
        
        self.num_points = num_points
        self.lr = 0.1

        # ここにレイヤーを定義
        self.model = ResNet50(4, 10)
        
        # self.criterion を定義
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # ここで入力画像から点の座標を推定
        x = self.model.forward(x)
        x = torch.reshape(x, (x.shape[0], 5, 2))

        out = x
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-09)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer



class ResidualBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        dim_inter = outdim //4
        self.shortcut = self._shortcut(indim, outdim)
        
        self.relu = nn.ReLU()

        # 1 x 1 
        self.conv1 = nn.Conv2d(indim, dim_inter, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(dim_inter)

        # 3 x 3
        self.conv2 = nn.Conv2d(dim_inter, dim_inter, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim_inter)

        # 1 x 1
        self.conv3 = nn.Conv2d(dim_inter, outdim, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(outdim)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        out = self.relu(x+shortcut)

        return out

    def _shortcut(self, indim, outdim):
        if indim != outdim:
            return nn.Conv2d(indim, outdim, kernel_size=1, padding=0)
        else:
            return lambda x: x



class ResNet50(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(indim, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Block 1
        self.block0 = self._building_block(256, indim=64)
        self.block1 = nn.ModuleList([self._building_block(256) for _ in range(2)])
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=2)

        # Block 2
        self.block2 = nn.ModuleList([self._building_block(512) for _ in range(4)])
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=1, stride=2)

        # Block 3
        self.block3 = nn.ModuleList([self._building_block(1024) for _ in range(6)])
        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2)

        # Block 4
        self.block4 = nn.ModuleList([self._building_block(2048) for _ in range(3)])
        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Linear(2048, 1000)
        self.out = nn.Linear(1000, outdim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pool1(x)

        x = self.block0(x)
        for block in self.block1:
            x = block(x)
        x = self.conv2(x)

        for block in self.block2:
            x = block(x)
        x = self.conv3(x)

        for block in self.block3:
            x = block(x)
        x = self.conv4(x)

        for block in self.block4:
            x = block(x)

        x = self.avg_pool(x)
        x = self.fc(x)
        out = self.out(x)

        return out
    
    def _building_block(self, outdim, indim=None):
        if indim is None:
            indim = outdim
        return ResidualBlock(indim, outdim)


class GlobalAvgPool2d(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))

# test 
if __name__ == '__main__':
    model = PointDetecter(num_points=5)
    print(model.training_step((model.img, model.label), 0))