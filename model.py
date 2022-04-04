import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_pretrained_model(model_name, pretrained=False):
    if model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        model = nn.Sequential(
            *list(model.children())[:-2])
        fc_in_features = 512
        fn = model.__call__

    if model_name.startswith("resnet"):
        model = getattr(models, model_name)(pretrained=pretrained)
        fc_in_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-2])
        fn = model.__call__
    return model, fc_in_features, fn


class PointDetecter(pl.LightningModule):

    def __init__(self, num_points, base_model="resnet50", dropout_rate=0.2):
        super().__init__()
        #self.save_hyperparameters()
        
        self.num_points = num_points
        self.lr = 1e-3

        # ここにレイヤーを定義
        self.model, fc_in_features, self.extract_feature = get_pretrained_model(base_model, pretrained=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(fc_in_features, self.num_points*2)
        # self.criterion を定義
        self.criterion = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.extract_feature(x)
        x = self.swish(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = x.view((-1, self.num_points, 2))
        return x

    def swish(self, x):
        return x * torch.sigmoid(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-09)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# test 
if __name__ == '__main__':
    model = PointDetecter(num_points=5)
    print(model.training_step((model.img, model.label), 0))