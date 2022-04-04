import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import glob
import json
import os

np.set_printoptions(precision=2)

class BoltPointDataset(Dataset):
    def __init__(self, dataset_dir, num_points=5, img_suffix="-img.png", gt_suffix="-gt.json", device="cuda"):
        super().__init__()
        
        self.num_points = num_points
        self.dataset_dir = dataset_dir
        self.img_suffix = img_suffix
        self.gt_suffix = gt_suffix
        self.device = device

        self.data = glob.glob(os.path.join(dataset_dir, f"*{img_suffix}"))

        self.resize = 100
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.resize, scale=(1.0, 1.0), ratio=(1.0, 1.0)), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Image
        img_path = self.data[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        # Ground Truth
        gt_path = img_path.replace(self.img_suffix, self.gt_suffix)
        with open(gt_path) as f:
            gt = json.load(f)
        gt = torch.tensor(gt)
        gt = gt / 1000.0

        return img, gt # 1x4x100x100, 5x2


# test
if __name__ == '__main__':
    dataset = BoltPointDataset(dataset_dir="/home/gecs/datasets/bolt-points-dataset")
    print(dataset.__getitem__(0)[0].shape,dataset.__getitem__(0)[1].shape, dataset.__getitem__(0))
