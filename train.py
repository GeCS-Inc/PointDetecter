from argparse import Namespace
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader
from data import BoltPointDataset
from options import get_options
from model import PointDetecter

def train_model(args: Namespace, epoch=300) -> None:
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = args.batch_size
    num_points = 5
    
    dataset    = BoltPointDataset(dataset_dir="/home/gecs/datasets/bolt-points-dataset", num_points=num_points)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    model = PointDetecter(num_points=num_points)
    #model.to(device)

    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    trainer = Trainer(
        gpus=args.gpus, 
        max_epochs=epoch
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    options = get_options() # GPU, batch size, logger(TorF)
    train_model(options)