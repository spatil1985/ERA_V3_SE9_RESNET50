from data.dataset import ImageNetDataModule

class Config:
    def __init__(self):
        self.data_dir = '/mnt/efs/imagenet/processed'
        self.batch_size = 32
        self.num_workers = 4

config = Config()
data_module = ImageNetDataModule(config)
data_module.setup()
train_loader, val_loader = data_module.get_dataloaders()
