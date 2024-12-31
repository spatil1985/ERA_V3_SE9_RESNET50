import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

class ImageNetDataModule:
    def __init__(self, config):
        self.config = config
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _verify_dataset(self):
        """Verify that the dataset directory exists and has the expected structure"""
        if not os.path.exists(self.config.data_dir):
            raise RuntimeError(f"Dataset directory {self.config.data_dir} does not exist")
        
        train_dir = os.path.join(self.config.data_dir, 'train')
        val_dir = os.path.join(self.config.data_dir, 'val')
        
        if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
            raise RuntimeError(
                f"Expected ImageNet directory structure:\n"
                f"{self.config.data_dir}/\n"
                f"  train/\n"
                f"    n01440764/\n"
                f"    n01443537/\n"
                f"    ...\n"
                f"  val/\n"
                f"    n01440764/\n"
                f"    n01443537/\n"
                f"    ..."
            )
    
    def setup(self):
        """Setup the ImageNet dataset"""
        try:
            self._verify_dataset()
            
            self.train_dataset = datasets.ImageFolder(
                os.path.join(self.config.data_dir, 'train'),
                transform=self.train_transforms
            )
            
            self.val_dataset = datasets.ImageFolder(
                os.path.join(self.config.data_dir, 'val'),
                transform=self.val_transforms
            )
            
            print(f"Dataset loaded successfully:")
            print(f"Training samples: {len(self.train_dataset)}")
            print(f"Validation samples: {len(self.val_dataset)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ImageNet dataset: {str(e)}")
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader