import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet import ResNet50  # Import your ResNet model
from trainer import Trainer

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = ResNet50(num_classes=1000).to(device)  # ImageNet has 1000 classes
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder('/mnt/efs/imagenet/processed/train', train_transform)
    val_dataset = datasets.ImageFolder('/mnt/efs/imagenet/processed/val', val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )
    
    # Define training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.1
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir='checkpoints'
    )
    
    # Start training
    trainer.train(num_epochs=90)

if __name__ == '__main__':
    main() 