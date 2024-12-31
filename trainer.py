import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import os

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_dir='checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return running_loss/len(self.train_loader), 100.*correct/total

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return running_loss/len(self.val_loader), 100.*correct/total

    def train(self, num_epochs):
        best_acc = 0.0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            if self.scheduler:
                self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f'\nEpoch {epoch} Summary:')
            print(f'Time: {epoch_time:.2f}s')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                print(f'New best accuracy: {best_acc:.2f}%')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_acc': best_acc,
                }, os.path.join(self.save_dir, 'best_model.pth'))
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_acc': best_acc,
                }, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
            
            print('-' * 60)

# Example usage
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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