import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss/len(val_loader), 100.*correct/total

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'Time: {epoch_time:.2f}s')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            print(f'New best accuracy: {best_acc:.2f}%')
            torch.save(model.state_dict(), 'best_model.pth')
        
        print('-' * 60)

# Example usage:
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
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    
    # Initialize model, criterion, optimizer, and scheduler
    model = YourModel().to(device)  # Replace with your model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=90) 