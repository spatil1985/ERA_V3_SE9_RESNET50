import torch
import torchvision.models as models

def ResNet50(num_classes: int = 1000) -> torch.nn.Module:
    """
    Creates a ResNet50 model from torchvision.
    
    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
        
    Returns:
        ResNet50 model with random initialization
    """
    # Create ResNet50 with random initialization (weights=None)
    model = models.resnet50(weights=None)
    
    # Modify the final fully connected layer if num_classes is different
    if num_classes != 1000:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
    return model 