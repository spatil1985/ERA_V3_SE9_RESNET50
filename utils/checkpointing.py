import os
import torch
from typing import Dict, Any

class CheckpointHandler:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, state: Dict[str, Any], epoch: int, is_best: bool = False):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        return torch.load(path) 