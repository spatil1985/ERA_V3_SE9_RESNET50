import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict
import logging

class MetricLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f'{prefix}/{name}', value, step)
            logging.info(f'{prefix}/{name}: {value:.4f}')
    
    def close(self):
        self.writer.close() 