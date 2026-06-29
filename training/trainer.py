import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
import os
from .metrics import calculate_metrics

class Trainer:
    def __init__(self, model: nn.Module, device: str, logger: Any):
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                self.logger.info(f"Train Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")
                
        metrics = calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        metrics = calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics

    def save_checkpoint(self, state: dict, save_dir: str, filename: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, filename)
        torch.save(state, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
