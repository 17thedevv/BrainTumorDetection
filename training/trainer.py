import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any
import os
from tqdm import tqdm
from .metrics import calculate_metrics

class Trainer:
    def __init__(self, model: nn.Module, device: str, logger: Any):
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Automatic Mixed Precision (AMP) - only enabled on CUDA
        self.use_amp = device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        if self.use_amp:
            self.logger.info("Automatic Mixed Precision (AMP) enabled.")
        
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        start_time = time.time()
        
        pbar = tqdm(dataloader, desc="  Training", unit="batch", leave=False,
                    bar_format="{l_bar}{bar:30}{r_bar}")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
                
        elapsed = time.time() - start_time
        metrics = calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(dataloader)
        metrics['epoch_time'] = elapsed
        
        return metrics

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc="  Validating", unit="batch", leave=False,
                    bar_format="{l_bar}{bar:30}{r_bar}")
        
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
        metrics = calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics

    def save_checkpoint(self, state: dict, save_dir: str, filename: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, filename)
        torch.save(state, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")

