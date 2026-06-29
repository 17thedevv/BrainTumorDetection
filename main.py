import argparse
import os
import torch
from configs.config import Config
from datasets.data_module import DataModule
from models.cnn import BaselineCNN
from training.trainer import Trainer
from utils.logger import get_logger
from utils.seed import set_seed

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Detection Pipeline")
    parser.add_argument('--config', type=str, default='experiments/baseline.yaml', help='Path to config yaml file')
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        return

    config = Config.from_yaml(args.config)
    
    # Initialize logger
    logger = get_logger(name=config.project_name, log_file=f"{config.experiment_name}.log")
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    logger.info(f"Random seed set to {config.seed}")
    
    # Initialize DataModule
    data_module = DataModule(config.data)
    
    # Get DataLoaders
    train_loader = data_module.get_train_dataloader()
    val_loader = data_module.get_val_dataloader()
    
    logger.info(f"Initialized Dataset 1 (Supervised) - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = BaselineCNN(
        name=config.model.name, 
        num_classes=config.model.num_classes, 
        pretrained=config.model.pretrained
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )
    
    # Trainer
    trainer = Trainer(model=model, device=device, logger=logger)
    
    # Training Loop
    best_val_loss = float('inf')
    save_dir = "saved_model"
    
    logger.info("Starting Phase 2: Supervised Baseline Training")
    
    for epoch in range(1, config.training.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{config.training.epochs} ---")
        
        train_metrics = trainer.train_epoch(train_loader, optimizer)
        logger.info(f"Train Metrics: Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        
        val_metrics = trainer.validate(val_loader)
        logger.info(f"Val Metrics: Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Checkpointing
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            trainer.save_checkpoint(
                state={
                    'epoch': epoch,
                    'state_dict': trainer.model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict(),
                },
                save_dir=save_dir,
                filename="best_baseline_model.pth"
            )

    logger.info("Phase 2 Training Completed successfully!")

if __name__ == '__main__':
    main()
