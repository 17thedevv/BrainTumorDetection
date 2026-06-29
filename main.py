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
    mode = config.active_mode
    mode_name = "Development" if config.development.enabled else "Research"
    
    # Initialize logger
    logger = get_logger(name=config.project_name, log_file=f"{config.experiment_name}.log")
    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Running in [{mode_name} Mode] | epochs={mode.epochs} | image_size={mode.image_size} | batch_size={mode.batch_size} | subset_ratio={mode.subset_ratio}")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    logger.info(f"Random seed set to {config.seed}")
    
    # Initialize DataModule
    data_module = DataModule(config)
    
    # Get DataLoaders
    train_loader = data_module.get_train_dataloader()
    val_loader = data_module.get_val_dataloader()
    
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    logger.info(f"Train samples: {train_samples} ({len(train_loader)} batches) | Val samples: {val_samples} ({len(val_loader)} batches)")
    
    # Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = BaselineCNN(
        name=config.model.name, 
        num_classes=config.model.num_classes, 
        pretrained=config.model.pretrained
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: CustomCNN | Trainable parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )
    
    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=mode.epochs)
    
    # Trainer
    trainer = Trainer(model=model, device=device, logger=logger)
    
    # Early Stopping settings
    early_stopping_patience = getattr(mode, 'early_stopping_patience', 7)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    save_dir = "saved_model"
    
    logger.info(f"Starting Phase 2: Supervised Baseline Training | Early stopping patience: {early_stopping_patience}")
    
    for epoch in range(1, mode.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{mode.epochs} ---")
        
        train_metrics = trainer.train_epoch(train_loader, optimizer)
        logger.info(
            f"[Train] Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"F1: {train_metrics['f1']:.4f} | "
            f"Prec: {train_metrics['precision']:.4f} | "
            f"Rec: {train_metrics['recall']:.4f} | "
            f"Time: {train_metrics.get('epoch_time', 0):.1f}s"
        )
        
        val_metrics = trainer.validate(val_loader)
        logger.info(
            f"[Val]   Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | "
            f"Rec: {val_metrics['recall']:.4f}"
        )
        
        scheduler.step()
        
        # Save best model checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            epochs_no_improve = 0
            trainer.save_checkpoint(
                state={
                    'epoch': epoch,
                    'state_dict': trainer.model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict(),
                    'config': args.config,
                },
                save_dir=save_dir,
                filename="best_model.pth"
            )
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve}/{early_stopping_patience} epoch(s).")
            if epochs_no_improve >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    logger.info(f"Phase 2 Training Completed. Best Val Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()

