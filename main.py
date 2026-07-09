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
    
    # Save run info to docs/
    import datetime
    os.makedirs("docs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_info_file = f"docs/run_info_{timestamp}.txt"
    with open(run_info_file, "w", encoding="utf-8") as f:
        f.write(f"=== CHI TIẾT LẦN CHẠY ({timestamp}) ===\n")
        f.write(f"Chế độ chạy (Mode): {mode_name} Mode\n")
        f.write(f"Số lượng Epochs tối đa: {mode.epochs}\n")
        f.write(f"Kích thước ảnh (Image Size): {mode.image_size}x{mode.image_size}\n")
        f.write(f"Kích thước Batch (Batch Size): {mode.batch_size}\n")
        f.write(f"Learning Rate: {config.training.learning_rate}\n")
        f.write(f"Weight Decay: {config.training.weight_decay}\n")
        f.write(f"Tỷ lệ dữ liệu sử dụng (Subset Ratio): {mode.subset_ratio * 100:.0f}%\n")
        f.write(f"Tổng số ảnh dùng để Train: {train_samples} ảnh\n")
        f.write(f"Tổng số ảnh dùng để Validation: {val_samples} ảnh\n")
        f.write(f"\n{'='*90}\n")
        f.write(f"{'Epoch':>6} | {'T-Loss':>8} | {'T-Acc':>7} | {'T-F1':>7} | {'T-Prec':>7} | {'T-Rec':>7} | {'V-Loss':>8} | {'V-Acc':>7} | {'V-F1':>7} | {'V-Prec':>7} | {'V-Rec':>7} | {'Time(s)':>8} | {'Best':>5}\n")
        f.write(f"{'='*90}\n")
    logger.info(f"Đã lưu thông tin cấu hình chạy vào: {run_info_file}")
    

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
    
    import time
    total_train_start = time.time()

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
        epoch_time = train_metrics.get('epoch_time', 0)
        
        # Save best model checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            best_val_acc = val_metrics['accuracy']
            best_val_f1 = val_metrics['f1']
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
        
        # Ghi chi tiết epoch vào file run_info
        with open(run_info_file, "a", encoding="utf-8") as f:
            best_marker = " <--" if is_best else ""
            f.write(
                f"{epoch:>6} | "
                f"{train_metrics['loss']:>8.4f} | "
                f"{train_metrics['accuracy']:>7.4f} | "
                f"{train_metrics['f1']:>7.4f} | "
                f"{train_metrics['precision']:>7.4f} | "
                f"{train_metrics['recall']:>7.4f} | "
                f"{val_metrics['loss']:>8.4f} | "
                f"{val_metrics['accuracy']:>7.4f} | "
                f"{val_metrics['f1']:>7.4f} | "
                f"{val_metrics['precision']:>7.4f} | "
                f"{val_metrics['recall']:>7.4f} | "
                f"{epoch_time:>8.1f}{best_marker}\n"
            )
        
        if not is_best and epochs_no_improve >= early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break
    
    total_train_time = time.time() - total_train_start

    logger.info(f"Phase 2 Training Completed. Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    
    # Ghi kết quả tổng kết vào file run_info
    hours, rem = divmod(int(total_train_time), 3600)
    minutes, seconds = divmod(rem, 60)
    with open(run_info_file, "a", encoding="utf-8") as f:
        f.write(f"{'='*90}\n")
        f.write(f"\n=== KẾT QUẢ TỔNG KẾT ===\n")
        f.write(f"Tổng thời gian huấn luyện : {hours:02d}h {minutes:02d}m {seconds:02d}s ({total_train_time:.1f}s)\n")
        f.write(f"Dừng ở Epoch              : {epoch}/{mode.epochs}\n")
        f.write(f"Epoch tốt nhất            : {best_epoch}\n")
        f.write(f"Validation Loss tốt nhất  : {best_val_loss:.4f}\n")
        f.write(f"Validation Accuracy       : {best_val_acc:.4f}\n")
        f.write(f"Validation F1-Score       : {best_val_f1:.4f}\n")
        f.write(f"Model đã được lưu tại     : {save_dir}/best_model.pth\n")
    logger.info(f"Đã cập nhật kết quả huấn luyện vào file: {run_info_file}")

if __name__ == '__main__':
    main()

