"""main_ssl.py — Entry point cho Semi-Supervised Learning Pipeline.

Sử dụng:
    python main_ssl.py --config experiments/ssl_experiment.yaml

Luồng:
    1. Load config → init DataModule → lấy labeled / unlabeled / val loaders
    2. Khởi tạo ImprovedCNN + AdamW + CosineAnnealingLR
    3. Phase A: Supervised Warmup (config.ssl.warmup_epochs)
    4. Phase B: Pseudo-Labeling SSL (config.ssl.ssl_epochs)
       - Generate pseudo-labels với curriculum threshold
       - Combine labeled + pseudo → train
    5. Evaluate trên val set → lưu kết quả
    6. Vẽ tất cả charts (training curves, confusion matrix, metrics bar)
"""
import argparse
import os
import time
import datetime

import torch
from sklearn.metrics import classification_report

from configs.config import Config
from datasets.data_module import DataModule
from models.cnn import build_model
from training.ssl_trainer import SSLTrainer, get_curriculum_threshold
from utils.logger import get_logger
from utils.seed import set_seed
from visualization.plot_metrics import save_all_plots


CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


def collect_val_predictions(trainer: SSLTrainer, val_loader):
    """Chạy final evaluation trên val set, trả về y_true và y_pred."""
    import torch
    trainer.model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(trainer.device)
            outputs = trainer.model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


def write_run_header(f, config, mode_name, info, ssl_cfg):
    """Ghi header thông tin chạy vào file log."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = config.active_mode
    f.write(f"=== SSL RUN ({ts}) ===\n")
    f.write(f"Experiment       : {config.experiment_name}\n")
    f.write(f"Mode             : {mode_name}\n")
    f.write(f"Model            : {config.model.name}\n")
    f.write(f"Labeled per class: {ssl_cfg.labeled_per_class} → {info['labeled']} total\n")
    f.write(f"Unlabeled total  : {info['unlabeled_total']} "
            f"(ds1={info['unlabeled_ds1']}, ds2={info['unlabeled_ds2']})\n")
    f.write(f"Val samples      : {info['val']}\n")
    f.write(f"Warmup epochs    : {ssl_cfg.warmup_epochs}\n")
    f.write(f"SSL epochs       : {ssl_cfg.ssl_epochs}\n")
    f.write(f"Threshold        : {ssl_cfg.pseudo_threshold_start} → {ssl_cfg.pseudo_threshold_end}\n")
    f.write(f"Image size       : {mode.image_size}×{mode.image_size}\n")
    f.write(f"Batch size       : {mode.batch_size}\n")
    f.write(f"LR               : {config.training.learning_rate}\n")
    f.write(f"\n{'='*100}\n")
    header = (f"{'Phase':>6} | {'Epoch':>5} | {'T-Loss':>8} | {'T-Acc':>7} | "
              f"{'T-F1':>7} | {'V-Loss':>8} | {'V-Acc':>7} | {'V-F1':>7} | "
              f"{'#Pseudo':>8} | {'Thresh':>6} | {'Time':>7} | Note\n")
    f.write(header)
    f.write(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor SSL Pipeline")
    parser.add_argument('--config', type=str, default='experiments/ssl_experiment.yaml')
    parser.add_argument('--skip_warmup', action='store_true',
                        help='Skip Phase A warmup, load checkpoint and go straight to Phase B SSL')
    parser.add_argument('--checkpoint', type=str, default='saved_model/ssl_best_model.pth',
                        help='Checkpoint path to load when --skip_warmup is used')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Config & Seed
    # ------------------------------------------------------------------ #
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return

    config = Config.from_yaml(args.config)
    mode = config.active_mode
    ssl_cfg = config.ssl
    mode_name = "Development" if config.development.enabled else "Research"

    logger = get_logger(
        name=config.project_name,
        log_file=f"{config.experiment_name}.log",
    )
    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Mode: [{mode_name}] | image={mode.image_size} | batch={mode.batch_size}")
    logger.info(f"SSL: labeled/class={ssl_cfg.labeled_per_class} | warmup={ssl_cfg.warmup_epochs} | ssl_epochs={ssl_cfg.ssl_epochs}")

    set_seed(config.seed)
    logger.info(f"Seed: {config.seed}")

    # ------------------------------------------------------------------ #
    # 2. DataModule
    # ------------------------------------------------------------------ #
    data_module = DataModule(config)
    info = data_module.get_info()
    logger.info(
        f"Data split -> labeled: {info['labeled']} | "
        f"unlabeled: {info['unlabeled_total']} | val: {info['val']}"
    )

    labeled_loader   = data_module.get_labeled_dataloader()
    unlabeled_loader = data_module.get_unlabeled_dataloader()
    val_loader       = data_module.get_val_dataloader()

    # ------------------------------------------------------------------ #
    # 3. Model, Optimizer, Scheduler
    # ------------------------------------------------------------------ #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    model = build_model(config.model.name, num_classes=config.model.num_classes)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config.model.name} | Params: {total_params:,}")

    total_epochs = ssl_cfg.warmup_epochs + ssl_cfg.ssl_epochs
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    trainer = SSLTrainer(model=model, device=device, logger=logger)

    # ------------------------------------------------------------------ #
    # 4. Run info file
    # ------------------------------------------------------------------ #
    os.makedirs("docs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_info_file = f"docs/ssl_run_{ts}.txt"
    with open(run_info_file, "w", encoding="utf-8") as f:
        write_run_header(f, config, mode_name, info, ssl_cfg)

    # ------------------------------------------------------------------ #
    # 5. Tracking
    # ------------------------------------------------------------------ #
    warmup_history = []
    ssl_history    = []
    pseudo_stats_list = []

    best_val_loss   = float('inf')
    best_val_acc    = 0.0
    best_val_f1     = 0.0
    best_epoch      = 0
    epochs_no_improve = 0
    patience = getattr(mode, 'early_stopping_patience', 7)
    save_dir = "saved_model"
    total_start = time.time()

    # ================================================================== #
    # PHASE A — Supervised Warmup  (bo qua neu --skip_warmup)
    # ================================================================== #
    if args.skip_warmup:
        ckpt_path = args.checkpoint
        if not os.path.exists(ckpt_path):
            logger.info(f"ERROR: Checkpoint not found: {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path, map_location=device)
        trainer.model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        # Advance scheduler for warmup_epochs steps that were already done
        for _ in range(ssl_cfg.warmup_epochs):
            scheduler.step()
        # Lay best_val_loss tu checkpoint neu co
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        logger.info(f"[SKIP WARMUP] Loaded checkpoint from: {ckpt_path}")
        logger.info(f"[SKIP WARMUP] best_val_loss from ckpt = {best_val_loss:.4f}")
        logger.info(f"[SKIP WARMUP] Jumping straight to Phase B SSL")
        warmup_history = []  # Khong co warmup history khi skip
    else:
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE A — Supervised Warmup ({ssl_cfg.warmup_epochs} epochs)")
        logger.info(f"{'='*60}")

    if not args.skip_warmup:
        for epoch in range(1, ssl_cfg.warmup_epochs + 1):
            logger.info(f"[Warmup] Epoch {epoch}/{ssl_cfg.warmup_epochs}")

            train_m = trainer.train_epoch(labeled_loader, optimizer, phase_label=f"W-{epoch}")
            val_m   = trainer.validate(val_loader)
            scheduler.step()

            warmup_history.append({
                'train_loss': train_m['loss'], 'train_acc': train_m['accuracy'],
                'train_f1':   train_m['f1'],
                'val_loss':   val_m['loss'],   'val_acc': val_m['accuracy'],
                'val_f1':     val_m['f1'],
            })

            logger.info(
                f"  Train -> Loss:{train_m['loss']:.4f} Acc:{train_m['accuracy']:.4f} F1:{train_m['f1']:.4f} | "
                f"Val -> Loss:{val_m['loss']:.4f} Acc:{val_m['accuracy']:.4f} F1:{val_m['f1']:.4f}"
            )

            is_best = val_m['loss'] < best_val_loss
            note = ""
            if is_best:
                best_val_loss = val_m['loss']
                best_val_acc  = val_m['accuracy']
                best_val_f1   = val_m['f1']
                best_epoch    = epoch
                epochs_no_improve = 0
                note = " <-- BEST"
                trainer.save_checkpoint(
                    {'epoch': epoch, 'state_dict': trainer.model.state_dict(),
                     'best_val_loss': best_val_loss, 'optimizer': optimizer.state_dict()},
                    save_dir, "ssl_best_model.pth",
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered at warmup epoch {epoch}.")
                    with open(run_info_file, "a", encoding="utf-8") as f:
                        f.write(f"{'  A':>6} | {epoch:>5} | {train_m['loss']:>8.4f} | {train_m['accuracy']:>7.4f} | "
                                f"{train_m['f1']:>7.4f} | {val_m['loss']:>8.4f} | {val_m['accuracy']:>7.4f} | "
                                f"{val_m['f1']:>7.4f} | {'':>8} | {'':>6} | {train_m.get('epoch_time',0):>6.1f}s | EARLY STOP\n")
                    break

            with open(run_info_file, "a", encoding="utf-8") as f:
                f.write(f"{'  A':>6} | {epoch:>5} | {train_m['loss']:>8.4f} | {train_m['accuracy']:>7.4f} | "
                        f"{train_m['f1']:>7.4f} | {val_m['loss']:>8.4f} | {val_m['accuracy']:>7.4f} | "
                        f"{val_m['f1']:>7.4f} | {'':>8} | {'':>6} | {train_m.get('epoch_time',0):>6.1f}s |{note}\n")


    # ================================================================== #
    # PHASE B — SSL Pseudo-Labeling
    # ================================================================== #
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE B — SSL Pseudo-Labeling ({ssl_cfg.ssl_epochs} epochs)")
    logger.info(f"{'='*60}")

    epochs_no_improve = 0  # reset

    for ssl_ep in range(1, ssl_cfg.ssl_epochs + 1):
        global_epoch = ssl_cfg.warmup_epochs + ssl_ep
        threshold = get_curriculum_threshold(
            ssl_ep, ssl_cfg.ssl_epochs,
            start=ssl_cfg.pseudo_threshold_start,
            end=ssl_cfg.pseudo_threshold_end,
        )
        logger.info(f"[SSL] Epoch {ssl_ep}/{ssl_cfg.ssl_epochs} | threshold={threshold:.3f}")

        # --- Generate pseudo-labels ---
        pseudo_dataset, p_stats = trainer.generate_pseudo_labels(unlabeled_loader, threshold)
        pseudo_stats_list.append(p_stats)

        logger.info(
            f"  Pseudo-labels: {p_stats['selected']}/{p_stats['total_unlabeled']} "
            f"({p_stats['selection_rate']*100:.1f}%) | "
            f"per class: {p_stats['per_class_count']}"
        )

        # --- Combine & train ---
        if len(pseudo_dataset) > 0:
            combined_loader = trainer.build_combined_loader(
                labeled_loader, pseudo_dataset,
                batch_size=mode.batch_size,
                num_workers=config.data.num_workers,
            )
            train_m = trainer.train_epoch(combined_loader, optimizer, phase_label=f"B-{ssl_ep}")
        else:
            logger.info("  No pseudo-labels selected, training on labeled only.")
            train_m = trainer.train_epoch(labeled_loader, optimizer, phase_label=f"B-{ssl_ep}")

        val_m = trainer.validate(val_loader)
        scheduler.step()

        ssl_history.append({
            'train_loss': train_m['loss'], 'train_acc': train_m['accuracy'],
            'train_f1':   train_m['f1'],
            'val_loss':   val_m['loss'],   'val_acc': val_m['accuracy'],
            'val_f1':     val_m['f1'],
        })

        logger.info(
            f"  Train → Loss:{train_m['loss']:.4f} Acc:{train_m['accuracy']:.4f} F1:{train_m['f1']:.4f} | "
            f"Val → Loss:{val_m['loss']:.4f} Acc:{val_m['accuracy']:.4f} F1:{val_m['f1']:.4f}"
        )

        is_best = val_m['loss'] < best_val_loss
        note = ""
        if is_best:
            best_val_loss = val_m['loss']
            best_val_acc  = val_m['accuracy']
            best_val_f1   = val_m['f1']
            best_epoch    = global_epoch
            epochs_no_improve = 0
            note = " <-- BEST"
            trainer.save_checkpoint(
                {'epoch': global_epoch, 'state_dict': trainer.model.state_dict(),
                 'best_val_loss': best_val_loss, 'optimizer': optimizer.state_dict()},
                save_dir, "ssl_best_model.pth",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at SSL epoch {ssl_ep}.")
                break

        with open(run_info_file, "a", encoding="utf-8") as f:
            f.write(
                f"{'  B':>6} | {global_epoch:>5} | {train_m['loss']:>8.4f} | {train_m['accuracy']:>7.4f} | "
                f"{train_m['f1']:>7.4f} | {val_m['loss']:>8.4f} | {val_m['accuracy']:>7.4f} | "
                f"{val_m['f1']:>7.4f} | {p_stats['selected']:>8} | {threshold:>6.3f} | "
                f"{train_m.get('epoch_time',0):>6.1f}s |{note}\n"
            )

    # ================================================================== #
    # 6. Final Evaluation & Summary
    # ================================================================== #
    total_time = time.time() - total_start
    hours, rem = divmod(int(total_time), 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training Completed in {hours:02d}h {minutes:02d}m {seconds:02d}s")
    logger.info(f"Best Val Loss: {best_val_loss:.4f} | Acc: {best_val_acc:.4f} | F1: {best_val_f1:.4f} (epoch {best_epoch})")

    # Load best model để final eval
    best_ckpt = os.path.join(save_dir, "ssl_best_model.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        trainer.model.load_state_dict(ckpt['state_dict'])
        logger.info(f"Loaded best model from epoch {ckpt['epoch']}")

    y_true, y_pred = collect_val_predictions(trainer, val_loader)
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    logger.info(f"\nFinal Classification Report:\n{report_str}")

    with open(run_info_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*100}\n")
        f.write(f"=== KẾT QUẢ TỔNG KẾT ===\n")
        f.write(f"Tổng thời gian     : {hours:02d}h {minutes:02d}m {seconds:02d}s\n")
        f.write(f"Epoch tốt nhất     : {best_epoch}\n")
        f.write(f"Best Val Loss      : {best_val_loss:.4f}\n")
        f.write(f"Best Val Accuracy  : {best_val_acc:.4f}\n")
        f.write(f"Best Val F1        : {best_val_f1:.4f}\n")
        f.write(f"Model saved        : {best_ckpt}\n")
        f.write(f"\n--- Final Classification Report ---\n{report_str}\n")

    logger.info(f"Run info saved → {run_info_file}")

    # ------------------------------------------------------------------ #
    # 7. Visualization
    # ------------------------------------------------------------------ #
    plot_dir = "visualization/results"
    save_all_plots(
        warmup_history=warmup_history,
        ssl_history=ssl_history,
        pseudo_stats_list=pseudo_stats_list,
        y_true_final=y_true,
        y_pred_final=y_pred,
        output_dir=plot_dir,
        experiment_name=config.experiment_name,
    )
    logger.info(f"All plots saved in: {plot_dir}/")


if __name__ == '__main__':
    main()
