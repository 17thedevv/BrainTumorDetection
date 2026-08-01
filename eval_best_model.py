"""eval_best_model.py — Đánh giá model tốt nhất đã lưu và xuất toàn bộ biểu đồ báo cáo ngay lập tức.

Sử dụng khi không muốn đợi chạy hết 35 epochs:
    python -X utf8 eval_best_model.py --config experiments/ssl_experiment.yaml
"""
import argparse
import os
import datetime

import torch
from sklearn.metrics import classification_report

from configs.config import Config
from datasets.data_module import DataModule
from models.cnn import build_model
from utils.logger import get_logger
from visualization.plot_metrics import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    CLASS_NAMES,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate best SSL model & generate plots immediately")
    parser.add_argument('--config', type=str, default='experiments/ssl_experiment.yaml')
    parser.add_argument('--ckpt', type=str, default='saved_model/ssl_best_model.pth')
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found at {args.ckpt}")
        return

    config = Config.from_yaml(args.config)
    logger = get_logger("eval_best_model", "eval_best_model.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"=== ĐÁNH GIÁ VÀ XUẤT BIỂU ĐỒ TỪ CHECKPOINT ===")
    logger.info(f"Checkpoint: {args.ckpt} | Device: {device}")

    # 1. Load Data
    data_module = DataModule(config)
    val_loader = data_module.get_val_dataloader()

    # 2. Load Model & Weights
    model = build_model(config.model.name, num_classes=config.model.num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    best_epoch = ckpt.get('epoch', 'N/A')
    best_loss = ckpt.get('best_val_loss', 'N/A')
    logger.info(f"Loaded weights from epoch {best_epoch} (Best Val Loss: {best_loss})")

    # 3. Evaluate on Validation Set (1,600 ảnh)
    model.eval()
    y_true, y_pred = [], []
    logger.info("Đang chạy dự đoán trên tập kiểm thử 1,600 ảnh...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    # 4. In báo cáo phân loại
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    logger.info(f"\nFinal Classification Report:\n{report_str}")

    # 5. Lưu báo cáo vào docs/
    os.makedirs("docs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"docs/final_eval_{ts}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"=== SSL BEST CHECKPOINT EVALUATION ({ts}) ===\n")
        f.write(f"Model            : {config.model.name}\n")
        f.write(f"Best Epoch       : {best_epoch}\n")
        f.write(f"Best Val Loss    : {best_loss}\n\n")
        f.write(report_str)
    logger.info(f"Đã lưu báo cáo văn bản vào: {report_file}")

    # 6. Vẽ biểu đồ Confusion Matrix & Per-class metrics
    out_dir = "visualization/results"
    os.makedirs(out_dir, exist_ok=True)
    exp_name = config.experiment_name

    logger.info("Đang xuất biểu đồ Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, exp_name, out_dir)

    logger.info("Đang xuất biểu đồ Per-Class Metrics...")
    plot_per_class_metrics(y_true, y_pred, CLASS_NAMES, exp_name, out_dir)

    logger.info(f"\n✅ HOÀN TẤT! Tất cả biểu đồ đã được lưu vào: {out_dir}")
    logger.info(f"  - {out_dir}/{exp_name}_confusion_matrix.png")
    logger.info(f"  - {out_dir}/{exp_name}_per_class_metrics.png")


if __name__ == "__main__":
    main()
