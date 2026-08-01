"""Visualization — Vẽ các bảng chỉ số sau quá trình huấn luyện SSL.

Hàm chính:
    plot_training_curves()  : Loss & Accuracy theo epoch (warmup + SSL)
    plot_confusion_matrix() : Confusion matrix heatmap
    plot_per_class_metrics(): Bar chart Precision / Recall / F1 theo từng class
    plot_pseudo_stats()     : Số pseudo-labels được chọn mỗi SSL epoch
    save_all_plots()        : Lưu tất cả vào thư mục output
"""
import os
from typing import List, Dict, Optional, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (không cần màn hình)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# Màu sắc nhất quán
COLORS = {
    'warmup_train': '#4A90D9',
    'warmup_val':   '#E67E22',
    'ssl_train':    '#27AE60',
    'ssl_val':      '#8E44AD',
    'boundary':     '#E74C3C',
    'glioma':       '#E74C3C',
    'meningioma':   '#3498DB',
    'notumor':      '#2ECC71',
    'pituitary':    '#F39C12',
}
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


def _style():
    """Áp dụng style mặc định."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 120,
    })


def plot_training_curves(
    warmup_history: List[Dict],
    ssl_history: List[Dict],
    save_path: str,
):
    """Vẽ Loss và Accuracy theo epoch cho cả warmup và SSL phase.

    Args:
        warmup_history : list of dict {'train_loss', 'val_loss', 'train_acc', 'val_acc'}
        ssl_history    : list of dict tương tự, cho Phase B
        save_path      : đường dẫn file .png để lưu
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves — SSL Pseudo-Labeling', fontsize=14, fontweight='bold', y=1.02)

    n_warm = len(warmup_history)
    epochs_warm = list(range(1, n_warm + 1))
    epochs_ssl  = list(range(n_warm + 1, n_warm + len(ssl_history) + 1))

    # ---- Loss ----
    ax = axes[0]
    if warmup_history:
        ax.plot(epochs_warm, [h['train_loss'] for h in warmup_history],
                color=COLORS['warmup_train'], lw=2, label='Warmup Train Loss')
        ax.plot(epochs_warm, [h['val_loss'] for h in warmup_history],
                color=COLORS['warmup_val'], lw=2, linestyle='--', label='Warmup Val Loss')
    if ssl_history:
        ax.plot(epochs_ssl, [h['train_loss'] for h in ssl_history],
                color=COLORS['ssl_train'], lw=2, label='SSL Train Loss')
        ax.plot(epochs_ssl, [h['val_loss'] for h in ssl_history],
                color=COLORS['ssl_val'], lw=2, linestyle='--', label='SSL Val Loss')
    if n_warm > 0 and ssl_history:
        ax.axvline(x=n_warm + 0.5, color=COLORS['boundary'], linestyle=':', lw=1.5,
                   label='SSL Start')
    ax.set_title('Loss', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend(fontsize=8, loc='upper right')

    # ---- Accuracy ----
    ax = axes[1]
    if warmup_history:
        ax.plot(epochs_warm, [h['train_acc'] for h in warmup_history],
                color=COLORS['warmup_train'], lw=2, label='Warmup Train Acc')
        ax.plot(epochs_warm, [h['val_acc'] for h in warmup_history],
                color=COLORS['warmup_val'], lw=2, linestyle='--', label='Warmup Val Acc')
    if ssl_history:
        ax.plot(epochs_ssl, [h['train_acc'] for h in ssl_history],
                color=COLORS['ssl_train'], lw=2, label='SSL Train Acc')
        ax.plot(epochs_ssl, [h['val_acc'] for h in ssl_history],
                color=COLORS['ssl_val'], lw=2, linestyle='--', label='SSL Val Acc')
    if n_warm > 0 and ssl_history:
        ax.axvline(x=n_warm + 0.5, color=COLORS['boundary'], linestyle=':', lw=1.5,
                   label='SSL Start')
    ax.set_ylim(0, 1.05)
    ax.set_title('Accuracy', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Training curves saved → {save_path}")


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: str,
    title: str = 'Confusion Matrix',
):
    """Vẽ confusion matrix dạng heatmap với seaborn."""
    _style()
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_pct, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor='white',
        annot_kws={'size': 11},
        ax=ax,
    )
    ax.set_title(f'{title}\n(% per true class)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)

    # Overlay số thực tế ở góc dưới mỗi ô
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j + 0.5, i + 0.75, f'n={cm[i, j]}',
                    ha='center', va='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")


def plot_per_class_metrics(
    y_true: List[int],
    y_pred: List[int],
    save_path: str,
    title: str = 'Per-Class Metrics',
):
    """Bar chart Precision / Recall / F1-Score cho từng class."""
    _style()
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)

    precisions = [report[c]['precision'] for c in CLASS_NAMES]
    recalls    = [report[c]['recall']    for c in CLASS_NAMES]
    f1s        = [report[c]['f1-score']  for c in CLASS_NAMES]

    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#4A90D9', alpha=0.85)
    bars2 = ax.bar(x,         recalls,    width, label='Recall',    color='#27AE60', alpha=0.85)
    bars3 = ax.bar(x + width, f1s,        width, label='F1-Score',  color='#E67E22', alpha=0.85)

    # Hiển thị giá trị trên mỗi bar
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend(fontsize=10)

    # Macro avg line
    macro_f1 = report['macro avg']['f1-score']
    ax.axhline(y=macro_f1, color='red', linestyle='--', lw=1.2,
               label=f'Macro F1 = {macro_f1:.3f}')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Per-class metrics saved → {save_path}")


def plot_pseudo_stats(
    pseudo_stats_list: List[Dict],
    warmup_epochs: int,
    save_path: str,
):
    """Vẽ số pseudo-labels được chọn mỗi SSL epoch, chia theo class.

    pseudo_stats_list: mỗi phần tử là dict từ SSLTrainer.generate_pseudo_labels()
        {'selected', 'threshold', 'per_class_count', ...}
    """
    if not pseudo_stats_list:
        return
    _style()
    ssl_epochs = list(range(warmup_epochs + 1, warmup_epochs + len(pseudo_stats_list) + 1))

    class_counts = {cn: [] for cn in CLASS_NAMES}
    thresholds   = []
    totals       = []

    for stats in pseudo_stats_list:
        per_class = stats.get('per_class_count', {})
        for i, cn in enumerate(CLASS_NAMES):
            class_counts[cn].append(per_class.get(i, 0))
        thresholds.append(stats['threshold'])
        totals.append(stats['selected'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Pseudo-Label Statistics — Curriculum Threshold', fontsize=13, fontweight='bold')

    # ---- Stacked bar: số ảnh per class ----
    bottom = np.zeros(len(ssl_epochs))
    class_colors = [COLORS['glioma'], COLORS['meningioma'],
                    COLORS['notumor'], COLORS['pituitary']]
    for cn, color in zip(CLASS_NAMES, class_colors):
        vals = np.array(class_counts[cn])
        ax1.bar(ssl_epochs, vals, bottom=bottom, label=cn, color=color, alpha=0.8)
        bottom += vals
    ax1.set_ylabel('# Pseudo-labels Selected')
    ax1.set_title('Số pseudo-labels được chọn mỗi epoch (chia theo class)')
    ax1.legend(fontsize=9, loc='upper left')

    # Ghi tổng trên đầu mỗi bar
    for e, t in zip(ssl_epochs, totals):
        ax1.text(e, t + 5, str(t), ha='center', va='bottom', fontsize=8)

    # ---- Line: threshold curriculum ----
    ax2.plot(ssl_epochs, thresholds, color='#E74C3C', lw=2, marker='o', markersize=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Confidence Threshold')
    ax2.set_title('Curriculum Threshold tăng dần qua các SSL epochs')
    ax2.set_ylim(0.80, 1.0)
    for e, th in zip(ssl_epochs, thresholds):
        ax2.text(e, th + 0.002, f'{th:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Pseudo-label stats saved → {save_path}")


def save_all_plots(
    warmup_history: List[Dict],
    ssl_history: List[Dict],
    pseudo_stats_list: List[Dict],
    y_true_final: List[int],
    y_pred_final: List[int],
    output_dir: str,
    experiment_name: str = 'ssl',
):
    """Gọi tất cả các hàm vẽ và lưu vào output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, experiment_name)

    plot_training_curves(
        warmup_history, ssl_history,
        save_path=f"{prefix}_training_curves.png",
    )
    plot_confusion_matrix(
        y_true_final, y_pred_final,
        save_path=f"{prefix}_confusion_matrix.png",
        title=f'Confusion Matrix — {experiment_name}',
    )
    plot_per_class_metrics(
        y_true_final, y_pred_final,
        save_path=f"{prefix}_per_class_metrics.png",
        title=f'Per-Class Metrics — {experiment_name}',
    )
    plot_pseudo_stats(
        pseudo_stats_list,
        warmup_epochs=len(warmup_history),
        save_path=f"{prefix}_pseudo_stats.png",
    )
    print(f"[Plot] All plots saved in: {output_dir}/")
