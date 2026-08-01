"""Dataset Overview Visualization.

Tao mot bang visual tong quan ve cau truc du lieu trong pipeline SSL:
  - So sanh toan bo dataset (DS1 + DS2)
  - SSL split: Labeled vs Unlabeled vs Val
  - Phan bo class trong tung pool
  - Sample images tu moi class

Chay: python visualization/dataset_overview.py
"""
import os
import sys
import glob
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============================================================
# CONFIG
# ============================================================
ROOT = os.path.join(os.path.dirname(__file__), '..')
DS1_TRAIN = os.path.join(ROOT, 'data/dataset1/Training')
DS1_TEST  = os.path.join(ROOT, 'data/dataset1/Testing')
DS2_ROOT  = os.path.join(ROOT, 'data/dataset2')

CLASSES      = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
LABELED_PER_CLASS = 250

# Color palette
BG_COLOR     = '#0F1117'
PANEL_COLOR  = '#1A1D27'
ACCENT1      = '#4F8EF7'   # blue
ACCENT2      = '#F7934C'   # orange
ACCENT3      = '#56C26D'   # green
ACCENT4      = '#C256C2'   # purple
UNLABELED_C  = '#3A3F5C'
VAL_C        = '#F7D354'
LABELED_C    = '#56C26D'

CLASS_COLORS = [ACCENT1, ACCENT2, ACCENT3, ACCENT4]


def count_images(folder):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    total = 0
    for ext in exts:
        total += len(glob.glob(os.path.join(folder, ext)))
    return total


def get_random_sample_image(folder, seed=42):
    """Lay 1 anh ngau nhien tu folder."""
    random.seed(seed)
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    if not paths:
        return None
    return random.choice(paths)


# ============================================================
# COUNT DATA
# ============================================================
train_per_class = {c: count_images(os.path.join(DS1_TRAIN, c)) for c in CLASSES}
test_per_class  = {c: count_images(os.path.join(DS1_TEST, c))  for c in CLASSES}

ds2_yes = count_images(os.path.join(DS2_ROOT, 'yes'))
ds2_no  = count_images(os.path.join(DS2_ROOT, 'no'))
ds2_total = ds2_yes + ds2_no

total_train = sum(train_per_class.values())
total_test  = sum(test_per_class.values())

labeled_total   = LABELED_PER_CLASS * 4
unlabeled_ds1   = total_train - labeled_total
unlabeled_ds2   = ds2_total
unlabeled_total = unlabeled_ds1 + unlabeled_ds2
grand_total     = labeled_total + unlabeled_total + total_test


# ============================================================
# FIGURE
# ============================================================
fig = plt.figure(figsize=(20, 14), facecolor=BG_COLOR)
gs = gridspec.GridSpec(
    3, 4,
    figure=fig,
    hspace=0.48,
    wspace=0.35,
    left=0.05, right=0.97,
    top=0.90, bottom=0.07
)

title_y = 0.955
fig.text(
    0.5, title_y,
    'Brain Tumor Detection — Dataset Overview',
    ha='center', va='center',
    fontsize=22, fontweight='bold',
    color='white',
    fontfamily='DejaVu Sans'
)
fig.text(
    0.5, title_y - 0.028,
    f'Semi-Supervised Learning Pipeline  |  Total: {grand_total:,} images',
    ha='center', va='center',
    fontsize=12, color='#9099B2',
)


def styled_ax(ax, title=''):
    ax.set_facecolor(PANEL_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2A2F45')
    ax.tick_params(colors='#9099B2', labelsize=9)
    if title:
        ax.set_title(title, color='white', fontsize=11,
                     fontweight='bold', pad=10)
    return ax


# ============================================================
# PANEL 1 (row0, col0-1): SSL Split Donut Chart
# ============================================================
ax1 = fig.add_subplot(gs[0, :2])
styled_ax(ax1, 'SSL Data Split')

sizes   = [labeled_total, unlabeled_ds1, unlabeled_ds2, total_test]
colors  = [LABELED_C, ACCENT1, ACCENT2, VAL_C]
labels  = [
    f'Labeled\n{labeled_total:,} imgs',
    f'Unlabeled (DS1)\n{unlabeled_ds1:,} imgs',
    f'Unlabeled (DS2 Br35H)\n{unlabeled_ds2:,} imgs',
    f'Validation\n{total_test:,} imgs',
]
explode = (0.05, 0.02, 0.02, 0.05)

wedges, texts, autotexts = ax1.pie(
    sizes, labels=None, colors=colors, explode=explode,
    autopct='%1.1f%%', startangle=140,
    wedgeprops=dict(width=0.55, edgecolor=BG_COLOR, linewidth=2),
    pctdistance=0.78,
    textprops=dict(color='white', fontsize=10, fontweight='bold'),
)

# Draw center text
ax1.text(0, 0, f'{grand_total:,}\nimages', ha='center', va='center',
         fontsize=13, fontweight='bold', color='white')

# Custom legend
legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
ax1.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.18),
           ncol=2, fontsize=9, frameon=False,
           labelcolor='#C8CDE0', handlelength=1.2)


# ============================================================
# PANEL 2 (row0, col2-3): Per-Class Distribution Bar
# ============================================================
ax2 = fig.add_subplot(gs[0, 2:])
styled_ax(ax2, 'Per-Class Distribution — Dataset 1')

x = np.arange(len(CLASSES))
w = 0.38
bars_train = ax2.bar(x - w/2, [train_per_class[c] for c in CLASSES],
                     width=w, color=ACCENT1, alpha=0.85, label='Training (all)')
bars_test  = ax2.bar(x + w/2, [test_per_class[c] for c in CLASSES],
                     width=w, color=VAL_C, alpha=0.85, label='Testing / Val')

# Labeled line
ax2.axhline(y=LABELED_PER_CLASS, color=LABELED_C, linestyle='--', lw=1.8,
            label=f'Labeled threshold ({LABELED_PER_CLASS}/class)')

for bar in bars_train:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             f'{int(bar.get_height()):,}', ha='center', va='bottom',
             fontsize=8, color='#B0B8D0')
for bar in bars_test:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             f'{int(bar.get_height()):,}', ha='center', va='bottom',
             fontsize=8, color='#B0B8D0')

ax2.set_xticks(x)
ax2.set_xticklabels(CLASS_LABELS, color='#C8CDE0', fontsize=10)
ax2.set_ylabel('# Images', color='#9099B2', fontsize=9)
ax2.yaxis.label.set_color('#9099B2')
ax2.legend(fontsize=9, frameon=False, labelcolor='#C8CDE0',
           loc='upper right')
ax2.set_ylim(0, max(train_per_class.values()) * 1.2)
ax2.tick_params(axis='y', colors='#9099B2')
ax2.grid(axis='y', color='#2A2F45', linewidth=0.8)


# ============================================================
# PANEL 3 (row1, col0): Labeled Pool per class
# ============================================================
ax3 = fig.add_subplot(gs[1, 0])
styled_ax(ax3, 'Labeled Pool\n(1,000 images)')

vals   = [LABELED_PER_CLASS] * 4
bars3  = ax3.barh(CLASS_LABELS, vals, color=CLASS_COLORS, alpha=0.85,
                  edgecolor=BG_COLOR, linewidth=1.5)
for bar, v in zip(bars3, vals):
    ax3.text(v + 3, bar.get_y() + bar.get_height()/2,
             f'{v}', va='center', color='white', fontsize=10, fontweight='bold')
ax3.set_xlim(0, 320)
ax3.set_xlabel('# Labeled images', color='#9099B2', fontsize=9)
ax3.tick_params(axis='x', colors='#9099B2')
ax3.yaxis.set_tick_params(labelcolor='#C8CDE0', labelsize=10)
ax3.grid(axis='x', color='#2A2F45', linewidth=0.8)


# ============================================================
# PANEL 4 (row1, col1): Unlabeled Pool breakdown
# ============================================================
ax4 = fig.add_subplot(gs[1, 1])
styled_ax(ax4, 'Unlabeled Pool\n(8,461 images)')

unlabeled_per_class = {c: train_per_class[c] - LABELED_PER_CLASS for c in CLASSES}
ds2_contributions = {'Br35H Yes': ds2_yes, 'Br35H No': ds2_no}

# Stacked bar: DS1 unlabeled per class + DS2
cat_names  = CLASS_LABELS + ['Br35H Yes', 'Br35H No']
cat_vals   = [unlabeled_per_class[c] for c in CLASSES] + [ds2_yes, ds2_no]
cat_colors = CLASS_COLORS + ['#F7934C', '#C256C2']

bars4 = ax4.barh(cat_names, cat_vals, color=cat_colors, alpha=0.85,
                 edgecolor=BG_COLOR, linewidth=1.5)
for bar, v in zip(bars4, cat_vals):
    ax4.text(v + 10, bar.get_y() + bar.get_height()/2,
             f'{v:,}', va='center', color='white', fontsize=9)
ax4.set_xlabel('# Unlabeled images', color='#9099B2', fontsize=9)
ax4.tick_params(axis='x', colors='#9099B2')
ax4.yaxis.set_tick_params(labelcolor='#C8CDE0', labelsize=9)
ax4.set_xlim(0, max(cat_vals) * 1.22)
ax4.grid(axis='x', color='#2A2F45', linewidth=0.8)


# ============================================================
# PANEL 5 (row1, col2-3): SSL Pipeline Flow Diagram
# ============================================================
ax5 = fig.add_subplot(gs[1, 2:])
styled_ax(ax5, 'Semi-Supervised Learning Pipeline')
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 5)
ax5.axis('off')

def draw_box(ax, x, y, w, h, text, color, fontsize=9):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle='round,pad=0.08',
                         facecolor=color, edgecolor='white',
                         linewidth=1.2, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            color='white', fontsize=fontsize, fontweight='bold',
            multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, color='#9099B2'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Boxes
draw_box(ax5, 0.2, 3.2, 2.2, 1.0, f'Dataset 1 Training\n{total_train:,} images\n(4 classes)', ACCENT1, 8)
draw_box(ax5, 0.2, 1.5, 2.2, 1.0, f'Dataset 2 (Br35H)\n{ds2_total:,} images\n(yes/no)', ACCENT2, 8)

draw_box(ax5, 3.0, 3.2, 2.0, 0.8, f'Labeled Pool\n{labeled_total:,} imgs', LABELED_C, 8)
draw_box(ax5, 3.0, 1.8, 2.0, 0.8, f'Unlabeled Pool\n{unlabeled_total:,} imgs', UNLABELED_C, 8)

draw_box(ax5, 5.7, 3.5, 2.0, 0.8, f'Phase A\nWarmup (15 ep)\nSupervised', '#3A6BC2', 8)
draw_box(ax5, 5.7, 2.0, 2.0, 0.8, f'Phase B\nSSL (20 ep)\nPseudo-Label', '#8B3AC2', 8)

draw_box(ax5, 8.2, 2.7, 1.5, 0.8, f'Val Set\n{total_test:,} imgs', VAL_C, 8)

# Arrows
draw_arrow(ax5, 2.4, 3.7,  3.0, 3.6)
draw_arrow(ax5, 2.4, 1.95, 3.0, 2.2)
draw_arrow(ax5, 2.4, 3.7,  3.0, 2.2)  # DS1 -> unlabeled too

draw_arrow(ax5, 5.0, 3.6,  5.7, 3.9)
draw_arrow(ax5, 5.0, 2.2,  5.7, 2.4)
draw_arrow(ax5, 5.7 + 1.0, 3.5, 5.7 + 1.0, 2.8)  # Phase A -> Phase B

draw_arrow(ax5, 7.7, 3.9, 8.2, 3.1)
draw_arrow(ax5, 7.7, 2.4, 8.2, 2.9)

# Threshold label
ax5.text(5.5, 1.6,
         'Curriculum threshold: 0.85 -> 0.95',
         ha='center', color='#9099B2', fontsize=8, style='italic')


# ============================================================
# PANEL 6 (row2, col0-3): Sample Images
# ============================================================
sample_axes = []
for i, cls in enumerate(CLASSES):
    ax = fig.add_subplot(gs[2, i])
    ax.set_facecolor(PANEL_COLOR)
    for sp in ax.spines.values():
        sp.set_edgecolor(CLASS_COLORS[i])
        sp.set_linewidth(2)

    img_path = get_random_sample_image(os.path.join(DS1_TRAIN, cls), seed=i*7+1)
    if img_path and os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert('RGB').resize((180, 180))
            ax.imshow(np.array(img))
        except Exception:
            ax.set_facecolor(PANEL_COLOR)
    else:
        ax.set_facecolor(PANEL_COLOR)

    ax.axis('off')
    ax.set_title(
        f'{CLASS_LABELS[i]}\nTrain: {train_per_class[cls]:,}  |  Labeled: {LABELED_PER_CLASS}',
        color='white', fontsize=10, fontweight='bold', pad=6,
    )
    # Colored bottom bar
    ax.plot([0, 1], [-0.03, -0.03], color=CLASS_COLORS[i], lw=3,
            transform=ax.transAxes, clip_on=False)

    sample_axes.append(ax)


# ============================================================
# SAVE
# ============================================================
output_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, 'dataset_overview.png')
fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print(f'[OK] Dataset overview saved -> {out_path}')
