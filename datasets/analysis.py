import os
import glob
from collections import defaultdict
from PIL import Image
import numpy as np

def analyze_dataset(base_path, dataset_name):
    print(f"\n--- Analyzing {dataset_name} at {base_path} ---")
    
    # Identify classes based on subdirectories
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    total_images = 0
    class_counts = {}
    widths, heights = [], []
    
    for cls in classes:
        cls_path = os.path.join(base_path, cls)
        # Handle sub-splits like Training/Testing in Dataset 1
        subdirs = [d for d in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path, d))]
        
        image_paths = []
        if len(subdirs) > 0:
             for subdir in subdirs:
                 image_paths.extend(glob.glob(os.path.join(cls_path, subdir, '*.*')))
        else:
             image_paths.extend(glob.glob(os.path.join(cls_path, '*.*')))
             
        # Filter valid images
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        class_counts[cls] = len(image_paths)
        total_images += len(image_paths)
        
        # Sample images to get size and intensity distributions (up to 100 per class to save time)
        sample_paths = np.random.choice(image_paths, min(100, len(image_paths)), replace=False) if image_paths else []
        for path in sample_paths:
            try:
                with Image.open(path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
            except Exception as e:
                pass
                
    print(f"Total Images: {total_images}")
    print(f"Classes found: {classes}")
    for cls, count in class_counts.items():
        print(f"  - {cls}: {count} images ({count/total_images*100:.2f}% if total>0 else 0)")
        
    if widths and heights:
        print(f"Image Width Distribution: min={np.min(widths)}, max={np.max(widths)}, mean={np.mean(widths):.2f}")
        print(f"Image Height Distribution: min={np.min(heights)}, max={np.max(heights)}, mean={np.mean(heights):.2f}")
    
    return {
        'total_images': total_images,
        'class_counts': class_counts,
        'classes': classes
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds1', default='data/dataset1')
    parser.add_argument('--ds2', default='data/dataset2')
    args = parser.parse_args()
    
    np.random.seed(42)
    
    ds1_path = os.path.join(os.getcwd(), args.ds1)
    ds2_path = os.path.join(os.getcwd(), args.ds2)
    
    if os.path.exists(ds1_path):
        # Dataset 1 has Training and Testing subdirs at root
        # Let's check structure
        if os.path.exists(os.path.join(ds1_path, 'Training')):
            print("Dataset 1 Structure: Root -> Training/Testing -> Classes")
            analyze_dataset(os.path.join(ds1_path, 'Training'), "Dataset 1 (Training)")
            analyze_dataset(os.path.join(ds1_path, 'Testing'), "Dataset 1 (Testing)")
        else:
            analyze_dataset(ds1_path, "Dataset 1")
    else:
        print(f"Dataset 1 path not found: {ds1_path}")
        
    if os.path.exists(ds2_path):
        analyze_dataset(ds2_path, "Dataset 2")
    else:
        print(f"Dataset 2 path not found: {ds2_path}")
