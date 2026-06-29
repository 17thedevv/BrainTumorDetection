import argparse
import os
import torch
from configs.config import Config
from datasets.data_module import DataModule
from models.cnn import BaselineCNN
from training.metrics import calculate_metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(config_path: str, checkpoint_path: str, output_dir: str = "evaluation_results"):
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return
        
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} not found.")
        return

    config = Config.from_yaml(config_path)
    
    # Initialize Dataset
    data_module = DataModule(config.data)
    # The testing data is what we care about here.
    # In DataModule, get_val_dataloader actually returns the Test split according to Phase 1 logic.
    test_loader = data_module.get_val_dataloader()
    
    print(f"Loaded Test Dataset. Total batches: {len(test_loader)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize Model
    model = BaselineCNN(
        name=config.model.name, 
        num_classes=config.model.num_classes, 
        pretrained=False # No need to download pretrained weights when loading a checkpoint
    )
    
    # Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')} with validation loss {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    all_preds = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate Metrics
    metrics = calculate_metrics(all_labels, all_preds)
    print("\n--- Overall Metrics ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    # Detailed Classification Report
    classes = test_loader.dataset.classes
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Confusion Matrix
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Baseline CNN')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Baseline Model")
    parser.add_argument('--config', type=str, default='experiments/baseline.yaml', help='Path to config yaml file')
    parser.add_argument('--checkpoint', type=str, default='saved_model/best_baseline_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Directory to save evaluation artifacts')
    args = parser.parse_args()
    
    evaluate_model(args.config, args.checkpoint, args.output)
