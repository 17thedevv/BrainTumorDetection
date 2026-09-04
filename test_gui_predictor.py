import os
import random
from inference.predictor import Predictor

def test_predictor():
    predictor = Predictor(model_path="saved_model/ssl_best_model.pth", image_size=224)
    test_dir = "data/Dataset1/Testing"
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    correct = 0
    total = 0
    
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        images = os.listdir(cls_dir)
        random.shuffle(images)
        subset = images[:20]  # Test 20 images per class
        
        for img_name in subset:
            img_path = os.path.join(cls_dir, img_name)
            result = predictor.predict(img_path)
            
            pred_cls = result['class'].lower().replace(" ", "")
            if pred_cls == cls:
                correct += 1
            total += 1
            
    print(f"Accuracy on random test images: {correct}/{total} ({(correct/total)*100:.2f}%)")

if __name__ == "__main__":
    test_predictor()
