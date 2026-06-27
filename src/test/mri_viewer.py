import cv2
import numpy as np
import argparse
import os
import tkinter as tk
from tkinter import filedialog

def dummy_mri_image(width=512, height=512):
    # Create a dummy image for testing if no image is provided
    img = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(img, (width//2, height//2), 150, 255, -1)
    cv2.circle(img, (width//2, height//2), 100, 100, -1)
    cv2.circle(img, (width//2 - 50, height//2 - 20), 20, 50, -1)
    cv2.circle(img, (width//2 + 50, height//2 - 20), 20, 50, -1)
    # Add some noise
    noise = np.random.randint(0, 50, (height, width), dtype=np.uint8)
    img = cv2.add(img, noise)
    return img

def main():
    parser = argparse.ArgumentParser(description="Basic MRI Image Viewer and Editor using OpenCV")
    parser.add_argument("-i", "--image", help="Path to the MRI image", default=None)
    args = parser.parse_args()

    image_path = args.image

    if not image_path:
        root = tk.Tk()
        root.withdraw() # Hide the main window
        print("Please select an MRI image in the file dialog...")
        image_path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"), ("All files", "*.*")]
        )
        if not image_path:
            print("No file selected.")

    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image at {image_path}. Using dummy image instead.")
            img = dummy_mri_image()
    else:
        print("No valid image provided. Generating dummy MRI image...")
        img = dummy_mri_image()

    original_img = img.copy()
    current_img = img.copy()
    
    # Trackbar callback function (does nothing but required by createTrackbar)
    def on_trackbar(val):
        pass

    window_name = 'MRI Viewer and Editor'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Create trackbars for basic adjustments
    cv2.createTrackbar('Brightness', window_name, 50, 100, on_trackbar)
    cv2.createTrackbar('Contrast', window_name, 50, 100, on_trackbar)
    cv2.createTrackbar('Blur', window_name, 0, 20, on_trackbar)

    print("--- MRI Viewer Controls ---")
    print("Adjust sliders for Brightness, Contrast, and Blur.")
    print("Press 'r' to reset to original image.")
    print("Press 'e' to toggle Edge Detection (Canny).")
    print("Press 's' to save the current image as 'edited_mri.png'.")
    print("Press 'q' or 'ESC' to quit.")

    show_edges = False

    while True:
        # Get trackbar values
        brightness = cv2.getTrackbarPos('Brightness', window_name) - 50
        contrast = cv2.getTrackbarPos('Contrast', window_name)
        blur_val = cv2.getTrackbarPos('Blur', window_name)

        # Apply Contrast and Brightness
        # Formula: new_img = alpha * old_img + beta
        alpha = contrast / 50.0  # Simple contrast adjustment 
        beta = brightness * 2    # Simple brightness adjustment
        
        current_img = cv2.convertScaleAbs(original_img, alpha=alpha, beta=beta)
        
        # Apply Blur
        if blur_val > 0:
            ksize = blur_val * 2 + 1 # Kernel size must be odd
            current_img = cv2.GaussianBlur(current_img, (ksize, ksize), 0)

        # Apply Edge Detection
        if show_edges:
            current_img = cv2.Canny(current_img, 100, 200)

        cv2.imshow(window_name, current_img)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('r'):
            # Reset
            cv2.setTrackbarPos('Brightness', window_name, 50)
            cv2.setTrackbarPos('Contrast', window_name, 50)
            cv2.setTrackbarPos('Blur', window_name, 0)
            show_edges = False
            print("Reset to original image.")
        elif key == ord('e'):
            show_edges = not show_edges
            print(f"Edge Detection: {'ON' if show_edges else 'OFF'}")
        elif key == ord('s'):
            cv2.imwrite('edited_mri.png', current_img)
            print("Image saved as 'edited_mri.png'.")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
