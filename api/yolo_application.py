from flask import Flask, request, Response, send_from_directory, jsonify
import os
import cv2
import time
from datetime import datetime
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Directory to save camouflaged images
SAVE_DIR = './static/camafalgues'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load YOLO model
model = YOLO('yolov8n-seg.pt')


def load_images(env_image, camo_image):
    """Function to load the environment and camouflage images"""
    return env_image, camo_image


def segment_objects(image, selected_objects):
    """Function to segment the objects from the environment image using YOLO"""
    results = model(image)
    img_height, img_width = image.shape[:2]
    
    # Extract masks for selected objects
    masks = []
    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            for mask, box in zip(result.masks.data, result.boxes):
                label = result.names[int(box.cls)]
                confidence = box.conf
                
                # Only consider objects of interest and high confidence
                if label in selected_objects and confidence > 0.5:
                    mask_resized = cv2.resize(mask.cpu().numpy(), (img_width, img_height))
                    masks.append(mask_resized)
                    
    return masks


def apply_step1_highlight(image, masks):
    """Function to apply mask highlight (green) to segmented areas (Step 1)"""
    # Keep the original image in BGR
    img_rgb = image.copy()  # Use a copy of the original image
    
    # Apply color changes only to masked areas
    for mask in masks:
        img_rgb[mask > 0.5] = (0, 255, 0)  # Only apply green to masked areas
    
    return img_rgb


def apply_camouflage_pattern(image, mask, pattern):
    """Function to apply the camouflage pattern to the segmented mask"""
    y_indices, x_indices = np.where(mask > 0.5)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return image  # Return original if no valid mask
    
    # Calculate the bounding box of the mask
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    
    # Resize the camouflage pattern to match the size of the object
    box_width = max_x - min_x
    box_height = max_y - min_y
    pattern_resized = cv2.resize(pattern, (box_width, box_height))
    
    # Apply camouflage pattern only to the masked area
    img_copy = image.copy()  # Ensure we don't modify the entire image
    roi = img_copy[min_y:max_y, min_x:max_x]
    roi[mask[min_y:max_y, min_x:max_x] > 0.5] = pattern_resized[mask[min_y:max_y, min_x:max_x] > 0.5]
    
    # Place the modified ROI back in the image
    img_copy[min_y:max_y, min_x:max_x] = roi
    
    return img_copy


def apply_camouflage_blending(image, mask, pattern, proximity_factor=1.0, alpha=0.7):
    """Function to apply camouflage with blending for realistic transitions (Step 3)"""
    y_indices, x_indices = np.where(mask > 0.5)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return image  # Return original if no valid mask
    
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    
    box_width = max_x - min_x
    box_height = max_y - min_y
    
    # Resize camouflage pattern based on proximity factor
    pattern_resized = cv2.resize(pattern, (int(box_width), int(box_height)))
    
    img_copy = image.copy()  # Ensure we don't modify the entire image
    
    roi = img_copy[min_y:max_y, min_x:max_x]
    
    mask_resized = cv2.resize(mask[min_y:max_y, min_x:max_x].astype(np.uint8), 
                              (int(box_width), int(box_height)))
    
    # Blend the pattern and the original image only in the masked area
    blended_roi = cv2.addWeighted(roi, 1 - alpha, pattern_resized, alpha, 0)
    
    roi[mask_resized > 0.5] = blended_roi[mask_resized > 0.5]
    img_copy[min_y:max_y, min_x:max_x] = roi
    
    return img_copy


def save_image(image, step, timestamp):
    """Function to save the image at different steps"""
    filename = f'camouflaged_{step}_{timestamp}.png'
    save_path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(save_path, image)
    return save_path


def yolo_application(env_image, camo_image, selected_objects, base_url):
    """Main application function to process the images"""
    # Load the images
    env_img, camo_img = load_images(env_image, camo_image)
    
    # Step 1: Segment objects
    masks = segment_objects(env_img, selected_objects)
    
    # Step 2: Highlight segmentation
    img_step1 = apply_step1_highlight(env_img, masks)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    step1_path = save_image(img_step1, 'green', timestamp)
    
    # Step 3: Apply camouflage pattern to segmented areas
    for mask in masks:
        img_step2 = apply_camouflage_pattern(img_step1, mask, camo_img)
    step2_path = save_image(img_step2, 'step2', timestamp)

    # Step 4: Apply camouflage with blending for a more realistic look
    for mask in masks:
        img_step3 = apply_camouflage_blending(img_step2, mask, camo_img)
    step3_path = save_image(img_step3, 'step3', timestamp)

    return step3_path


    
def check_detection(camo_applied_image, selected_objects):
    camo_applied_image = cv2.imread(camo_applied_image)
    img_rgb = cv2.cvtColor(camo_applied_image, cv2.COLOR_BGR2RGB)
    results_after_camouflage = model(img_rgb)

    # Check if any selected objects were detected
    no_detections = True  # Flag to check if there are no detections of selected objects

    # Iterate through results to see if any selected objects were detected
    for result in results_after_camouflage:
        if result.boxes:  # Check if there are any bounding boxes
            for box in result.boxes:
                label = result.names[int(box.cls)]
                if label in selected_objects:
                    no_detections = False
                    confidence = box.conf.item()  # Convert tensor to float
                    return f"Detected {label} with confidence {confidence:.2f}"

    # If no selected objects were detected
    if no_detections:
        return f"No detections: YOLO was unable to detect any of the selected objects ({', '.join(selected_objects)}) in the image."
