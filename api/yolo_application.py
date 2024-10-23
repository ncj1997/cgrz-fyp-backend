from flask import Flask, request, Response, send_from_directory,jsonify
import os
import cv2
import time
from datetime import datetime
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Path to save camouflaged images
SAVE_DIR = './static/camafalgues'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load YOLO model
model = YOLO('yolov8n-seg.pt')

# Function to apply camouflage with blending
def apply_camouflage_with_blending(img, mask, pattern, proximity_factor=1.0, alpha=0.7):
    y_indices, x_indices = np.where(mask > 0.5)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return img

    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    box_width = max_x - min_x
    box_height = max_y - min_y

    # Adjust scaling based on object size
    if box_width * box_height < 10000:
        scaling_factor = 0.5 * proximity_factor
    else:
        scaling_factor = 1.5 * proximity_factor

    pattern_resized = cv2.resize(pattern, (int(box_width), int(box_height)))
    roi = img[min_y:max_y, min_x:max_x]
    mask_resized = cv2.resize(mask[min_y:max_y, min_x:max_x].astype(np.uint8), (int(box_width), int(box_height)))

    blended_roi = cv2.addWeighted(roi, 1 - alpha, pattern_resized, alpha, 0)
    roi[mask_resized > 0.5] = blended_roi[mask_resized > 0.5]
    img[min_y:max_y, min_x:max_x] = roi

    return img

# SSE stream to send real-time updates
def sse_stream(env_image, camo_image, selected_objects, base_url):
                # Load YOLOv8 model (segmentation)
        model = YOLO('yolov8n-seg.pt')

        # Load the image (replace with your image path in Colab)
        img = env_image

        # Get the original image dimensions
        img_height, img_width = img.shape[:2]

        # Convert BGR (OpenCV format) to RGB (Matplotlib format)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run YOLO model for segmentation
        results = model(img)

        # Define the object types you are interested in
        object_types = selected_objects

        # Extract masks, boxes, labels, and confidences
        masks = []
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
                    # box.cls is the index of the class, map it to the actual name
                    label = result.names[int(box.cls)]
                    confidence = box.conf

                    # Check if the object type matches and confidence is above threshold
                    if label in object_types and confidence > 0.5:
                        # Resize the mask to match the original image size
                        mask_resized = cv2.resize(mask.cpu().numpy(), (img_width, img_height))  # Convert tensor to numpy and resize
                        masks.append(mask_resized)
       
        # Apply masks to the image (visualization)
        for mask in masks:
            img_rgb[mask > 0.5] = (0, 255, 0)  # Mark the masked area in green

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        save_path_one = os.path.join(SAVE_DIR, f'camouflaged_green_{timestamp}.png')
        cv2.imwrite(save_path_one, img_rgb)


        # Load the camouflage pattern (replace with your own pattern)
        pattern = camo_image

    # Function to apply camouflage to a masked area
        def apply_camouflage(img, mask, pattern):
            # Get the bounding box of the mask (to resize the pattern)
            y_indices, x_indices = np.where(mask > 0.5)
            
            # If no valid indices are found (mask too small), return original image
            if len(y_indices) == 0 or len(x_indices) == 0:
                return img
            
            min_x, max_x = np.min(x_indices), np.max(x_indices)
            min_y, max_y = np.min(y_indices), np.max(y_indices)

            # Calculate the size of the bounding box
            box_width = max_x - min_x
            box_height = max_y - min_y

            # Resize the camouflage pattern to fit the size of the object
            pattern_resized = cv2.resize(pattern, (box_width, box_height))

            # Extract the region of interest (ROI) from the original image where the mask is
            roi = img[min_y:max_y, min_x:max_x]

            # Apply the camouflage pattern to the masked area in the ROI
            roi[mask[min_y:max_y, min_x:max_x] > 0.5] = pattern_resized[mask[min_y:max_y, min_x:max_x] > 0.5]

            # Place the modified ROI back in the original image
            img[min_y:max_y, min_x:max_x] = roi

            return img

        # Apply the camouflage pattern to all masks
        for mask in masks:
            img_step_two = apply_camouflage(img_rgb, mask, pattern)

        save_path_two = os.path.join(SAVE_DIR, f'camouflaged_step2_{timestamp}.png')
        cv2.imwrite(save_path_two, img_step_two)
        
                # Function to apply camouflage with blending for more realistic transitions
        def apply_camouflage_with_blending(img, mask, pattern, proximity_factor=1.0, alpha=0.7):
            # Get the bounding box of the mask (to resize the pattern)
            y_indices, x_indices = np.where(mask > 0.5)
            
            # If no valid indices are found (mask too small), return original image
            if len(y_indices) == 0 or len(x_indices) == 0:
                return img
            
            min_x, max_x = np.min(x_indices), np.max(x_indices)
            min_y, max_y = np.min(y_indices), np.max(y_indices)

            # Calculate the size of the bounding box
            box_width = max_x - min_x
            box_height = max_y - min_y

            # Adjust the pattern size based on the bounding box size (proximity scaling)
            if box_width * box_height < 10000:  # Smaller objects, scale down the pattern
                scaling_factor = 0.5 * proximity_factor  # Scale down for distant objects
            else:  # Larger objects, use the original or even scale up slightly
                scaling_factor = 1.5 * proximity_factor  # Scale up for nearer objects

            # Resize the camouflage pattern based on scaling factor
            pattern_resized = cv2.resize(pattern, 
                                        (int(box_width), int(box_height)))

            # Extract the region of interest (ROI) from the original image where the mask is
            roi = img[min_y:max_y, min_x:max_x]

            # Resize the mask to match the pattern size
            mask_resized = cv2.resize(mask[min_y:max_y, min_x:max_x].astype(np.uint8), 
                                    (int(box_width), int(box_height)))

            # Apply alpha blending between the pattern and the original image
            blended_roi = cv2.addWeighted(roi, 1 - alpha, pattern_resized, alpha, 0)

            # Apply the camouflage pattern only where the mask is true
            roi[mask_resized > 0.5] = blended_roi[mask_resized > 0.5]

            # Place the modified ROI back in the original image
            img[min_y:max_y, min_x:max_x] = roi

            return img
        
        # Apply the camouflage pattern with blending to all masks

        for mask in masks:
            img_step3 = apply_camouflage_with_blending(img_rgb, mask, pattern)

        save_path_step3 = os.path.join(SAVE_DIR, f'camouflaged_step3_{timestamp}.png')

        cv2.imwrite(save_path_step3, img_step3)

        return save_path_step3
    
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


@app.route('/apply_camouflage', methods=['POST'])
def apply_camouflage():
    base_url = request.host_url.rstrip('/')  # Get the base URL
    
    # Get the images and object type
    env_image_file = request.files['environment_image']
    camo_image_file = request.files['camouflage_image']
    object_type = request.form['object_type']

    # Load the images
    env_image = cv2.imdecode(np.frombuffer(env_image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    camo_image = cv2.imdecode(np.frombuffer(camo_image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Object type mapping
    object_types = {
        'humans': ['person'],
        'vehicles': ['car', 'bus', 'truck', 'motorcycle']
    }
    selected_objects = object_types.get(object_type, [])

    final_applied_images = sse_stream(env_image, camo_image, selected_objects, base_url)


    detection_result = check_detection(final_applied_images,selected_objects)

        # Use os.path.relpath to get the relative path from the static folder
    relative_path = os.path.relpath(final_applied_images, start='static')
    # image_url = f"{base_url}/static/{relative_path.replace(os.sep, '/')}"
    image_url = f"{base_url}/static/camafalgues/camouflaged_step3_20241023211000.png"
    # Return the image URL as JSON
    # detection_result = ""
    return jsonify({'image_url': image_url,'detection_result': detection_result})

# Serve the camouflaged image
@app.route('/static/camafalgues/<filename>')
def serve_image(filename):
    return send_from_directory(SAVE_DIR, filename)

