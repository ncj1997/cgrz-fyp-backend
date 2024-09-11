from datetime import datetime
import io
import os
import numpy as np 
import cv2 # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import camogen # Replace with the actual camouflage generator library
from sklearn.cluster import KMeans # type: ignore
import time
from PIL import Image # type: ignore

# Step 1: Extract dominant colors from multiple images using K-means clustering
def extract_colors_kmeans(image_list, num_colors=3):
    """
    Use K-means clustering to extract dominant colors from multiple images.
    
    Parameters:
    - image_list (list of numpy arrays): List of images as numpy arrays.
    - num_colors (int): Number of dominant colors to extract.
    
    Returns:
    - list of hex color strings.
    """
    all_pixels = []

    for img in image_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1], 3))  # Flatten image into list of pixels
        all_pixels.extend(img)

    all_pixels = np.array(all_pixels)

    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(all_pixels)
    colors = kmeans.cluster_centers_.astype(int)

    # Convert to HEX color format
    return [f"#{b:02x}{g:02x}{r:02x}" for r, g, b in colors]

# Step 2: Extract deep features using a pre-trained CNN (VGG16)
def extract_deep_features(image_list):
    """
    Extract deep features from multiple images using a pretrained VGG16 model.
    
    Parameters:
    - image_list (list of numpy arrays): List of images as numpy arrays.
    
    Returns:
    - Aggregated deep features (numpy array).
    """
    model = VGG16(weights='imagenet', include_top=False)
    all_features = []

    for img in image_list:
        img = cv2.resize(img, (224, 224))  # Resize to the input size for VGG16
        img_data = np.expand_dims(img, axis=0)
        img_data = preprocess_input(img_data)
        
        features = model.predict(img_data)
        all_features.append(features.flatten())
    
    # Average the features across all images
    return np.mean(all_features, axis=0)

# Step 3: Use deep features to dynamically set spots, pixel, and other parameters
def determine_dynamic_params(deep_features, base_spot_amount=10, base_pixel_amount=0.5):
    """
    Dynamically adjust the spots, pixelization, polygon size, color bleed, and max depth
    parameters based on extracted deep features.
    
    Parameters:
    - deep_features (numpy array): Extracted deep features from the environment images.
    - base_spot_amount (int): Base number of spots.
    - base_pixel_amount (float): Base amount of pixelization.
    
    Returns:
    - spots_params (dict): Parameters for spots.
    - pixelize_params (dict): Parameters for pixelization.
    - polygon_size (int): Dynamically determined polygon size.
    - color_bleed (int): Dynamically determined color bleed.
    - max_depth (int): Dynamically determined max depth.
    """
    feature_variance = np.var(deep_features)  # Use variance as a measure of complexity
    feature_mean = np.mean(deep_features)    # Use mean to determine general texture complexity
    
    # Adjust spots based on texture complexity (high variance -> more spots)
    spot_amount = int(base_spot_amount + feature_variance * 80)
    spot_radius_min = max(5, 10 + int(feature_mean * 20))  # Minimum spot size related to mean feature
    spot_radius_max = spot_radius_min + 20  # Max radius is a bit larger than min
    
    spots_params = {
        "amount": spot_amount,
        "radius": {'min': spot_radius_min, 'max': spot_radius_max},
        "sampling_variation": int(feature_variance * 5)  # Adjust variation based on feature variance
    }
    
    # Adjust pixelization based on complexity (more complex textures -> more pixelization)
    pixelize_amount = min(1.0, base_pixel_amount + feature_variance / 50)
    pixel_density_x = 50 + int(feature_mean * 100)  # Increase pixel density based on feature mean
    pixel_density_y = pixel_density_x
    
    pixelize_params = {
        "percentage": pixelize_amount,
        "sampling_variation": int(feature_variance * 5),
        "density": {'x': pixel_density_x, 'y': pixel_density_y}
    }
    
    # Dynamically set polygon size (smaller for higher variance)
    polygon_size = int(100 / (feature_variance + 1))  # Larger variance -> smaller polygons
    
    # Dynamically set color bleed (smoother textures -> higher bleed)
    color_bleed = int(10 * (1 - feature_mean))  # Lower mean -> more bleed

    # Dynamically set max depth (higher for more complex textures)
    max_depth = int(15 + feature_variance * 10)

    return spots_params, pixelize_params, polygon_size, color_bleed, max_depth


def generate_camouflage(image_list,num_colors):
    
    environment_colors = extract_colors_kmeans(image_list, num_colors=num_colors)
    deep_features = extract_deep_features(image_list)
    
    # Determine dynamic parameters (spots, pixelization, polygon size, color bleed, max depth)
    spots_params, pixelize_params, polygon_size, color_bleed, max_depth = determine_dynamic_params(deep_features)
    
    # Define base camouflage parameters
    camouflage_params = {
        "width": 500,
        "height": 500,
        "polygon_size": polygon_size,
        "color_bleed": color_bleed,
        "colors": environment_colors,
        "max_depth": max_depth,
        "spots": spots_params,
        "pixelize": pixelize_params
    }
    print(camouflage_params)
    # Generate the camouflage using the camouflage generator library
    # Generate the camouflage using the camouflage generator library
    camouflage_image = camogen.generate(camouflage_params)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # # Save the image to a BytesIO stream
    #     img_io = io.BytesIO()
    #     camouflage_image.save(img_io, 'PNG')
    #     img_io.seek(0)


    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    output_file_path = f'./static/images/patterns/camouflaged_{timestamp}.png'

    camouflage_image.save(output_file_path)

    # IMAGE_FOLDER = os.path.join('static','images', 'patterns')

     # Output path for the camouflaged image (it will be saved in the /images/pattern folder)
    # final_output_path = os.path.join(IMAGE_FOLDER, output_file_path)

    path_img = f'images/patterns/camouflaged_{timestamp}.png'

    return path_img
