import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import time
import camogen


# Step 1: Extract dominant colors from multiple images using K-means clustering
def extract_colors_kmeans(image_paths, num_colors=5):
    """
    Use K-means clustering to extract dominant colors from multiple images.

    Parameters:
    - image_paths (list of str): List of paths to the images.
    - num_colors (int): Number of dominant colors to extract.

    Returns:
    - list of hex color strings.
    """
    all_pixels = []

    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1], 3))  # Flatten image into list of pixels
        all_pixels.extend(img)

    all_pixels = np.array(all_pixels)

    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(all_pixels)
    colors = kmeans.cluster_centers_.astype(int)

    # Convert to HEX color format
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]


# Step 2: Extract deep features using a pre-trained CNN (VGG16)
def extract_deep_features(image_paths):
    """
    Extract deep features from multiple images using a pretrained VGG16 model.

    Parameters:
    - image_paths (list of str): List of paths to environment images.

    Returns:
    - Aggregated deep features (numpy array).
    """
    model = VGG16(weights='imagenet', include_top=False)
    all_features = []

    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        features = model.predict(img_data)
        all_features.append(features.flatten())

    # Average the features across all images
    return np.mean(all_features, axis=0)


# Step 3: Use deep features to dynamically set spots and pixel parameters
def determine_spot_and_pixel_params(deep_features, base_spot_amount=10, base_pixel_amount=0.5):
    """
    Dynamically adjust the spots and pixelization parameters based on extracted deep features.

    Parameters:
    - deep_features (numpy array): Extracted deep features from the environment images.
    - base_spot_amount (int): Base number of spots.
    - base_pixel_amount (float): Base amount of pixelization.

    Returns:
    - spots_params (dict): Parameters for spots.
    - pixelize_params (dict): Parameters for pixelization.
    """
    # Analyze feature vector to adjust spot amount and size
    feature_variance = np.var(deep_features)  # Use variance as a measure of complexity
    feature_mean = np.mean(deep_features)  # Use mean to determine general texture complexity

    # Adjust spots based on texture complexity (high variance -> more spots)
    spot_amount = int(base_spot_amount + feature_variance * 50)
    spot_radius_min = max(5, 10 + int(feature_mean * 20))  # Minimum spot size related to mean feature
    spot_radius_max = spot_radius_min + 20  # Max radius is a bit larger than min

    spots_params = {
        "amount": spot_amount,
        "radius": {'min': spot_radius_min, 'max': spot_radius_max},
        "sampling_variation": int(feature_variance * 5)  # Adjust variation based on feature variance
    }

    # Adjust pixelization based on complexity (more complex textures -> more pixelization)
    pixelize_amount = min(1.0, base_pixel_amount + feature_variance / 10)
    pixel_density_x = 100 + int(feature_mean * 100)  # Increase pixel density based on feature mean
    pixel_density_y = pixel_density_x

    pixelize_params = {
        "percentage": pixelize_amount,
        "sampling_variation": int(feature_variance * 5),
        "density": {'x': pixel_density_x, 'y': pixel_density_y}
    }

    return spots_params, pixelize_params


# Step 4: Generate camouflage with dynamically determined spots and pixel parameters
def generate_camouflage_with_deep_features(image_paths, output_path, num_colors=5):
    """
    Generate camouflage pattern based on environment images, using deep features to set spots and pixels.

    Parameters:
    - image_paths (list of str): Paths to environment images.
    - output_path (str): Path to save the generated camouflage image.
    - num_colors (int): Number of dominant colors to use in camouflage.

    Returns:
    - None
    """
    # Extract dominant colors across all images
    environment_colors = extract_colors_kmeans(image_paths, num_colors=num_colors)

    # Extract deep features from environment images
    deep_features = extract_deep_features(image_paths)

    # Determine spot and pixel parameters based on deep features
    spots_params, pixelize_params = determine_spot_and_pixel_params(deep_features)

    # Define base camouflage parameters
    camouflage_params = {
        "width": 500,
        "height": 500,
        "polygon_size": 50,
        "color_bleed": 5,
        "colors": environment_colors,
        "max_depth": 10,
        "spots": spots_params,
        "pixelize": pixelize_params
    }
    print(camouflage_params)
    # Generate the camouflage
    camouflage_image = camogen.generate(camouflage_params)

    # Save the generated image
    camouflage_image.save(output_path)
    camouflage_image.show()






# Step 5: Example usage for generating camouflage based on a set of images
if __name__ == "__main__":
    # Specify the folder containing environment images
    image_folder = "Datasets/Data/forest_20"

    # Collect all image paths from the folder
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')]

    # Generate a timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")

# Create a unique filename using the timestamp
    output_filename = f"./outputs/adaptive_camouflage_{timestamp}.png"


    # Generate the adaptive camouflage pattern
    generate_camouflage_with_deep_features(image_paths, output_filename, num_colors=5)
