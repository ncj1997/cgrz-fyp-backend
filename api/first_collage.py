import os
from PIL import Image
import math
import random
import numpy as np
import cv2

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


def detect_and_remove_sky_from_list(image_list, threshold=0.1):
    # Load the pre-trained ResNet50 model
    model = ResNet50(weights='imagenet')

    for i, img in enumerate(image_list):
        # Preprocess the image for ResNet50
        img_resnet = cv2.resize(img, (224, 224))
        img_resnet = cv2.cvtColor(img_resnet, cv2.COLOR_BGR2RGB)
        x = image.img_to_array(img_resnet)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make predictions
        preds = model.predict(x)

        # Decode the predictions
        decoded_preds = decode_predictions(preds, top=3)[0]

        # Check if 'sky' is in the top predictions
        sky_detected = False
        for _, label, score in decoded_preds:
            if 'sky' in label and score > threshold:
                sky_detected = True
                break

        if sky_detected or check_blue_dominance(img):
            # Convert to HSV color space
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Define range of blue color in HSV
            lower_blue = np.array([100,50,50])
            upper_blue = np.array([130,255,255])

            # Threshold the HSV image to get only blue colors
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Create a black image of the same size as original
            black = np.zeros(img.shape, np.uint8)

            # Copy the non-sky parts of the original image
            result = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

            # Add the black sky
            result = cv2.add(result, black, mask=mask)

            # Replace the original image in the list with the processed image
            image_list[i] = result

    return image_list

def check_blue_dominance(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    lower_blue = np.array([100,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Calculate the percentage of blue pixels
    blue_ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])

    # If more than 30% of the image is blue, it's likely to contain sky
    return blue_ratio > 0.3

def generate_first_collage(images, timestamp):
    resized_images = []

    for img in images:
        # image = timestamp + "/" + img
        im = Image.open(img)
        im_resize = im.resize(
            (400, 400)
        )
        resized_images.append(im_resize)

    # resized_images = detect_and_remove_sky_from_list(resized_images)

    aspect = 1.77  # Aspect ratio of the output image

    cols = int(math.sqrt(len(images) * aspect))
    rows = int(math.ceil(float(len(images)) / float(cols)))

    random.shuffle(images)
    (w, h) = (400, 400)

    (width, height) = (w * cols, h * rows)

    first_collage_img = Image.new("RGB", (width, height))
    for y in range(rows):
        for x in range(cols):
            i = y * cols + x
            # Fill in extra images by duplicating some images randomly
            if i >= len(images):
                i = random.randrange(len(images))
            first_collage_img.paste(resized_images[i], (x * w, y * h))

    first_collage_img.resize((500, 500))

    folder_path = os.path.join("./static/images/patterns/", timestamp)

    # Make the directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    first_collage_path = f"{folder_path}/first_collage_image.png"

    first_collage_img.save(first_collage_path, 'PNG')

    return first_collage_path, first_collage_img
