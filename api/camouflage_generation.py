import os
import numpy as np
from opensimplex import OpenSimplex
from noise import pnoise2
import time
from PIL import Image
import random
import math
import io
from colorthief import ColorThief


# images = os.listdir(initial_images_path)


def camouflage_generation(folder_id=timestamp, noise_image_path=noise_blend_image):

    initial_images_path = "timestamp"
    images = os.listdir(initial_images_path)

    resized_images = []

    for img in images:
        image = initial_images_path + "/" + img
        im = Image.open(image)
        im_resize = im.resize(
        (400, 400)
    )
    resized_images.append(im_resize)

    aspect = 1.77  # Aspect ratio of the output image

    cols = int(math.sqrt(len(images) * aspect))
    rows = int(math.ceil(float(len(images)) / float(cols)))

    random.shuffle(images)
    (w, h) = (400, 400)

    (width, height) = (w * cols, h * rows)

    collage = Image.new("RGB", (width, height))
    for y in range(rows):
        for x in range(cols):
            i = y * cols + x
            # Fill in extra images by duplicating some images randomly
            if i >= len(images):
                i = random.randrange(len(images))
            collage.paste(resized_images[i], (x * w, y * h))

    collage_img_resize = collage.resize((500, 500))

    return generate_color_palette(collage_img_resize)


    ######  return the collage image to the front end

def generate_color_palette(collage_img_resize):
    with io.BytesIO() as file_object:
        collage_img_resize.save(file_object, "PNG")
        color_thief = ColorThief(file_object)

        barColors = color_thief.get_palette(color_count=24, quality=1)
        barColors = (np.array(barColors)).astype(np.uint8)
        cols = len(barColors)
        rows = max([1, int(cols / 2.5)])

        # Create color Array
        barFullData = np.tile(barColors, (rows, 1)).reshape(rows, cols, 3)
        # Create Image from Array
        barImg = Image.fromarray(barFullData, "RGB")

    return barColors
