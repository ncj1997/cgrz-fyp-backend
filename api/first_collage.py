import os
from random import random
from PIL import Image
import math


def generate_first_collage(images, folder_path):
    resized_images = []
    for img in images:
        image = folder_path + "/" + img
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

    first_collage_img = Image.new("RGB", (width, height))
    for y in range(rows):
        for x in range(cols):
            i = y * cols + x
            # Fill in extra images by duplicating some images randomly
            if i >= len(images):
                i = random.randrange(len(images))
            first_collage_img.paste(resized_images[i], (x * w, y * h))

    folder_path = os.path.join("./static//", folder_path)

    first_collage_path = f"{folder_path}/first_collage_image.png"

    first_collage_img.save(first_collage_path, 'PNG')

    return first_collage_img