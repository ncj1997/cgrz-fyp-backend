import os
import numpy as np
from opensimplex import OpenSimplex
from noise import pnoise2
import time
from PIL import Image

def generateRandomNoise():

    (w, h) = (1000,1000)
    num_samples = w * h
    # for pix in num_samples:
    return np.random.random(size=num_samples)


def generateSimplexNoise():

    (w, h) = (1000, 1000)
    arr = []
    simp = OpenSimplex(seed=int(time.time()))
    for y in range(0, h):
        for x in range(0, w):
            arr.append(simp.noise2(x / 40, y / 40))
    return arr


def generatePerlinNoise():

    (w, h) = (1000,1000)
    octaves = 10
    freq = 30.0 * octaves
    arr = []
    for y in range(0, h):
        for x in range(0, w):
            arr.append(pnoise2(x / freq, y / freq, octaves))
    return arr



def generateNoiseImage(folder_id, existing_image_path=None ):
    (w, h) = (1000, 1000)  # Dimensions for noise generation
    num_samples = w * h
    final_values = []

    # Process the existing image if provided
    if existing_image_path:
        existing_image = Image.open(existing_image_path)
        existing_image = existing_image.resize((w, h))  # Resize to match noise size
        existing_image_np = np.array(existing_image) / 255.0  # Normalize to [0, 1] range

    # Generate random noise (assuming generateRandomNoise, generatePerlinNoise, generateSimplexNoise exist)
    random = generateRandomNoise()  # Generate random noise
    perlin = generatePerlinNoise()  # Generate Perlin noise
    simplex = generateSimplexNoise()  # Generate Simplex noise

    # Static values for HSV
    static_val = np.empty(num_samples)
    static_val.fill(0.5)

    static_val2 = np.empty(num_samples)
    static_val2.fill(1.0)

    # Blend the noises
    for a in range(0, w * h):
        final_values.append(
            (
                (random[a] * 0.02)
                + (((simplex[a] / 2) + 0.5) * 0.48)
                + (((perlin[a] / 2) + 0.5) * 0.5)
            )
            / 3
        )

    h_samples = np.reshape(np.asarray(final_values), (w, h))
    s_samples = np.reshape(static_val, (w, h))
    v_samples = np.reshape(static_val2, (w, h))

    # Stack HSV channels to create the HSV image
    hsv_img = np.dstack([h_samples, s_samples, v_samples])

    # Convert HSV to RGB
    noise_image = Image.fromarray(np.uint8(hsv_img * 255), mode="HSV").convert("RGB")

    # Optional: blend with existing image
    if existing_image:
        # Blend noise image and the existing image using a weighted sum
        noise_image_np = np.array(noise_image) / 255.0
        blended_image = (0.7 * noise_image_np + 0.3 * existing_image_np)
        # blended_image = noise_image_np

        blended_image = np.uint8(blended_image * 255)  # Convert back to [0, 255] range
        blended_image = Image.fromarray(blended_image)
    else:
        blended_image = noise_image

    # Return the blended or generated image directly (skip quantization)
    blended_image = blended_image.quantize(colors=12)
    blended_image.convert(mode="RGB")
    
    folder_path = os.path.join("./static/images/patterns/", folder_id)

    noise_img_path = f"{folder_path}/generated_noise_blended_image.png"
    print("noise image path", noise_img_path)
    # Save the final collage image

    blended_image.save(noise_img_path, 'PNG')

    return noise_img_path


