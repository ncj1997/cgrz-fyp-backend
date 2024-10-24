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

    (w, h) = (1000,1000)

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

def generateNoiseImage():

    (w, h) = (1000, 1000)
    num_samples = w * h
    final_values = []

    random = generateRandomNoise()
    perlin = generatePerlinNoise()
    simplex = generateSimplexNoise()

    static_val = np.empty(num_samples)
    static_val.fill(0.5)

    static_val2 = np.empty(num_samples)
    static_val2.fill(1.0)

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

    hsv_img = np.dstack([h_samples, s_samples, v_samples])

    image = Image.fromarray(np.uint8(hsv_img * 255), mode="HSV").convert("RGB")
    image = image.quantize(colors=12)
    image.convert(mode="RGB")

    return image