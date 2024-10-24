import os
from colorthief import ColorThief
import numpy as np
from PIL import Image
import io


def color_platte_generation(collage_im_resize, folder_id):
# Save the image to an in-memory file
    with io.BytesIO() as file_object:
        collage_im_resize.save(file_object, "PNG")
        color_thief = ColorThief(file_object)

        # Get the palette of 24 colors
        barColors = color_thief.get_palette(color_count=24, quality=1)
        barColors = (np.array(barColors)).astype(np.uint8)

        # Calculate grid size (rows and columns) for a square-like layout
        num_colors = len(barColors)
        grid_size = int(np.ceil(np.sqrt(num_colors)))  # Ensures a square grid

        # Create a new array for the color grid
        barFullData = np.zeros((grid_size * 50, grid_size * 50, 3), dtype=np.uint8)  # Each square is 50x50 pixels

        # Fill the grid with colors
        for idx, color in enumerate(barColors):
            row = idx // grid_size
            col = idx % grid_size
            barFullData[row*50:(row+1)*50, col*50:(col+1)*50] = color

        # Create Image from Array
        barImg = Image.fromarray(barFullData, "RGB")

        # Display the image
        # display(barImg)
        folder_path = os.path.join("./static/images/patterns/", folder_id)

        color_platte_img_path = f"{folder_path}/colour_platte_image.png"
 
        barImg.save(color_platte_img_path, 'PNG')

        return color_platte_img_path, barColors
    

def apply_colors_noise_image(noise_image, barcolors,folder_id):

    swapped_colors = noise_image.convert("P")

    swapped_colors.putpalette(barcolors)

    quantized_swapped_colors = swapped_colors.quantize(colors=16)

    folder_path = os.path.join("./static/images/patterns/", folder_id)

    colour_noise_img_path = f"{folder_path}/colour_noise_image.png"

    quantized_swapped_colors.save(colour_noise_img_path, 'PNG')

    return colour_noise_img_path, swapped_colors

    