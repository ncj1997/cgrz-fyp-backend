##########################################################################
#            Step 5: First Iteration of Tessellation            #
##########################################################################

import io
import math
import os
import random
from tqdm import tqdm
from PIL import Image
from colorthief import ColorThief
import copy
from api.utils import generateNoiseImage


def first_iteration(quantized_swapped_colors,folder_id):
    height = 1000
    width = 1000
    num_cells = 1000

    rgbbase = quantized_swapped_colors.convert("RGB")
    first_voronoi = Image.new("RGB", (width, height))
    putpixel = first_voronoi.putpixel
    imgx, imgy = first_voronoi.size
    nx = []
    ny = []
    nr = []
    ng = []
    nb = []
    for i in tqdm(range(num_cells), desc="Creating Points"):
        nx.append(random.randrange(1, imgx))
        ny.append(random.randrange(1, imgy))
        rgb = rgbbase.getpixel((nx[i], ny[i]))
        nr.append(rgb[0])
        ng.append(rgb[1])
        nb.append(rgb[2])
    for y in tqdm(range(imgy), desc="Tessellating"):
        for x in range(imgx):
            dmin = math.hypot(imgx - 1, imgy - 1)
            j = -1
            for i in range(num_cells):
                d = math.hypot(nx[i] - x, ny[i] - y)
                if d < dmin:
                    dmin = d
                    j = i
            putpixel((x, y), (nr[j], ng[j], nb[j]))

    # display(first_voronoi)

    folder_path = os.path.join("./static/images/patterns/", folder_id)

    first_veronoi_path = f"{folder_path}/first_veronoi_image.png"

    first_voronoi.save(first_veronoi_path, 'PNG')

    return first_veronoi_path, first_voronoi






##########################################################################
#            Step 6: Second Iteration of Tessellation            #
##########################################################################

def single_color_fun(first_voronoi):
    with io.BytesIO() as file_object:
        first_voronoi.save(file_object, "PNG")
        color_thief = ColorThief(file_object)

        # https://stackoverflow.com/questions/56069551/trying-to-display-list-of-rgb-values
        single_color = color_thief.get_color(quality=1)
        return single_color

def second_iteration(first_voronoi,single_color,folder_id):
    
    # single_color = single_color_fun(first_voronoi)
    new_noise = generateNoiseImage()
    thresh = 222
    fn = lambda x : 255 if x > thresh else 0
    r = new_noise.convert('L').point(fn, mode='1')

    # display(r)
    height = 1000
    width = 1000
    num_cells = 1000
    tmp_thresh = 8
    darkness = 30

    if single_color[0] - darkness > 0:
        single_color_dark_r = single_color[0] - darkness
    else:
        single_color_dark_r = 0

    if single_color[1] - darkness > 0:
        single_color_dark_g = single_color[1] - darkness
    else:
        single_color_dark_g = 0

    if single_color[2] - darkness > 0:
        single_color_dark_b = single_color[2] - darkness
    else:
        single_color_dark_b = 0

    single_color_dark = (single_color_dark_r, single_color_dark_g, single_color_dark_b)

    second_voronoi = Image.new("RGBA", (width, height))

    putpixel = second_voronoi.putpixel
    imgx, imgy = second_voronoi.size
    nx = []
    ny = []
    nr = []
    ng = []
    nb = []
    na = []
    for i in tqdm(range(num_cells), desc="Creating Points"):
        nx.append(random.randrange(1, imgx))
        ny.append(random.randrange(1, imgy))
        bw = new_noise.getpixel((nx[i], ny[i]))
        nr.append(single_color_dark[0])
        ng.append(single_color_dark[1])
        nb.append(single_color_dark[2])
        if bw > tmp_thresh:
            na.append(255)
        else:
            na.append(0)

    for y in tqdm(range(imgy), desc="Tessellating"):
        for x in range(imgx):
            dmin = math.hypot(imgx - 1, imgy - 1)
            j = -1
            for i in range(num_cells):
                d = math.hypot(nx[i] - x, ny[i] - y)
                if d < dmin:
                    dmin = d
                    j = i
            putpixel((x, y), (nr[j], ng[j], nb[j], na[j]))

    # display(second_voronoi)

    bg = copy.deepcopy(first_voronoi)
    bg.paste(second_voronoi, (0, 0), second_voronoi)
    # display(bg)
    
    folder_path = os.path.join("./static/images/patterns/", folder_id)

    second_veronoi_path = f"{folder_path}/second_veronoi_image.png"

    bg.save(second_veronoi_path, 'PNG')

    return second_veronoi_path,bg

      
##########################################################################
#            Step 7: Second Iteration of Tessellation                    #
##########################################################################
def final_comouflague(single_color,second_veranoi, folder_id):
    new_noise = generateNoiseImage()
    thresh = 222
    fn = lambda x : 255 if x > thresh else 0
    r = new_noise.convert('L').point(fn, mode='1')

    # display(r)

    height = 1000
    width = 1000
    num_cells = 1000
    tmp_thresh = 10
    lightness = 30
    darkness = 30


    if single_color[0] + lightness < 255:
        single_color_light_r = single_color[0] + lightness
    else:
        single_color_light_r = 255

    if single_color[1] - darkness < 255:
        single_color_light_g = single_color[1] + lightness
    else:
        single_color_light_g = 255

    if single_color[2] - darkness < 255:
        single_color_light_b = single_color[2] + lightness
    else:
        single_color_light_b = 255

    single_color_light = (single_color_light_r, single_color_light_g, single_color_light_b)

    third_voronoi = Image.new("RGBA", (width, height))

    putpixel = third_voronoi.putpixel
    imgx, imgy = third_voronoi.size
    nx = []
    ny = []
    nr = []
    ng = []
    nb = []
    na = []
    for i in tqdm(range(num_cells), desc="Creating Points"):
        nx.append(random.randrange(1, imgx))
        ny.append(random.randrange(1, imgy))
        bw = new_noise.getpixel((nx[i], ny[i]))
        nr.append(single_color_light[0])
        ng.append(single_color_light[1])
        nb.append(single_color_light[2])
        if bw > tmp_thresh:
            na.append(255)
        else:
            na.append(0)

    for y in tqdm(range(imgy), desc="Tessellating"):
        for x in range(imgx):
            dmin = math.hypot(imgx - 1, imgy - 1)
            j = -1
            for i in range(num_cells):
                d = math.hypot(nx[i] - x, ny[i] - y)
                if d < dmin:
                    dmin = d
                    j = i
            putpixel((x, y), (nr[j], ng[j], nb[j], na[j]))

    # display(third_voronoi)
    import copy
    bg2 = copy.deepcopy(bg)
    bg2.paste(third_voronoi, (0, 0), third_voronoi)
    # display(bg2)

    folder_path = os.path.join("./static/images/patterns/", folder_id)

    final_pattern_path = f"{folder_path}/final_pattern_image.png"
 
    bg2.save(final_pattern_path, 'PNG')

    return final_pattern_path