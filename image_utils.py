from PIL import Image, ImageDraw
import numpy as np
from patchmatch import patch_match
import time

def fill_outpaint_area(image, outpaint_sizes, color, is_mask=False):
    original_width, original_height = image.size
    new_size = (original_width + outpaint_sizes["right"] + outpaint_sizes["left"], original_height + outpaint_sizes["up"] + outpaint_sizes["down"])

    if not is_mask:
        new_image = Image.new("RGB", new_size, "black")
        paste_position = (outpaint_sizes["left"], outpaint_sizes["up"])
        new_image.paste(image, paste_position)
    else:
        new_image = Image.new("RGB", new_size, "white")
    if color == 'patch':
        return fill_with_patchmatch(new_image, outpaint_sizes)
    else:
        return fill_with_color(new_image, outpaint_sizes, color)

def patchmatch(image, mask):
    if patch_match.patchmatch_available:
        start_time = time.time() * 1000  # Get the current time in milliseconds
        print("Running PatchMatch")
        result = patch_match.inpaint(np.array(image), np.array(mask), patch_size=2)
        end_time = time.time() * 1000  # Get the current time again after the function has completed
        elapsed_time_ms = end_time - start_time
        print(f"PatchMatch completed, time taken: {elapsed_time_ms} ms")
        return Image.fromarray(result)

    else:
        print("PatchMatch is not available.")
        return image
    

def fill_with_patchmatch(image, outpaint_sizes):
    original_width, original_height = image.size

    mask = Image.new("L", image.size, 0)  # Entirely black

    if outpaint_sizes["left"] != 0:
        mask_area = (0, 0, outpaint_sizes["left"], original_height)
        mask.paste(255, mask_area)  # White in the area to be outpainted
        image = patchmatch(image, mask)
        original_width, original_height = image.size
        
    if outpaint_sizes["right"] != 0:
        mask_area = (original_width - outpaint_sizes["right"], 0, original_width, original_height)
        mask.paste(255, mask_area)
        image = patchmatch(image, mask)
        original_width, original_height = image.size
        
    if outpaint_sizes["down"] != 0:
        mask_area = (0, original_height - outpaint_sizes["down"], original_width, original_height)
        mask.paste(255, mask_area)
        image = patchmatch(image, mask)
        original_width, original_height = image.size
        
    if outpaint_sizes["up"] != 0:
        mask_area = (0, 0, original_width, outpaint_sizes["up"])
        mask.paste(255, mask_area)
        image = patchmatch(image, mask)
   
    return image

def fill_with_color(image, outpaint_sizes, color):

    original_width, original_height = image.size

    color_area = (outpaint_sizes["left"], outpaint_sizes["up"], original_width - outpaint_sizes["right"], original_height - outpaint_sizes["down"])
    draw = ImageDraw.Draw(image)
    draw.rectangle(color_area, fill=color)
            
    return image
