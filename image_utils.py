from PIL import Image, ImageDraw
import numpy as np
from patchmatch import patch_match
import time

def fill_outpaint_area(image, outpaint_direction, outpaint_size, color, is_mask=False):
    original_width, original_height = image.size

    new_size = {
        'left': (original_width + outpaint_size, original_height),
        'right': (original_width + outpaint_size, original_height),
        'up': (original_width, original_height + outpaint_size),
        'down': (original_width, original_height + outpaint_size)
    }[outpaint_direction]
    new_image = Image.new("RGB", new_size, "black")

    if not is_mask:
        paste_position = {
            'left': (outpaint_size, 0),
            'right': (0, 0),
            'up': (0, outpaint_size),
            'down': (0, 0)
        }[outpaint_direction]
        new_image.paste(image, paste_position)

    if color == 'patch':
        print("color == patch")
        return fill_with_patchmatch(new_image, outpaint_direction, outpaint_size)
    else:

        print("color!=patch")
        return fill_with_color(new_image, outpaint_direction, outpaint_size, color)

def fill_with_patchmatch(image, outpaint_direction, outpaint_size):
    image.save('input_patch_image.png')
    original_width, original_height = image.size
    new_size = image.size

    mask = Image.new("L", new_size, 0)  # Entirely black
    mask_area = {
        'left': (0, 0, outpaint_size, original_height),
        'right': (original_width - outpaint_size, 0, original_width, original_height),
        'up': (0, 0, original_width, outpaint_size),
        'down': (0, original_height - outpaint_size, original_width, original_height)
    }[outpaint_direction]
    mask.paste(255, mask_area)  # White in the area to be outpainted

    if patch_match.patchmatch_available:
        start_time = time.time() * 1000  # Get the current time in milliseconds
        print("Running PatchMatch")
        result = patch_match.inpaint(np.array(image), np.array(mask), patch_size=2)
        end_time = time.time() * 1000  # Get the current time again after the function has completed
        elapsed_time_ms = end_time - start_time
        print(f"PatchMatch completed, time taken: {elapsed_time_ms} ms")
        Image.fromarray(result).save('PatchMatch_image.png')
        return Image.fromarray(result)
    else:
        print("PatchMatch is not available.")
        return image

def fill_with_color(image, outpaint_direction, outpaint_size, color):
    image.save('color_input.png')
    color_area = {
        'left': (0, 0, outpaint_size, image.height),
        'right': (image.width - outpaint_size, 0, image.width, image.height),
        'up': (0, 0, image.width, outpaint_size),
        'down': (0, image.height - outpaint_size, image.width, image.height)
    }[outpaint_direction]
    draw = ImageDraw.Draw(image)
    image.save('draw.png')
    draw.rectangle(color_area, fill=color)
    return image
