from PIL import Image, ImageDraw
import numpy as np
from patchmatch import patch_match


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
        return fill_with_patchmatch(new_image, outpaint_direction, outpaint_size)
    else:
        return fill_with_color(new_image, outpaint_direction, outpaint_size, color)

def fill_with_patchmatch(image, outpaint_direction, outpaint_size):
    original_width, original_height = image.size
    new_size = image.size

    mask = Image.new("L", new_size, 0)  # Entirely black
    mask_area = {
        'left': (0, 0, outpaint_size, original_height),
        'right': (original_width, 0, original_width + outpaint_size, original_height),
        'up': (0, 0, original_width, outpaint_size),
        'down': (0, original_height, original_width, original_height + outpaint_size)
    }[outpaint_direction]
    mask.paste(255, mask_area)  # White in the area to be outpainted

    if patch_match.patchmatch_available:
        result = patch_match.inpaint(np.array(image), np.array(mask), patch_size=3)
        return Image.fromarray(result)
    else:
        print("PatchMatch is not available.")
        return image

def fill_with_color(image, outpaint_direction, outpaint_size, color):
    color_area = {
        'left': (0, 0, outpaint_size, image.height),
        'right': (image.width - outpaint_size, 0, image.width, image.height),
        'up': (0, 0, image.width, outpaint_size),
        'down': (0, image.height - outpaint_size, image.width, image.height)
    }[outpaint_direction]
    draw = ImageDraw.Draw(image)
    draw.rectangle(color_area, fill=color)
    return image
