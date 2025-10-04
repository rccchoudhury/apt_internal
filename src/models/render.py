import math
from tokenize import group
from typing import Optional, Union

import torch
from PIL import Image, ImageDraw
from torch import FloatTensor, LongTensor, Tensor
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F


def tensor_to_pil_image(image: FloatTensor) -> Image.Image:
    assert image.ndim == 3  # [C,H,W]
    if image.min() < 0:
        image = image - image.min()
        image = image / image.max()
    image = to_pil_image(image)
    return image


def render_initgroup(image: Union[FloatTensor, Image.Image], group_assignment, min_patch_size: int, line_color_rgba: Optional[tuple] = (0, 0, 0, 85)) -> Image.Image:

    if isinstance(image, Tensor):
        image = tensor_to_pil_image(image)

    vis = Image.new('RGB', (image.width, image.height), (255, 255, 255))
    draw = ImageDraw.Draw(vis, "RGBA")
    line_kwargs = dict(width=3, fill=line_color_rgba)
    
    vis.paste(image, (0, 0))
    
    h, w = group_assignment.shape
    padded_group = F.pad(group_assignment, (1, 1, 1, 1), 'constant', -1)
    
    row_flag = padded_group[1:, 1:-1] != padded_group[:-1, 1:-1]
    col_flag = padded_group[1:-1, 1:] != padded_group[1:-1, :-1]
    
    for i in range(h):
        for j in range(w):
            if row_flag[i, j]:
                draw.line(((j * min_patch_size, i * min_patch_size), ((j + 1) * min_patch_size, i * min_patch_size)), **line_kwargs)
            if col_flag[i, j]:
                draw.line(((j * min_patch_size, i * min_patch_size), (j * min_patch_size, (i + 1) * min_patch_size)), **line_kwargs)
                
    return vis


def hstack_images(images: list[Image.Image], gap: int = 20) -> Image.Image:
    prev_concat = images[0]
    for image in images[1:]:
        concat = Image.new("RGB", (prev_concat.width + gap +
                        image.width, image.height), (255, 255, 255))
        concat.paste(prev_concat, (0, 0))
        concat.paste(image, (prev_concat.width + gap, 0))
        prev_concat = concat
    return concat