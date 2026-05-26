import numpy as np
from typing import Tuple, Union
import torch

def xy_center(xyxy: Union[np.ndarray, torch.Tensor]):
    if isinstance(xyxy, torch.Tensor):
        xyxy = xyxy.numpy()
    
    assert (
        xyxy.shape[0] == 4
    ), f"Invalid xyxy input: xyxy.shape[0] = {xyxy.shape[0]}"
    
    
    x_center = (xyxy[0]+xyxy[2])/2
    y_center = (xyxy[1]+xyxy[3])/2

    return np.array([x_center, y_center]).astype(np.int32)

def inverse_letterbox(
    coords: Union[np.ndarray, torch.Tensor],
    original_shape: Tuple[int, int],
    target_size: int = 640,
) -> np.ndarray:
    """
    Scale bounding box coordinates from letterboxed model space
    back to original image space.

    :param coords: Array of shape (n, 4) with [x1, y1, x2, y2] in model space.
    :param original_shape: (height, width) of the original image before letterboxing.
    :param target_size: The square size used during letterboxing (default 640).
    :return: Array of shape (n, 4) with coordinates in original image space.
    """

    if isinstance(coords, torch.Tensor):
        coords = coords.numpy()

    orig_h, orig_w = original_shape

    # Reproduce the exact same scale and padding letterbox() applied
    scale = target_size / max(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top  = pad_h // 2
    left = pad_w // 2

    # Undo padding offset
    coords = coords.copy().astype(np.float32)
    coords[:, [0, 2]] -= left  # x1, x2
    coords[:, [1, 3]] -= top   # y1, y2

    # Undo scaling
    coords[:, :4] /= scale

    # Clip to original image boundaries
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, orig_w)
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, orig_h)

    return coords