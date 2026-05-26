import cv2
import numpy as np
from typing import Tuple


def preprocess_image_yolov8_format(
    bgr_img: np.ndarray, in_size: Tuple[int, int] = (640, 640)
) -> np.ndarray:
    """
    Preprocess a BGR cv2 image for YOLOv8 inference.

    Converts BGR -> RGB, resizes to in_size, normalizes to [0, 1],
    and transposes to CHW format.

    :param bgr_img: Input image in BGR format (as returned by cv2.imread).
    :param in_size: (width, height) target input size for the model.
    :return: Float32 array of shape [C, H, W], values in [0, 1].
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, in_size).astype(np.float32)
    chw = np.transpose(resized, (2, 0, 1))
    chw /= 255.0
    return chw


def pad_to_square(image: np.ndarray, pad_value: int = 114) -> np.ndarray:
    """
    Pad an image to a square by adding pixels to the shorter axis.
    Preserves aspect ratio.

    :param image: Input BGR image.
    :param pad_value: Pixel fill value (default black).
    :return: Square padded image.
    """
    h, w = image.shape[:2]
    diff = abs(h - w)
    half = diff // 2
    remainder = diff % 2

    if h < w:
        top, bottom = half, half + remainder
        left, right = 0, 0
    else:
        top, bottom = 0, 0
        left, right = half, half + remainder

    return cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[pad_value] * 3
    )

def letterbox(
    bgr_img: np.ndarray,
    target_size: int = 640,
    pad_value: int = 114
) -> np.ndarray:
    h, w = bgr_img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(bgr_img, (new_w, new_h))
    
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=[pad_value] * 3
    )
    
    return padded

def preserve_aspect_ratio_resize(img: np.ndarray, 
                                  target_width = 640) -> np.ndarray:
    """
    Resizing the image to desired width while preserving the aspect ratio
    
    :param img: Image
    :type img: np.ndarray
    :param target_width: desired width
    :return: resized image with its aspect ratio preserved
    :rtype: ndarray[Any, Any]
    """
    aspect_ratio = img.shape[0]/img.shape[1]

    # if aspect_ratio > 1:
    transformed_img = cv2.resize(img, 
                                (target_width, int(target_width*aspect_ratio)), 
                                interpolation=cv2.INTER_LINEAR)
    # else:
    #     transformed_img = cv.resize(img, 
    #                                 (target_width, target_width/aspect_ratio), 
    #                                 interpolation=cv.INTER_LINEAR)        

    return transformed_img

# from torchvision import transforms
# from PIL import Image
# import cv2
# import numpy as np
# from typing import Tuple


# image_transform = transforms.Compose(
#     [
#         transforms.Resize((640, 640)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )


# def convert_to_PIL(image):
#     img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img_PIL = Image.fromarray(img_RGB)

#     return img_PIL


# def transform_image(image):
#     if isinstance(image, np.ndarray):
#         image = convert_to_PIL(image)

#     transformed_image = image_transform(image)
#     input_image = np.expand_dims(transformed_image.numpy(), axis=0)

#     return input_image


# def preprocess_image_yolov8_format(
#     cv2_img: np.ndarray, in_size: Tuple[int, int] = (640, 640)
# ) -> np.ndarray:
#     """preprocesses cv2 image and returns a norm np.ndarray
#     (For yolov5 model)

#      cv2_img = cv2 image
#      in_size: in_width, in_height
#     """
#     # print(in_size)
#     resized = cv2.resize(cv2_img, in_size).astype(np.float32)
#     # cv2.imshow("test", resized)
#     # cv2.waitKey(0)
#     img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
#     img_in /= 255.0
#     return img_in




# def add_zero_pixels_atas(image, num_pixels):
#     top, bottom = num_pixels, num_pixels
#     left, right = 0, 0

#     bordered_image = cv2.copyMakeBorder(
#         image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
#     )

#     return bordered_image
