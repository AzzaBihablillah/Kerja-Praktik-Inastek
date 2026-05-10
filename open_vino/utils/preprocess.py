from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from typing import Tuple


image_transform = transforms.Compose(
    [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def convert_to_PIL(image):
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_PIL = Image.fromarray(img_RGB)

    return img_PIL


def transform_image(image):
    if isinstance(image, np.ndarray):
        image = convert_to_PIL(image)

    transformed_image = image_transform(image)
    input_image = np.expand_dims(transformed_image.numpy(), axis=0)

    return input_image


def preprocess_image_yolov8_format(
    cv2_img: np.ndarray, in_size: Tuple[int, int] = (640, 640)
) -> np.ndarray:
    """preprocesses cv2 image and returns a norm np.ndarray
    (For yolov5 model)

     cv2_img = cv2 image
     in_size: in_width, in_height
    """
    # print(in_size)
    resized = cv2.resize(cv2_img, in_size).astype(np.float32)
    # cv2.imshow("test", resized)
    # cv2.waitKey(0)
    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in /= 255.0
    return img_in


def add_zero_pixels_atas(image, num_pixels):
    top, bottom = num_pixels, num_pixels
    left, right = 0, 0

    bordered_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return bordered_image
