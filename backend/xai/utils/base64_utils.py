import base64
import cv2
import numpy as np


def encode_image_to_base64(image_np):
    """
    Convert numpy image to base64 string
    """
    _, buffer = cv2.imencode('.jpg', image_np)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


def decode_base64_to_image(base64_string):
    """
    Convert base64 string back to numpy image
    """
    img_bytes = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image