import cv2
import numpy as np
import os


def downsample_from_data(image: np.ndarray, factor: int) -> np.ndarray:
    if len(image.shape) == 2:
        h, w = image.shape
        c = 1
    else:
        h, w, c = image.shape
    new_h = h // factor
    new_w = w // factor
    new_img = np.zeros((new_h, new_w, c), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            new_img[i, j] = image[i * factor, j * factor]
    return new_img


def downsample(img_path, factor):
    """
    下采样图像
    """
    img = cv2.imread(img_path)
    return downsample_from_data(img, factor)


if __name__ == "__main__":
    files = os.listdir("./images/512")
    for file in files:
        file_path = os.path.join("./images/512", file)
        new_img = downsample(file_path, 2)
        cv2.imwrite("./images/256/" + file, new_img)
