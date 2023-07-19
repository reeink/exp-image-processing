import os
from typing import Iterable, Callable
import cv2
import numpy as np
from rich.progress import track
from rich import print
from downsample import downsample_from_data as downsample
from upsample import (
    nearest_interpolation,
    bilinear_interpolation,
    bicubic_interpolation,
)


def compose(*functions):
    def inner(arg):
        for f in functions:
            arg = f(arg)
        return arg

    return inner


def neighbor_average(image: np.ndarray, radius: int = 1) -> np.ndarray:
    height, width = image.shape[:2]

    for y in range(height):
        for x in range(width):
            neighbor = image[
                max(0, y - radius) : min(height, y + radius + 1),
                max(0, x - radius) : min(width, x + radius + 1),
            ]
            image[y, x] = np.mean(neighbor)

    return image


def gaussian_lowpass_filter(image, kernel_size, sigma):
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image


def upsample_by_2_nearest(image: np.ndarray) -> np.ndarray:
    return nearest_interpolation(image, image.shape[0] * 2, image.shape[1] * 2)


def upsample_by_2_bilinear(image: np.ndarray) -> np.ndarray:
    return bilinear_interpolation(image, image.shape[0] * 2, image.shape[1] * 2)


def upsample_by_2_bicubic(image: np.ndarray) -> np.ndarray:
    return bicubic_interpolation(image, image.shape[0] * 2, image.shape[1] * 2)


def pyramid_builder(
    image: np.ndarray,
    approximation_filter: Callable[[np.ndarray], np.ndarray],
    interpolaion_filter: Callable[[np.ndarray], np.ndarray],
    downsample: Callable[[np.ndarray], np.ndarray] = lambda x: downsample(x, 2),
    upsample: Callable[[np.ndarray], np.ndarray] = lambda x: x,
) -> Iterable[np.ndarray]:
    image = image.reshape(image.shape[0], image.shape[1], -1)
    while image.shape[0] > 1 and image.shape[1] > 1:
        next_level = compose(approximation_filter, downsample)(image)
        next_level = next_level.reshape(next_level.shape[0], next_level.shape[1], -1)
        now_level = image - compose(upsample, interpolaion_filter)(next_level).reshape(image.shape)
        yield now_level
        image = next_level
    yield image

if __name__ == "__main__":
    SOURCE_PATH = "images/512"
    
    # DIST_PATH = "images/512_pyramid"

    # approximation_filters = {
    #     "mean": lambda x: neighbor_average(x, 1),
    #     "gaussian": lambda x: gaussian_lowpass_filter(x, 5, 1),
    #     "subsampling": lambda x: x,
    # }

    # interpolation_filters = {
    #     "nearest": lambda x: nearest_interpolation(x, x.shape[0] * 2, x.shape[1] * 2),
    #     "bilinear": lambda x: bilinear_interpolation(x, x.shape[0] * 2, x.shape[1] * 2),
    #     "bicubic": lambda x: bicubic_interpolation(x, x.shape[0] * 2, x.shape[1] * 2),
    # }
    
    DIST_PATH = "images/512_pyramid_opencv"
    
    approximation_filters = {
        "mean": lambda x: cv2.blur(x, (3, 3)),
        "gaussian": lambda x: cv2.GaussianBlur(x, (3, 3), 1),
        "subsampling": lambda x: x,
    }
    
    interpolation_filters = {
        "nearest": lambda x: cv2.resize(x, (x.shape[0] * 2, x.shape[1] * 2), interpolation=cv2.INTER_NEAREST),
        "bilinear": lambda x: cv2.resize(x, (x.shape[0] * 2, x.shape[1] * 2), interpolation=cv2.INTER_LINEAR),
        "bicubic": lambda x: cv2.resize(x, (x.shape[0] * 2, x.shape[1] * 2), interpolation=cv2.INTER_CUBIC),
    }

    for apf_name, apf in approximation_filters.items():
        for inf_name, inf in interpolation_filters.items():
            dist_path = os.path.join(DIST_PATH, f"{apf_name}_{inf_name}")
            os.makedirs(dist_path, exist_ok=True)
            for filename in track(os.listdir(SOURCE_PATH), description=f"{apf_name}_{inf_name}"):
                image = cv2.imread(os.path.join(SOURCE_PATH, filename), cv2.IMREAD_GRAYSCALE)
                pyramid = reversed(list(pyramid_builder(image, apf, inf)))
                for level, image in enumerate(pyramid):
                    cv2.imwrite(
                        os.path.join(
                            dist_path, f"{filename.split('.')[0]}_l{level+1}.bmp"
                        ),
                        image,
                    )
