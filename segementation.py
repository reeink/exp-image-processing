import os
import cv2
import numpy as np
from typing import List
from itertools import product

SOURCE_PATH = "images/segementation"
DIST_PATH = "images/segementation-results"


# Otsu's Method
def otsu(image: np.ndarray) -> np.ndarray:
    assert len(image.shape) == 2, "must be grayscale image"

    # get histogram
    hist, _ = np.histogram(image, bins=256, range=(0, 255))
    hist_p = hist / hist.sum()  # probability density

    # set initial values
    w0 = 0.0
    sum_ipi = 0.0
    sum_ipi_all = hist_p.dot(np.arange(256))
    sigma_b_max = 0.0
    sigma_b_max_t = 0

    # find optimal threshold
    for t in range(256):
        w0 += hist_p[t]
        sum_ipi += t * hist_p[t]
        w1 = 1 - w0
        mu0 = sum_ipi / w0 if w0 > 0 else 0
        mu1 = (sum_ipi_all - sum_ipi) / w1 if w1 > 0 else 0

        sigma_b = w0 * w1 * (mu0 - mu1) ** 2
        if sigma_b > sigma_b_max:
            sigma_b_max = sigma_b
            sigma_b_max_t = t

    # return mask
    return (image > sigma_b_max_t).astype(np.uint8) * 255


def region_growing(image: np.ndarray, seed: List, threshold: int) -> np.ndarray:
    assert len(image.shape) == 2, "must be grayscale image"

    mask = np.zeros_like(image, dtype=np.uint8)
    stack = seed.copy()

    while len(stack) > 0:
        x, y = stack.pop()
        mask[x, y] = 255

        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if i < 0 or j < 0 or i >= image.shape[0] or j >= image.shape[1]:
                    continue
                if (
                    mask[i, j] == 0
                    and abs(int(image[i, j]) - int(image[x, y])) < threshold
                ):
                    stack.append([i, j])

    return mask


def get_edge_from_mask(mask_image: np.ndarray) -> np.ndarray:
    assert len(mask_image.shape) == 2, "must be grayscale image"

    # edge = np.zeros_like(mask_image, dtype=np.uint8)
    #
    # for i in range(mask_image.shape[0]):
    #     for j in range(mask_image.shape[1]):
    #         if mask_image[i, j] > 0:
    #             for x,y in product(range(i - 1, i + 2), range(j - 1, j + 2)):
    #                 if x < 0 or y < 0 or x >= mask_image.shape[0] or y >= mask_image.shape[1]:
    #                     continue
    #                 if mask_image[x, y] == 0:
    #                     edge[x, y] = 255
    #                     break
    #
    # return edge

    return cv2.Canny(mask_image, 100, 200)


if __name__ == "__main__":
    os.makedirs(DIST_PATH, exist_ok=True)

    for filename in os.listdir(SOURCE_PATH):
        img = cv2.imread(os.path.join(SOURCE_PATH, filename), cv2.IMREAD_GRAYSCALE)

        # method otsu
        ostu_mask = otsu(img)
        ostu_edge = get_edge_from_mask(ostu_mask)
        os.makedirs(os.path.join(DIST_PATH, "otsu"), exist_ok=True)
        cv2.imwrite(
            os.path.join(DIST_PATH, "otsu", filename.split(".")[0] + "-mask.jpg"),
            ostu_mask,
        )
        cv2.imwrite(
            os.path.join(DIST_PATH, "otsu", filename.split(".")[0] + "-edge.jpg"),
            ostu_edge,
        )

        # method region growing
        threshold = 8

        rg_mask = region_growing(img, [[0, 0]], threshold)
        rg_edge = get_edge_from_mask(rg_mask)
        os.makedirs(
            os.path.join(DIST_PATH, f"region-growing-{threshold}"), exist_ok=True
        )
        cv2.imwrite(
            os.path.join(
                DIST_PATH,
                f"region-growing-{threshold}",
                filename.split(".")[0] + "-mask.jpg",
            ),
            rg_mask,
        )
        cv2.imwrite(
            os.path.join(
                DIST_PATH,
                f"region-growing-{threshold}",
                filename.split(".")[0] + "-edge.jpg",
            ),
            rg_edge,
        )

        # use different color to label edge in original image
        new_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        new_img[rg_edge > 0] = [0, 255, 0]
        new_img[ostu_edge > 0] = [0, 0, 255]
        os.makedirs(os.path.join(DIST_PATH, "compare"), exist_ok=True)
        cv2.imwrite(
            os.path.join(DIST_PATH, "compare", filename.split(".")[0] + ".jpg"), new_img
        )
