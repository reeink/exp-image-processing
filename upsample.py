import numpy as np
import cv2


# 最近邻插值
def nearest_interpolation(img: np.ndarray, dstH: int, dstW: int) -> np.ndarray:
    if len(img.shape) == 2:
        scrH, scrW = img.shape
        c = 1
    else:
        scrH, scrW, c = img.shape
    retimg = np.zeros((dstH, dstW, c), dtype=np.uint8)
    for i in range(dstH - 1):
        for j in range(dstW - 1):
            scrx = round(i * (scrH / dstH))
            scry = round(j * (scrW / dstW))
            retimg[i, j] = img[scrx, scry]
    return retimg


# 双线性插值
def bilinear_interpolation(img: np.ndarray, dstH: int, dstW: int) -> np.ndarray:
    if len(img.shape) == 2:
        scrH, scrW = img.shape
        c = 1
    else:
        scrH, scrW, c = img.shape
    retimg = np.zeros((dstH, dstW, c), dtype=np.uint8)
    for i in range(dstH - 1):
        for j in range(dstW - 1):
            scrx = i * (scrH / dstH)
            scry = j * (scrW / dstW)
            x = int(scrx)
            y = int(scry)
            u = scrx - x
            v = scry - y
            # avoid out of index
            if x + 1 > scrH - 1:
                x = scrH - 2
            if y + 1 > scrW - 1:
                y = scrW - 2
            retimg[i, j] = (
                img[x, y] * (1 - u) * (1 - v)
                + img[x + 1, y] * u * (1 - v)
                + img[x, y + 1] * (1 - u) * v
                + img[x + 1, y + 1] * u * v
            )
    return retimg


# 双三次内插
def bicubic_interpolation(img: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    if len(img.shape) == 2:
        src_h, src_w = img.shape
        c = 1
    else:
        src_h, src_w, c = img.shape
    ret_img = np.zeros((dst_h, dst_w, c), dtype=np.uint8)
    for i in range(dst_h):
        for j in range(dst_w):
            src_x = i * (src_h / dst_h)
            src_y = j * (src_w / dst_w)
            x = int(src_x)
            y = int(src_y)
            u = src_x - x
            v = src_y - y
            tmp = 0
            for ii in range(-1, 3):
                for jj in range(-1, 3):
                    if (
                        x + ii < 0
                        or y + jj < 0
                        or x + ii > src_h - 1
                        or y + jj > src_w - 1
                    ):
                        continue
                    weight = bicubic_weight(u - ii) * bicubic_weight(v - jj)
                    tmp += weight * img[x + ii, y + jj]
            ret_img[i, j] = np.uint8(np.clip(tmp, 0, 255))
    return ret_img


def bicubic_weight(x: float, a: float = -0.5) -> float:
    if abs(x) <= 1:
        return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
    elif 1 < abs(x) <= 2:
        return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
    else:
        return 0


def cv2_nearest_interpolation(img: np.ndarray, dstH: int, dstW: int) -> np.ndarray:
    return cv2.resize(img, (dstH, dstW), interpolation=cv2.INTER_NEAREST)


def cv2_bilinear_interpolation(img: np.ndarray, dstH: int, dstW: int) -> np.ndarray:
    return cv2.resize(img, (dstH, dstW), interpolation=cv2.INTER_LINEAR)


def cv2_bicubic_interpolation(img: np.ndarray, dstH: int, dstW: int) -> np.ndarray:
    return cv2.resize(img, (dstH, dstW), interpolation=cv2.INTER_CUBIC)


if __name__ == "__main__":
    import os
    import json

    img256px_folder = "./images/256"
    img512px_folder = "./images/512"
    dstW, dstH = 512, 512
    psnr = {}
    for filename in os.listdir(img256px_folder):
        original_img = cv2.imread(os.path.join(img512px_folder, filename))
        downsample_img = cv2.imread(os.path.join(img256px_folder, filename))
        psnr[filename] = {}
        for method in [
            nearest_interpolation,
            bilinear_interpolation,
            bicubic_interpolation,
            cv2_nearest_interpolation,
            cv2_bilinear_interpolation,
            cv2_bicubic_interpolation,
        ]:
            new_img = method(downsample_img, dstH, dstW)
            if not os.path.exists(f"./images/{method.__name__}"):
                os.mkdir(f"./images/{method.__name__}")
            cv2.imwrite(f"./images/{method.__name__}/{filename}", new_img)
            psnr[filename][method.__name__] = cv2.PSNR(original_img, new_img)
            print(
                f"{method.__name__} {filename} PSNR: {psnr[filename][method.__name__]}"
            )
    json.dump(psnr, open("./images/psnr.json", "w"))
