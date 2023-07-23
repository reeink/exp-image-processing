import numpy as np
import argparse
import cv2
import os
from rich.progress import track
from rich import print
from matplotlib import pyplot as plt
import seaborn as sns


def add_blur_and_noise(img, kernel_size=19, blur_sigma=2, noise_sigma=5):
    kernel = cv2.getGaussianKernel(kernel_size, blur_sigma)
    kernel = kernel @ kernel.T
    noise = np.random.normal(0, noise_sigma, img.shape)

    kernel_freq = np.fft.fft2(kernel, s=img.shape)
    image_freq = np.fft.fft2(img)

    blur_image_freq = image_freq * kernel_freq

    out_image = np.fft.ifft2(blur_image_freq).real + noise
    return out_image.clip(0, 255).astype(np.uint8), noise, kernel


# 逆滤波
def inverse_filter(image, kernel, eps=1e-1):
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, s=image.shape) + eps

    result_fft = image_fft / kernel_fft
    result = np.fft.ifft2(result_fft).real
    return result.clip(0, 255).astype(np.uint8)


# Wiener 滤波
def wiener_filter(image, kernel, k):
    image_freq = np.fft.fft2(image)
    kernel_freq = np.fft.fft2(kernel, s=image.shape)

    kernel_power = np.abs(kernel_freq) ** 2

    wiener_filter = np.conj(kernel_freq) / (kernel_power + k)

    filtered_freq = image_freq * wiener_filter
    filtered_image = np.fft.ifft2(filtered_freq)
    filtered_image = np.real(filtered_image)
    return filtered_image.clip(0, 255).astype(np.uint8)


# 最大后验估计
def map_filter(image, kernel, lambda_):
    image_freq = np.fft.fft2(image)
    kernel_freq = np.fft.fft2(kernel, s=image.shape)

    kernel_power = np.abs(kernel_freq) ** 2

    n = image.shape[0]
    x = np.concatenate((np.arange(0, n // 2 + 1), np.arange(-n // 2 + 1, 0)))
    Y, X = np.meshgrid(x, x)
    S = (X**2 + Y**2) * (2 / n) * 2
    map_filter = np.conj(kernel_freq) / (kernel_power + lambda_ * S)
    result_freq = image_freq * map_filter
    result = np.fft.ifft2(result_freq)
    return result.real.clip(0, 255).astype(np.uint8)


def cal_psnr(f0, f):
    return cv2.PSNR(f0, f)


def save_psnr(save_path, psnr):
    with open(save_path, "w") as f:
        f.write(f"PSNR: {psnr:.2f}")


if __name__ == "__main__":
    SRC_IMAGE = "images/512"
    DEST_IMAGE = "images/deblurring"

    sns.set()

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sigma", type=float, default=5.0)

    images = {}
    sigma = argparser.parse_args().sigma

    noise_with_blur_save_path = os.path.join(
        DEST_IMAGE, "noise_with_blur", f"sigma={sigma}"
    )
    wiener_save_path = os.path.join(DEST_IMAGE, "wiener", f"sigma={sigma}")
    inverse_filter_save_path = os.path.join(
        DEST_IMAGE, "inverse_filter", f"sigma={sigma}"
    )
    map_filter_save_path = os.path.join(DEST_IMAGE, "map_filter", f"sigma={sigma}")
    os.makedirs(noise_with_blur_save_path, exist_ok=True)
    os.makedirs(wiener_save_path, exist_ok=True)
    os.makedirs(inverse_filter_save_path, exist_ok=True)
    os.makedirs(map_filter_save_path, exist_ok=True)

    # experiment the best lambda for MAP filter
    print("Experiment the best lambda for MAP filter...")
    image = cv2.imread(os.path.join(SRC_IMAGE, "barb.bmp"), cv2.IMREAD_GRAYSCALE)
    best_lambda = 0
    best_psnr = 0
    lambdas = np.arange(0, 10, 0.1)
    psnrs = []
    for k in track(lambdas):
        noise_with_blur_image, noise, kernel = add_blur_and_noise(
            image, noise_sigma=sigma
        )
        filtered_image = map_filter(noise_with_blur_image, kernel, k)
        psnr = cal_psnr(image, filtered_image)
        psnrs.append(psnr)
        if psnr > best_psnr:
            best_psnr = psnr
            best_lambda = k
    print(f"best lambda: {best_lambda:.2f}, best PSNR: {best_psnr:.2f}")

    plt.figure(figsize=(10, 10))
    figure = sns.lineplot(x=lambdas, y=psnrs)
    figure.set_xlabel("lambda")
    figure.set_ylabel("PSNR")
    figure.set_title("PSNR-lambda")
    figure.text(
        best_lambda + 0.1,
        best_psnr + 0.1,
        f"lambda={best_lambda:.2f}, PSNR={best_psnr:.2f}",
    )
    figure.axvline(x=best_lambda, color="r", linestyle="--")
    plt.savefig(os.path.join(map_filter_save_path, "psnr-lambda.png"))

    for image_name in track(os.listdir(SRC_IMAGE)):
        image_path = os.path.join(SRC_IMAGE, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        noise_blur_image, noise, kernel = add_blur_and_noise(image, noise_sigma=sigma)
        noise_blur_psnr = cal_psnr(image, noise_blur_image)

        K = np.abs(np.fft.fft2(noise)) ** 2 / np.abs(np.fft.fft2(image)) ** 2
        inverse_filtered_image = inverse_filter(noise_blur_image, kernel)
        inverse_filtered_psnr = cal_psnr(image, inverse_filtered_image)
        wiener_filtered_image = wiener_filter(noise_blur_image, kernel, K)
        wiener_filtered_psnr = cal_psnr(image, wiener_filtered_image)
        map_filtered_image = map_filter(noise_blur_image, kernel, best_lambda)
        map_filtered_psnr = cal_psnr(image, map_filtered_image)

        images[image_name] = {
            "original": image,
            "kernel": kernel,
            "noise_blur": noise_blur_image,
            "noise_blur_psnr": noise_blur_psnr,
            "inverse_filtered": inverse_filtered_image,
            "inverse_filtered_psnr": inverse_filtered_psnr,
            "wiener_filtered": wiener_filtered_image,
            "wiener_filtered_psnr": wiener_filtered_psnr,
            "map_filtered": map_filtered_image,
            "map_filtered_psnr": map_filtered_psnr,
        }

        print(image_name)
        print(f"noise_blur_psnr: {noise_blur_psnr:.2f}")
        print(f"inverse_filtered_psnr: {inverse_filtered_psnr:.2f}")
        print(f"wiener_filtered_psnr: {wiener_filtered_psnr:.2f}")
        print(f"map_filtered_psnr: {map_filtered_psnr:.2f}")
        print("+++++++++++++++++++++")

        cv2.imwrite(
            os.path.join(noise_with_blur_save_path, image_name), noise_blur_image
        )
        cv2.imwrite(os.path.join(wiener_save_path, image_name), wiener_filtered_image)
        cv2.imwrite(
            os.path.join(inverse_filter_save_path, image_name), inverse_filtered_image
        )
        cv2.imwrite(os.path.join(map_filter_save_path, image_name), map_filtered_image)
        save_psnr(
            os.path.join(noise_with_blur_save_path, f"{image_name.split('.')[0]}.txt"),
            noise_blur_psnr,
        )
        save_psnr(
            os.path.join(wiener_save_path, f"{image_name.split('.')[0]}.txt"),
            wiener_filtered_psnr,
        )
        save_psnr(
            os.path.join(inverse_filter_save_path, f"{image_name.split('.')[0]}.txt"),
            inverse_filtered_psnr,
        )
        save_psnr(
            os.path.join(map_filter_save_path, f"{image_name.split('.')[0]}.txt"),
            map_filtered_psnr,
        )
