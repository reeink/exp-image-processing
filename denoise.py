import os
import cv2
import numpy as np
from rich.progress import track
from rich import print
from multiprocessing import Pool
from typing import Iterable, Callable
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
from scipy.ndimage import sobel


# add noise to image
def add_noise(img: np.ndarray, sigma=5) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape)
    img_noise = img + noise
    return img_noise


def sobolev(g, lambda_):
    fg = np.fft.fft2(g)
    n = g.shape[0]
    x = np.concatenate((np.arange(0, n // 2 + 1), np.arange(-n // 2 + 1, 0)))
    Y, X = np.meshgrid(x, x)
    S = (X**2 + Y**2) * (2 / n) * 2
    lambda_ = lambda_ * 2 / n
    f = np.real(np.fft.ifft2((fg / (1 + lambda_ * S))))
    f = f.clip(0, 255).astype(np.uint8)
    return f


def tv(g, lambda_, alpha=0.9, epsilon=0.05, max_iter=1000, f0=None):
    g = g.astype(np.float64)
    f = g.copy()
    alpha = alpha * 2 / (1 + lambda_ * 8 / epsilon)
    his = []

    def div(p):
        return -sum([sobel(p[i, ...], axis=i) for i in range(p.shape[0])])

    for _ in range(max_iter):
        f_gradient = np.gradient(f)
        f_gradient_norm = np.sqrt(
            np.abs(f_gradient[0]) ** 2 + np.abs(f_gradient[1]) ** 2 + epsilon**2
        )
        divergence_term = lambda_ * div(f_gradient / f_gradient_norm)
        f = f - alpha * (2 * (f - g) + divergence_term)
        if f0 is not None:
            f_clipped = np.clip(f, 0, 255).astype(np.uint8)
            his.append(cv2.PSNR(f0, f_clipped))

    f_clipped = np.clip(f, 0, 255).astype(np.uint8)
    if f0 is not None:
        return f_clipped, his
    else:
        return f_clipped


def cal_psnr(data):
    original_image = data[0]
    noise_image = data[1]
    method = data[2]
    lambda_ = data[3]
    denoised_image = method(noise_image, lambda_)

    psnr = cv2.PSNR(original_image, denoised_image)
    return {"lambda": lambda_, "psnr": psnr, "image": denoised_image}


def calc_best_lambda(
    original_image: np.ndarray,
    noise_image: np.ndarray,
    method: Callable,
    lambdas: Iterable[int | float],
):
    results = Pool(os.cpu_count()).map(
        cal_psnr,
        [(original_image, noise_image, method, lambda_) for lambda_ in lambdas],
    )
    best = max(results, key=lambda x: x["psnr"])
    return best, results


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sigma", type=float, default=20)
    argparser.add_argument("--sobolev", action="store_true")
    argparser.add_argument("--tv", action="store_true")

    # define path
    FOLDER = "images/denoising"  # folder to save results
    IMGAES = "images/512"  # folder to load images
    NOISE_IMAGES = os.path.join(FOLDER, "noise")  # folder to save noise images
    SOBOLEV_PATH = os.path.join(FOLDER, "sobolev")
    TV_PATH = os.path.join(FOLDER, "tv")
    os.makedirs(NOISE_IMAGES, exist_ok=True)
    os.makedirs(SOBOLEV_PATH, exist_ok=True)
    os.makedirs(TV_PATH, exist_ok=True)

    sns.set()
    os.makedirs(NOISE_IMAGES, exist_ok=True)

    noise_sigma = argparser.parse_args().sigma

    original_images = {
        img_name: cv2.imread(os.path.join(IMGAES, img_name), cv2.IMREAD_GRAYSCALE)
        for img_name in os.listdir(IMGAES)
    }

    images = {}

    save_path = os.path.join(NOISE_IMAGES, f"sigma-{noise_sigma}")
    os.makedirs(save_path, exist_ok=True)
    for img_name, img in track(original_images.items(), description="Add noise"):
        images[img_name] = {"original": img, "noise": add_noise(img, noise_sigma)}
        cv2.imwrite(os.path.join(save_path, img_name), images[img_name]["noise"])

    # Sobolev denoising

    if argparser.parse_args().sobolev:
        # get first image
        print("Calculating best lambda for Sobolev denoising...")
        first_image = list(images.values())[0]
        lambdas = np.arange(0, 10, 0.01)
        best, results = calc_best_lambda(
            first_image["original"], first_image["noise"], sobolev, lambdas
        )
        print(f"Best lambda: {best['lambda']:.2f}, PSNR: {best['psnr']:.2f}")

        # plot psnr-lambda
        plt.figure(figsize=(10, 10))
        figure1 = sns.lineplot(x=lambdas, y=[result["psnr"] for result in results])
        figure1.set(xlabel="lambda", ylabel="PSNR")
        figure1.set_title("PSNR-lambda curve of Sobolev denoising")
        # label best lambda
        figure1.text(
            best["lambda"] + 0.1,
            best["psnr"] + 0.1,
            f"lambda={best['lambda']:.2f}, PSNR={best['psnr']:.2f}",
        )
        figure1.axvline(best["lambda"], color="red", linestyle="--")
        plt.savefig(os.path.join(SOBOLEV_PATH, f"psnr-lambda-sigma={noise_sigma}.png"))

        # use best lambda to denoise
        lambda_ = best["lambda"]
        save_path = os.path.join(SOBOLEV_PATH, f"sigma-{noise_sigma}")
        os.makedirs(save_path, exist_ok=True)
        for img_name, img in track(images.items(), description="Sobolev denoising"):
            denoised_image = sobolev(img["noise"], lambda_)
            psnr = cv2.PSNR(img["original"], denoised_image)
            cv2.imwrite(os.path.join(save_path, img_name), denoised_image)
            with open(
                os.path.join(save_path, f"psnr-{img_name.split('.')[0]}.txt"), "w"
            ) as f:
                f.write(f"PSNR: {psnr:.2f}")

    # TV denoising

    if argparser.parse_args().tv:
        print("Calculating best lambda for TV denoising...")
        first_image = list(images.values())[0]
        lambdas = np.arange(0, 30, 0.1)
        best, results = calc_best_lambda(
            first_image["original"], first_image["noise"], tv, lambdas
        )
        print(f"Best lambda: {best['lambda']:.1f}, PSNR: {best['psnr']:.2f}")

        # plot psnr-lambda
        plt.figure(figsize=(10, 10))
        figure1 = sns.lineplot(x=lambdas, y=[result["psnr"] for result in results])
        figure1.set(xlabel="lambda", ylabel="PSNR")
        figure1.set_title("PSNR-lambda curve of TV denoising")
        # label best lambda
        figure1.text(
            best["lambda"] + 0.1,
            best["psnr"] + 0.1,
            f"lambda={best['lambda']:.1f}, PSNR={best['psnr']:.2f}",
        )
        figure1.axvline(best["lambda"], color="red", linestyle="--")
        plt.savefig(os.path.join(TV_PATH, f"psnr-lambda-sigma={noise_sigma}.png"))

        best_lambda = best["lambda"]
        # best_lambda = 7
        save_path = os.path.join(TV_PATH, f"sigma-{noise_sigma}")
        os.makedirs(save_path, exist_ok=True)
        for img_name, img in track(images.items(), description="TV denoising"):
            denoised_image, his = tv(img["noise"], best_lambda, f0=img["original"])
            cv2.imwrite(os.path.join(save_path, img_name), denoised_image)
            plt.figure(figsize=(10, 10))
            figure = sns.lineplot(x=np.arange(len(his)), y=his)
            figure.set(xlabel="iteration", ylabel="loss")
            figure.set_title(
                f"PSNR-iteration curve of TV denoising, sigma={noise_sigma}"
            )
            figure.axhline(max(his), color="red", linestyle="--")
            figure.text(
                0,
                max(his) + 0.01,
                f"PSNR={max(his):.2f}",
            )
            plt.savefig(
                os.path.join(save_path, f"psnr-iteration-{img_name.split('.')[0]}.jpg")
            )
