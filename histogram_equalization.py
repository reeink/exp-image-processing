from typing import Optional
import numpy as np
import gradio as gr
import seaborn as sns
import matplotlib.pyplot as plt
import cv2


# 灰度图像的直方图均衡
def process_grayscale(image: np.ndarray) -> np.ndarray:
    assert len(image.shape) == 2, "输入图像必须为灰度图像！"
    hist, _ = np.histogram(image, bins=256, range=(0, 255))  # 计算灰度直方图
    cdf = hist.cumsum()  # 计算累积直方图
    cdf_norm = (cdf * 255 / cdf.max()).astype(np.uint8)  # 归一化累积直方图
    map_table = np.vectorize(lambda x: cdf_norm[x])  # 创建映射表
    return map_table(image)  # 返回均衡化后的图像


# 彩色图像的直方图均衡
def process_color(image):
    assert len(image.shape) == 3, "输入图像必须为彩色图像！"
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # 将图像转换为HSV色彩空间
    hsv_image[:, :, 2] = process_grayscale(hsv_image[:, :, 2])  # 对明度通道进行直方图均衡
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)  # 将图像转换回BGR色彩空间


def get_hist_plot(image: Optional[np.ndarray], gray_type: str = "灰度图像"):
    if image is None:
        return None
    if gray_type == "灰度图像":
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist, _ = np.histogram(image, bins=256, range=(0, 255))  # type: ignore
        plot1 = sns.lineplot(x=range(256), y=hist)
        return plot1.figure
    elif gray_type == "彩色图像":
        plot2 = plt.figure()
        COLOR = ("r", "g", "b")
        for i in range(image.shape[2]):
            hist, _ = np.histogram(image[:, :, i], bins=256, range=(0, 255))
            sns.lineplot(x=range(256), y=hist, color=COLOR[i])
        return plot2


def process_all(
    image: Optional[np.ndarray], gray_type: str = "灰度图像"
) -> Optional[np.ndarray]:
    if image is None:
        return None

    if gray_type == "灰度图像":
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return process_grayscale(image)  # type: ignore
    elif gray_type == "彩色图像":
        return process_color(image)
    else:
        raise ValueError("图像类型必须为灰度图像或彩色图像！")


with gr.Blocks(title="直方图均衡") as demo:
    with gr.Row():
        gray_type = gr.Radio(["灰度图像", "彩色图像"], label="图像类型", value="灰度图像")
        btn = gr.Button("均衡化处理")
        btn1 = gr.Button("原图像直方图")
        btn2 = gr.Button("处理图像直方图")
    with gr.Row():
        in_img = gr.components.Image(label="原图像")
        in_hist = gr.components.Plot(label="直方图")
    with gr.Row():
        out_img = gr.components.Image(type="numpy", label="处理后图像", interactive=False)
        out_hist = gr.components.Plot(label="直方图")
    btn.click(fn=process_all, inputs=[in_img, gray_type], outputs=out_img)
    btn1.click(fn=get_hist_plot, inputs=[in_img, gray_type], outputs=in_hist)
    btn2.click(fn=get_hist_plot, inputs=[out_img, gray_type], outputs=out_hist)

if __name__ == "__main__":
    sns.set()
    demo.launch()
