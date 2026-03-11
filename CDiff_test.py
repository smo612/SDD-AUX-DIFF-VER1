import os
import matplotlib
matplotlib.use('Agg')  # ✅ 改為 non-GUI backend，防止 core dumped
import torch.nn as nn
import torch
import argparse
import pandas as pd
from modules_CDiff import UNet
from PIL import Image
from torchvision.utils import save_image
from CDiff import Diffusion
from utils import get_data
from collections import OrderedDict
import torch.nn.functional as F  # 用於計算 MSE
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from torch.utils.tensorboard import SummaryWriter

# 設置隨機種子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 設定隨機種子
#set_seed(50)


# 初始化參數
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 1
args.image_size = 64
#args.dataset_path = "/home/lab715/code/Howen/Diffusion-Models-pytorch-main/unconditional_images"
#args.dataset_path = "/home/lab715/code/Howen/CDiff/cifar10-64/test"
args.dataset_path = "/home/ee715/code/Howen/test"

# 設定裝置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 設置保存路徑的根目錄
output_dir = "test_image"
os.makedirs(output_dir, exist_ok=True)

# 初始化 Diffusion 類
diffusion = Diffusion(noise_steps=1000, beta_start=1e-4, beta_end=0.0095, img_size=64, device=device)

# 加載數據
dataloader = get_data(args)
image = next(iter(dataloader))[0].to(device)  # 從 dataloader 中獲取一張圖片

# 加噪圖片
save_image(image.add(1).mul(0.5), os.path.join(output_dir, "input.jpg"))  # 保存加噪的圖片


# 假設舊模型權重文件路徑
#ckpt = torch.load("/home/lab715/code/Howen/CDiff/CDiff_train/ckpt.pt")
#ckpt = torch.load("/home/lab715/code/Howen/CDiff/CDiff_train/t=300y/ckpt.pt")
#ckpt = torch.load("/home/lab715/code/Howen/CDiff/ckpt.pt")
ckpt = torch.load("/home/ee715/code/Howen/300T_0dB.pt", weights_only=True)

# 初始化新模型
model = UNet().to("cuda")

# 將新 state_dict 加載到模型
model.load_state_dict(ckpt)

t = torch.Tensor([0, 150, 300, 450, 600, 750, 900, 999]).long().to(device)

#t_hat = torch.full((image.size(0),), 50, dtype=torch.long).to(device)
x_hat = diffusion.noise_images_x_hat(image, t=300)


x_t, _, _, _ = diffusion.noise_images(image, x_hat, t)
_, _, noised_images, _= diffusion.noise_images(image, x_hat, t)
_, _, _, y = diffusion.noise_images(image, x_hat, t)

# Lambda schedule 可以像這樣定義：
lambda_schedule = torch.linspace(0., 1, diffusion.noise_steps).to(device)

# 呼叫 sample
x, intermediate_images = diffusion.sample(model, n=1, x_hat=x_hat, lambda_schedule=lambda_schedule)

for step, img in enumerate(intermediate_images):
    if step % 50 == 0:  # 每隔 50 步保存一次
        save_image(img, os.path.join(output_dir, f"step_{step}.jpg"))
        
x_hat = x_hat - x_hat.min()
x_hat = x_hat / (x_hat.max() + 1e-8)

x_t = x_t - x_t.min()
x_t = x_t / (x_t.max() + 1e-8)

y = y - y.min()
y = y / (y.max() + 1e-8)

#save_image(x_hat, os.path.join(output_dir, "x_hat.jpg"))
#save_image(x_t, os.path.join(output_dir, "x_t.jpg"))
save_image(x, os.path.join(output_dir, "denoised.jpg"))
#save_image(y, os.path.join(output_dir, "y_image.jpg"))
#print("x_hat max/min:", x_hat.max().item(), x_hat.min().item())
#print("x_t max/min:", x_t.max().item(), x_t.min().item())
#print("x max/min:", x.max().item(), x.min().item())

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

psnr_list = []

with torch.no_grad():  # ✅ 包整個推論，省記憶體

    for i in range(1000):
        image = next(iter(dataloader))[0].to(device)

        x_hat = diffusion.noise_images_x_hat(image, t=300)
        x, _ = diffusion.sample(model, n=1, x_hat=x_hat, lambda_schedule=lambda_schedule)

        image_norm = (image + 1) / 2
        x_clamped = torch.clamp(x, 0., 1.)

        # ✅ 可選：不每張都存圖（可改回你原本的 if 有需要）
        # save_image(x, os.path.join(output_dir, f"denoised{i}.jpg"))

        psnr = calculate_psnr(image_norm, x_clamped)
        psnr_list.append(psnr)

        print(f"[{i+1}/1000] PSNR: {psnr:.2f} dB")

output_dir = "metrics_2"
os.makedirs(output_dir, exist_ok=True)
psnr_df = pd.DataFrame({'Run': list(range(1, len(psnr_list) + 1)),
                        'PSNR(dB)': psnr_list})
psnr_df.to_csv(os.path.join(output_dir, "20dB"), index=False)

print(f"\n✅ Saved all PSNR values to: {os.path.join(output_dir, 'Power')}")        

# 畫 PSNR 分佈圖
import os
plt.figure()
plt.hist(psnr_list, bins=50, color='skyblue', edgecolor='black')
plt.title('PSNR Distribution over 1000 Runs')
plt.xlabel('PSNR(dB)')
plt.ylabel('Frequency')
plt.grid(True)
output_dir = "metrics_2"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Power.png"))
plt.close()

# 輸出平均值
avg_psnr = sum(psnr_list) / len(psnr_list)
print(f"\n✅ Finished 1000 runs.")
print(f"Average PSNR: {avg_psnr:.2f} dB")



'''
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def calculate_snr(clean, noisy):
    # 假設 clean 和 noisy 都已經是 [0,1] 範圍
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean((clean - noisy) ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

psnr_list = []
snr_list = []

for i in range(1):
    # 重新取一張圖
    image = next(iter(dataloader))[0].to(device)

    # 加噪 x_hat
    x_hat = diffusion.noise_images_x_hat(image, t=300)

    # 去噪
    x, _ = diffusion.sample(model, n=1, x_hat=x_hat, lambda_schedule=lambda_schedule)

    # 正規化
    image_norm = (image + 1) / 2
    x_clamped = torch.clamp(x, 0., 1.)
    x_hat_clamped = torch.clamp(x_hat, 0., 1.)

    # 計算 PSNR
    psnr = calculate_psnr(image_norm, x_clamped)
    psnr_list.append(psnr)

    # 計算 SNR（原圖 vs 加噪圖）
    snr = calculate_snr(image_norm, x_hat_clamped)
    snr_list.append(snr)

    print(f"Run {i+1}: SNR = {snr:.2f} dB, PSNR = {psnr:.2f} dB")

# 繪圖
plt.figure()
plt.scatter(snr_list, psnr_list, marker='o')
plt.title('PSNR vs. SNR')
plt.xlabel('SNR (dB)')
plt.ylabel('PSNR (dB)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, "psnr_vs_snr_curve.png"))
plt.close()

# 平均值列印
avg_psnr = sum(psnr_list) / len(psnr_list)
avg_snr = sum(snr_list) / len(snr_list)
print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
print(f"Average SNR: {avg_snr:.2f} dB")
'''