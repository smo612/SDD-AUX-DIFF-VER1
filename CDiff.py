import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from modules_CDiff import UNet
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class SDDDataset(Dataset):
    def __init__(self, pt_file):
        logging.info(f"載入 SDD 資料集: {pt_file}")
        self.data = torch.load(pt_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['target'], item['cond'], item['noisy']

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"): # 🌟 移除了用不到的 img_size
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def add_noise_ddpm(self, x, t):
        # 🌟 降維廣播：移除一個 None，對應 1D 的 (Batch, Channel, Length)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        noise = torch.randn_like(x)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_t, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=250, size=(n,))    

def train(args):
    device = args.device
    dataset = SDDDataset("sdd_diffusion_dataset.pt")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    diffusion = Diffusion(device=device)
    model = UNet(c_in=6, c_out=2, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    all_epoch_losses = []

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        batch_losses = []
        
        for i, (target, cond, noisy) in enumerate(pbar):
            target = target.to(device)
            cond = cond.to(device)
            noisy = noisy.to(device)

            t = diffusion.sample_timesteps(target.shape[0]).to(device)
            x_t, noise = diffusion.add_noise_ddpm(target, t)

            condition = torch.cat([noisy, cond], dim=1)
            predicted_noise = model(x_t, t, x_hat=condition)

            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            batch_losses.append(loss_val)
            pbar.set_postfix(MSE=loss_val)
            logger.add_scalar("MSE", loss_val, global_step=epoch * l + i)

        all_epoch_losses.append(batch_losses)

        if epoch % 10 == 0:
            os.makedirs(os.path.join("ddpm_models", args.run_name), exist_ok=True)
            torch.save(model.state_dict(), os.path.join("ddpm_models", args.run_name, f"ckpt_epoch_{epoch}.pt"))
            logging.info(f"模型權重已儲存: ckpt_epoch_{epoch}.pt")

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "SDD_Diffusion_VER4_RSI315W_1D" # 🌟 換個新名字，紀念 1D 時代的開始！
    args.epochs = 200 
    args.batch_size = 12
    args.device = "cuda"
    args.lr = 3e-4
    train(args)

if __name__ == '__main__':
    launch()