import torch
import numpy as np
from tqdm import tqdm
from modules_CDiff import UNet
from CDiff import Diffusion

def process_complex_to_2d(npy_path, target_len=8704, H=64, W=136):
    """讀取 1D 複數並轉為 (1, 2, H, W) Tensor (加上 Batch 維度)"""
    signal = np.load(npy_path)
    pad_size = target_len - len(signal)
    signal_padded = np.pad(signal, (0, pad_size), mode='constant')
    real_part = np.real(signal_padded)
    imag_part = np.imag(signal_padded)
    tensor_2d = torch.tensor(np.stack([real_part, imag_part]), dtype=torch.float32)
    return tensor_2d.view(1, 2, H, W)

def process_2d_to_complex(tensor_2d, original_len=8319):
    """將去噪後的 (1, 2, H, W) Tensor 還原為 1D 複數陣列"""
    tensor_2d = tensor_2d.squeeze(0).cpu().numpy() # 變成 (2, 8704)
    tensor_1d = tensor_2d.reshape(2, -1)           # 展平
    complex_signal = tensor_1d[0] + 1j * tensor_1d[1] # 合併實部與虛部
    return complex_signal[:original_len]           # 砍掉當初補的零

def main():
    device = "cuda"
    
    print("🔄 正在載入實體層特徵...")
    # 讀取條件 (本地干擾源) 與帶噪訊號
    cond = process_complex_to_2d('bridge_tx/x_tx.npy').to(device)
    noisy = process_complex_to_2d('bridge/y_adc.npy').to(device)
    
    # 拼接成 4 通道條件
    condition = torch.cat([noisy, cond], dim=1) 

    print("🧠 正在載入訓練好的 Diffusion 模型...")
    # 初始化模型並載入剛剛訓練好的權重 (根據你的 log，這裡抓 epoch_390)
    model = UNet(c_in=6, c_out=2, device=device).to(device)
    model.load_state_dict(torch.load("ddpm_models/SDD_Diffusion_VER4/ckpt_epoch_390.pt"))
    model.eval()

    diffusion = Diffusion(device=device)

    print("🚀 開始反向去噪採樣 (Denoising Sampling)...")
    # 從純高斯雜訊 x_T 開始
    x = torch.randn((1, 2, 64, 136)).to(device) 
    
    # 標準 DDPM 逆向採樣迴圈
    with torch.no_grad():
        for i in tqdm(reversed(range(1, diffusion.noise_steps)), total=diffusion.noise_steps-1):
            t = (torch.ones(1) * i).long().to(device)
            predicted_noise = model(x, t, x_hat=condition)
            
            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

    print("✨ 去噪完成！正在還原物理層維度...")
    denoised_complex = process_2d_to_complex(x)
    
    # 存成新的 numpy 檔案供 NTSCC 解碼
    output_path = "bridge/y_adc_denoised.npy"
    np.save(output_path, denoised_complex)
    print(f"✅ 成功儲存乾淨特徵至：{output_path}")

if __name__ == "__main__":
    main()