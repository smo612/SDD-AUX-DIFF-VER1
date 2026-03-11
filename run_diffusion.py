import os
import torch
import numpy as np
from tqdm import tqdm
from modules_CDiff import UNet
from CDiff import Diffusion
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def process_complex_to_1d(signal, target_len=8704):
    pad_size = target_len - len(signal)
    signal_padded = np.pad(signal, (0, pad_size), mode='constant')
    real_part = np.real(signal_padded)
    imag_part = np.imag(signal_padded)
    tensor_1d = torch.tensor(np.stack([real_part, imag_part]), dtype=torch.float32)
    # 🌟 為了推論時的 Batch 維度，加上 unsqueeze(0) 變成 (1, 2, 8704)
    return tensor_1d.unsqueeze(0)

def process_1d_to_complex(tensor_1d, original_len=8319):
    # tensor_1d 輸入維度為 (1, 2, 8704)
    tensor_1d = tensor_1d.squeeze(0).cpu().numpy() # 變成 (2, 8704)
    complex_signal = tensor_1d[0] + 1j * tensor_1d[1] # 合併實部與虛部
    return complex_signal[:original_len]

def main():
    device = "cuda"
    
    cond_np = np.load('bridge_tx/x_tx.npy')
    noisy_np = np.load('bridge/y_adc.npy')
    
    # 計算 STD 並將推論特徵正規化
    scale = np.std(noisy_np) + 1e-8
    
    # 🌟 換成 1D 處理函數
    cond = process_complex_to_1d(cond_np / scale).to(device)
    noisy = process_complex_to_1d(noisy_np / scale).to(device)
    condition = torch.cat([noisy, cond], dim=1) 

    # 🌟 記得改成 1D 專屬權重資料夾名稱！
    model_path = "ddpm_models/SDD_Diffusion_VER4_RSI315W_1D/ckpt_epoch_190.pt"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型權重: {model_path}")
        exit(1)
        
    model = UNet(c_in=6, c_out=2, device=device).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    diffusion = Diffusion(device=device)

    # 從環境變數動態讀取 T_start (相容之前的 GP6 掃描腳本)，預設為 200
    T_start = int(os.environ.get("T_START", 200))
    print(f"🚀 啟動 1D AI-SIC SDEdit (從 T={T_start} 開始精修)...")
    
    t_start_tensor = (torch.ones(1) * T_start).long().to(device)
    alpha_hat_t = diffusion.alpha_hat[t_start_tensor][:, None, None] # 🌟 降維：移除一個 None
    noise_init = torch.randn_like(noisy)
    
    # 在 ADC 接收訊號上加噪，產生 x_{T_start}
    x = torch.sqrt(alpha_hat_t) * noisy + torch.sqrt(1 - alpha_hat_t) * noise_init

    with torch.no_grad():
        for i in tqdm(reversed(range(1, T_start)), total=T_start-1, desc="1D AI-SIC Denoising", leave=False):
            t = (torch.ones(1) * i).long().to(device)
            predicted_noise = model(x, t, x_hat=condition)
            
            alpha = diffusion.alpha[t][:, None, None]         # 🌟 降維：移除一個 None
            alpha_hat = diffusion.alpha_hat[t][:, None, None] # 🌟 降維：移除一個 None
            beta = diffusion.beta[t][:, None, None]           # 🌟 降維：移除一個 None
            
            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

    # 🌟 換成 1D 還原函數，並乘回 scale
    denoised_complex = process_1d_to_complex(x) * scale
    
    os.makedirs('bridge_digital', exist_ok=True)
    output_path = "bridge_digital/y_clean.npy"
    np.save(output_path, denoised_complex)
    
if __name__ == "__main__":
    main()