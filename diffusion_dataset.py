import os
import subprocess
import numpy as np
import torch
from tqdm import tqdm
import itertools

def process_complex_to_1d(signal, target_len=8704):
    pad_size = target_len - len(signal)
    signal_padded = np.pad(signal, (0, pad_size), mode='constant')
    real_part = np.real(signal_padded)
    imag_part = np.imag(signal_padded)
    # 🌟 直接回傳 1D 特徵，不折疊成 2D。輸出 shape 為 (2, 8704)
    tensor_1d = torch.tensor(np.stack([real_part, imag_part]), dtype=torch.float32)
    return tensor_1d

def generate_dataset(output_file="sdd_diffusion_dataset.pt"):
    img_names = [f"kodim{str(i).zfill(2)}" for i in range(1, 25)]
    pairs = list(itertools.permutations(img_names, 2)) 
    
    dataset = []
    print(f"🚀 準備生成 {len(pairs)} 筆【極端地獄難度 RSI=315W】的 1D 訓練資料...")
    
    for local_img, remote_img in tqdm(pairs):
        cmd = f"python run_sdd_final.py --local {local_img} --remote {remote_img} --no-digital-sic --rsi-scale 3150000 --aux-disable-iqpa False"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\n❌ 執行失敗 ({local_img} vs {remote_img})！錯誤訊息：")
            print(result.stderr)
            print("🛑 停止生成，請先修復上述錯誤。")
            return
            
        path_target = 'bridge_tx_remote/x_tx.npy'
        path_cond = 'bridge_tx/x_tx.npy'
        path_noisy = 'bridge/y_adc.npy'
        
        if not (os.path.exists(path_target) and os.path.exists(path_cond) and os.path.exists(path_noisy)):
            continue
            
        x_target_np = np.load(path_target)
        x_cond_np = np.load(path_cond)
        y_noisy_np = np.load(path_noisy)
        
        scale = np.std(y_noisy_np) + 1e-8
        
        # 🌟 換成 1D 處理函數
        dataset.append({
            'target': process_complex_to_1d(x_target_np / scale),
            'cond': process_complex_to_1d(x_cond_np / scale),
            'noisy': process_complex_to_1d(y_noisy_np / scale)
        })
        
    if len(dataset) > 0:
        print(f"\n✅ 成功儲存 {len(dataset)} 筆資料至 {output_file}！")
        torch.save(dataset, output_file)

if __name__ == "__main__":
    generate_dataset("sdd_diffusion_dataset.pt")