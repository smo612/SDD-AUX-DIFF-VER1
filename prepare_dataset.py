import numpy as np
import torch

def process_complex_to_2d(npy_path, target_len=8704, H=64, W=136):
    """將 1D 複數陣列轉換為 2D PyTorch Tensor (2, H, W)"""
    # 1. 讀取 1D 複數訊號 (8319,)
    signal = np.load(npy_path)
    
    # 2. 補零 (Zero-padding) 到 8704
    pad_size = target_len - len(signal)
    signal_padded = np.pad(signal, (0, pad_size), mode='constant')
    
    # 3. 拆分實部與虛部
    real_part = np.real(signal_padded)
    imag_part = np.imag(signal_padded)
    
    # 4. 堆疊並 Reshape 成 (2, 64, 136)
    tensor_2d = torch.tensor(np.stack([real_part, imag_part]), dtype=torch.float32)
    tensor_2d = tensor_2d.view(2, H, W)
    
    return tensor_2d

if __name__ == "__main__":
    # 測試轉換
    x_target = process_complex_to_2d('bridge_tx_remote/x_tx.npy') # 遠端乾淨特徵 (Ground Truth)
    x_cond = process_complex_to_2d('bridge_tx/x_tx.npy')          # 本地干擾特徵 (Condition)
    y_noisy = process_complex_to_2d('bridge/y_adc.npy')           # 帶噪接收特徵 (Noisy Input)
    
    print(f"✅ Target (Clean) Tensor Shape: {x_target.shape}")
    print(f"✅ Condition (Local) Tensor Shape: {x_cond.shape}")
    print(f"✅ Noisy (ADC) Tensor Shape: {y_noisy.shape}")