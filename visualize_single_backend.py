import numpy as np
import matplotlib.pyplot as plt
import os

def load_complex(path):
    if not os.path.exists(path):
        print(f"⚠️ 找不到檔案: {path}")
        return None
    return np.load(path)

def main():
    # 1. 讀取四個關鍵節點的訊號
    target = load_complex('bridge_tx_remote/x_tx.npy')        # 遠端完美訊號 (Ground Truth)
    y_adc = load_complex('bridge/y_adc.npy')                  # 類比消除後 (未做數位處理)
    y_mp = load_complex('bridge_digital/y_clean_mp.npy')      # 傳統數學 MP 去噪結果
    y_diff = load_complex('bridge_digital/y_clean_diff.npy')  # 1D Diffusion 去噪結果

    if any(x is None for x in [target, y_adc, y_mp, y_diff]):
        print("❌ 請確保已經分別跑過 MP 與 Diffusion 並將檔案正確命名！")
        return

    # 為了對齊長度，取最小值
    min_len = min(len(target), len(y_adc), len(y_mp), len(y_diff))
    target, y_adc, y_mp, y_diff = target[:min_len], y_adc[:min_len], y_mp[:min_len], y_diff[:min_len]

    # ==========================================
    # 📊 圖表 1：時域波形圖 (Time-Domain Waveform)
    # ==========================================
    plt.figure(figsize=(14, 6))
    L = 150 # 只取前 150 個採樣點以便看清波形細節
    t_axis = np.arange(L)

    # 畫圖順序：背景雜訊 -> MP -> Diffusion -> 目標訊號
    plt.plot(t_axis, np.real(y_adc[:L]), label='Analog Only (Noisy ADC)', color='lightgray', linewidth=2, alpha=0.8)
    plt.plot(t_axis, np.real(y_mp[:L]), label='Digital SIC (WL-MP)', color='orange', linewidth=1.5, alpha=0.9)
    plt.plot(t_axis, np.real(y_diff[:L]), label='AI-SIC (1D Diffusion)', color='blue', linewidth=2)
    plt.plot(t_axis, np.real(target[:L]), label='Ground Truth (Target)', color='green', linestyle='dashed', linewidth=2)

    plt.title('Time-Domain Waveform Comparison (Real Part / I-Channel)', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plot_waveform_comparison.png', dpi=300)
    print("✅ 時域波形圖已儲存為 plot_waveform_comparison.png")

    # ==========================================
    # 📊 圖表 2：頻譜圖 (Power Spectral Density, PSD)
    # ==========================================
    plt.figure(figsize=(12, 7))
    
    # 使用 matplotlib 內建的 psd 函數
    plt.psd(y_adc, NFFT=1024, Fs=1.0, color='lightgray', label='Analog Only', linewidth=2)
    plt.psd(y_mp, NFFT=1024, Fs=1.0, color='orange', label='Digital SIC (WL-MP)', linewidth=1.5)
    plt.psd(y_diff, NFFT=1024, Fs=1.0, color='blue', label='AI-SIC (1D Diffusion)', linewidth=2)
    plt.psd(target, NFFT=1024, Fs=1.0, color='green', label='Ground Truth', linestyle='dashed', linewidth=2)

    plt.title('Power Spectral Density (PSD) Comparison', fontsize=14)
    plt.xlabel('Normalized Frequency', fontsize=12)
    plt.ylabel('Power/Frequency (dB/Hz)', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('plot_spectrum_comparison.png', dpi=300)
    print("✅ 頻譜對照圖已儲存為 plot_spectrum_comparison.png")

if __name__ == "__main__":
    main()