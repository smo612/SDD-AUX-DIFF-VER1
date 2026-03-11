#!/usr/bin/env python3
"""
visualize_physics_waveforms.py
專門用來視覺化 Fig. 1 物理層各個 Block 的波形變化。
重點展示：Main PA 與 Aux PA 的 Mismatch，以及消除前後的差異。
"""

import numpy as np
import matplotlib.pyplot as plt
from sdd_channel_model_v5 import simulate_full_receive_signal
import config as C

def plot_psd(ax, signal, label, color, linestyle='-'):
    """畫功率頻譜密度 (PSD)"""
    # 使用 Hamming Window 減少頻譜洩漏
    f, Pxx = plt.psd(signal, NFFT=1024, Fs=1.0, window=np.hamming(1024), 
                     scale_by_freq=False, return_line=None)
    # 轉換成 dB
    Pxx = 10 * np.log10(Pxx + 1e-18)
    # Shift frequency to center (0在中間)
    Pxx = np.fft.fftshift(Pxx)
    f = np.fft.fftshift(f)
    
    ax.plot(f, Pxx, label=label, color=color, linestyle=linestyle, linewidth=1.5)

def main():
    print("🚀 Running Single-Shot Physics Simulation for Visualization...")
    
    # 1. 產生測試訊號 (OFDM-like or Random)
    N = 4096
    x_self = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
    # 稍微過採樣或濾波讓頻譜好看一點 (Optional)
    
    # 2. 執行模擬
    out = simulate_full_receive_signal(
        x_remote=np.zeros(N), # 這裡先不看遠端訊號，專注看干擾
        x_self=x_self,
        snr_db=30,
        rsi_scale=100.0, # 開大一點讓非線性明顯
        sic_db=30,
        use_realistic_analog_sic=True,
        enable_pa_nonlinearity=True
    )
    
    waves = out['debug_waveforms']
    
    # 3. 開始繪圖 (模仿 Fig. 1 的 Layout)
    fig = plt.figure(figsize=(16, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # --- Plot 1: Main Path Physics (時域振幅) ---
    ax1 = plt.subplot(2, 2, 1)
    # 取前 100 點來看波形細節
    t = np.arange(100)
    ax1.plot(t, np.abs(waves['input'][0:100]), 'k--', label='Input (Baseband)', alpha=0.5)
    ax1.plot(t, np.abs(waves['main_after_pa'][0:100]), 'r-', label='After Main PA (p=2.2)')
    ax1.set_title("Main Path: Nonlinear Distortion (Time Domain)", fontweight='bold')
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time Sample")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: The Mismatch (Aux PA vs Main PA) ---
    # 這是最關鍵的一張圖！展示為什麼消不乾淨
    ax2 = plt.subplot(2, 2, 2)
    # 畫 AM-AM 曲線
    vin = np.abs(waves['main_after_iq'])
    vout_main = np.abs(waves['main_after_pa'])
    vout_aux = np.abs(waves['aux_after_pa'])
    
    # Sort for plotting curve
    idx = np.argsort(vin)
    ax2.plot(vin[idx], vout_main[idx], 'r-', linewidth=2, label='Main PA Output (p=2.2)')
    ax2.plot(vin[idx], vout_aux[idx], 'b--', linewidth=2, label='Aux PA Output (p=2.4)')
    
    # 畫出差值 (Mismatch)
    diff = np.abs(vout_main - vout_aux)
    ax2.fill_between(vin[idx], vout_main[idx], vout_aux[idx], color='purple', alpha=0.3, label='Mismatch Area')
    
    ax2.set_title("Physical Mismatch: Main PA vs Aux PA (AM-AM)", fontweight='bold')
    ax2.set_xlabel("Input Amplitude")
    ax2.set_ylabel("Output Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Cancellation Process (PSD) ---
    # 頻域看消除效果
    ax3 = plt.subplot(2, 2, 3)
    plot_psd(ax3, waves['main_arrived_at_rx'], "Received SI (Interference)", 'red')
    plot_psd(ax3, waves['aux_cancellation_signal'], "Aux Cancellation Signal", 'blue', '--')
    plot_psd(ax3, waves['residual_after_analog'], "Residual (After Cancellation)", 'green')
    
    ax3.set_title("RF Cancellation Spectrum (PSD)", fontweight='bold')
    ax3.set_xlabel("Normalized Frequency")
    ax3.set_ylabel("Power (dB)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Residual Signal Analysis ---
    ax4 = plt.subplot(2, 2, 4)
    # 畫殘留訊號的時域波形
    ax4.plot(t, np.abs(waves['residual_after_analog'][0:100]), 'g-', label='Residual Error', linewidth=2)
    ax4.set_title("What's Left for Digital? (Residual Waveform)", fontweight='bold')
    ax4.set_xlabel("Time Sample")
    ax4.set_ylabel("Amplitude")
    
    # 標註：這就是 MP 要做的事
    ax4.text(50, np.max(np.abs(waves['residual_after_analog'][0:100]))*0.8, 
             "This specific pattern\nneeds MP to fix!", 
             ha='center', color='darkgreen', fontweight='bold', fontsize=12)
    
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle("Physics Engine Visualization: Inside the Aux-TX Architecture", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("fig1_physics_waveforms.png", dpi=300)
    print("✅ Visualization saved to fig1_physics_waveforms.png")

if __name__ == "__main__":
    main()