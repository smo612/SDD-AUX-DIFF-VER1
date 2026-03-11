#!/usr/bin/env python3
"""驗證 RSI_SCALE 修復是否生效"""
import numpy as np
import sys
sys.path.insert(0, '.')

from sdd_channel_model import simulate_full_receive_signal

# 生成測試信號
N = 1000
x_remote = (np.random.randn(N) + 1j*np.random.randn(N)).astype(np.complex64)
x_self = (np.random.randn(N) + 1j*np.random.randn(N)).astype(np.complex64)

# 歸一化到單位功率
x_remote /= np.sqrt(np.mean(np.abs(x_remote)**2))
x_self /= np.sqrt(np.mean(np.abs(x_self)**2))

print("="*60)
print("RSI_SCALE 修復驗證")
print("="*60)
print(f"\n輸入信號功率: {np.mean(np.abs(x_self)**2):.6f}")
print()

# 測試不同的 RSI_SCALE
rsi_scales = [5, 10, 20, 50, 100, 200]
results = []

for rsi in rsi_scales:
    rx_out = simulate_full_receive_signal(
        x_remote=x_remote,
        x_self=x_self,
        snr_db=22.0,
        rsi_scale=rsi,
        sic_db=23.0,
        use_realistic_analog_sic=True,
        enable_pa_nonlinearity=False
    )
    
    P_si_before = np.mean(np.abs(rx_out['y_rsi_before_analog'])**2)
    saturated = rx_out['analog_sic_info']['saturated']
    
    results.append((rsi, P_si_before, saturated))
    print(f"RSI_SCALE={rsi:3d}: P_si_before={P_si_before:.3e}, Saturated={saturated}")

print()
print("="*60)
print("驗證結果")
print("="*60)

# 檢查是否線性增長
base_power = results[0][1]  # RSI_SCALE=5 的功率
expected_ratios = [1, 2, 4, 10, 20, 40]  # 5→5,  10→10, 20→20...

print(f"\nRSI_SCALE 增長比 vs 功率增長比：")
all_pass = True

for i, (rsi, power, saturated) in enumerate(results):
    actual_ratio = power / base_power
    expected_ratio = expected_ratios[i]
    ratio_error = abs(actual_ratio - expected_ratio) / expected_ratio
    
    if ratio_error < 0.2:  # 允許 20% 誤差
        status = "✓"
    else:
        status = "✗"
        all_pass = False
    
    print(f"  {status} {rsi:3d}/5 = {rsi/5:4.0f}x → 功率增長 {actual_ratio:5.1f}x (預期 {expected_ratio:4.0f}x)")

# 檢查飽和
print(f"\nAnalog SIC 飽和檢查：")
for i, (rsi, power, saturated) in enumerate(results):
    if rsi >= 50:
        expected_saturated = True
    else:
        expected_saturated = False
    
    if saturated == expected_saturated:
        status = "✓"
    else:
        status = "✗"
        all_pass = False
    
    print(f"  {status} RSI_SCALE={rsi:3d}: Saturated={saturated} (預期 {expected_saturated})")

print()
if all_pass:
    print("🎉 修復成功！所有測試通過")
else:
    print("❌ 修復失敗！需要重新檢查代碼")
    print("\n可能的問題：")
    print("  1. 文件沒有完全替換")
    print("  2. Python 緩存了舊版模塊")
    print("  3. 代碼中還有其他歸一化")