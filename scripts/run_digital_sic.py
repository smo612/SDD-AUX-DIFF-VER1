#!/usr/bin/env python3
"""
Digital SIC - Unified Runner (MP & WLLS)
支援後端切換: --backend [mp|wlls]
修正: 防止 SINR 為 None 時導致 Crash
"""

import numpy as np
import json
import argparse
from pathlib import Path
import sys

# 將專案根目錄加入路徑
sys.path.append(str(Path(__file__).parent.parent))

from utils.wlls_wrapper import WLLSDigitalSIC, sweep_wlls_parameters

# 嘗試導入 MP
try:
    from SIC.mp import MPBackend
    HAS_MP = True
except ImportError:
    print("⚠️ 找不到 SIC.mp 模組，MP 後端將不可用")
    HAS_MP = False

# ✅ 導入導頻校正工具 (選擇性)
try:
    from utils.pilot_correction import pilot_based_correction
    PILOT_CORRECTION_AVAILABLE = True
except ImportError:
    PILOT_CORRECTION_AVAILABLE = False


def load_analog_output(bridge_dir='bridge'):
    """載入 analog 階段輸出"""
    bridge_path = Path(bridge_dir)
    y_adc = np.load(bridge_path / 'y_adc.npy')
    
    # 載入 SI-only 波形（如果存在）
    # 嘗試多種可能的命名
    possible_names = ['y_si_after_analog.npy', 'y_rsi_after_analog.npy']
    y_si_after_analog = None
    for name in possible_names:
        p = bridge_path / name
        if p.exists():
            y_si_after_analog = np.load(p)
            break
            
    with open(bridge_path / 'meta.json', 'r') as f:
        meta = json.load(f)
        
    P_signal = meta.get('P_main', None)
    return y_adc, y_si_after_analog, meta, P_signal


def load_tx_signal(bridge_tx_dir='bridge_tx'):
    """載入 TX 信號"""
    path = Path(bridge_tx_dir)
    x_tx = np.load(path / 'x_tx.npy')
    return x_tx


def align_lengths(y, x):
    n_min = min(len(y), len(x))
    return y[:n_min], x[:n_min]


def to_json_serializable(value):
    if value is None:
        return None
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.complex64) or isinstance(value, np.complex128):
        return {'real': float(value.real), 'imag': float(value.imag)}
    else:
        return value


def save_digital_output(y_clean, h_hat, metrics, meta_analog, backend_name, output_dir='bridge_digital'):
    """保存 Digital SIC 輸出"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / 'y_clean.npy', y_clean)
    if h_hat is not None:
        np.save(output_path / 'h_hat.npy', h_hat)
    
    # 計算總抑制
    analog_supp = meta_analog.get('Supp_analog', 0.0)
    digital_supp_si = metrics.get('Digital_supp_si', 0.0)
    
    total_supp_si_only = None
    if analog_supp is not None and digital_supp_si is not None:
        try:
            total_supp_si_only = float(analog_supp + digital_supp_si)
        except:
            total_supp_si_only = 0.0
    
    sinr_final = metrics.get('SINR_after_digital') or metrics.get('SINR_after')
    
    meta_digital = {
        'backend': backend_name,
        'snr_db': to_json_serializable(meta_analog.get('snr_db')),
        'Supp_analog': to_json_serializable(analog_supp),
        'SINR_analog': to_json_serializable(meta_analog.get('SINR_analog')),
        
        'Digital_gain': to_json_serializable(metrics.get('Digital_gain')),
        'Digital_supp_si': to_json_serializable(digital_supp_si),
        'SINR_after_digital': to_json_serializable(sinr_final),
        
        'Total_supp_SI_only': to_json_serializable(total_supp_si_only),
        
        'params': {
            'L': metrics.get('L'),
            'lambda': metrics.get('lambda')
        },
        'pilot_correction': metrics.get('pilot_correction_info')
    }
    
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(meta_digital, f, indent=2)
    
    print(f"\n✅ {backend_name} Digital SIC 完成")
    print(f"  Analog Gain: {analog_supp:.2f} dB")
    print(f"  Digital Gain: {metrics.get('Digital_gain', 0):.2f} dB")
    if sinr_final is not None:
        print(f"  Final SINR: {sinr_final:.2f} dB")
    else:
        print(f"  Final SINR: N/A (SI reference missing)")


def main():
    parser = argparse.ArgumentParser(description='Digital SIC Runner')
    
    # 新增 backend 參數
    parser.add_argument('--backend', type=str, default='wlls', choices=['wlls', 'mp'],
                       help='選擇數位消除演算法: wlls 或 mp')

    # 通用參數
    parser.add_argument('--bridge-dir', type=str, default='bridge')
    parser.add_argument('--bridge-tx-dir', type=str, default='bridge_tx')
    parser.add_argument('--output-dir', type=str, default='bridge_digital')
    
    # 演算法參數
    parser.add_argument('--L', type=int, default=5, help='記憶長度/通道階數')
    parser.add_argument('--lambda-reg', type=float, default=0.01, help='正則化係數')
    parser.add_argument('--widely-linear', action='store_true', default=True)
    
    # 導頻校正
    parser.add_argument('--pilot-correction', action='store_true', default=True)
    parser.add_argument('--pilot-period', type=int, default=64)
    parser.add_argument('--n-pilots', type=int, default=127)
    
    # WLLS 專用
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--holdout-ratio', type=float, default=0.2)
    parser.add_argument('--skip-samples', type=int, default=64)
    parser.add_argument('--version', type=str, default='conservative')

    args = parser.parse_args()
    
    print("="*60)
    print(f"Digital SIC Runner - Backend: {args.backend.upper()}")
    print("="*60)

    # 1. 載入數據
    y_adc, y_si_after_analog, meta_analog, P_signal = load_analog_output(args.bridge_dir)
    x_tx = load_tx_signal(args.bridge_tx_dir)
    y_adc, x_tx = align_lengths(y_adc, x_tx)
    
    if y_si_after_analog is not None:
        n_min = min(len(y_adc), len(y_si_after_analog))
        y_si_after_analog = y_si_after_analog[:n_min]
        y_adc = y_adc[:n_min]
        x_tx = x_tx[:n_min]
    
    # 2. 導頻校正 (共通步驟)
    pilot_info = None
    if args.pilot_correction and PILOT_CORRECTION_AVAILABLE:
        try:
            print("[導頻校正] 執行中...")
            y_adc, alpha_est = pilot_based_correction(y_adc, args.pilot_period, args.n_pilots)
            if y_si_after_analog is not None:
                y_si_after_analog = y_si_after_analog / alpha_est
            pilot_info = {'enabled': True, 'alpha_abs': float(np.abs(alpha_est))}
            print("  ✅ 校正完成")
        except Exception as e:
            print(f"  ⚠️ 校正失敗: {e}")
            pilot_info = {'enabled': False, 'error': str(e)}

    noise_var = meta_analog.get('noise_var', 0.0)
    amp_scale = meta_analog.get('amp_scale', 1.0)

    # 3. 執行選定的後端
    if args.backend == 'mp':
        if not HAS_MP:
            raise ImportError("無法載入 MP Backend")
            
        print(f"[MP] 開始執行 Memory Polynomial SIC (Order=[1,3,5], M={args.L})...")
        
        # 初始化 MP
        mp = MPBackend(
            poly_orders=[1, 3, 5], 
            memory_len=args.L, 
            ridge_lambda=args.lambda_reg
        )
        
        # 準備資料
        data_train = {
            'y': y_adc,
            'x': x_tx,
            'si_after_analog': y_si_after_analog
        }
        
        # 訓練
        mp.fit(data_train)
        
        # 預測
        batch_predict = {
            'y': y_adc,
            'x': x_tx,
            'si_after_analog': y_si_after_analog,
            'P_signal': P_signal,
            'P_noise': noise_var
        }
        
        r_hat, metrics = mp.predict(batch_predict)
        h_hat = np.array([0]) 
        
        # ✅ 安全計算 Digital Gain (防止 NoneType error)
        sinr_after = metrics.get('SINR_after_digital')
        sinr_analog = meta_analog.get('SINR_analog', 0)
        
        if sinr_after is not None and sinr_analog is not None:
            metrics['Digital_gain'] = sinr_after - sinr_analog
        else:
            metrics['Digital_gain'] = 0.0
            
        metrics['pilot_correction_info'] = pilot_info
        
        y_clean = y_adc - r_hat

    elif args.backend == 'wlls':
        print(f"[WLLS] 開始執行 (L={args.L}, lambda={args.lambda_reg})...")
        sic = WLLSDigitalSIC(
            L=args.L,
            lambda_reg=args.lambda_reg,
            use_widely_linear=args.widely_linear,
            version=args.version
        )
        
        y_clean, metrics, info = sic.process(
            y_adc, x_tx, noise_var, amp_scale,
            y_si_after_analog=y_si_after_analog,
            P_signal=P_signal,
            return_full_info=True
        )
        h_hat = info['h_hat']
        metrics['pilot_correction_info'] = pilot_info

    # 4. 保存結果
    save_digital_output(
        y_clean, h_hat, metrics, meta_analog, 
        backend_name=args.backend.upper(),
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()