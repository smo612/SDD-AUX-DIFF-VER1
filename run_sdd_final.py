#!/usr/bin/env python3
"""
run_sdd_final.py - SDD Final Experiment (VER4: AI-SIC Diffusion Integration)
修復與新增: 
1. 容忍 SINR 為 None 的情況 (避免 Final Report 崩潰)
2. 確保一定會印出 PSNR 結果
3. 新增: 執行與報告時印出 AUX_DISABLE_IQPA 狀態
4. 支援透過 --aux-disable-iqpa 直接從終端機覆寫 config
5. 🌟 新增: --use-diffusion 參數，以條件擴散模型取代傳統數位 SIC
"""
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import subprocess
import shutil
from pathlib import Path
import json
import re
import numpy as np
import sys

sys.path.append(os.getcwd())

try:
    import config
    default_aux_disable = getattr(config, 'AUX_DISABLE_IQPA', 'Unknown')
except ImportError:
    default_aux_disable = 'Unknown'

def run_command(cmd, desc):
    print(f"➜ {desc}...", end=" ", flush=True)
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print("❌ 失敗")
        print("="*40 + " ERROR LOG " + "="*40)
        print(res.stderr)
        print(res.stdout) 
        print("="*90)
        exit(1)
    print("✅")

def force_cleanup(path_str):
    p = Path(path_str)
    if p.exists():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', type=str, required=True)
    parser.add_argument('--remote', type=str, required=True)
    parser.add_argument('--backend', type=str, default='mp', choices=['wlls', 'mp'])
    parser.add_argument('--rsi-scale', type=float, default=20.0)
    parser.add_argument('--no-digital-sic', action='store_true')
    parser.add_argument('--no-normalize', action='store_true')
    # 🌟 新增 Diffusion 參數
    parser.add_argument('--use-diffusion', action='store_true', help="使用 Conditional Diffusion 取代傳統數位 SIC")
    
    parser.add_argument('--aux-disable-iqpa', type=str, choices=['True', 'False'], default=None,
                        help="強制覆寫 AUX_DISABLE_IQPA (True:理想硬體不帶瑕疵, False:真實硬體帶預補償)")
    args = parser.parse_args()

    if args.aux_disable_iqpa is not None:
        run_aux_status = True if args.aux_disable_iqpa == 'True' else False
        status_source = "Argparse 覆寫"
    else:
        run_aux_status = default_aux_disable
        status_source = "Config 預設"

    print("\n" + "="*60)
    print(f"🚀 SDD Final Physics Simulation (RSI={args.rsi_scale}x)")
    print(f"⚙️  硬體設定: AUX_DISABLE_IQPA = {run_aux_status} ({status_source})")
    print(f"🧠 AI-SIC (Diffusion) 啟用狀態: {args.use_diffusion}")
    print("="*60 + "\n")

    # 1. 準備 TX
    norm_flag = "--no-normalize" if args.no_normalize else ""
    run_command(f"python scripts/run_tx_kodak_batch.py --img {args.local} {norm_flag}", "生成干擾源 (Local TX)")
    force_cleanup('bridge_tx_local')
    shutil.copytree('bridge_tx', 'bridge_tx_local')
    
    run_command(f"python scripts/run_tx_kodak_batch.py --img {args.remote} {norm_flag}", "生成期望訊號 (Remote TX)")
    force_cleanup('bridge_tx_remote')
    shutil.move('bridge_tx', 'bridge_tx_remote')
    
    force_cleanup('bridge_tx')
    shutil.move('bridge_tx_local', 'bridge_tx')

    # 2. 執行類比模擬
    cfg_path = Path('config.py')
    cfg_orig = cfg_path.read_text()
    cfg_new = re.sub(r'RSI_SCALE\s*=\s*[\d.]+', f'RSI_SCALE = {args.rsi_scale}', cfg_orig)
    if args.aux_disable_iqpa is not None:
        cfg_new = re.sub(r'AUX_DISABLE_IQPA\s*=\s*(True|False)', f'AUX_DISABLE_IQPA = {args.aux_disable_iqpa}', cfg_new)
    cfg_path.write_text(cfg_new)
    try:
        run_command("python run_analog_semantic.py", "執行 Aux-TX 類比物理模擬")
    finally:
        cfg_path.write_text(cfg_orig)

    # 3. 執行數位模擬 或 Diffusion 去噪
    force_cleanup('bridge_digital')
    Path('bridge_digital').mkdir(exist_ok=True)
    
    if args.use_diffusion:
        # 🌟 執行 Diffusion
        run_command("python run_diffusion.py", "執行 Diffusion AI-SIC 去噪精修")
        with open('bridge/meta.json') as f: m = json.load(f)
        d_meta = {'Digital_supp_note': 'Diffusion AI-SIC', 'SINR_after_digital': None, 'backend': 'Diffusion'}
        with open('bridge_digital/metrics.json', 'w') as f: json.dump(d_meta, f)
        
    elif args.no_digital_sic:
        print("➜ Digital SIC... (已跳過: Analog Only Mode)")
        shutil.copy('bridge/y_adc.npy', 'bridge_digital/y_clean.npy')
        with open('bridge/meta.json') as f: m = json.load(f)
        d_meta = {'Digital_supp_note': 'SKIPPED', 'SINR_after_digital': m['SINR_analog'], 'backend': 'None'}
        with open('bridge_digital/metrics.json', 'w') as f: json.dump(d_meta, f)
        
    else:
        if Path("scripts/run_digital_sic.py").exists():
            run_command(f"python scripts/run_digital_sic.py --backend {args.backend}", f"執行傳統數位消除 ({args.backend.upper()})")
        else:
             print("⚠️ 警告: 找不到 scripts/run_digital_sic.py，跳過數位部分。")

    # 4. 執行 RX 解碼
    rx_script = f"""
import os, sys, warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

import numpy as np, json
from PIL import Image
from src.semantic.ntscc_rx_wrapper import NTSCCRXWrapper
import logging
logging.getLogger().setLevel(logging.ERROR)

y = np.load('bridge_digital/y_clean.npy')
with open('bridge/meta_tx_remote.json') as f: tx = json.load(f)

gt_path = 'data/kodak/{args.remote}.png'
if os.path.exists(gt_path):
    img_gt = Image.open(gt_path).convert('RGB')
    img_gt = img_gt.resize((128, 128), Image.LANCZOS)
    img_gt = np.array(img_gt).astype(np.float32) / 255.0
else:
    img_gt = None

dec = NTSCCRXWrapper(ckpt_path="checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth", device='cuda')
img, m = dec.decode(y, original_img=img_gt, img_size=(128,128), cbr=1/16, meta_tx=tx)

from pathlib import Path
Path('bridge_rx').mkdir(exist_ok=True)
Image.fromarray((img*255).astype(np.uint8)).save('bridge_rx/img_recon_remote.png')
with open('bridge_rx/metrics_remote.json','w') as f: json.dump(m, f)
"""
    Path('_rx_tmp.py').write_text(rx_script)
    run_command("python _rx_tmp.py", "執行 NTSCC 語意解碼")
    Path('_rx_tmp.py').unlink()

    # 5. 最終報告
    print("\n" + "="*60)
    print("📊 最終物理實驗報告 (Final Report)")
    print("="*60)
    try:
        with open('bridge/meta.json') as f: am = json.load(f)
        with open('bridge_digital/metrics.json') as f: dm = json.load(f)
        with open('bridge_rx/metrics_remote.json') as f: rm = json.load(f)

        sinr_pre = am.get('SINR_pre', 0)
        sinr_ana = am.get('SINR_analog', 0)
        sinr_fin = dm.get('SINR_after_digital')
        
        gain_ana = sinr_ana - sinr_pre
        backend_name = dm.get('backend', 'None')
        
        if sinr_fin is not None:
            gain_dig = sinr_fin - sinr_ana
            total_gain = sinr_fin - sinr_pre
            sinr_fin_str = f"{sinr_fin:6.2f} dB"
            gain_dig_str = f"{gain_dig:+.2f} dB"
            total_gain_str = f"{total_gain:.2f} dB"
        else:
            sinr_fin_str = "(Diffusion 隱式去噪，無計算 SINR)"
            gain_dig_str = "N/A"
            total_gain_str = "N/A (直接觀察下方 PSNR)"
            
        print(f"【實驗配置】")
        print(f"  干擾強度: {am.get('rsi_scale', 0)}x (Scale)")
        print(f"  AUX 硬體: {run_aux_status} {'(理想硬體)' if run_aux_status else '(預補償真實硬體)'}")
        print(f"  後端去噪: {backend_name.upper()}")
        
        print(f"\n【SINR 物理演進】")
        print(f"  1. 原始輸入: {sinr_pre:6.2f} dB")
        print(f"  2. 類比消除: {sinr_ana:6.2f} dB  (Gain: {gain_ana:+.2f} dB) [Aux-TX]")
        print(f"  3. 數位消除: {sinr_fin_str}  [{backend_name}]")
        
        print(f"\n【最終性能】")
        print(f"  Total SIC: {total_gain_str}")
        
        psnr_val = rm.get('psnr', -1.0)
        if psnr_val < 0:
            print(f"  PSNR:      N/A (GT missing?)")
        else:
            print(f"  PSNR:      {psnr_val:.2f} dB  <-- 關鍵結果！")
        
        if psnr_val > 30.0:
            print(f"\n✅ 實驗成功！高畫質還原 (PSNR > 30dB)")
            
    except Exception as e:
        print(f"❌ 無法讀取報告: {e}")

if __name__ == '__main__':
    main()