#!/usr/bin/env python3
# sweep.py - SDD 物理層極限壓力測試 (支援 AI Diffusion 模式與安全存檔防呆)
# 用法：
#   python sweep.py --mode ai_comp

import argparse
import json
import time
import shutil
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# ✅ 測試環境參數設定
# =========================
LOCAL_IMG = "kodim01"      # 干擾源 (Local TX)
REMOTE_IMG = "kodim24"     # 期望訊號 (Remote TX)

RSI_SCALE_LIST = [10000, 50000, 100000, 250000, 1000000, 3150000,10000000]
USE_NO_NORMALIZE = True    # 極端測試時關閉歸一化


def get_cmd_args(run_tag):
    if run_tag == 'ideal_analog': return ['--no-digital-sic', '--aux-disable-iqpa', 'True']
    elif run_tag == 'precomp_analog': return ['--no-digital-sic', '--aux-disable-iqpa', 'False']
    elif run_tag == 'ideal_digital': return ['--backend', 'mp', '--aux-disable-iqpa', 'True']
    elif run_tag == 'precomp_digital': return ['--backend', 'mp', '--aux-disable-iqpa', 'False']
    elif run_tag == 'ideal_diffusion': return ['--use-diffusion', '--aux-disable-iqpa', 'True']
    elif run_tag == 'precomp_diffusion': return ['--use-diffusion', '--aux-disable-iqpa', 'False']
    else: raise ValueError(f"Unknown run_tag: {run_tag}")

def safe_float(v):
    """安全轉換為 float，防止 NoneType 造成格式化崩潰"""
    if v is None or isinstance(v, str):
        return np.nan
    try:
        return float(v)
    except:
        return np.nan

def run_one(rsi_scale: float, run_tag: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python", "run_sdd_final.py",
        "--local", LOCAL_IMG,
        "--remote", REMOTE_IMG,
        "--rsi-scale", str(rsi_scale)
    ] + get_cmd_args(run_tag)
    
    if USE_NO_NORMALIZE:
        cmd.append("--no-normalize")

    # 執行並捕獲輸出
    with open(out_dir / "run_stdout.txt", "w") as f_out:
        t0 = time.time()
        res = subprocess.run(cmd, stdout=f_out, stderr=subprocess.STDOUT, text=True)
        t1 = time.time()

    if res.returncode != 0:
        print(f"  [ERROR] Command failed. See {out_dir / 'run_stdout.txt'}")

    row = {
        "rsi_scale": rsi_scale,
        "run_tag": run_tag,
        "time_sec": t1 - t0,
        "sinr_pre": np.nan,
        "sinr_after_analog": np.nan,
        "sinr_after_digital": np.nan,
        "analog_supp_db": np.nan,
        "digital_supp_db": np.nan,
        "psnr": np.nan,
    }

    try:
        with open("bridge/meta.json", "r") as f:
            am = json.load(f)
            row["sinr_pre"] = safe_float(am.get("SINR_pre"))
            row["sinr_after_analog"] = safe_float(am.get("SINR_analog"))
    except: pass

    try:
        with open("bridge_digital/metrics.json", "r") as f:
            dm = json.load(f)
            row["sinr_after_digital"] = safe_float(dm.get("SINR_after_digital"))
    except: pass

    try:
        with open("bridge_rx/metrics_remote.json", "r") as f:
            rm = json.load(f)
            row["psnr"] = safe_float(rm.get("psnr"))
    except: pass

    # 備份檔案
    if Path("bridge/meta.json").exists(): shutil.copy("bridge/meta.json", out_dir / "meta_analog.json")
    if Path("bridge_digital/metrics.json").exists(): shutil.copy("bridge_digital/metrics.json", out_dir / "metrics_digital.json")
    if Path("bridge_rx/metrics_remote.json").exists(): shutil.copy("bridge_rx/metrics_remote.json", out_dir / "metrics_rx.json")
    if Path("bridge_rx/img_recon_remote.png").exists(): shutil.copy("bridge_rx/img_recon_remote.png", out_dir / "img_recon_remote.png")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(row, f, indent=2)

    return row

def load_existing_row(out_dir: Path):
    """如果已經跑過，直接讀取 summary.json 避免重跑"""
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                return json.load(f)
        except:
            return None
    return None

def main():
    parser = argparse.ArgumentParser(description="SDD 壓力測試腳本")
    parser.add_argument('--mode', type=str, required=True, choices=['tf', 'ideal_comp', 'precomp_comp', 'ai_comp', 'all'])
    args = parser.parse_args()

    if args.mode == 'ai_comp':
        tags = ['precomp_analog', 'precomp_digital', 'precomp_diffusion']
        labels = {'precomp_analog': 'Analog Only (Aux-TX)', 'precomp_digital': 'Analog SIC + MP', 'precomp_diffusion': 'Analog SIC +  Diffusion'}
        colors = {'precomp_analog': 'gray', 'precomp_digital': 'orange', 'precomp_diffusion': 'blue'}
        markers = {'precomp_analog': 'v', 'precomp_digital': 's', 'precomp_diffusion': '*'}
        title_suffix = "Ablation Study: Mathematical MP vs AI-SIC"
    else:
        print("目前僅展示 ai_comp 模式，若需其他模式請參考原版設定。")
        return

    out_root = Path(f"results_sweep_{args.mode}")
    out_root.mkdir(exist_ok=True)

    print("=" * 80)
    print(f"🚀 SDD STRESS TEST START (MODE: {args.mode.upper()})")
    print("=" * 80)

    rows = []

    for s in RSI_SCALE_LIST:
        for t in tags:
            run_name = f"rsi{s:g}_{t}"
            out_dir = out_root / run_name
            
            # 🌟 智慧判斷：如果剛剛已經成功跑完並存檔了，就直接讀取，不重跑！
            row = load_existing_row(out_dir)
            if row is not None:
                print(f"--- [SKIPPED] {run_name} already exists, loaded from cache ---")
            else:
                print(f"\n--- Running {run_name} ---")
                row = run_one(s, t, out_dir)
                
            rows.append(row)

    # =========================
    # 安全儲存 CSV (防呆)
    # =========================
    csv_path = out_root / f"sweep_results_{args.mode}.csv"
    with open(csv_path, "w") as f:
        f.write("rsi_scale,run_tag,time_sec,sinr_pre,sinr_after_analog,sinr_after_digital,psnr\n")
        for r in rows:
            # 強制轉換為安全浮點數
            s_pre = safe_float(r.get('sinr_pre'))
            s_ana = safe_float(r.get('sinr_after_analog'))
            s_dig = safe_float(r.get('sinr_after_digital'))
            psnr_v = safe_float(r.get('psnr'))
            time_v = safe_float(r.get('time_sec', 0))
            
            f.write(
                f"{r['rsi_scale']},{r['run_tag']},{time_v:.2f},"
                f"{s_pre:.4f},{s_ana:.4f},{s_dig:.4f},{psnr_v:.4f}\n"
            )

    # =========================
    # 畫圖 (Final SINR / PSNR vs RSI)
    # =========================
    print("\nGenerating Plots...")

    # 1. Plot Final Clean SINR
    plt.figure(figsize=(10, 7))
    for t in tags:
        if 'diffusion' in t: continue 
        x_vals = [r["rsi_scale"] for r in rows if r["run_tag"] == t]
        y_vals = [safe_float(r.get("sinr_after_digital")) if not np.isnan(safe_float(r.get("sinr_after_digital"))) else safe_float(r.get("sinr_after_analog")) for r in rows if r["run_tag"] == t]
        plt.plot(x_vals, y_vals, marker=markers[t], color=colors[t], label=labels[t], linewidth=2.5, markersize=9)
    
    plt.xscale('log')
    plt.xlabel("RSI Scale (Interference Strength)", fontsize=12)
    plt.ylabel("Final Cleaned SINR before RX (dB)", fontsize=12)
    plt.title(f"Final SINR Capability vs Extreme Interference\n{title_suffix}", fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_root / f"plot_sinr_{args.mode}.png", dpi=300)
    plt.close()

    # 2. Plot PSNR
    plt.figure(figsize=(10, 7))
    for t in tags:
        x_vals = [r["rsi_scale"] for r in rows if r["run_tag"] == t]
        y_vals = [safe_float(r.get("psnr")) for r in rows if r["run_tag"] == t]
        lw = 3.5 if 'diffusion' in t else 2.5
        ms = 12 if 'diffusion' in t else 9
        plt.plot(x_vals, y_vals, marker=markers[t], color=colors[t], label=labels[t], linewidth=lw, markersize=ms)

    plt.xscale('log')
    plt.axhline(y=30.0, color='green', linestyle=':', label='High Quality Threshold (30dB)', linewidth=2)
    plt.xlabel("RSI Scale (Interference Strength)", fontsize=12)
    plt.ylabel("Image PSNR (dB)", fontsize=12)
    plt.title(f"End-to-End Image Quality vs Extreme Interference\n{title_suffix}", fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_root / f"plot_psnr_{args.mode}.png", dpi=300)
    plt.close()

    print(f"\n[OK] Saved successfully to '{out_root}' directory:")
    print(f"  - sweep_results_{args.mode}.csv")
    print(f"  - plot_sinr_{args.mode}.png")
    print(f"  - plot_psnr_{args.mode}.png")
    print("SWEEP DONE")

if __name__ == "__main__":
    main()