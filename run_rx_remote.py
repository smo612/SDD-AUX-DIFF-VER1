"""
run_rx_remote.py - RX解碼（遠端語意版）

修改自 run_rx.py
主要變更：使用遠端的meta來解碼
"""
import numpy as np
import json
from pathlib import Path
from PIL import Image

from src.semantic.ntscc_rx_wrapper import NTSCCRXWrapper

def main():
    print("="*60)
    print("RX 解碼 - 遠端語意")
    print("="*60)
    
    CKPT = "checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth"
    IMG_PATH = "data/kodak/kodim15.png"  # 遠端圖像（用於PSNR）
    
    # ========================================
    # 1. 載入 Digital SIC 輸出
    # ========================================
    print("\n[1/4] 載入 Digital SIC 輸出...")
    
    y_clean = np.load("bridge_digital/y_clean.npy")
    with open("bridge_digital/metrics.json", 'r') as f:
        digital_metrics = json.load(f)
    
    print(f"  ✓ y_clean: {y_clean.shape}")
    
    # 安全地打印SINR（可能是None或字符串）
    sinr_dig = digital_metrics.get('SINR_digital', None)
    if isinstance(sinr_dig, (int, float)):
        print(f"  ✓ SINR_digital: {sinr_dig:.2f} dB")
    else:
        print(f"  ✓ SINR_digital: {sinr_dig}")
    
    # ========================================
    # 2. 載入遠端meta（關鍵！）
    # ========================================
    print("\n[2/4] 載入遠端TX元數據...")
    
    meta_remote_path = Path("bridge/meta_tx_remote.json")
    if not meta_remote_path.exists():
        print(f"❌ 找不到 {meta_remote_path}")
        print("請確認已執行：python run_analog_semantic.py")
        return
    
    with open(meta_remote_path, 'r') as f:
        meta_tx_remote = json.load(f)
    
    print(f"  ✓ meta_tx_remote.json")
    print(f"    tx_scale: {meta_tx_remote['signal_info']['tx_scale']:.6f}")
    print(f"    n_data_symbols: {meta_tx_remote['signal_info']['n_data_symbols']}")
    
    # ========================================
    # 3. 載入原始圖像
    # ========================================
    print("\n[3/4] 載入遠端原始圖像...")
    
    if Path(IMG_PATH).exists():
        img = Image.open(IMG_PATH).convert('RGB').resize((128, 128), Image.LANCZOS)
        original_img = np.array(img).astype(np.float32) / 255.0
        print(f"  ✓ {IMG_PATH}")
    else:
        print(f"  ⚠️  找不到 {IMG_PATH}")
        original_img = None
    
    # ========================================
    # 4. 解碼
    # ========================================
    print("\n[4/4] 解碼...")
    
    decoder = NTSCCRXWrapper(ckpt_path=CKPT, device='cuda')
    
    img_recon, metrics = decoder.decode(
        y_clean=y_clean,
        original_img=original_img,
        img_size=(128, 128),
        cbr=1/16,
        meta_tx=meta_tx_remote  # ✅ 使用遠端的meta！
    )
    
    # ========================================
    # 保存
    # ========================================
    print("\n[保存] RX 輸出...")
    
    output_dir = Path("bridge_rx")
    output_dir.mkdir(exist_ok=True)
    
    img_recon_uint8 = (img_recon * 255).astype(np.uint8)
    Image.fromarray(img_recon_uint8).save(output_dir / "img_recon_remote.png")
    
    if original_img is not None:
        comparison = np.hstack([original_img, img_recon])
        comparison_uint8 = (comparison * 255).astype(np.uint8)
        Image.fromarray(comparison_uint8).save(output_dir / "comparison_remote.png")
    
    with open(output_dir / "metrics_remote.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  ✓ {output_dir}/img_recon_remote.png")
    print(f"  ✓ {output_dir}/comparison_remote.png")
    print(f"  ✓ {output_dir}/metrics_remote.json")
    
    # ========================================
    # E2E結果
    # ========================================
    print("\n" + "="*60)
    print("E2E 測試結果（雙端語意）")
    print("="*60)
    
    # 讀取analog metrics
    with open("bridge/meta.json", 'r') as f:
        analog_meta = json.load(f)
    
    # 安全地打印SINR值
    def safe_print_db(value, name):
        if isinstance(value, (int, float)):
            return f"{value:.2f} dB"
        else:
            return str(value)
    
    print(f"SINR 演進:")
    print(f"  TX前: {safe_print_db(analog_meta.get('SINR_pre', 'N/A'), 'SINR_pre')}")
    print(f"  Analog SIC後: {safe_print_db(analog_meta.get('SINR_analog', 'N/A'), 'SINR_analog')}")
    print(f"  Digital SIC後: {safe_print_db(digital_metrics.get('SINR_digital', 'N/A'), 'SINR_digital')}")
    
    print(f"\n圖像品質:")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  MS-SSIM: {metrics.get('ms_ssim', 'N/A'):.4f}")
    
    print(f"\n判讀:")
    if metrics['psnr'] > 30:
        print("  ✅ 優秀！接近AWGN測試水平")
    elif metrics['psnr'] > 25:
        print("  ✅ 良好！雙端語意系統運作正常")
    elif metrics['psnr'] > 20:
        print("  ⚠️  可接受，但可能需要優化：")
        print("     • 開啟Widely-Linear (run_digital_sic.py)")
        print("     • 調整L, λ參數")
    else:
        print("  ❌ 低於預期，建議檢查：")
        print("     • Digital SIC是否正常運作")
        print("     • 通道等化是否需要")
    
    print("\n對比:")
    print(f"  自環路PSNR: 35.93 dB (無通道)")
    print(f"  AWGN PSNR:  35.31 dB (SNR=20dB)")
    print(f"  E2E PSNR:   {metrics['psnr']:.2f} dB (RSI+SIC)")
    print("="*60)

if __name__ == "__main__":
    main()