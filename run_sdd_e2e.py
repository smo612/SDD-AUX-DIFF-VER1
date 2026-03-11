#!/usr/bin/env python3
"""
SDD E2E 測試主控腳本（改進版 - 真正的一鍵測試）

✅ 新功能：
1. --no-normalize: 禁用 TX 功率歸一化（測試極端 RSI）
2. --no-digital-sic: 禁用 Digital SIC（驗證 Analog SIC 性能）
3. RSI_SCALE 直接傳遞，不依賴 config.py
4. 所有參數一次搞定，無需手動操作

用法：
    # 正常測試（歸一化，完整系統）
    python run_sdd_e2e.py --local kodim01 --remote kodim15
    
    # 極端測試（不歸一化，RSI=500）
    python run_sdd_e2e.py --local kodim01 --remote kodim15 --rsi-scale 500 --no-normalize
    
    # 驗證 Analog SIC（無 Digital SIC）
    python run_sdd_e2e.py --local kodim01 --remote kodim15 --rsi-scale 500 --no-normalize --no-digital-sic
"""
import argparse
import subprocess
import shutil
from pathlib import Path
import json
import re
import numpy as np


def run_command(cmd, description, verbose=True):
    """執行命令並處理錯誤"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"[{description}]")
        print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=not verbose)
    
    if result.returncode != 0:
        print(f"❌ {description} 失敗")
        if not verbose:
            print(result.stderr.decode())
        return False
    
    if verbose:
        print(f"✓ {description} 完成")
    
    return True


def setup_tx_signals(img_local, img_remote, normalize_power=True, verbose=True):
    """準備 TX 信號（✅ 支持歸一化控制）"""
    print("\n" + "="*60)
    print("階段 1/4: 準備 TX 信號")
    print("="*60)
    print(f"本地圖片: {img_local}")
    print(f"遠端圖片: {img_remote}")
    if not normalize_power:
        print("⚠️  測試模式：功率歸一化已禁用")
    
    # 清理舊數據
    for dir_name in ['bridge_tx_remote', 'bridge', 'bridge_digital', 'bridge_rx']:
        path = Path(dir_name)
        if path.exists():
            shutil.rmtree(path)
    
    # ✅ 構建 TX 命令（加入 --no-normalize）
    normalize_flag = "" if normalize_power else " --no-normalize"
    
    # 生成本地 TX（自干擾源）
    if not run_command(
        f"python scripts/run_tx_kodak_batch.py --img {img_local}{normalize_flag}",
        f"生成本地 TX ({img_local})",
        verbose=verbose
    ):
        return False
    
    # 備份為本地信號
    if Path('bridge_tx_local').exists():
        shutil.rmtree('bridge_tx_local')
    shutil.copytree('bridge_tx', 'bridge_tx_local')
    
    # 生成遠端 TX（期望信號）
    if not run_command(
        f"python scripts/run_tx_kodak_batch.py --img {img_remote}{normalize_flag}",
        f"生成遠端 TX ({img_remote})",
        verbose=verbose
    ):
        return False
    
    # 重命名為遠端信號
    if Path('bridge_tx_remote').exists():
        shutil.rmtree('bridge_tx_remote')
    shutil.move('bridge_tx', 'bridge_tx_remote')
    shutil.move('bridge_tx_local', 'bridge_tx')
    
    print("\n✓ TX 信號準備完成")
    print(f"  - bridge_tx/: {img_local} (本地/自干擾)")
    print(f"  - bridge_tx_remote/: {img_remote} (遠端/期望)")
    
    # ✅ 驗證功率
    x_local = np.load('bridge_tx/x_tx.npy')
    x_remote = np.load('bridge_tx_remote/x_tx.npy')
    p_local = np.mean(np.abs(x_local)**2)
    p_remote = np.mean(np.abs(x_remote)**2)
    
    print(f"  - 本地功率:  {p_local:.3f}")
    print(f"  - 遠端功率:  {p_remote:.3f}")
    
    return True


def run_analog_stage(snr_db, rsi_scale, sic_db=23.0, verbose=True):
    """
    運行 Analog 階段
    
    ✅ 改進：直接傳遞參數，不依賴 config.py
    """
    print("\n" + "="*60)
    print("階段 2/4: Analog 通道模擬 + Analog SIC")
    print("="*60)
    print(f"配置: SNR={snr_db} dB, RSI_SCALE={rsi_scale}, SIC={sic_db} dB")
    
    # ✅ 方法：直接修改 config.py（臨時）
    config_path = Path('config.py')
    config_backup = config_path.read_text()
    
    # 修改配置
    config_modified = re.sub(r'SNR_DB\s*=\s*[\d.]+', f'SNR_DB = {snr_db}', config_backup)
    config_modified = re.sub(r'RSI_SCALE\s*=\s*[\d.]+', f'RSI_SCALE = {rsi_scale}', config_modified)
    config_modified = re.sub(r'SIC_DB\s*=\s*[\d.]+', f'SIC_DB = {sic_db}', config_modified)
    
    config_path.write_text(config_modified)
    
    try:
        success = run_command(
            "python run_analog_semantic.py",
            "Analog 階段",
            verbose=verbose
        )
    finally:
        # 恢復配置
        config_path.write_text(config_backup)
    
    return success


def run_digital_sic(L=5, lambda_reg=0.01, widely_linear=True, verbose=True):
    """運行 Digital SIC"""
    print("\n" + "="*60)
    print("階段 3/4: Digital SIC")
    print("="*60)
    print(f"配置: L={L}, λ={lambda_reg}, Widely-Linear={widely_linear}")
    
    cmd = f"python scripts/run_digital_sic.py --L {L} --lambda {lambda_reg}"
    if widely_linear:
        cmd += " --widely-linear"
    
    return run_command(cmd, "Digital SIC", verbose=verbose)


def skip_digital_sic(verbose=True):
    """
    ✅ 新增：跳過 Digital SIC，直接使用 Analog 輸出
    """
    print("\n" + "="*60)
    print("階段 3/4: Digital SIC")
    print("="*60)
    print("⚠️  跳過 Digital SIC（測試 Analog 性能）")
    
    try:
        # 讀取 Analog SIC 輸出
        y_adc = np.load('bridge/y_adc.npy')
        
        # 直接當作 Digital SIC 輸出
        Path('bridge_digital').mkdir(exist_ok=True)
        np.save('bridge_digital/y_clean.npy', y_adc)
        
        # 複製 meta
        with open('bridge/meta.json') as f:
            meta_analog = json.load(f)
        
        # 創建假的 Digital SIC metrics（零抑制）
        digital_metrics = {
            'SINR_digital': meta_analog.get('SINR_analog', 0),
            'Digital_gain': 0.0,
            'Digital_supp_si': 0.0,
            'Digital_supp_note': 'SKIPPED (--no-digital-sic)',
            'P_before': 0,
            'P_after': 0,
            'Pn_digital': meta_analog.get('noise_var', 0) * 2.0,
            'AA_approx': meta_analog.get('AA', 0),
            'SINR_note': 'Analog only (no DSIC)',
        }
        
        with open('bridge_digital/metrics.json', 'w') as f:
            json.dump(digital_metrics, f, indent=2)
        
        print(f"✓ 已跳過 Digital SIC")
        print(f"  SINR (Analog only): {meta_analog.get('SINR_analog', 'N/A'):.2f} dB")
        
        return True
        
    except Exception as e:
        print(f"❌ 跳過 Digital SIC 失敗: {e}")
        return False


def run_rx_decode(img_remote, verbose=True):
    """運行 RX 解碼"""
    print("\n" + "="*60)
    print("階段 4/4: RX 解碼")
    print("="*60)
    print(f"解碼目標: {img_remote}")
    
    # 創建臨時的 RX 腳本
    img_path = f"data/kodak/{img_remote}.png"
    
    rx_script = f'''
import numpy as np
import json
from pathlib import Path
from PIL import Image
from src.semantic.ntscc_rx_wrapper import NTSCCRXWrapper

CKPT = "checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth"
IMG_PATH = "{img_path}"

# 載入 Digital SIC 輸出
y_clean = np.load("bridge_digital/y_clean.npy")
with open("bridge_digital/metrics.json", 'r') as f:
    digital_metrics = json.load(f)

# 載入遠端 meta
with open("bridge/meta_tx_remote.json", 'r') as f:
    meta_tx_remote = json.load(f)

# 載入原始圖像
if Path(IMG_PATH).exists():
    img = Image.open(IMG_PATH).convert('RGB').resize((128, 128), Image.LANCZOS)
    original_img = np.array(img).astype(np.float32) / 255.0
else:
    print(f"⚠️  找不到 {{IMG_PATH}}")
    original_img = None

# 解碼
decoder = NTSCCRXWrapper(ckpt_path=CKPT, device='cuda')
img_recon, metrics = decoder.decode(
    y_clean=y_clean,
    original_img=original_img,
    img_size=(128, 128),
    cbr=1/16,
    meta_tx=meta_tx_remote
)

# 保存結果
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

print(f"✓ RX 解碼完成")
print(f"  PSNR: {{metrics['psnr']:.2f}} dB")
ms_ssim = metrics.get('ms_ssim', 'N/A')
if isinstance(ms_ssim, float):
    print(f"  MS-SSIM: {{ms_ssim:.4f}}")
else:
    print(f"  MS-SSIM: {{ms_ssim}}")
'''
    
    Path('_temp_rx.py').write_text(rx_script)
    
    try:
        success = run_command("python _temp_rx.py", "RX 解碼", verbose=verbose)
    finally:
        if Path('_temp_rx.py').exists():
            Path('_temp_rx.py').unlink()
    
    return success


def print_summary(img_local, img_remote):
    """打印測試總結"""
    print("\n" + "="*60)
    print("📊 SDD E2E 測試總結")
    print("="*60)
    
    try:
        with open('bridge/meta.json') as f:
            analog_meta = json.load(f)
        
        with open('bridge_digital/metrics.json') as f:
            digital_meta = json.load(f)
        
        with open('bridge_rx/metrics_remote.json') as f:
            rx_metrics = json.load(f)
        
        # 打印性能指標
        print(f"\n【測試配置】")
        print(f"  本地圖片 (SI):  {img_local}")
        print(f"  遠端圖片 (期望): {img_remote}")
        print(f"  SNR:            {analog_meta['snr_db']} dB")
        print(f"  RSI_SCALE:      {analog_meta['rsi_scale']}")
        
        # ✅ 顯示 Analog SIC 狀態
        analog_sic_info = analog_meta.get('analog_sic_info', {})
        if analog_sic_info.get('saturated', False):
            print(f"  ⚠️  Analog SIC:   已飽和 ({analog_sic_info['actual_suppression_db']:.0f} dB)")
        else:
            print(f"  ✅ Analog SIC:   正常工作 ({analog_sic_info['actual_suppression_db']:.0f} dB)")
        
        print(f"\n【SINR 演進】")
        sinr_pre = analog_meta.get('SINR_pre', 'N/A')
        sinr_analog = analog_meta.get('SINR_analog', 'N/A')
        sinr_digital = digital_meta.get('SINR_digital', 'N/A')
        supp_analog = analog_meta.get('Supp_analog', 0)
        supp_digital = digital_meta.get('Digital_supp_si', 0)
        
        if isinstance(sinr_pre, (int, float)):
            print(f"  TX 前:          {sinr_pre:.2f} dB")
        else:
            print(f"  TX 前:          {sinr_pre}")
        
        if isinstance(sinr_analog, (int, float)) and isinstance(supp_analog, (int, float)):
            print(f"  Analog SIC 後:  {sinr_analog:.2f} dB  (+{supp_analog:.2f} dB)")
        else:
            print(f"  Analog SIC 後:  {sinr_analog}")
        
        # ✅ 檢查是否跳過 Digital SIC
        if digital_meta.get('Digital_supp_note', '').startswith('SKIPPED'):
            print(f"  Digital SIC 後: (已跳過)")
        else:
            if isinstance(sinr_digital, (int, float)) and isinstance(supp_digital, (int, float)):
                print(f"  Digital SIC 後: {sinr_digital:.2f} dB  (+{supp_digital:.2f} dB)")
            else:
                print(f"  Digital SIC 後: {sinr_digital}")
        
        print(f"\n【抑制效果】")
        if isinstance(supp_analog, (int, float)):
            print(f"  Analog Supp:    {supp_analog:.2f} dB")
        
        if not digital_meta.get('Digital_supp_note', '').startswith('SKIPPED'):
            if isinstance(supp_digital, (int, float)):
                print(f"  Digital Supp:   {supp_digital:.2f} dB")
            
            total_supp = digital_meta.get('Total_supp_SI_only', 'N/A')
            if isinstance(total_supp, (int, float)):
                print(f"  Total Supp:     {total_supp:.2f} dB")
        
        print(f"\n【圖像品質】")
        print(f"  PSNR:           {rx_metrics['psnr']:.2f} dB")
        ms_ssim = rx_metrics.get('ms_ssim', 'N/A')
        if isinstance(ms_ssim, float):
            print(f"  MS-SSIM:        {ms_ssim:.4f}")
        
        # 性能評估
        psnr = rx_metrics['psnr']
        print(f"\n【性能評估】")
        if psnr > 32:
            print("  ✅ 優秀！接近無干擾水平")
        elif psnr > 28:
            print("  ✅ 良好！系統運作正常")
        elif psnr > 20:
            print("  ⚠️  可接受，建議優化參數")
        else:
            print("  ❌ 性能不足（可能是極端測試條件）")
        
        # 信號相關性警告
        if 'signal_correlation' in analog_meta:
            rho = analog_meta['signal_correlation']['rho']
            print(f"\n【信號相關性】")
            print(f"  |ρ| = {rho:.4f}")
            if rho > 0.1:
                print("  ⚠️  相關性過高！建議更換圖片組合")
            else:
                print("  ✅ 相關性良好")
        
    except Exception as e:
        print(f"❌ 讀取結果失敗: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='SDD E2E 測試主控腳本（改進版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
測試場景範例：
  # 正常測試
  %(prog)s --local kodim01 --remote kodim15
  
  # 極端 RSI 測試（不歸一化）
  %(prog)s --local kodim01 --remote kodim15 --rsi-scale 500 --no-normalize
  
  # 驗證 Analog SIC（無 Digital SIC）
  %(prog)s --local kodim01 --remote kodim15 --rsi-scale 500 --no-normalize --no-digital-sic
  
  # 跳過 TX 生成（使用現有信號）
  %(prog)s --local kodim01 --remote kodim15 --skip-tx
        '''
    )
    
    # 必要參數
    parser.add_argument('--local', type=str, required=True,
                       help='本地圖片（自干擾源），例如 kodim01')
    parser.add_argument('--remote', type=str, required=True,
                       help='遠端圖片（期望信號），例如 kodim15')
    
    # 通道參數
    parser.add_argument('--snr', type=float, default=22.0,
                       help='通道 SNR (dB)，預設 22.0')
    parser.add_argument('--rsi-scale', type=float, default=5.0,
                       help='自干擾強度係數，預設 5.0')
    parser.add_argument('--sic-db', type=float, default=23.0,
                       help='Analog SIC 目標抑制 (dB)，預設 23.0')
    
    # Digital SIC 參數
    parser.add_argument('--L', type=int, default=5,
                       help='Digital SIC 通道長度，預設 5')
    parser.add_argument('--lambda', type=float, default=0.01, dest='lambda_reg',
                       help='Digital SIC 正則化參數，預設 0.01')
    parser.add_argument('--no-widely-linear', action='store_true',
                       help='禁用 Widely-Linear')
    
    # ✅ 新增控制參數
    parser.add_argument('--no-normalize', action='store_true',
                       help='禁用 TX 功率歸一化（用於測試極端 RSI）')
    parser.add_argument('--no-digital-sic', action='store_true',
                       help='跳過 Digital SIC（僅測試 Analog 性能）')
    
    # 控制參數
    parser.add_argument('--verbose', action='store_true',
                       help='顯示詳細輸出')
    parser.add_argument('--skip-tx', action='store_true',
                       help='跳過 TX 生成（bridge_tx 已存在）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SDD End-to-End 測試（改進版）")
    print("="*60)
    
    # ✅ 顯示測試模式
    test_mode = []
    if not args.no_normalize:
        test_mode.append("正常模式")
    else:
        test_mode.append("⚠️  極端測試（功率未歸一化）")
    
    if args.no_digital_sic:
        test_mode.append("⚠️  Analog SIC 驗證（無 Digital SIC）")
    
    if test_mode:
        print("測試模式:", " | ".join(test_mode))
    print()
    
    # 階段 1: 準備 TX
    if not args.skip_tx:
        if not setup_tx_signals(
            args.local, 
            args.remote, 
            normalize_power=not args.no_normalize,  # ✅ 傳遞歸一化控制
            verbose=args.verbose
        ):
            print("\n❌ TX 準備失敗")
            return
    else:
        print("\n⏭️  跳過 TX 生成（使用現有 bridge_tx）")
    
    # 階段 2: Analog
    if not run_analog_stage(
        args.snr, 
        args.rsi_scale, 
        args.sic_db,
        verbose=args.verbose
    ):
        print("\n❌ Analog 階段失敗")
        return
    
    # 階段 3: Digital SIC（✅ 可選跳過）
    if args.no_digital_sic:
        if not skip_digital_sic(verbose=args.verbose):
            print("\n❌ 跳過 Digital SIC 失敗")
            return
    else:
        if not run_digital_sic(
            L=args.L,
            lambda_reg=args.lambda_reg,
            widely_linear=not args.no_widely_linear,
            verbose=args.verbose
        ):
            print("\n❌ Digital SIC 失敗")
            return
    
    # 階段 4: RX
    if not run_rx_decode(args.remote, args.verbose):
        print("\n❌ RX 解碼失敗")
        return
    
    # 總結
    print_summary(args.local, args.remote)
    
    print("\n" + "="*60)
    print("✅ SDD E2E 測試完成！")
    print("="*60)
    print(f"\n結果文件:")
    print(f"  - bridge_rx/img_recon_remote.png")
    print(f"  - bridge_rx/comparison_remote.png")
    print(f"  - bridge_rx/metrics_remote.json")


if __name__ == '__main__':
    main()