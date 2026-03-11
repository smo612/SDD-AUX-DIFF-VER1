"""
run_tx_kodak_batch.py - 批量處理 Kodak 圖片（添加歸一化控制）

✅ 新增：--no-normalize 選項來測試 RSI_SCALE 真實影響

用法：
    # 正常模式（歸一化）
    python scripts/run_tx_kodak_batch.py --img kodim01
    
    # 測試模式（不歸一化）
    python scripts/run_tx_kodak_batch.py --img kodim01 --no-normalize
    
    # 批量處理
    python scripts/run_tx_kodak_batch.py --all
"""
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.tx_semantic import SemanticTX

# NTSCC Checkpoint 路徑（全域設定）
NTSCC_CKPT = "checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth"


def load_kodak_image(img_name: str, kodak_dir: str = 'data/kodak', target_size=(128, 128)):
    """
    載入並前處理 Kodak 影像
    
    Args:
        img_name: 影像名稱（例如 'kodim01' 或 'kodim01.png'）
        kodak_dir: Kodak 資料夾路徑
        target_size: 目標尺寸
    
    Returns:
        img: [H, W, 3], 0~1 範圍, float32
        img_path: 完整路徑
    """
    # 處理檔名
    if not img_name.endswith('.png'):
        img_name = f"{img_name}.png"
    
    img_path = Path(kodak_dir) / img_name
    
    if not img_path.exists():
        raise FileNotFoundError(f"找不到影像: {img_path}")
    
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    
    print(f"[Load Image] {img_path}")
    print(f"  Shape: {img_np.shape}, Range: [{img_np.min():.3f}, {img_np.max():.3f}]")
    
    return img_np, str(img_path)


def process_single_image(img_name: str, 
                         ckpt_path: str, 
                         output_dir: str = 'bridge_tx', 
                         cbr: float = 1/16,
                         normalize_power: bool = True):  # ✅ 新增參數
    """處理單張影像"""
    print("=" * 60)
    print(f"處理影像: {img_name}")
    if not normalize_power:
        print("⚠️  測試模式：功率歸一化已禁用")
    print("=" * 60)
    
    # 1. 載入影像
    img, img_path = load_kodak_image(img_name)
    
    # 2. 建立 TX（✅ 使用 normalize_power 參數）
    print("\n建立 TX 管線（Real NTSCC Encoder）...")
    tx = SemanticTX(
        ntscc_ckpt=ckpt_path,
        use_pilot=True,
        pilot_period=64,
        normalize_power=normalize_power  # ✅ 傳入參數
    )
    
    # 3. 執行 TX
    print("\n執行 TX 管線...")
    result = tx.transmit(img, cbr=cbr, sps=1)
    
    # 4. 補充 source_image 到 meta
    result['meta']['source_info'] = {
        'source_image': img_path,
        'image_name': img_name,
        'dataset': 'Kodak',
    }
    
    # 5. 儲存
    tx.save_bridge(
        x_tx=result['x_tx'],
        meta=result['meta'],
        output_dir=output_dir
    )
    
    # 6. 顯示統計
    print(f"\n✓ {img_name} 處理完成！")
    print(f"  輸出: {output_dir}/")
    print(f"    - x_tx.npy:    {len(result['x_tx'])} 個符號")
    print(f"    - meta_tx.json")
    
    # ✅ 顯示功率資訊
    signal_info = result['meta']['signal_info']
    print(f"  功率資訊:")
    print(f"    - 原始功率: {signal_info['original_power']:.6f}")
    print(f"    - 最終功率: {signal_info['mean_power']:.6f}")
    print(f"    - TX Scale:  {signal_info['tx_scale']:.6f}")
    print(f"    - 歸一化:   {result['meta']['tx_info']['normalize_power']}")
    
    print(f"  NTSCC: {result['meta']['tx_info']['ntscc_mode']}")
    
    return result


def process_all_kodak(ckpt_path: str, 
                     output_base: str = 'bridge_tx_kodak', 
                     cbr: float = 1/16,
                     normalize_power: bool = True):  # ✅ 新增參數
    """處理所有 Kodak 圖片（1-24）"""
    results = {}
    
    for i in range(1, 25):
        img_name = f"kodim{i:02d}"
        output_dir = f"{output_base}/{img_name}"
        
        try:
            result = process_single_image(
                img_name, 
                ckpt_path, 
                output_dir=output_dir, 
                cbr=cbr,
                normalize_power=normalize_power  # ✅ 傳入參數
            )
            results[img_name] = {
                'status': 'success',
                'n_symbols': len(result['x_tx']),
                'power': float(np.mean(np.abs(result['x_tx'])**2)),
                'normalized': normalize_power
            }
            print("")  # 空行分隔
        except Exception as e:
            print(f"\n✗ {img_name} 處理失敗: {e}")
            results[img_name] = {'status': 'failed', 'error': str(e)}
    
    # 儲存統計
    summary_path = f"{output_base}/summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("批量處理完成！")
    print(f"統計資料: {summary_path}")
    print("=" * 60)
    
    # 顯示成功/失敗統計
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"  成功: {success_count} / {len(results)}")


def main():
    parser = argparse.ArgumentParser(description='Kodak 圖片批量 TX 處理（Real NTSCC）')
    parser.add_argument('--img', type=str, help='指定單張圖片（例如 kodim01）')
    parser.add_argument('--all', action='store_true', help='處理全部 24 張')
    parser.add_argument('--cbr', type=float, default=1/16, help='CBR 參數（預設 1/16）')
    parser.add_argument('--output', type=str, default='bridge_tx', help='輸出資料夾')
    parser.add_argument('--ckpt', type=str, default=NTSCC_CKPT, 
                        help=f'NTSCC checkpoint 路徑（預設: {NTSCC_CKPT}）')
    
    # ✅ 新增歸一化控制選項
    parser.add_argument('--no-normalize', action='store_true',
                        help='禁用功率歸一化（用於測試 RSI_SCALE 真實影響）')
    
    args = parser.parse_args()
    
    # 使用命令列指定的 checkpoint 路徑
    ckpt_path = args.ckpt
    
    # ✅ 計算 normalize_power（預設 True，除非指定 --no-normalize）
    normalize_power = not args.no_normalize
    
    # 檢查 checkpoint 是否存在
    if not Path(ckpt_path).exists():
        print(f"✗ Checkpoint 不存在: {ckpt_path}")
        print(f"\n請確認路徑，或使用 --ckpt 指定其他位置")
        sys.exit(1)
    
    # ✅ 顯示歸一化設定
    print("=" * 60)
    print("TX 配置")
    print("=" * 60)
    print(f"  Checkpoint:  {ckpt_path}")
    print(f"  CBR:         {args.cbr}")
    print(f"  功率歸一化:  {'啟用' if normalize_power else '禁用 (測試模式)'}")
    print("=" * 60)
    print()
    
    if args.all:
        process_all_kodak(
            ckpt_path, 
            output_base='bridge_tx_kodak', 
            cbr=args.cbr,
            normalize_power=normalize_power
        )
    elif args.img:
        process_single_image(
            args.img, 
            ckpt_path, 
            output_dir=args.output, 
            cbr=args.cbr,
            normalize_power=normalize_power
        )
    else:
        # 預設處理 kodim01
        process_single_image(
            'kodim01', 
            ckpt_path, 
            output_dir=args.output, 
            cbr=args.cbr,
            normalize_power=normalize_power
        )


if __name__ == "__main__":
    main()