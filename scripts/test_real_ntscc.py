"""
test_real_ntscc.py - 測試真實 NTSCC encoder
包含 3 個測試：
1. 分析 checkpoint 結構
2. 比較 stub vs real 輸出
3. 完整 TX 管線測試
"""
import numpy as np
import torch
from pathlib import Path
import argparse
import sys

# 假設從專案根目錄執行
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.semantic.ntscc_wrapper import NTSCCWrapper


def test1_analyze_checkpoint(ckpt_path: str):
    """測試 1: 分析 checkpoint 結構"""
    print("\n" + "="*60)
    print("測試 1: 分析 Checkpoint 結構")
    print("="*60)
    
    if not Path(ckpt_path).exists():
        print(f"  ✗ Checkpoint 不存在: {ckpt_path}")
        return False
    
    print(f"  載入 checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    print(f"\n  Top-level keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"    - {key}: dict with {len(checkpoint[key])} items")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"    - {key}: tensor {checkpoint[key].shape}")
        else:
            print(f"    - {key}: {type(checkpoint[key])}")
    
    # 找到 state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"\n  State dict 分析:")
    print(f"    總共 {len(state_dict)} 個權重")
    
    # 統計不同模組的權重數量
    modules = {}
    for key in state_dict.keys():
        module_name = key.split('.')[0]
        if module_name not in modules:
            modules[module_name] = 0
        modules[module_name] += 1
    
    print(f"\n  模組統計:")
    for module, count in sorted(modules.items()):
        print(f"    - {module}: {count} 個權重")
    
    # 檢查是否有 ga (encoder)
    ga_keys = [k for k in state_dict.keys() if k.startswith('ga.')]
    print(f"\n  Encoder (ga) 權重:")
    print(f"    找到 {len(ga_keys)} 個 'ga.*' 權重")
    
    if len(ga_keys) > 0:
        print(f"    前 5 個 key:")
        for i, key in enumerate(ga_keys[:5]):
            tensor = state_dict[key]
            print(f"      {i+1}. {key}: {tensor.shape}")
        print(f"    ...")
        print(f"    後 3 個 key:")
        for i, key in enumerate(ga_keys[-3:]):
            tensor = state_dict[key]
            print(f"      {len(ga_keys)-2+i}. {key}: {tensor.shape}")
        
        print(f"\n  ✓ Checkpoint 包含 encoder 權重！")
        return True
    else:
        print(f"\n  ✗ Checkpoint 不包含 'ga.*' 權重")
        print(f"  可能的原因：")
        print(f"    1. Checkpoint 格式不同")
        print(f"    2. 命名慣例不同（試試 'analysis_transform' 或 'encoder'）")
        return False


def test2_stub_vs_real(ckpt_path: str):
    """測試 2: 比較 stub 和 real encoder 的輸出"""
    print("\n" + "="*60)
    print("測試 2: Stub vs Real Encoder")
    print("="*60)
    
    # 測試影像
    img = np.random.rand(128, 128, 3).astype(np.float32)
    
    # === Stub Encoder ===
    print("\n  [2.1] Stub Encoder")
    wrapper_stub = NTSCCWrapper(mode='stub')
    latent_stub, ctx_stub = wrapper_stub.encode(img, cbr=1/16)
    
    print(f"    Latent shape: {latent_stub.shape}")
    print(f"    Latent range: [{latent_stub.min():.3f}, {latent_stub.max():.3f}]")
    print(f"    Latent mean:  {latent_stub.mean():.3f}")
    print(f"    Latent std:   {latent_stub.std():.3f}")
    
    # === Real Encoder ===
    print("\n  [2.2] Real Encoder")
    try:
        wrapper_real = NTSCCWrapper(mode='real', ckpt_path=ckpt_path)
        latent_real, ctx_real = wrapper_real.encode(img, cbr=1/16)
        
        print(f"    Latent shape: {latent_real.shape}")
        print(f"    Latent range: [{latent_real.min():.3f}, {latent_real.max():.3f}]")
        print(f"    Latent mean:  {latent_real.mean():.3f}")
        print(f"    Latent std:   {latent_real.std():.3f}")
        
        # === 比較 ===
        print("\n  [2.3] 比較")
        print(f"    Shape 相同: {latent_stub.shape == latent_real.shape}")
        
        if latent_stub.shape == latent_real.shape:
            diff = (latent_stub - latent_real).abs().mean().item()
            print(f"    平均差異:   {diff:.6f}")
            
            if diff > 0.01:
                print(f"\n  ✓ 真實 encoder 載入成功！")
                print(f"    差異顯著，說明不是 stub（預期行為）")
                return True
            else:
                print(f"\n  ✗ 警告：差異太小，可能仍在使用 stub")
                return False
        else:
            print(f"\n  ✗ Shape 不匹配")
            return False
    
    except Exception as e:
        print(f"\n  ✗ 載入失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test3_tx_pipeline(ckpt_path: str):
    """測試 3: 完整 TX 管線（使用真實 encoder）"""
    print("\n" + "="*60)
    print("測試 3: 完整 TX 管線（Real NTSCC）")
    print("="*60)
    
    # 載入真實影像（如果有的話）
    img_path = "data/kodak/kodim01.png"
    
    if not Path(img_path).exists():
        print(f"  找不到測試影像: {img_path}")
        print(f"  使用隨機影像代替")
        img = np.random.rand(128, 128, 3).astype(np.float32)
    else:
        from PIL import Image
        img_pil = Image.open(img_path).convert('RGB').resize((128, 128))
        img = np.array(img_pil).astype(np.float32) / 255.0
        print(f"  載入影像: {img_path}")
    
    # 建立 wrapper（真實模式）
    print(f"\n  建立 NTSCCWrapper (mode='real')...")
    wrapper = NTSCCWrapper(mode='real', ckpt_path=ckpt_path)
    
    # 執行編碼
    print(f"  執行編碼...")
    latent, context = wrapper.encode(img, cbr=1/16)
    
    print(f"\n  結果:")
    print(f"    Latent shape: {latent.shape}")
    print(f"    Latent dtype: {latent.dtype}")
    print(f"    Latent device: {latent.device}")
    print(f"    Latent stats:")
    print(f"      - min:  {latent.min():.4f}")
    print(f"      - max:  {latent.max():.4f}")
    print(f"      - mean: {latent.mean():.4f}")
    print(f"      - std:  {latent.std():.4f}")
    
    print(f"\n  Context:")
    for key, value in context.items():
        print(f"    - {key}: {value}")
    
    # 檢查是否真的是 real 模式
    if context['mode'] == 'real':
        print(f"\n  ✓ 成功！TX 正在使用真實 NTSCC encoder")
        return True
    else:
        print(f"\n  ✗ 警告：TX 仍在使用 {context['mode']} encoder")
        return False


def main():
    parser = argparse.ArgumentParser(description='測試真實 NTSCC encoder')
    parser.add_argument('--ckpt', type=str, 
                        default='checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth',
                        help='NTSCC checkpoint 路徑')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', '1', '2', '3'],
                        help='執行哪個測試（1=分析checkpoint, 2=比較輸出, 3=TX管線）')
    
    args = parser.parse_args()
    
    # 檢查檔案是否存在
    if not Path(args.ckpt).exists():
        print(f"✗ Checkpoint 不存在: {args.ckpt}")
        print(f"\n建議的 checkpoint 位置：")
        print(f"  - checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth")
        print(f"  - models/pretrained/ntscc/ntscc_hyperprior_quality_4_psnr.pth")
        sys.exit(1)
    
    # 執行測試
    results = {}
    
    if args.test in ['all', '1']:
        results['test1'] = test1_analyze_checkpoint(args.ckpt)
    
    if args.test in ['all', '2']:
        results['test2'] = test2_stub_vs_real(args.ckpt)
    
    if args.test in ['all', '3']:
        results['test3'] = test3_tx_pipeline(args.ckpt)
    
    # 總結
    print("\n" + "="*60)
    print("測試總結")
    print("="*60)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 所有測試通過！可以開始使用真實 NTSCC encoder")
        print("\n下一步：")
        print("  1. 更新 run_tx_kodak_batch.py，使用 mode='real'")
        print("  2. 重新生成 bridge_tx/")
        print("     python scripts/run_tx_kodak_batch.py --img kodim01")
        print("  3. 驗證類比層仍能命中工作窗")
        print("     python run_analog.py")
    else:
        print("\n需要檢查失敗的測試項目")


if __name__ == "__main__":
    main()