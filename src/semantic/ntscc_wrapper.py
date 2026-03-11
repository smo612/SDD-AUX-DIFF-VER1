"""
ntscc_wrapper.py - NTSCC Encoder Wrapper
只使用真實 NTSCC encoder (Swin Transformer)
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# 導入真實 NTSCC encoder
from layer.analysis_transform import AnalysisTransform


class NTSCCWrapper:
    """
    NTSCC Encoder 包裝器（簡化版）
    
    直接使用預訓練的 Swin Transformer encoder
    
    用法：
        wrapper = NTSCCWrapper(
            ckpt_path='checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth'
        )
        latent, context = wrapper.encode(img, cbr=1/16)
    """
    
    def __init__(self, 
                 ckpt_path: str,
                 device: str = None):
        """
        Args:
            ckpt_path: NTSCC checkpoint 路徑（quality_4）
            device: 'cuda' 或 'cpu'
        """
        self.ckpt_path = ckpt_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[NTSCCWrapper] 載入真實 NTSCC encoder (Swin Transformer)")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Device: {self.device}")
        
        self.model = self._load_encoder(ckpt_path)
        self.model.eval()
    
    def _load_encoder(self, ckpt_path: str):
        """載入 NTSCC encoder"""
        ckpt_path = Path(ckpt_path)
        
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
        
        print(f"  [1/4] 建立 AnalysisTransform...")
        
        # NTSCC quality_4 配置
        ga_kwargs = {
            'img_size': (128, 128),
            'embed_dims': [256, 256, 256, 256],
            'depths': [1, 1, 2, 4],
            'num_heads': [8, 8, 8, 8],
            'window_size': 8,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'qk_scale': None,
            'norm_layer': nn.LayerNorm,
            'patch_norm': True
        }
        
        model = AnalysisTransform(**ga_kwargs).to(self.device)
        print(f"    ✓ AnalysisTransform 建立完成")
        
        print(f"  [2/4] 載入 checkpoint...")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        print(f"    ✓ Checkpoint 載入完成")
        
        print(f"  [3/4] 提取 encoder 權重...")
        encoder_state_dict = self._extract_encoder_weights(checkpoint)
        print(f"    ✓ 提取到 {len(encoder_state_dict)} 個權重")
        
        print(f"  [4/4] 載入權重到模型...")
        missing, unexpected = model.load_state_dict(encoder_state_dict, strict=False)
        
        if len(missing) == 0 and len(unexpected) == 0:
            print(f"    ✓ 權重完美匹配！")
        else:
            if len(missing) > 0:
                print(f"    ⚠ Missing keys: {len(missing)}")
            if len(unexpected) > 0:
                print(f"    ⚠ Unexpected keys: {len(unexpected)}")
        
        print(f"  [Resolution] 調整到 128×128...")
        model.update_resolution(128, 128)
        print(f"    ✓ Resolution adaptation 完成")
        
        return model
    
    def _extract_encoder_weights(self, checkpoint: Dict) -> Dict:
        """從完整 checkpoint 提取 encoder 權重"""
        # 找到 state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and any('ga.' in k for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            raise ValueError("無法在 checkpoint 中找到 state_dict")
        
        # 提取 'ga.' 開頭的 keys
        encoder_dict = {}
        for key in state_dict.keys():
            if key.startswith('ga.'):
                new_key = key[3:]  # 去掉 'ga.'
                encoder_dict[new_key] = state_dict[key]
            elif key.startswith('module.ga.'):
                new_key = key[10:]  # 去掉 'module.ga.'
                encoder_dict[new_key] = state_dict[key]
        
        if len(encoder_dict) == 0:
            raise ValueError("無法找到 encoder 權重（'ga.' 前綴）")
        
        return encoder_dict
    
    def encode(self, img: np.ndarray, cbr: float = 1/16) -> Tuple[torch.Tensor, Dict]:
        """
        編碼影像為 latent representation
        
        Args:
            img: [H, W, 3] RGB 影像, 值域 0~1, dtype=float32
            cbr: Channel Bandwidth Ratio（記錄用）
        
        Returns:
            latent: [1, 256, 8, 8] latent tensor (on device)
            context: metadata dict
        """
        # 前處理
        x = self._preprocess(img)
        
        # 前向
        with torch.no_grad():
            latent = self.model(x)
        
        # Context
        context = {
            'ntscc_mode': 'real',
            'cbr': cbr,
            'latent_shape': list(latent.shape),
            'img_shape': list(img.shape),
        }
        
        return latent, context
    
    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """影像前處理：HWC → BCHW"""
        H, W = img.shape[:2]
        assert (H, W) == (128, 128), f"影像必須是 128×128，當前 {(H, W)}"
        assert img.shape[2] == 3, "需要 RGB 3 通道"
        assert img.dtype == np.float32, "需要 float32 類型"
        
        # HWC → CHW → add batch
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return x.to(self.device)


if __name__ == "__main__":
    # 快速測試
    print("="*60)
    print("NTSCC Wrapper 測試")
    print("="*60)
    
    ckpt_path = "checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth"
    
    if not Path(ckpt_path).exists():
        print(f"✗ Checkpoint 不存在: {ckpt_path}")
    else:
        wrapper = NTSCCWrapper(ckpt_path=ckpt_path)
        
        # 測試編碼
        img = np.random.rand(128, 128, 3).astype(np.float32)
        latent, ctx = wrapper.encode(img, cbr=1/16)
        
        print(f"\n✓ 測試通過")
        print(f"  Latent shape: {latent.shape}")
        print(f"  Latent range: [{latent.min():.3f}, {latent.max():.3f}]")
        print(f"  Context: {ctx}")