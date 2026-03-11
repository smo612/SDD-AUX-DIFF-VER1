"""
NTSCC RX Wrapper - 接收端解碼器（最終修復版）
從 Digital SIC 輸出重建圖像

✅ 完整修復：
1. 正確移除導頻（精確收集到8192個data）
2. TX縮放係數反向還原
3. I/Q 兩路正確還原
4. 過濾 attn_mask 避免形狀不匹配
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# 導入 NTSCC decoder
from layer.synthesis_transform import SynthesisTransform


class NTSCCRXWrapper:
    """NTSCC Decoder 包裝器"""
    
    def __init__(self, 
                 ckpt_path: str,
                 device: str = None):
        self.ckpt_path = ckpt_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[NTSCCRXWrapper] 初始化 NTSCC decoder")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Device: {self.device}")
        
        self.model = self._load_decoder(ckpt_path)
        self.model.eval()
    
    def _load_decoder(self, ckpt_path: str):
        """載入 NTSCC decoder"""
        ckpt_path = Path(ckpt_path)
        
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
        
        print(f"  [1/5] 建立 SynthesisTransform...")
        
        gs_kwargs = {
            'img_size': (128, 128),
            'embed_dims': [256, 256, 256, 256],
            'depths': [4, 2, 1, 1],
            'num_heads': [8, 8, 8, 8],
            'window_size': 8,
            'mlp_ratio': 4.0,
            'norm_layer': nn.LayerNorm,
            'patch_norm': True
        }
        
        model = SynthesisTransform(**gs_kwargs).to(self.device)
        print(f"    ✓ SynthesisTransform 建立完成")
        
        print(f"  [2/5] 載入 checkpoint...")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        print(f"    ✓ Checkpoint 載入完成")
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print(f"    ℹ️  Checkpoint 包含 'state_dict' key")
            state_dict = checkpoint['state_dict']
        else:
            print(f"    ℹ️  Checkpoint 為直接 state_dict")
            state_dict = checkpoint
        
        print(f"  [3/5] 提取 decoder 權重...")
        gs_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('gs.'):
                new_key = key[3:]
                gs_state_dict[new_key] = value
        
        print(f"    ✓ 提取到 {len(gs_state_dict)} 個權重")
        
        print(f"  [4/5] 過濾 attn_mask...")
        gs_state_dict_filtered = {}
        attn_mask_count = 0
        
        for key, value in gs_state_dict.items():
            if 'attn_mask' in key:
                attn_mask_count += 1
            else:
                gs_state_dict_filtered[key] = value
        
        if attn_mask_count > 0:
            print(f"    ✓ 過濾掉 {attn_mask_count} 個 attn_mask（將重新計算）")
        
        print(f"  [5/5] 載入權重到模型...")
        
        missing_keys, unexpected_keys = model.load_state_dict(
            gs_state_dict_filtered, 
            strict=False
        )
        
        missing_keys_real = [k for k in missing_keys if 'attn_mask' not in k]
        
        if len(missing_keys_real) == 0 and len(unexpected_keys) == 0:
            print(f"    ✓ 權重完美匹配！")
        elif len(missing_keys) == attn_mask_count:
            print(f"    ✓ 權重匹配！（僅缺少 attn_mask，將重新計算）")
        else:
            print(f"    ⚠️  權重部分匹配（可能影響效果）")
        
        print(f"  [Resolution] 調整到 128×128...")
        model.update_resolution(8, 8)
        print(f"    ✓ Resolution adaptation 完成（attn_mask 已重新計算）")
        
        return model
    
    def decode(self, 
               y_clean: np.ndarray,
               original_img: np.ndarray = None,
               img_size: Tuple[int, int] = (128, 128),
               cbr: float = 1/16,
               meta_tx: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        解碼：y_clean → 重建圖像
        
        Args:
            y_clean: Digital SIC 後的複數符號 (N,)
            original_img: 原始圖像，shape (H, W, 3)
            img_size: 圖像大小
            cbr: Channel Bandwidth Ratio
            meta_tx: TX端元數據
        
        Returns:
            img_recon: 重建圖像 (H, W, 3)
            metrics: 指標字典
        """
        print(f"\n{'='*60}")
        print(f"執行 NTSCC Decoder (RX)")
        print(f"{'='*60}")
        
        # ========================================
        # 步驟 1：移除導頻
        # ========================================
        print(f"[1/4] 移除導頻...")
        
        if meta_tx and 'pilot_info' in meta_tx:
            pilot_info = meta_tx['pilot_info']
            pilot_enabled = pilot_info.get('pilot_enabled', False)
            pilot_period = int(pilot_info.get('pilot_period', 64) or 64)
            n_pilots = int(pilot_info.get('n_pilots', 0) or 0)
        else:
            pilot_enabled = (len(y_clean) > 8192)
            pilot_period = 64
            n_pilots = len(y_clean) - 8192 if pilot_enabled else 0
            print(f"  ⚠️  未提供 meta_tx，使用推斷配置")
        
        # 從 meta 讀取期望的 data 符號數
        n_data_expected = None
        if meta_tx and 'signal_info' in meta_tx:
            n_data_expected = int(meta_tx['signal_info'].get('n_data_symbols', 0) or 0)
        if not n_data_expected:
            H, W = img_size
            n_data_expected = (H // 16) * (W // 16) * 256 // 2
        
        print(f"  導頻配置：enabled={pilot_enabled}, period={pilot_period}, n_pilots={n_pilots}")
        print(f"  期望資料符號數：{n_data_expected}")
        print(f"  輸入符號數：{len(y_clean)} 複數")
        
        if pilot_enabled and n_pilots > 0:
            y_data = self._strip_pilots(y_clean, pilot_period, n_pilots, n_data_expected)
            print(f"  ✓ 移除 {n_pilots} 個導頻後：{len(y_data)} 個資料符號")
            
            if len(y_data) != n_data_expected:
                raise ValueError(
                    f"❌ 導頻移除後符號數不符！\n"
                    f"   期望: {n_data_expected}\n"
                    f"   實際: {len(y_data)}\n"
                    f"   差異: {len(y_data) - n_data_expected}"
                )
        else:
            y_data = y_clean[:n_data_expected] if len(y_clean) > n_data_expected else y_clean
            print(f"  ℹ️  無導頻，直接使用原始符號")
        
        # ========================================
        # 步驟 2：反縮放
        # ========================================
        print(f"[2/4] 反縮放處理...")
        
        if meta_tx and 'signal_info' in meta_tx:
            tx_scale = meta_tx['signal_info'].get('tx_scale', 0.0)
        else:
            tx_scale = 0.0
        
        if tx_scale > 0:
            print(f"  使用 meta 中的 tx_scale: {tx_scale:.6f}")
            y_data = y_data / tx_scale
        else:
            rms = np.sqrt(np.mean(np.abs(y_data)**2))
            if rms > 1e-8:
                y_data = y_data / rms
                print(f"  ⚠️  未提供 tx_scale，使用 RMS 歸一化: {rms:.6f}")
            else:
                print(f"  ⚠️  符號功率過低，跳過歸一化")
        
        # ========================================
        # 步驟 3：I/Q 還原為 latent
        # ========================================
        print(f"[3/4] 整理 latent representation...")
        y_hat = self._symbols_to_latent(y_data, img_size, cbr)
        print(f"  ✓ y_hat shape: {y_hat.shape}")
        
        # ========================================
        # 步驟 4：解碼
        # ========================================
        print(f"[4/4] 執行 SynthesisTransform...")
        with torch.no_grad():
            y_hat_tensor = torch.from_numpy(y_hat).to(self.device)
            x_hat_tensor = self.model(y_hat_tensor, out_conv=True)
        
        x_hat = x_hat_tensor.cpu().numpy()
        print(f"  ✓ x_hat shape: {x_hat.shape}")
        
        img_recon = self._postprocess(x_hat)
        print(f"  ✓ img_recon shape: {img_recon.shape}, range: [{img_recon.min():.3f}, {img_recon.max():.3f}]")
        
        metrics = {}
        if original_img is not None:
            metrics = self._compute_metrics(original_img, img_recon)
            
            print(f"\n{'='*60}")
            print(f"===== NTSCC RX REPORT =====")
            print(f"{'='*60}")
            print(f"PSNR: {metrics['psnr']:.2f} dB")
            if 'ms_ssim' in metrics:
                print(f"MS-SSIM: {metrics['ms_ssim']:.4f}")
            print(f"{'='*60}\n")
        
        return img_recon, metrics
    
    def _strip_pilots(self, 
                      y: np.ndarray, 
                      period: int, 
                      n_pilots: int,
                      n_data_expected: int) -> np.ndarray:
        """
        按交錯模式移除導頻：每 period 個 data 後跟 1 個 pilot
        
        ✅ 核心修復：以 n_data_expected 為準，精確收集
        
        Args:
            y: 含導頻的符號序列，例如 8319 = 8192 data + 127 pilot
            period: 導頻週期
            n_pilots: 總導頻數量
            n_data_expected: 期望的資料符號數（從 meta 讀取）
        
        Returns:
            y_data: 純資料符號 (n_data_expected,)
        """
        if period <= 0 or n_pilots <= 0:
            return y[:n_data_expected].astype(np.complex64, copy=False)
        
        out = []
        i = 0
        total = len(y)
        pilots_removed = 0
        
        # 逐塊收集 data 直到湊滿 n_data_expected
        while i < total and sum(len(c) for c in out) < n_data_expected:
            # 收集一段 data（最多 period 個）
            chunk_end = min(i + period, total)
            if chunk_end > i:
                out.append(y[i:chunk_end])
            i = chunk_end
            
            # 跳過一個 pilot（如果還有）
            if pilots_removed < n_pilots and i < total:
                i += 1
                pilots_removed += 1
        
        y_data = np.concatenate(out) if out else np.empty(0, dtype=y.dtype)
        
        # 最終保險：精準裁到期望長度
        if len(y_data) > n_data_expected:
            y_data = y_data[:n_data_expected]
        
        return y_data.astype(np.complex64, copy=False)
    
    def _symbols_to_latent(self, 
                           y_data: np.ndarray, 
                           img_size: Tuple[int, int],
                           cbr: float) -> np.ndarray:
        """
        將純資料符號轉換為 latent representation
        
        Args:
            y_data: 純資料複數符號 (8192,)
            img_size: 原始圖像大小
            cbr: Channel Bandwidth Ratio
        
        Returns:
            y_hat: latent representation (1, 256, 8, 8)
        """
        H, W = img_size
        latent_h = H // 16
        latent_w = W // 16
        latent_c = 256
        
        expected_complex = (latent_h * latent_w * latent_c) // 2
        
        if len(y_data) != expected_complex:
            raise ValueError(
                f"❌ 資料符號數量不匹配！\n"
                f"   期望: {expected_complex} 個複數\n"
                f"   實際: {len(y_data)} 個複數\n"
                f"   差異: {len(y_data) - expected_complex}"
            )
        
        # I/Q 分離
        I = np.real(y_data).astype(np.float32)
        Q = np.imag(y_data).astype(np.float32)
        
        # 交錯還原
        v = np.empty(I.size * 2, dtype=np.float32)
        v[0::2] = I
        v[1::2] = Q
        
        # Reshape
        y_hat = v.reshape(1, latent_c, latent_h, latent_w)
        
        return y_hat
    
    def _postprocess(self, x_hat: np.ndarray) -> np.ndarray:
        """後處理"""
        img_recon = x_hat[0].transpose(1, 2, 0)
        img_recon = np.clip(img_recon, 0.0, 1.0)
        return img_recon
    
    def _compute_metrics(self, 
                        img_original: np.ndarray,
                        img_recon: np.ndarray) -> dict:
        """計算 PSNR 和 MS-SSIM"""
        img_original = np.clip(img_original, 0.0, 1.0)
        img_recon = np.clip(img_recon, 0.0, 1.0)
        
        mse = np.mean((img_original - img_recon) ** 2)
        
        if mse < 1e-10:
            psnr = 100.0
        else:
            psnr = 10 * np.log10(1.0 / mse)
        
        metrics = {
            'psnr': float(psnr),
            'mse': float(mse)
        }
        
        try:
            from skimage.metrics import structural_similarity as ssim
            try:
                ms_ssim = ssim(img_original, img_recon, 
                              data_range=1.0, channel_axis=2)
            except TypeError:
                ms_ssim = ssim(img_original, img_recon, 
                              data_range=1.0, multichannel=True)
            metrics['ms_ssim'] = float(ms_ssim)
        except ImportError:
            pass
        
        return metrics