#!/usr/bin/env python3
"""
SDD Final Visualization Script (v5-4 Optimized)
保留原版所有排版與功能，僅安全支援 Diffusion 模式。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path
from PIL import Image
import argparse

def safe_format(value, format_spec='.2f', unit='', na_text='N/A'):
    if value is None or value == 'N/A':
        return na_text + (f' {unit}' if unit else '')
    if isinstance(value, (int, float)):
        return f"{value:{format_spec}}" + (f' {unit}' if unit else '')
    return str(value) + (f' {unit}' if unit else '')

def load_results(bridge_rx='bridge_rx', bridge_digital='bridge_digital', bridge_analog='bridge'):
    results = {}
    
    # 1. 載入圖像
    img_recon_path = Path(bridge_rx) / 'img_recon_remote.png'
    if img_recon_path.exists():
        results['img_recon'] = np.array(Image.open(img_recon_path))
    
    # 2. 載入 metrics
    with open(Path(bridge_rx) / 'metrics_remote.json') as f:
        results['rx_metrics'] = json.load(f)
    
    # 3. 載入 digital metrics
    with open(Path(bridge_digital) / 'metrics.json') as f:
        results['digital_metrics'] = json.load(f)
    
    # 4. 載入 analog meta
    with open(Path(bridge_analog) / 'meta.json') as f:
        results['analog_meta'] = json.load(f)
        
    # 5. 判斷模式 (最小幅度新增 Diffusion 判斷)
    backend = results['digital_metrics'].get('backend', 'None')
    results['is_diffusion'] = False
    
    if backend == 'None' or backend is None:
        results['mode'] = 'Analog Only'
        results['has_dsic'] = False
    elif str(backend).lower() == 'diffusion':
        results['mode'] = 'AI Assisted (DIFFUSION)'
        results['has_dsic'] = True
        results['is_diffusion'] = True
    else:
        results['mode'] = f'Digital Assisted ({backend.upper()})'
        results['has_dsic'] = True
        
    return results

def create_professional_figure(results, gt_path=None, output_path='sdd_final_v5.png', dpi=300):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    
    fig = plt.figure(figsize=(13, 10), dpi=dpi)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 0.8], hspace=0.3, wspace=0.3)
    
    mode_name = results['mode']
    if not results['has_dsic']:
        mode_color = 'darkred'
    elif results['is_diffusion']:
        mode_color = 'blue'
    else:
        mode_color = 'darkgreen'
    
    # ==================== Row 1: 圖像對比 ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    img_original = None
    if gt_path and Path(gt_path).exists():
        try:
            img_original = np.array(Image.open(gt_path))
        except:
            pass
            
    if img_original is not None:
        if 'img_recon' in results:
            recon_h, recon_w = results['img_recon'].shape[:2]
            orig_h, orig_w = img_original.shape[:2]
            if (orig_h != recon_h) or (orig_w != recon_w):
                img_original_pil = Image.fromarray(img_original)
                img_original = np.array(img_original_pil.resize((recon_w, recon_h), Image.Resampling.LANCZOS))
        
        ax1.imshow(img_original)
        ax1.set_title(f'Original Image (Input)\n{img_original.shape[1]}x{img_original.shape[0]}', fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, "Original Image\n(Not loaded)", ha='center', va='center')
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    if 'img_recon' in results:
        ax2.imshow(results['img_recon'])
        psnr = results['rx_metrics']['psnr']
        h, w = results['img_recon'].shape[:2]
        title = f'Reconstructed Image\nPSNR: {psnr:.2f} dB ({w}x{h})'
        ax2.set_title(title, fontsize=12, fontweight='bold', color=mode_color)
    else:
        ax2.text(0.5, 0.5, "Reconstruction Failed", ha='center', va='center')
    ax2.axis('off')
    
    # ==================== Row 2: SINR Evolution ====================
    ax3 = fig.add_subplot(gs[1, :])
    
    analog_meta = results['analog_meta']
    digital_meta = results['digital_metrics']
    
    sinr_pre = analog_meta.get('SINR_pre', -20)
    sinr_analog = analog_meta.get('SINR_analog', 0)
    
    if results['has_dsic'] and not results['is_diffusion']:
        sinr_digital = digital_meta.get('SINR_after_digital', sinr_analog)
        if isinstance(sinr_digital, str): sinr_digital = sinr_analog
        
        stages = ['Input (TX)', 'After Analog SIC\n(Aux-TX)', 'After Digital SIC\n(MP)']
        values = [sinr_pre, sinr_analog, sinr_digital]
        colors = ['gray', '#3498db', '#2ecc71'] 
    else:
        stages = ['Input (TX)', 'After Analog SIC\n(Aux-TX)']
        values = [sinr_pre, sinr_analog]
        if results['is_diffusion']:
            colors = ['gray', '#3498db'] 
        else:
            colors = ['gray', '#e74c3c'] 
    
    x_pos = np.arange(len(stages))
    bars = ax3.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', width=0.6)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        y_pos = height + 2 if height > 0 else height - 5
        ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.2f} dB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    v_min = min(values) if values else 0
    v_max = max(values) if values else 0
    y_lower = min(0, v_min) - 15  
    y_upper = max(0, v_max) + 10
    ax3.set_ylim(y_lower, y_upper)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(stages, fontsize=11, fontweight='bold')
    
    if len(values) >= 2:
        gain_ana = values[1] - values[0]
        mid_y = (values[0] + values[1]) / 2
        ax3.annotate(f'+{gain_ana:.1f} dB', xy=(0.5, mid_y), xytext=(0.8, mid_y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                    fontsize=10, fontweight='bold', color='blue')
        
    if len(values) >= 3:
        gain_dig = values[2] - values[1]
        mid_y = (values[1] + values[2]) / 2
        ax3.annotate(f'+{gain_dig:.1f} dB', xy=(1.5, mid_y), xytext=(1.8, mid_y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                    fontsize=10, fontweight='bold', color='green')
    
    ax3.set_ylabel('SINR (dB)', fontsize=12, fontweight='bold')
    ax3.set_title(f'SINR Evolution ({mode_name})', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.axhline(y=0, color='black', linewidth=0.8)
    
    # ==================== Row 3: Key Metrics Table ====================
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    rsi_scale = analog_meta.get('rsi_scale', 'N/A')
    snr_db = analog_meta.get('snr_db', 'N/A')
    supp_ana = analog_meta.get('Supp_analog', 0)
    
    supp_dig = digital_meta.get('Digital_gain', 0) 
    if not results['has_dsic']: 
        supp_dig_str = "N/A (Skipped)"
    elif results['is_diffusion']:
        supp_dig_str = "AI Implicit"
    else: 
        try:
            supp_dig_str = f"{float(supp_dig):.2f} dB"
        except:
            supp_dig_str = str(supp_dig)
    
    psnr_val = results['rx_metrics']['psnr']
    status = "SUCCESS" if psnr_val > 30 else "LOW QUALITY"
    
    table_data = [
        ['Configuration', '', 'Performance', ''],
        ['Experiment Mode', mode_name, 'Analog Supp. (Aux-TX)', f"{supp_ana:.2f} dB"],
        ['RSI Scale', f"{rsi_scale}x", 'Digital Supp. (MP/AI)', supp_dig_str],
        ['SNR', f"{snr_db} dB", 'Final PSNR', f"{psnr_val:.2f} dB"],
        ['Hardware Mismatch', 'p=2.2 vs 2.4', 'Result Status', status]
    ]
    
    table = ax4.table(cellText=table_data, 
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    for i in range(5):
        if i == 0:
            for j in range(4):
                cell = table[(i, j)]
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
        else:
            if i == 4 and table_data[i][3] == "SUCCESS":
                table[(i, 3)].set_text_props(weight='bold', color='green')
            elif i == 4 and table_data[i][3] == "LOW QUALITY":
                table[(i, 3)].set_text_props(weight='bold', color='red')
                
            if i == 1:
                table[(i, 1)].set_text_props(weight='bold', color=mode_color)

            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')

    fig.suptitle(f'SDD End-to-End Performance: {mode_name}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 對比圖已成功儲存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='SDD V5 Final Visualization')
    parser.add_argument('--output', type=str, default='sdd_final_result.png')
    # 🌟 預設值還原回你一開始的 kodim24.png
    parser.add_argument('--gt', type=str, default='data/kodak/kodim24.png', help='Path to original image (for display)')
    args = parser.parse_args()
    
    print(f"正在載入結果並準備輸出至 {args.output}...")
    try:
        results = load_results()
        create_professional_figure(results, gt_path=args.gt, output_path=args.output)
    except Exception as e:
        print(f"❌ 繪圖失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()