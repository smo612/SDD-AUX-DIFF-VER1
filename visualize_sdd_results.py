#!/usr/bin/env python3
"""
SDD Results Professional Visualization Script (修復版)

✅ 修復：處理 Digital SIC 被跳過時的格式化錯誤
✅ 改進：安全的數值格式化函數

Generate publication-quality comparison figures including:
- Original image vs Reconstructed image
- Key metrics annotation (PSNR, SINR, RSI_SCALE, etc.)
- Channel condition information

Usage:
    # Auto-read latest results
    python visualize_sdd_results.py
    
    # Specify result directories
    python visualize_sdd_results.py --bridge-rx bridge_rx --bridge-digital bridge_digital
    
    # Custom output
    python visualize_sdd_results.py --output paper_figure.png --dpi 300
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path
from PIL import Image
import argparse


# ============================================================
# 🔧 新增：安全格式化函數
# ============================================================
def safe_format(value, format_spec='.2f', unit='', na_text='N/A'):
    """
    安全地格式化數值，處理 None 和 'N/A' 的情況
    
    Args:
        value: 要格式化的值
        format_spec: 格式化規格（如 '.2f', '.4f'）
        unit: 單位字串（如 'dB', 'Hz'）
        na_text: 當值不可用時的顯示文字
    
    Returns:
        格式化後的字串
    """
    if value is None or value == 'N/A':
        return na_text + (f' {unit}' if unit else '')
    
    if isinstance(value, (int, float)):
        try:
            formatted = f"{value:{format_spec}}"
            return formatted + (f' {unit}' if unit else '')
        except:
            return na_text + (f' {unit}' if unit else '')
    
    # 如果是字串類型
    return str(value) + (f' {unit}' if unit else '')


def load_results(bridge_rx='bridge_rx', bridge_digital='bridge_digital', bridge_analog='bridge'):
    """Load all result data"""
    results = {}
    
    # 1. Load images
    img_recon_path = Path(bridge_rx) / 'img_recon_remote.png'
    comparison_path = Path(bridge_rx) / 'comparison_remote.png'
    
    if img_recon_path.exists():
        results['img_recon'] = np.array(Image.open(img_recon_path))
    
    if comparison_path.exists():
        comparison = np.array(Image.open(comparison_path))
        # Split into original and reconstructed
        h, w = comparison.shape[:2]
        results['img_original'] = comparison[:, :w//2]
        results['img_recon_full'] = comparison[:, w//2:]
    
    # 2. Load metrics
    with open(Path(bridge_rx) / 'metrics_remote.json') as f:
        results['rx_metrics'] = json.load(f)
    
    # 3. Load digital metrics
    with open(Path(bridge_digital) / 'metrics.json') as f:
        results['digital_metrics'] = json.load(f)
    
    # 4. Load analog meta
    with open(Path(bridge_analog) / 'meta.json') as f:
        results['analog_meta'] = json.load(f)
    
    return results


def create_professional_figure(results, output_path='sdd_results_professional.png', dpi=300):
    """
    Create professional results figure
    
    Layout:
    - Top: Original | Reconstructed (side by side)
    - Middle: SINR evolution curve + key metrics table
    """
    # Font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    
    # 創建圖形
    fig = plt.figure(figsize=(12, 10), dpi=dpi)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 0.8], hspace=0.3, wspace=0.3)
    
    # ==================== 圖像對比 ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Original Image
    if 'img_original' in results:
        ax1.imshow(results['img_original'])
        ax1.set_title('Original Image (Remote/Desired)', fontsize=12, fontweight='bold')
        ax1.axis('off')
    
    # Reconstructed Image
    if 'img_recon' in results:
        ax2.imshow(results['img_recon'])
        
        # Annotate PSNR
        psnr = results['rx_metrics']['psnr']
        ms_ssim = results['rx_metrics'].get('ms_ssim', None)
        
        title = f'Reconstructed Image\nPSNR: {psnr:.2f} dB'
        if ms_ssim is not None:
            title += f' | MS-SSIM: {ms_ssim:.4f}'
        
        ax2.set_title(title, fontsize=12, fontweight='bold')
        ax2.axis('off')
    
    # ==================== SINR Evolution ====================
    ax3 = fig.add_subplot(gs[1, :])
    
    analog_meta = results['analog_meta']
    digital_meta = results['digital_metrics']
    
    sinr_pre = analog_meta.get('SINR_pre', 0)
    sinr_analog = analog_meta.get('SINR_analog', 0)
    sinr_digital = digital_meta.get('SINR_digital')  # 可能是 None
    
    # 🔧 判斷是否跳過 Digital SIC
    digital_skipped = (sinr_digital is None or sinr_digital == 'N/A')
    
    if not digital_skipped:
        # 完整流程
        stages = ['Before\nAnalog', 'After\nAnalog', 'After\nDigital']
        sinr_values = [sinr_pre, sinr_analog, sinr_digital]
        colors = ['red', 'orange', 'green']
    else:
        # 跳過 Digital SIC
        stages = ['Before\nAnalog', 'After\nAnalog']
        sinr_values = [sinr_pre, sinr_analog]
        colors = ['red', 'orange']
    
    # Plot bars
    x_pos = np.arange(len(stages))
    bars = ax3.bar(x_pos, sinr_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Annotate values
    for i, (bar, val) in enumerate(zip(bars, sinr_values)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f} dB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(stages, fontsize=11, fontweight='bold')
    
    # Annotate gains
    if len(sinr_values) >= 2:
        # Analog SIC 增益
        analog_gain = sinr_values[1] - sinr_values[0]
        ax3.annotate('', xy=(0.5, sinr_values[1]), xytext=(0.5, sinr_values[0]),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        ax3.text(0.5, (sinr_values[0] + sinr_values[1])/2, f'+{analog_gain:.1f} dB',
                ha='right', va='center', fontsize=9, color='blue', fontweight='bold')
    
    if len(sinr_values) >= 3:
        # Digital SIC 增益
        digital_gain = sinr_values[2] - sinr_values[1]
        ax3.annotate('', xy=(1.5, sinr_values[2]), xytext=(1.5, sinr_values[1]),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        ax3.text(1.5, (sinr_values[1] + sinr_values[2])/2, f'+{digital_gain:.1f} dB',
                ha='right', va='center', fontsize=9, color='green', fontweight='bold')
    
    ax3.set_ylabel('SINR (dB)', fontsize=11, fontweight='bold')
    ax3.set_title('SINR Evolution Process', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # ==================== Key Metrics Table ====================
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Prepare table data
    analog_sic_info = analog_meta.get('analog_sic_info', {})
    
    # Check Analog SIC status
    analog_status = "Normal" if not analog_sic_info.get('saturated', False) else "Saturated"
    
    # 🔧 使用 safe_format 處理可能缺失的值
    analog_supp = analog_meta.get('Supp_analog')
    digital_supp = digital_meta.get('Digital_supp_si')
    total_supp = digital_meta.get('Total_supp_SI_only')
    signal_corr = analog_meta.get('signal_correlation', {}).get('rho')
    
    table_data = [
        ['Configuration', '', 'Performance', ''],
        ['SNR', f"{analog_meta.get('snr_db', 'N/A')} dB", 
         'Analog Supp.', f"{safe_format(analog_supp, '.2f', f'dB ({analog_status})')}"],
        ['RSI_SCALE', f"{analog_meta.get('rsi_scale', 'N/A')}", 
         'Digital Supp.', safe_format(digital_supp, '.2f', 'dB')],
        ['SIC Target', f"{analog_sic_info.get('target_suppression_db', 'N/A')} dB", 
         'Total Supp.', safe_format(total_supp, '.2f', 'dB')],
        ['Signal Corr.', f"|ρ| = {safe_format(signal_corr, '.4f')}",
         'PSNR', f"{results['rx_metrics']['psnr']:.2f} dB"],
    ]
    
    # 繪製表格
    table = ax4.table(cellText=table_data, 
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.2, 0.3, 0.2, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 設置表頭樣式
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')
    
    # 交替行顏色
    for i in range(1, len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
    
    # ==================== Overall Title ====================
    fig.suptitle('SDD End-to-End System Performance Evaluation', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Professional figure saved: {output_path}")
    
    return fig


def create_compact_figure(results, output_path='sdd_results_compact.png', dpi=300):
    """
    Create compact figure (suitable for papers)
    
    Layout: Original | Reconstructed | Metrics panel
    """
    fig = plt.figure(figsize=(15, 5), dpi=dpi)
    gs = GridSpec(1, 3, figure=fig, wspace=0.15)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    if 'img_original' in results:
        ax1.imshow(results['img_original'])
        ax1.set_title('Original', fontsize=14, fontweight='bold')
        ax1.axis('off')
    
    # Reconstructed image
    ax2 = fig.add_subplot(gs[0, 1])
    if 'img_recon' in results:
        ax2.imshow(results['img_recon'])
        psnr = results['rx_metrics']['psnr']
        ax2.set_title(f'Reconstructed\nPSNR: {psnr:.2f} dB', fontsize=14, fontweight='bold')
        ax2.axis('off')
    
    # Metrics panel
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    # Extract key data
    analog_meta = results['analog_meta']
    digital_meta = results['digital_metrics']
    analog_sic_info = analog_meta.get('analog_sic_info', {})
    
    sinr_pre = analog_meta.get('SINR_pre', 0)
    sinr_analog = analog_meta.get('SINR_analog', 0)
    sinr_digital = digital_meta.get('SINR_digital')
    
    analog_supp = analog_meta.get('Supp_analog', 0)
    digital_supp = digital_meta.get('Digital_supp_si')
    total_supp = digital_meta.get('Total_supp_SI_only')
    
    saturated = analog_sic_info.get('saturated', False)
    status_icon = "⚠ " if saturated else "✓"
    
    # 🔧 使用 safe_format 處理格式化
    sinr_digital_str = safe_format(sinr_digital, '.2f', 'dB')
    digital_supp_str = safe_format(digital_supp, '.2f', 'dB')
    total_supp_str = safe_format(total_supp, '.2f', 'dB')
    
    # Text annotation
    info_text = f"""
Channel Conditions
────────────────
SNR:            {analog_meta.get('snr_db')} dB
RSI_SCALE:      {analog_meta.get('rsi_scale')}
SIC Target:     {analog_sic_info.get('target_suppression_db', 23)} dB

SINR Evolution
────────────────
Before TX:      {sinr_pre:.2f} dB
Analog SIC:     {sinr_analog:.2f} dB
Digital SIC:    {sinr_digital_str}

Suppression
────────────────
Analog:         {analog_supp:.2f} dB {status_icon}
Digital:        {digital_supp_str}
Total:          {total_supp_str}

Image Quality
────────────────
PSNR:           {results['rx_metrics']['psnr']:.2f} dB
MS-SSIM:        {results['rx_metrics'].get('ms_ssim', 0):.4f}
    """
    
    ax3.text(0.1, 0.95, info_text.strip(), 
            transform=ax3.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Compact figure saved: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize SDD results (修復版)')
    
    parser.add_argument('--bridge-rx', type=str, default='bridge_rx',
                       help='RX output directory')
    parser.add_argument('--bridge-digital', type=str, default='bridge_digital',
                       help='Digital SIC output directory')
    parser.add_argument('--bridge-analog', type=str, default='bridge',
                       help='Analog domain output directory')
    
    parser.add_argument('--output', type=str, default='sdd_results_professional.png',
                       help='Output figure path')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Figure DPI')
    parser.add_argument('--compact', action='store_true',
                       help='Generate compact figure instead')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SDD Results Professional Visualization")
    print("=" * 60)
    
    # Load results
    print("Loading results...")
    try:
        results = load_results(args.bridge_rx, args.bridge_digital, args.bridge_analog)
        print("✓ Results loaded")
        print(f"  PSNR: {results['rx_metrics']['psnr']:.2f} dB")
        
        sinr_digital = results['digital_metrics'].get('SINR_digital')
        if sinr_digital is not None and sinr_digital != 'N/A':
            print(f"  SINR: {sinr_digital:.2f} dB")
        else:
            print(f"  SINR: N/A (Digital SIC skipped)")
    except Exception as e:
        print(f"✗ Failed to load results: {e}")
        return
    
    # Generate figure
    print(f"\nGenerating {'compact' if args.compact else 'professional'} style figure...")
    try:
        if args.compact:
            fig = create_compact_figure(results, args.output, args.dpi)
        else:
            fig = create_professional_figure(results, args.output, args.dpi)
    except Exception as e:
        print(f"✗ Failed to generate figure: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n✓ Visualization complete!")
    print(f"  Output: {args.output}")
    print(f"  Resolution: {args.dpi} DPI")


if __name__ == '__main__':
    main()