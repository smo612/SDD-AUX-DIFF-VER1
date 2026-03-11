#!/usr/bin/env python3
"""
SDD Results Comparison Visualization Script (V6.5版)

用於對比 WLLS 和 MP 兩種 Digital SIC Backend 的性能差異

✅ 核心功能：
1. 並排顯示：原始圖 | WLLS重建 | MP重建
2. SINR演進對比（WLLS vs MP）
3. 關鍵指標對比表格（Digital Supp, PSNR等）
4. 自動計算MP相對WLLS的增益

使用方式：
    # 基本用法（假設結果在預設目錄）
    python visualize_v65_comparison.py
    
    # 指定結果目錄
    python visualize_v65_comparison.py --wlls-dir results_wlls --mp-dir results_mp
    
    # 指定輸出檔案
    python visualize_v65_comparison.py --output wlls_vs_mp_comparison.png --dpi 300
    
    # 生成緊湊版（適合論文）
    python visualize_v65_comparison.py --compact --output paper_fig.png

工作流程：
    # 先運行WLLS測試
    python run_sdd_e2e_v65.py --local kodim01 --remote kodim24 --backend wlls --rsi-scale 20
    # 備份結果
    mv bridge_rx bridge_rx_wlls
    mv bridge_digital bridge_digital_wlls
    
    # 再運行MP測試
    python run_sdd_e2e_v65.py --local kodim01 --remote kodim24 --backend mp --rsi-scale 20 --skip-tx
    # 備份結果
    mv bridge_rx bridge_rx_mp
    mv bridge_digital bridge_digital_mp
    
    # 生成對比圖
    python visualize_v65_comparison.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path
from PIL import Image
import argparse


def safe_format(value, format_spec='.2f', unit='', na_text='N/A'):
    """安全地格式化數值"""
    if value is None or value == 'N/A':
        return na_text + (f' {unit}' if unit else '')
    
    if isinstance(value, (int, float)):
        try:
            formatted = f"{value:{format_spec}}"
            return formatted + (f' {unit}' if unit else '')
        except:
            return na_text + (f' {unit}' if unit else '')
    
    return str(value) + (f' {unit}' if unit else '')


def load_single_backend_results(rx_dir, digital_dir, analog_dir='bridge'):
    """載入單個backend的結果"""
    results = {}
    
    # 載入圖像
    img_recon_path = Path(rx_dir) / 'img_recon_remote.png'
    comparison_path = Path(rx_dir) / 'comparison_remote.png'
    
    if img_recon_path.exists():
        results['img_recon'] = np.array(Image.open(img_recon_path))
    
    if comparison_path.exists():
        comparison = np.array(Image.open(comparison_path))
        h, w = comparison.shape[:2]
        results['img_original'] = comparison[:, :w//2]
    
    # 載入metrics
    with open(Path(rx_dir) / 'metrics_remote.json') as f:
        results['rx_metrics'] = json.load(f)
    
    with open(Path(digital_dir) / 'metrics.json') as f:
        results['digital_metrics'] = json.load(f)
    
    with open(Path(analog_dir) / 'meta.json') as f:
        results['analog_meta'] = json.load(f)
    
    return results


def load_comparison_results(wlls_rx='bridge_rx_wlls', wlls_digital='bridge_digital_wlls',
                            mp_rx='bridge_rx_mp', mp_digital='bridge_digital_mp',
                            analog_dir='bridge'):
    """載入WLLS和MP的對比結果"""
    print("載入 WLLS 結果...")
    wlls_results = load_single_backend_results(wlls_rx, wlls_digital, analog_dir)
    
    print("載入 MP 結果...")
    mp_results = load_single_backend_results(mp_rx, mp_digital, analog_dir)
    
    return {
        'wlls': wlls_results,
        'mp': mp_results
    }


def create_comparison_figure(results, output_path='wlls_vs_mp_comparison.png', dpi=300):
    """
    建立專業對比圖（V6.5版）
    
    Layout:
    - Row 1: 原始圖 | WLLS重建 | MP重建（3張並排）
    - Row 2: SINR演進對比（左）| 關鍵指標表格（右）
    """
    wlls = results['wlls']
    mp = results['mp']
    
    # 字體設定
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    
    # 建立圖形
    fig = plt.figure(figsize=(16, 10), dpi=dpi)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1.2], hspace=0.25, wspace=0.25)
    
    # ==================== Row 1: 圖像對比 ====================
    # 原始圖
    ax1 = fig.add_subplot(gs[0, 0])
    if 'img_original' in wlls:
        ax1.imshow(wlls['img_original'])
        ax1.set_title('Original Image\n(Remote/Desired)', fontsize=12, fontweight='bold')
        ax1.axis('off')
    
    # WLLS重建
    ax2 = fig.add_subplot(gs[0, 1])
    if 'img_recon' in wlls:
        ax2.imshow(wlls['img_recon'])
        psnr_wlls = wlls['rx_metrics']['psnr']
        ms_ssim_wlls = wlls['rx_metrics'].get('ms_ssim', 0)
        title = f'WLLS Reconstruction\nPSNR: {psnr_wlls:.2f} dB | MS-SSIM: {ms_ssim_wlls:.4f}'
        ax2.set_title(title, fontsize=12, fontweight='bold', color='darkred')
        ax2.axis('off')
    
    # MP重建
    ax3 = fig.add_subplot(gs[0, 2])
    if 'img_recon' in mp:
        ax3.imshow(mp['img_recon'])
        psnr_mp = mp['rx_metrics']['psnr']
        ms_ssim_mp = mp['rx_metrics'].get('ms_ssim', 0)
        
        # 計算增益
        psnr_gain = psnr_mp - psnr_wlls
        title = f'MP Reconstruction\nPSNR: {psnr_mp:.2f} dB | MS-SSIM: {ms_ssim_mp:.4f}\n(+{psnr_gain:.2f} dB over WLLS)'
        ax3.set_title(title, fontsize=12, fontweight='bold', color='darkgreen')
        ax3.axis('off')
    
    # ==================== Row 2 Left: SINR演進對比 ====================
    ax4 = fig.add_subplot(gs[1, :2])
    
    # WLLS資料
    wlls_analog = wlls['analog_meta']
    wlls_digital = wlls['digital_metrics']
    wlls_sinr = [
        wlls_analog.get('SINR_pre', 0),
        wlls_analog.get('SINR_analog', 0),
        wlls_digital.get('SINR_after_digital', 0)
    ]
    
    # MP資料
    mp_analog = mp['analog_meta']
    mp_digital = mp['digital_metrics']
    mp_sinr = [
        mp_analog.get('SINR_pre', 0),
        mp_analog.get('SINR_analog', 0),
        mp_digital.get('SINR_after_digital', 0)
    ]
    
    stages = ['Before\nAnalog', 'After\nAnalog', 'After\nDigital']
    x_pos = np.arange(len(stages))
    width = 0.35
    
    # 繪製並排柱狀圖
    bars_wlls = ax4.bar(x_pos - width/2, wlls_sinr, width, 
                        label='WLLS', color='darkred', alpha=0.7, edgecolor='black')
    bars_mp = ax4.bar(x_pos + width/2, mp_sinr, width,
                      label='MP', color='darkgreen', alpha=0.7, edgecolor='black')
    
    # 標註數值
    for bars in [bars_wlls, bars_mp]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(stages, fontsize=11, fontweight='bold')
    ax4.set_ylabel('SINR (dB)', fontsize=11, fontweight='bold')
    ax4.set_title('SINR Evolution Comparison (WLLS vs MP)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # 標註Digital增益差異
    digital_gain_wlls = wlls_sinr[2] - wlls_sinr[1]
    digital_gain_mp = mp_sinr[2] - mp_sinr[1]
    gain_diff = digital_gain_mp - digital_gain_wlls
    
    ax4.text(2, max(wlls_sinr[2], mp_sinr[2]) + 2,
            f'MP Digital Gain: +{digital_gain_mp:.1f} dB\nWLLS Digital Gain: +{digital_gain_wlls:.1f} dB\nDifference: +{gain_diff:.1f} dB',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # ==================== Row 2 Right: 對比表格 ====================
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # 準備表格資料
    wlls_digital_supp = wlls_digital.get('Digital_supp_si', 0)
    mp_digital_supp = mp_digital.get('Digital_supp_si', 0)
    digital_supp_gain = mp_digital_supp - wlls_digital_supp
    
    wlls_total_supp = wlls_digital.get('Total_supp_SI_only', 0)
    mp_total_supp = mp_digital.get('Total_supp_SI_only', 0)
    total_supp_gain = mp_total_supp - wlls_total_supp
    
    psnr_wlls = wlls['rx_metrics']['psnr']
    psnr_mp = mp['rx_metrics']['psnr']
    psnr_gain = psnr_mp - psnr_wlls
    
    ms_ssim_wlls = wlls['rx_metrics'].get('ms_ssim', 0)
    ms_ssim_mp = mp['rx_metrics'].get('ms_ssim', 0)
    ms_ssim_gain = ms_ssim_mp - ms_ssim_wlls
    
    # 通道配置（從WLLS或MP任一取得，應相同）
    rsi_scale = wlls_analog.get('rsi_scale', 'N/A')
    snr_db = wlls_analog.get('snr_db', 'N/A')
    analog_supp = wlls_analog.get('Supp_analog', 0)
    
    table_data = [
        ['Metric', 'WLLS', 'MP', 'MP Gain'],
        ['Digital Supp.', f'{wlls_digital_supp:.2f} dB', f'{mp_digital_supp:.2f} dB', 
         f'+{digital_supp_gain:.2f} dB'],
        ['Total Supp.', f'{wlls_total_supp:.2f} dB', f'{mp_total_supp:.2f} dB',
         f'+{total_supp_gain:.2f} dB'],
        ['PSNR', f'{psnr_wlls:.2f} dB', f'{psnr_mp:.2f} dB',
         f'+{psnr_gain:.2f} dB'],
        ['MS-SSIM', f'{ms_ssim_wlls:.4f}', f'{ms_ssim_mp:.4f}',
         f'+{ms_ssim_gain:.4f}'],
        ['', '', '', ''],
        ['Channel Config', '', '', ''],
        ['RSI_SCALE', f'{rsi_scale}', '', ''],
        ['SNR', f'{snr_db} dB', '', ''],
        ['Analog Supp.', f'{analog_supp:.2f} dB', '', ''],
    ]
    
    # 繪製表格
    table = ax5.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # 表頭樣式
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')
    
    # 高亮增益行
    for i in [1, 2, 3, 4]:
        cell = table[(i, 3)]
        if 'gain' in table_data[i][3].lower() or '+' in table_data[i][3]:
            cell.set_facecolor('#2ecc71')
            cell.set_text_props(weight='bold')
    
    # 通道配置標題
    cell = table[(6, 0)]
    cell.set_facecolor('#95a5a6')
    cell.set_text_props(weight='bold')
    
    # 交替行顏色
    for i in range(1, 6):
        for j in range(3):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
    
    # ==================== 總標題 ====================
    analog_status = "Normal" if not wlls_analog.get('analog_sic_info', {}).get('saturated', False) else "Saturated"
    
    fig.suptitle(f'SDD V6.5: WLLS vs MP Comparison (RSI_SCALE={rsi_scale}, Analog SIC: {analog_status})',
                fontsize=16, fontweight='bold', y=0.98)
    
    # 保存
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ 對比圖已保存: {output_path}")
    
    return fig


def create_compact_comparison(results, output_path='wlls_vs_mp_compact.png', dpi=300):
    """
    建立緊湊版對比圖（適合論文）
    
    Layout: 原始 | WLLS | MP（單行，附metrics）
    """
    wlls = results['wlls']
    mp = results['mp']
    
    fig = plt.figure(figsize=(15, 5), dpi=dpi)
    gs = GridSpec(1, 3, figure=fig, wspace=0.15)
    
    # 原始圖
    ax1 = fig.add_subplot(gs[0, 0])
    if 'img_original' in wlls:
        ax1.imshow(wlls['img_original'])
        ax1.set_title('Original', fontsize=14, fontweight='bold')
        ax1.axis('off')
    
    # WLLS
    ax2 = fig.add_subplot(gs[0, 1])
    if 'img_recon' in wlls:
        ax2.imshow(wlls['img_recon'])
        psnr = wlls['rx_metrics']['psnr']
        digital_supp = wlls['digital_metrics'].get('Digital_supp_si', 0)
        ax2.set_title(f'WLLS\nPSNR: {psnr:.2f} dB\nDigital: {digital_supp:.2f} dB',
                     fontsize=14, fontweight='bold', color='darkred')
        ax2.axis('off')
    
    # MP
    ax3 = fig.add_subplot(gs[0, 2])
    if 'img_recon' in mp:
        ax3.imshow(mp['img_recon'])
        psnr = mp['rx_metrics']['psnr']
        digital_supp = mp['digital_metrics'].get('Digital_supp_si', 0)
        
        psnr_gain = psnr - wlls['rx_metrics']['psnr']
        digital_gain = digital_supp - wlls['digital_metrics'].get('Digital_supp_si', 0)
        
        ax3.set_title(f'MP\nPSNR: {psnr:.2f} dB (+{psnr_gain:.2f})\nDigital: {digital_supp:.2f} dB (+{digital_gain:.2f})',
                     fontsize=14, fontweight='bold', color='darkgreen')
        ax3.axis('off')
    
    # 總標題
    rsi_scale = wlls['analog_meta'].get('rsi_scale', 'N/A')
    fig.suptitle(f'WLLS vs MP Comparison (RSI_SCALE={rsi_scale})',
                fontsize=16, fontweight='bold')
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ 緊湊版對比圖已保存: {output_path}")
    
    return fig


def print_comparison_summary(results):
    """打印對比摘要"""
    wlls = results['wlls']
    mp = results['mp']
    
    print("\n" + "="*70)
    print("📊 WLLS vs MP 對比摘要")
    print("="*70)
    
    # 通道配置
    rsi_scale = wlls['analog_meta'].get('rsi_scale', 'N/A')
    snr_db = wlls['analog_meta'].get('snr_db', 'N/A')
    print(f"\n【通道配置】")
    print(f"  RSI_SCALE: {rsi_scale}")
    print(f"  SNR: {snr_db} dB")
    
    # Digital SIC性能
    wlls_digital_supp = wlls['digital_metrics'].get('Digital_supp_si', 0)
    mp_digital_supp = mp['digital_metrics'].get('Digital_supp_si', 0)
    digital_gain = mp_digital_supp - wlls_digital_supp
    
    print(f"\n【Digital SIC 抑制】")
    print(f"  WLLS: {wlls_digital_supp:.2f} dB")
    print(f"  MP:   {mp_digital_supp:.2f} dB")
    print(f"  增益:  +{digital_gain:.2f} dB {'✅' if digital_gain > 5 else '⚠️'}")
    
    # 圖像品質
    psnr_wlls = wlls['rx_metrics']['psnr']
    psnr_mp = mp['rx_metrics']['psnr']
    psnr_gain = psnr_mp - psnr_wlls
    
    ms_ssim_wlls = wlls['rx_metrics'].get('ms_ssim', 0)
    ms_ssim_mp = mp['rx_metrics'].get('ms_ssim', 0)
    ms_ssim_gain = ms_ssim_mp - ms_ssim_wlls
    
    print(f"\n【圖像品質】")
    print(f"  PSNR (WLLS): {psnr_wlls:.2f} dB")
    print(f"  PSNR (MP):   {psnr_mp:.2f} dB")
    print(f"  PSNR增益:     +{psnr_gain:.2f} dB {'✅' if psnr_gain > 1 else '⚠️'}")
    print(f"  MS-SSIM增益:  +{ms_ssim_gain:.4f}")
    
    # 結論
    print(f"\n【結論】")
    if digital_gain > 10 and psnr_gain > 2:
        print("  ✅ MP顯著優於WLLS，非線性抑制效果明顯")
    elif digital_gain > 5 and psnr_gain > 1:
        print("  ✅ MP優於WLLS，證明非線性SIC的必要性")
    else:
        print("  ⚠️  增益較小，可能非線性不強或參數需調整")


def main():
    parser = argparse.ArgumentParser(
        description='SDD V6.5版：WLLS vs MP 對比可視化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用範例：
  # 基本用法（預設目錄）
  %(prog)s
  
  # 指定結果目錄
  %(prog)s --wlls-dir results_wlls --mp-dir results_mp
  
  # 生成緊湊版
  %(prog)s --compact --output paper_comparison.png
        '''
    )
    
    # 結果目錄
    parser.add_argument('--wlls-rx', type=str, default='bridge_rx_wlls',
                       help='WLLS RX輸出目錄')
    parser.add_argument('--wlls-digital', type=str, default='bridge_digital_wlls',
                       help='WLLS Digital SIC輸出目錄')
    parser.add_argument('--mp-rx', type=str, default='bridge_rx_mp',
                       help='MP RX輸出目錄')
    parser.add_argument('--mp-digital', type=str, default='bridge_digital_mp',
                       help='MP Digital SIC輸出目錄')
    parser.add_argument('--analog', type=str, default='bridge',
                       help='Analog階段輸出目錄（共用）')
    
    # 輸出設定
    parser.add_argument('--output', type=str, default='wlls_vs_mp_comparison.png',
                       help='輸出檔案路徑')
    parser.add_argument('--dpi', type=int, default=300,
                       help='圖片DPI')
    parser.add_argument('--compact', action='store_true',
                       help='生成緊湊版（適合論文）')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SDD V6.5版：WLLS vs MP 對比可視化")
    print("="*70)
    
    # 載入結果
    print("\n載入結果...")
    try:
        results = load_comparison_results(
            args.wlls_rx, args.wlls_digital,
            args.mp_rx, args.mp_digital,
            args.analog
        )
        print("✓ 結果載入完成")
    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        print("\n請確認：")
        print("  1. 已運行WLLS測試並備份結果到 bridge_rx_wlls/ 和 bridge_digital_wlls/")
        print("  2. 已運行MP測試並備份結果到 bridge_rx_mp/ 和 bridge_digital_mp/")
        print("  3. bridge/ 目錄包含Analog階段結果")
        return
    
    # 打印摘要
    print_comparison_summary(results)
    
    # 生成圖表
    print(f"\n生成{'緊湊版' if args.compact else '完整版'}對比圖...")
    try:
        if args.compact:
            fig = create_compact_comparison(results, args.output, args.dpi)
        else:
            fig = create_comparison_figure(results, args.output, args.dpi)
    except Exception as e:
        print(f"❌ 生成失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n✓ 可視化完成！")
    print(f"  輸出: {args.output}")
    print(f"  解析度: {args.dpi} DPI")


if __name__ == '__main__':
    main()