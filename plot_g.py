import re
import matplotlib.pyplot as plt
import os

print("🚀 啟動繪圖腳本...")

# 1. 檢查檔案是否存在
if not os.path.exists("g.txt"):
    print("❌ 錯誤：在當前目錄找不到 g.txt！")
    exit()

print("📄 找到 g.txt，開始解析數據...")

bo_data = {}
current_bo = None
current_p = None
parse_count = 0

# 2. 讀取並解析 (使用 search 避開隱藏字元)
with open("g.txt", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        
        m_bo = re.search(r"BO\s*=\s*(\d+)\s*dB", line)
        if m_bo:
            current_bo = int(m_bo.group(1))
            if current_bo not in bo_data:
                bo_data[current_bo] = {}
            continue

        m_p = re.search(r"P\s*=\s*(\d+)", line)
        if m_p:
            current_p = int(m_p.group(1))
            continue

        m_can = re.search(r"Cancellation\s*=\s*([\d\.]+)\s*dB", line)
        if m_can and current_bo is not None and current_p is not None:
            val = float(m_can.group(1))
            bo_data[current_bo][current_p] = val
            parse_count += 1

print(f"✅ 解析完成！共抓取到 {parse_count} 筆有效數據。")

if parse_count == 0:
    print("⚠️ 警告：沒有抓到任何數據，請檢查 g.txt 的內容格式是否正確！")
    exit()

# 3. 開始畫圖
print("📊 開始繪製折線圖...")
p_levels = [1, 3, 5, 7]
bo_list = sorted(bo_data.keys())

plt.figure(figsize=(9, 6))

styles = {
    1: {'color': 'gray', 'marker': 'o', 'label': 'P = 1 (Linear)'},
    3: {'color': 'royalblue', 'marker': 's', 'label': 'P = 3'},
    5: {'color': 'darkorange', 'marker': '^', 'label': 'P = 5'},
    7: {'color': 'firebrick', 'marker': 'd', 'label': 'P = 7 (High Non-Linear)'}
}

for p in p_levels:
    y_vals = [bo_data[bo].get(p, None) for bo in bo_list]
    valid_bo = [bo for bo, y in zip(bo_list, y_vals) if y is not None]
    valid_y = [y for y in y_vals if y is not None]
    
    if valid_y:
        plt.plot(valid_bo, valid_y, marker=styles[p]['marker'], color=styles[p]['color'], 
                 linewidth=2.5, markersize=8, label=styles[p]['label'])

plt.xlabel("Power Amplifier Back-Off (dB)", fontsize=13, fontweight='bold')
plt.ylabel("Theoretical Cancellation Limit (dB)", fontsize=13, fontweight='bold')
plt.title("ASIC Theoretical Capacity (C++ Simulation)", fontsize=15, fontweight='bold')

plt.gca().invert_xaxis()
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=11, loc="lower left")
plt.tight_layout()

plt.savefig("plot_ASIC_theory.png", dpi=300)
print("🎉 成功生成圖表！請打開 plot_ASIC_theory.png 查看。")