import os
import subprocess
import json

def main():
    t_starts = [50, 100, 150, 200, 250]
    results = {}

    print("🚀 啟動 GP6: 擴散模型 T_start 參數全自動掃描 (Inference Sweep)")
    print("="*60)

    for t in t_starts:
        print(f"\n▶ 正在測試 T_start = {t} ...")
        
        # 設定環境變數傳遞給 run_diffusion.py
        env = os.environ.copy()
        env["T_START"] = str(t)
        
        # 執行端到端模擬 (加入 --use-diffusion 觸發 AI-SIC)
        cmd = "python run_sdd_final.py --local kodim01 --remote kodim15 --aux-disable-iqpa False --rsi-scale 3150000 --use-diffusion"
        
        result = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 執行失敗 (T_start={t})")
            print(result.stderr)
            results[t] = "Error"
            continue
            
        # 從 RX 端讀取最終 PSNR
        try:
            with open('bridge_rx/metrics_remote.json', 'r') as f:
                metrics = json.load(f)
                psnr = metrics.get('psnr', -1.0)
                results[t] = psnr
                print(f"✅ 完成！取得 PSNR = {psnr:.2f} dB")
        except Exception as e:
            print(f"⚠️ 無法讀取 PSNR 結果: {e}")
            results[t] = "N/A"

    print("\n" + "="*60)
    print("📊 GP6 Sweep 最終測試結果總結")
    print("="*60)
    for t in t_starts:
        val = results[t]
        if isinstance(val, float):
            print(f"  T_start = {t:<4} | PSNR = {val:.2f} dB")
        else:
            print(f"  T_start = {t:<4} | PSNR = {val}")
    
    print("-" * 60)
    print("  MP Baseline | PSNR = 27.72 dB")
    print("="*60)
    print("💡 結論：如果上述 T_start 都沒有突破 27.72 dB，代表 2D 模型遇到物理極限，請直接準備改 1D！")

if __name__ == "__main__":
    main()