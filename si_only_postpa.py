# si_only_postpa.py
# SI-only test for post-PA reference analog SIC (Aux-TX)
# - Reproducible runs via --seed
# - Keeps channel / aux mismatch / estimation noise deterministic per run_case
# - Prints suppression + key debug fields
#
# Usage:
#   python si_only_postpa.py --snr-db 22 --rsi-scale 20 --seed 0
#
# Notes:
# - This script assumes:
#     1) bridge_tx/x_tx.npy exists (self TX waveform)
#     2) sdd_channel_model_v5.py provides simulate_full_receive_signal(...)
# - SI-only means x_remote = 0

import argparse
import os
import numpy as np

from sdd_channel_model_v5 import simulate_full_receive_signal


def power(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x) ** 2) + 1e-18)


def suppression_db(y_before: np.ndarray, y_after: np.ndarray) -> float:
    pb = power(y_before)
    pa = power(y_after)
    return float(10.0 * np.log10(pb / (pa + 1e-18)))


def load_x_self(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 x_tx.npy: {path}")
    x = np.load(path)
    x = x.astype(np.complex128)
    if x.ndim != 1:
        x = x.reshape(-1).astype(np.complex128)
    return x


def print_debug(info: dict):
    print("\n=== DEBUG analog_sic_info ===")
    keys = [
        "mode",
        "AUX_DISABLE_IQPA",
        "ASIC_L", "ASIC_P", "ASIC_NSYM",
        "BO_DB",
        "enable_pa_nonlinearity",
        "alpha_real", "alpha_imag",
        "safety_flip",
        "P_rsi_before", "P_rsi_after",
        "analog_supp_db",
        "h_aux_flat_mag_db",
        "g_err", "p_err_rad",
        "EST_SNR_DB",
    ]
    for k in keys:
        if k in info:
            print(f"{k}: {info[k]}")
    # 若你想看有哪些欄位可用，取消下行註解：
    # print("keys:", sorted(list(info.keys())))


def run_case(
    x_self: np.ndarray,
    snr_db: float,
    rsi_scale: float,
    seed: int,
    enable_pa: bool,
    show_debug: bool = True,
):
    # 讓每次 run_case 都完全可重現
    # 若你希望 PA=ON 與 PA=OFF 用「同一組 channel/aux/noise」做公平比較，
    # 就讓兩個 case 使用同一個 seed（這裡預設就是同一個 seed）。
    np.random.seed(seed)

    x_remote = np.zeros_like(x_self, dtype=np.complex128)

    result = simulate_full_receive_signal(
        x_remote=x_remote,
        x_self=x_self,
        snr_db=snr_db,
        rsi_scale=rsi_scale,
        sic_db=0.0,  # SI-only + realistic analog SIC 時這個通常不使用
        main_channel_type="flat",
        rsi_channel_type="rayleigh",
        use_realistic_analog_sic=True,
        enable_pa_nonlinearity=enable_pa,
    )

    y_before = result["y_rsi_before_analog"]
    y_after = result["y_rsi_after_analog"]

    pb = power(y_before)
    pa = power(y_after)
    supp = suppression_db(y_before, y_after)

    tag = "PA=ON" if enable_pa else "PA=OFF"
    print(f"\n[{tag}]")
    print(f"P_before ≈ {pb:.6g}")
    print(f"P_after  ≈ {pa:.6g}")
    print(f"supp ≈ {supp:.3f} dB")

    if show_debug:
        info = result.get("analog_sic_info", {})
        print_debug(info)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr-db", type=float, default=22.0)
    parser.add_argument("--rsi-scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--x-tx-path",
        type=str,
        default=os.path.join("bridge_tx", "x_tx.npy"),
        help="self TX waveform path (npy)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="both",
        choices=["on", "off", "both"],
        help="run only PA=ON, only PA=OFF, or both",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print analog_sic_info debug fields",
    )
    args = parser.parse_args()

    x_self = load_x_self(args.x_tx_path)
    print(f"Loaded x_self: N={len(x_self)} from {args.x_tx_path}")
    print(f"Args: snr_db={args.snr_db}, rsi_scale={args.rsi_scale}, seed={args.seed}")

    show_debug = bool(args.debug)

    # 預設跑 both；兩個 case 用同一個 seed，確保 channel/aux/noise 一致
    # 如果你希望 ON/OFF 各自獨立但仍可重現，可改成 seed+0 / seed+1
    if args.only in ["both", "on"]:
        run_case(
            x_self=x_self,
            snr_db=args.snr_db,
            rsi_scale=args.rsi_scale,
            seed=args.seed,
            enable_pa=True,
            show_debug=show_debug,
        )

    if args.only in ["both", "off"]:
        run_case(
            x_self=x_self,
            snr_db=args.snr_db,
            rsi_scale=args.rsi_scale,
            seed=args.seed,
            enable_pa=False,
            show_debug=show_debug,
        )


if __name__ == "__main__":
    main()