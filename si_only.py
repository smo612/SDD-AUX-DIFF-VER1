#!/usr/bin/env python3
# test_asic_si_only.py
# Minimal SI-only test for Active Analog Cancellation (Aux-TX)
#
# Usage:
#   python test_asic_si_only.py --snr-db 22 --rsi-scale 20
#   python test_asic_si_only.py --snr-db 22 --rsi-scale 20 --n 65536
#
# Notes:
# - Uses bridge_tx/x_tx.npy if present; otherwise generates random QPSK symbols.
# - Sets x_remote = 0 to isolate self-interference cancellation.
# - Prints suppression and sanity correlation diagnostics.

import argparse
from pathlib import Path
import numpy as np

import sdd_channel_model_v5 as ch


def qpsk(N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits_i = rng.integers(0, 2, size=N) * 2 - 1
    bits_q = rng.integers(0, 2, size=N) * 2 - 1
    return (bits_i + 1j * bits_q).astype(np.complex128) / np.sqrt(2)


def power(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x) ** 2) + 1e-18)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr-db", type=float, default=22.0)
    ap.add_argument("--rsi-scale", type=float, default=20.0)
    ap.add_argument("--sic-db", type=float, default=23.0)  # only used by toy mode
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-toy", action="store_true", help="force toy analog (k_amp) instead of realistic ASIC")
    ap.add_argument("--no-pa", action="store_true", help="disable PA nonlinearity (debug)")
    args = ap.parse_args()

    base = Path(".").resolve()
    p_self = base / "bridge_tx" / "x_tx.npy"

    if p_self.exists():
        x_self = np.load(p_self).astype(np.complex128)
        src = str(p_self)
    else:
        x_self = qpsk(args.n, seed=args.seed)
        src = f"generated QPSK (N={args.n})"

    # truncate/shape
    N = min(len(x_self), args.n)
    x_self = x_self[:N].astype(np.complex128)

    # SI-only: desired signal is zero
    x_remote = np.zeros_like(x_self, dtype=np.complex128)

    print("==============================================")
    print("SI-only test (Active Analog Cancellation)")
    print("==============================================")
    print(f"channel file: {ch.__file__}")
    print(f"source x_self: {src}")
    print(f"N={N}, SNR={args.snr_db} dB, RSI_SCALE={args.rsi_scale}")
    print(f"mode: {'toy' if args.use_toy else 'realistic ASIC'} | PA={'off' if args.no_pa else 'on'}")
    print("")

    res = ch.simulate_full_receive_signal(
        x_remote=x_remote,
        x_self=x_self,
        snr_db=args.snr_db,
        rsi_scale=args.rsi_scale,
        sic_db=args.sic_db,
        use_realistic_analog_sic=(not args.use_toy),
        enable_pa_nonlinearity=(not args.no_pa),
    )

    y_before = res["y_rsi_before_analog"]
    y_after = res["y_rsi_after_analog"]
    info = res.get("analog_sic_info", {}) or {}
    wave = res.get("debug_waveforms", {}) or {}

    P_before = power(y_before)
    P_after = power(y_after)
    supp_db = 10 * np.log10(P_before / (P_after + 1e-18))

    print("[Power]")
    print(f"  P_before: {P_before:.6e}")
    print(f"  P_after : {P_after:.6e}")
    print(f"  supp_db : {supp_db:.2f} dB")

    # also show the model's own reported suppression if present
    model_supp = info.get("actual_suppression_db", info.get("analog_supp_db", None))
    if model_supp is not None:
        print(f"  (model) : {model_supp:.2f} dB")

    sinr_pre = info.get("SINR_pre_db", None)
    sinr_post = info.get("SINR_analog_db", None)
    if sinr_pre is not None and sinr_post is not None:
        print("[SINR] (note: desired=0, so SINR is only diagnostic)")
        print(f"  pre : {sinr_pre:.2f} dB")
        print(f"  post: {sinr_post:.2f} dB")

    # correlation sanity: if cancel exists, check if it tends to ADD energy
    y_cancel = wave.get("y_cancel_arrived", wave.get("y_cancellation_arrived", None))
    if isinstance(y_cancel, np.ndarray):
        # Check correlation sign (real part)
        corr = np.vdot(y_before, y_cancel)  # <before, cancel>
        print("[Correlation sanity]")
        print(f"  vdot(before, cancel) = {np.real(corr):.6e} + j{np.imag(corr):.6e}")
        if np.real(corr) > 0:
            print("  ⚠️ real(corr) > 0 suggests cancel tends to be in-phase (may add SI).")
        else:
            print("  ✅ real(corr) <= 0 suggests cancel tends to oppose SI (good sign).")
    else:
        print("[Correlation sanity]")
        print("  (no y_cancel_arrived found in debug_waveforms)")

    # show key ASIC debug fields
    print("[ASIC debug]")
    keys = [
        "mode", "BO_DB", "ASIC_L", "ASIC_P", "ASIC_NSYM", "EST_SNR_DB",
        "alpha_real", "alpha_imag", "safety_flip",
        "h_aux_flat_mag_db", "g_err", "p_err_rad"
    ]
    for k in keys:
        if k in info:
            print(f"  {k}: {info[k]}")

    print("==============================================")
    # Exit code hint (optional)
    if supp_db < 0:
        print("❌ SI-only suppression is NEGATIVE -> still adding SI. Fix estimator/forward-model alignment first.")
    elif supp_db < 10:
        print("⚠️ SI-only suppression is small (<10 dB). Direction ok, but still weak.")
    else:
        print("✅ SI-only suppression looks positive & meaningful.")
    print("==============================================")


if __name__ == "__main__":
    main()