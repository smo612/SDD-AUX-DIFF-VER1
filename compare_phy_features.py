import os
import numpy as np


def load_complex_npy(path):
    x = np.load(path)

    # 若原本就是 complex
    if np.iscomplexobj(x):
        return x.reshape(-1).astype(np.complex64)

    # 若是最後一維為 2，視為 [real, imag]
    if x.ndim >= 1 and x.shape[-1] == 2:
        xr = x[..., 0]
        xi = x[..., 1]
        return (xr + 1j * xi).reshape(-1).astype(np.complex64)

    raise ValueError(f"Unsupported array format in {path}, shape={x.shape}, dtype={x.dtype}")


def mse(a, b):
    return np.mean(np.abs(a - b) ** 2)


def nmse_db(a, b):
    err = np.mean(np.abs(a - b) ** 2)
    ref = np.mean(np.abs(b) ** 2) + 1e-12
    return 10 * np.log10(err / ref + 1e-12)


def power_db(x):
    p = np.mean(np.abs(x) ** 2) + 1e-12
    return 10 * np.log10(p)


def complex_corr(a, b):
    num = np.vdot(a, b)
    den = np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real) + 1e-12
    return num / den


def print_stats(name, x):
    print(f"[{name}]")
    print(f"  length      : {len(x)}")
    print(f"  power       : {power_db(x):.4f} dB")
    print(f"  mean(|x|)   : {np.mean(np.abs(x)):.6e}")
    print(f"  max(|x|)    : {np.max(np.abs(x)):.6e}")
    print()


def compare_pair(name_a, a, name_b, b):
    m = mse(a, b)
    n = nmse_db(a, b)
    c = complex_corr(a, b)

    print(f"{name_a} vs {name_b}")
    print(f"  MSE         : {m:.6e}")
    print(f"  NMSE        : {n:.4f} dB")
    print(f"  Corr(|rho|) : {np.abs(c):.6f}")
    print(f"  Corr(angle) : {np.angle(c):.6f} rad")
    print()


def main():
    target_path = "bridge_tx_remote/x_tx.npy"
    noisy_path = "bridge/y_adc.npy"
    denoised_path = "bridge_digital/y_clean.npy"

    print("=== Compare PHY Features ===")
    print(f"target   : {target_path}")
    print(f"noisy    : {noisy_path}")
    print(f"denoised : {denoised_path}")
    print()

    for p in [target_path, noisy_path, denoised_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    target = load_complex_npy(target_path)
    noisy = load_complex_npy(noisy_path)
    denoised = load_complex_npy(denoised_path)

    min_len = min(len(target), len(noisy), len(denoised))
    target = target[:min_len]
    noisy = noisy[:min_len]
    denoised = denoised[:min_len]

    print(f"Aligned length: {min_len}")
    print()

    print_stats("target", target)
    print_stats("noisy", noisy)
    print_stats("denoised", denoised)

    compare_pair("noisy", noisy, "target", target)
    compare_pair("denoised", denoised, "target", target)
    compare_pair("denoised", denoised, "noisy", noisy)

    mse_noisy = mse(noisy, target)
    mse_denoised = mse(denoised, target)

    improve_ratio = (mse_noisy - mse_denoised) / (mse_noisy + 1e-12)
    improve_db = 10 * np.log10((mse_noisy + 1e-12) / (mse_denoised + 1e-12))

    print("=== Summary ===")
    print(f"MSE(noisy, target)      = {mse_noisy:.6e}")
    print(f"MSE(denoised, target)   = {mse_denoised:.6e}")
    print(f"Relative improvement    = {improve_ratio * 100:.2f} %")
    print(f"MSE gain                = {improve_db:.4f} dB")
    print()

    if mse_denoised < mse_noisy:
        print("Result: denoised 比 noisy 更接近 target")
    elif mse_denoised > mse_noisy:
        print("Result: denoised 比 noisy 更遠離 target")
    else:
        print("Result: denoised 和 noisy 幾乎一樣")


if __name__ == "__main__":
    main()