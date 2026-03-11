"""
sdd_channel_model_v5.py - 最終版類比通道模型 (透明視窗版)
新增功能: 回傳 Fig. 1 各個 Block 的中間波形 (Debug Waveforms)
"""

import numpy as np
import config as C

# ... (保留原本的 helper functions: _rand_rayleigh_taps, _circ_conv 等，不需要改) ...
def _rand_rayleigh_taps(L: int) -> np.ndarray:
    h = (np.random.randn(L) + 1j*np.random.randn(L)).astype(np.complex64)
    h /= np.sqrt(np.sum(np.abs(h)**2) + 1e-18)
    return h

def _rand_rician_taps(L: int, K_dB: float) -> np.ndarray:
    K_lin = 10 ** (K_dB / 10.0)
    h_los = np.sqrt(K_lin / (K_lin + 1))
    h_nlos = (np.random.randn(L) + 1j * np.random.randn(L)).astype(np.complex64)
    h_nlos /= np.sqrt(np.sum(np.abs(h_nlos)**2) + 1e-18)
    h_nlos *= np.sqrt(1.0 / (K_lin + 1))
    h = h_nlos.copy()
    h[0] += h_los
    h /= np.sqrt(np.sum(np.abs(h)**2) + 1e-18)
    return h

def _circ_conv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.complex64)
    for k in range(len(h)):
        y += h[k] * np.roll(x, k)
    return y

def _iq_imbalance_widely_linear(x: np.ndarray, ambal: float, ph_deg: float) -> np.ndarray:
    phi = np.deg2rad(ph_deg)
    a = (1.0 + ambal) * np.exp(1j*phi/2)
    b = ambal * np.exp(-1j*phi/2) * 0.5
    return (a*x + b*np.conjugate(x)).astype(np.complex64)

def _rapp_pa(x: np.ndarray, p: float, Asat: float) -> np.ndarray:
    a = np.abs(x)
    ang = np.angle(x)
    # AM-AM
    gain = a / ((1.0 + (a/Asat)**(2.0*p))**(1.0/(2.0*p)) + 1e-18)
    # AM-PM
    phase_shift = 0.08 * (a/Asat)**2
    return (gain * np.exp(1j*(ang + phase_shift))).astype(np.complex64)

def simulate_full_receive_signal(x_remote: np.ndarray,
                                 x_self:   np.ndarray,
                                 snr_db:   float,
                                 rsi_scale: float,
                                 sic_db:   float,
                                 main_channel_type: str = 'flat',
                                 main_K_dB: float = 5.0,
                                 rsi_channel_type: str = 'rayleigh',
                                 rsi_K_dB: float = 0.0,
                                 use_realistic_analog_sic: bool = True,
                                 enable_pa_nonlinearity: bool = True):
    """
    完整接收訊號模擬 (含中間波形輸出)
    """
    x_remote = x_remote.astype(np.complex64)
    x_self   = x_self.astype(np.complex64)
    N = len(x_remote)
    
    # 儲存波形的容器
    waveforms = {}
    waveforms['input'] = x_self.copy()

    # 1. Main Channel (期望訊號)
    # ... (省略 Main Channel 計算，與原本相同，請保留) ...
    if main_channel_type == 'flat':
        y_main_no_noise = x_remote.copy()
        h_main = np.array([1.0], dtype=np.complex64)
    elif main_channel_type == 'rician':
        L_main = getattr(C, "MAIN_NUM_TAPS", 3)
        h_main = _rand_rician_taps(L_main, main_K_dB)
        y_main_no_noise = _circ_conv(x_remote, h_main)
    else:
        L_main = getattr(C, "MAIN_NUM_TAPS", 3)
        h_main = _rand_rayleigh_taps(L_main)
        y_main_no_noise = _circ_conv(x_remote, h_main)

    p_sig = float(np.mean(np.abs(y_main_no_noise)**2))
    snr_lin = 10.0**(snr_db/10.0)
    noise_var = p_sig / (snr_lin + 1e-18)
    w = (np.sqrt(noise_var/2.0) * (np.random.randn(N) + 1j*np.random.randn(N))).astype(np.complex64)
    y_main = (y_main_no_noise + w).astype(np.complex64)


    # 2. RSI Channel (自干擾路徑 - Main Path)
    # Block A: IQ Imbalance
    iq_ambal = 0.025
    iq_phase = 2.5
    y_rsi_iq = _iq_imbalance_widely_linear(x_self, iq_ambal, iq_phase)
    waveforms['main_after_iq'] = y_rsi_iq.copy() # Capture!
    
    # Block B: PA Nonlinearity
    ref_amplitude = np.sqrt(np.mean(np.abs(y_rsi_iq)**2) + 1e-12)
    Asat_abs = ref_amplitude * 2.0
    rapp_p_main = 2.2 
    
    if enable_pa_nonlinearity:
        y_rsi_pa = _rapp_pa(y_rsi_iq, rapp_p_main, Asat_abs)
    else:
        y_rsi_pa = y_rsi_iq.copy()
    waveforms['main_after_pa'] = y_rsi_pa.copy() # Capture!
    
    # Block C: Multipath Channel
    if rsi_channel_type == 'rician':
        L_rsi = getattr(C, "RSI_NUM_TAPS", 3)
        h_rsi = _rand_rician_taps(L_rsi, rsi_K_dB)
    else:
        L_rsi = getattr(C, "RSI_NUM_TAPS", 3)
        h_rsi = _rand_rayleigh_taps(L_rsi)
    
    y_rsi_channel = _circ_conv(y_rsi_pa, h_rsi)
    y_rsi_channel = y_rsi_channel * np.sqrt(rsi_scale)
    y_rsi_before_analog = y_rsi_channel.astype(np.complex64)
    waveforms['main_arrived_at_rx'] = y_rsi_before_analog.copy() # Capture! (干擾訊號)

    # 3. Analog SIC (Auxiliary Transmitter Path)
    y_cancellation = np.zeros_like(x_self)
    
    if use_realistic_analog_sic:
        # Saturation Logic... (保留原本邏輯)
        P_main = float(np.mean(np.abs(y_main)**2))
        P_rsi_before = float(np.mean(np.abs(y_rsi_before_analog)**2))
        rsi_to_main_db = 10.0 * np.log10(P_rsi_before / (P_main + 1e-18))
        saturation_threshold_db = 20.0
        if rsi_to_main_db > saturation_threshold_db:
            analog_sic_saturated = True
            actual_matching_db = sic_db - 6.0 
        else:
            analog_sic_saturated = False
            actual_matching_db = sic_db

        # Block D: Aux IQ
        y_aux_iq = _iq_imbalance_widely_linear(x_self, iq_ambal, iq_phase)
        waveforms['aux_after_iq'] = y_aux_iq.copy() # Capture!

        # Block E: Aux PA (Mismatch!)
        rapp_p_aux = 2.4 
        if enable_pa_nonlinearity:
            y_aux_pa = _rapp_pa(y_aux_iq, rapp_p_aux, Asat_abs)
        else:
            y_aux_pa = y_aux_iq.copy()
        waveforms['aux_after_pa'] = y_aux_pa.copy() # Capture!
        
        # Block F: Matching/Filtering
        y_aux_filt = _circ_conv(y_aux_pa, h_rsi) # 假設 Aux 經歷類似的 delay/channel
        y_aux_filt = y_aux_filt * np.sqrt(rsi_scale)
        
        k_match = 1.0 - 10.0**(-actual_matching_db/20.0)
        g_err = 1.0 + np.random.randn() * getattr(C, "ANA_GAIN_ERR_STD", 0.02)
        p_err = np.exp(1j*np.deg2rad(np.random.randn() * getattr(C, "ANA_PHASE_ERR_STD_DEG", 1.5)))
        
        y_cancellation = y_aux_filt * k_match * g_err * p_err
        waveforms['aux_cancellation_signal'] = y_cancellation.copy() # Capture! (消除訊號)
        
        # 相減
        y_rsi_after_analog = y_rsi_before_analog - y_cancellation
        actual_suppression_report = float(actual_matching_db)
        
    else:
        # Ideal...
        k_amp = 10.0**(-sic_db/20.0)
        y_rsi_after_analog = y_rsi_before_analog * k_amp
        analog_sic_saturated = False
        actual_suppression_report = float(sic_db)
        rapp_p_aux = 0.0

    waveforms['residual_after_analog'] = y_rsi_after_analog.copy() # Capture! (殘留訊號)

    # Metadata...
    channel_info = {
        'version': 'v5_final_physics_debug',
        'nonlinearity': {
            'enabled': enable_pa_nonlinearity,
            'rapp_p_main': rapp_p_main if enable_pa_nonlinearity else 0.0,
            'rapp_p_aux': rapp_p_aux if enable_pa_nonlinearity else 0.0,
        }
    }
    
    analog_sic_info = {
        'saturated': analog_sic_saturated if use_realistic_analog_sic else False,
        'matching_accuracy_db': float(actual_suppression_report),
        'architecture': 'Aux-TX Cancellation'
    }

    if use_realistic_analog_sic:
        analog_sic_info['rsi_to_main_db'] = float(10.0 * np.log10(P_rsi_before / (P_main + 1e-18)))

    return {
        "y_main": y_main,
        "y_rsi_before_analog": y_rsi_before_analog,
        "y_rsi_after_analog":  y_rsi_after_analog,
        "noise_var": float(noise_var),
        "channel_info": channel_info,
        "analog_sic_info": analog_sic_info,
        "debug_waveforms": waveforms # ✅ 回傳所有波形
    }