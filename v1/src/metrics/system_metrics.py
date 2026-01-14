# src/metrics/system_metrics.py
"""
System-Level Metrics: BER and EVM

Per advisor directive for TWC:
- Convert estimation-level metrics to system-level
- Show communication impact of tracking errors
- Demonstrate DU advantages in BER/EVM recovery

Implementation:
- Single-carrier QPSK
- 64 data symbols per frame
- Compensation using (τ̂, ν̂, φ̂) estimates

P0-3 Fix: All parameters now read from model.cfg to ensure consistency
with thz_isac_model.h() observation model.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.physics.thz_isac_model import THzISACConfig


@dataclass
class SystemMetricsConfig:
    """Configuration for system-level metrics."""
    
    # Modulation
    modulation: str = 'QPSK'
    
    # Symbols per frame
    n_data_symbols: int = 64
    
    # Symbol period (derived from frame duration / n_data_symbols)
    T_sym: float = 1e-6  # Default, will be overwritten from model.cfg
    
    # Physical parameters - defaults, should be read from model.cfg
    c: float = 3e8  # Speed of light (m/s)
    f_c: float = 300e9  # Carrier frequency (Hz) - THz band
    delay_scale: float = 1e-9  # Delay normalization scale (s)
    doppler_scale: float = 1e3  # Doppler normalization scale (Hz)
    frame_duration: float = 100e-6  # Frame duration (s)
    
    # ISAC mode: 'one_way' or 'two_way'
    isac_mode: str = 'one_way'
    
    @classmethod
    def from_model_cfg(cls, model_cfg: 'THzISACConfig') -> 'SystemMetricsConfig':
        """Create SystemMetricsConfig from THzISACConfig for consistency."""
        return cls(
            f_c=model_cfg.carrier_freq_hz,
            delay_scale=model_cfg.delay_scale,
            doppler_scale=model_cfg.doppler_scale,
            frame_duration=model_cfg.frame_duration_s,
            T_sym=model_cfg.frame_duration_s / 64,  # Assume 64 symbols per frame
        )


def generate_qpsk_symbols(n_symbols: int, rng: np.random.Generator = None) -> np.ndarray:
    """Generate random QPSK symbols."""
    if rng is None:
        rng = np.random.default_rng()
    
    # QPSK constellation: {±1 ± j} / sqrt(2)
    bits = rng.integers(0, 4, size=n_symbols)
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    return constellation[bits], bits


def apply_channel_distortion(
    symbols: np.ndarray,
    tau: float,  # Normalized delay
    nu: float,   # Normalized Doppler
    phi: float,  # Phase (rad, NOT normalized)
    snr_db: float,
    rng: np.random.Generator = None,
    cfg: SystemMetricsConfig = None,
) -> np.ndarray:
    """
    Apply channel distortion to transmitted symbols.
    
    Model consistent with thz_isac_model.h():
    y_k = s_k * exp(j * (phi + 2*pi*nu_phys*k*T_sym)) + noise
    
    NOTE: For data symbols, we use single-carrier model with time-varying phase.
    Delay effect is timing offset (omitted for simplicity in BER/EVM eval).
    
    Args:
        symbols: TX symbols
        tau: Normalized delay (unused in single-carrier, affects timing)
        nu: Normalized Doppler
        phi: Phase in radians (already denormalized, phase_scale=1.0)
        snr_db: SNR in dB
        rng: Random generator
        cfg: SystemMetricsConfig (for physical parameters)
    """
    if rng is None:
        rng = np.random.default_rng()
    if cfg is None:
        cfg = SystemMetricsConfig()
    
    n = len(symbols)
    k = np.arange(n)
    
    # Convert normalized Doppler to physical: nu_phys = nu * doppler_scale (Hz)
    nu_phys = nu * cfg.doppler_scale
    
    # Phase rotation: phi + 2*pi*nu_phys*k*T_sym
    # This is consistent with thz_isac_model.h() which uses 2*pi*nu*t
    phase = phi + 2.0 * np.pi * nu_phys * k * cfg.T_sym
    
    # Apply channel
    y = symbols * np.exp(1j * phase)
    
    # Add noise
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(symbols) ** 2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    
    return y + noise


def compensate_channel(
    y: np.ndarray,
    tau_hat: float,
    nu_hat: float,
    phi_hat: float,
    cfg: SystemMetricsConfig = None,
) -> np.ndarray:
    """
    Compensate channel using estimated parameters.
    
    De-rotation: y_comp = y * exp(-j * (phi_hat + 2*pi*nu_hat_phys*k*T_sym))
    
    Args:
        y: Received symbols
        tau_hat: Estimated normalized delay (unused)
        nu_hat: Estimated normalized Doppler
        phi_hat: Estimated phase (rad)
        cfg: SystemMetricsConfig (for physical parameters)
    """
    if cfg is None:
        cfg = SystemMetricsConfig()
    
    n = len(y)
    k = np.arange(n)
    
    # Convert normalized Doppler to physical
    nu_hat_phys = nu_hat * cfg.doppler_scale
    
    # Estimated phase trajectory
    phase_hat = phi_hat + 2.0 * np.pi * nu_hat_phys * k * cfg.T_sym
    
    # De-rotate
    y_comp = y * np.exp(-1j * phase_hat)
    
    return y_comp


def compute_evm(
    y_comp: np.ndarray,
    s_tx: np.ndarray,
) -> float:
    """
    Compute Error Vector Magnitude (EVM).
    
    EVM = sqrt(mean(|y_comp - s_tx|^2) / mean(|s_tx|^2)) * 100%
    
    Returns:
        EVM in percentage
    """
    error = y_comp - s_tx
    evm = np.sqrt(np.mean(np.abs(error) ** 2) / np.mean(np.abs(s_tx) ** 2)) * 100
    return evm


def compute_ber_qpsk(
    y_comp: np.ndarray,
    bits_tx: np.ndarray,
) -> float:
    """
    Compute Bit Error Rate for QPSK.
    
    QPSK decision regions:
    - I > 0, Q > 0 -> 00
    - I > 0, Q < 0 -> 01
    - I < 0, Q > 0 -> 10
    - I < 0, Q < 0 -> 11
    """
    # Decision
    I_sign = y_comp.real > 0
    Q_sign = y_comp.imag > 0
    
    # Map to bits: 00, 01, 10, 11 -> 0, 1, 2, 3
    bits_rx = (~I_sign).astype(int) * 2 + (~Q_sign).astype(int)
    
    # Count bit errors (2 bits per symbol)
    bit_errors = 0
    for tx, rx in zip(bits_tx, bits_rx):
        # XOR and count bits
        diff = tx ^ rx
        bit_errors += bin(diff).count('1')
    
    ber = bit_errors / (2 * len(bits_tx))
    return ber


def estimate_to_physical(
    tau_norm: float,
    nu_norm: float,
    cfg: SystemMetricsConfig,
) -> Tuple[float, float]:
    """
    Convert normalized estimates to physical units.
    
    Args:
        tau_norm: Normalized delay
        nu_norm: Normalized Doppler
        cfg: System configuration (with delay_scale, doppler_scale from model.cfg)
        
    Returns:
        range_m: Range in meters
        velocity_ms: Velocity in m/s
    """
    # Denormalize using scales from config
    tau_actual = tau_norm * cfg.delay_scale  # seconds
    
    if cfg.isac_mode == 'one_way':
        range_m = cfg.c * tau_actual
    else:  # two_way
        range_m = cfg.c * tau_actual / 2
    
    # Doppler: nu_norm * doppler_scale = physical frequency (Hz)
    nu_actual = nu_norm * cfg.doppler_scale  # Hz
    
    if cfg.isac_mode == 'one_way':
        velocity_ms = cfg.c * nu_actual / cfg.f_c
    else:
        velocity_ms = cfg.c * nu_actual / (2 * cfg.f_c)
    
    return range_m, velocity_ms


def rmse_to_physical(
    rmse_tau: float,
    rmse_nu: float,
    cfg: SystemMetricsConfig,
) -> Tuple[float, float]:
    """
    Convert RMSE in normalized units to physical units.
    
    Args:
        rmse_tau: RMSE of normalized delay
        rmse_nu: RMSE of normalized Doppler
        cfg: System configuration (with delay_scale, doppler_scale from model.cfg)
        
    Returns:
        range_error_m: Range error in meters
        velocity_error_ms: Velocity error in m/s
    """
    # Range error
    tau_error_s = rmse_tau * cfg.delay_scale
    if cfg.isac_mode == 'one_way':
        range_error_m = cfg.c * tau_error_s
    else:
        range_error_m = cfg.c * tau_error_s / 2
    
    # Velocity error
    nu_error_hz = rmse_nu * cfg.doppler_scale
    if cfg.isac_mode == 'one_way':
        velocity_error_ms = cfg.c * nu_error_hz / cfg.f_c
    else:
        velocity_error_ms = cfg.c * nu_error_hz / (2 * cfg.f_c)
    
    return range_error_m, velocity_error_ms


class SystemMetricsEvaluator:
    """
    Evaluator for system-level metrics (BER, EVM).
    
    Usage:
        evaluator = SystemMetricsEvaluator(cfg)
        ber, evm = evaluator.evaluate_frame(x_true, x_hat, snr_db)
    """
    
    def __init__(self, cfg: SystemMetricsConfig = None):
        self.cfg = cfg if cfg is not None else SystemMetricsConfig()
        self.rng = np.random.default_rng()
    
    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)
    
    def evaluate_frame(
        self,
        x_true: np.ndarray,  # [tau, nu, phi] true state
        x_hat: np.ndarray,   # [tau, nu, phi] estimated state
        snr_db: float,
    ) -> Tuple[float, float]:
        """
        Evaluate BER and EVM for one frame.
        
        Args:
            x_true: True state (normalized)
            x_hat: Estimated state (normalized)
            snr_db: SNR in dB
            
        Returns:
            ber: Bit error rate
            evm: Error vector magnitude (%)
        """
        # Generate transmitted symbols
        s_tx, bits_tx = generate_qpsk_symbols(self.cfg.n_data_symbols, self.rng)
        
        # Apply true channel (using cfg for physical parameters)
        y = apply_channel_distortion(
            s_tx, 
            x_true[0], x_true[1], x_true[2],
            snr_db, 
            self.rng,
            self.cfg,
        )
        
        # Compensate using estimates (using cfg for physical parameters)
        y_comp = compensate_channel(y, x_hat[0], x_hat[1], x_hat[2], self.cfg)
        
        # Compute metrics
        ber = compute_ber_qpsk(y_comp, bits_tx)
        evm = compute_evm(y_comp, s_tx)
        
        return ber, evm
    
    def evaluate_sequence(
        self,
        x_true_seq: List[np.ndarray],
        x_hat_seq: List[np.ndarray],
        snr_db: float,
    ) -> Tuple[List[float], List[float], float, float]:
        """
        Evaluate BER and EVM over a sequence.
        
        Returns:
            ber_seq: Per-frame BER
            evm_seq: Per-frame EVM
            avg_ber: Average BER
            avg_evm: Average EVM
        """
        ber_seq = []
        evm_seq = []
        
        for x_true, x_hat in zip(x_true_seq, x_hat_seq):
            ber, evm = self.evaluate_frame(x_true, x_hat, snr_db)
            ber_seq.append(ber)
            evm_seq.append(evm)
        
        avg_ber = np.mean(ber_seq)
        avg_evm = np.mean(evm_seq)
        
        return ber_seq, evm_seq, avg_ber, avg_evm
    
    def evaluate_around_slip(
        self,
        x_true_seq: List[np.ndarray],
        x_hat_seq: List[np.ndarray],
        slip_frames: List[int],
        snr_db: float,
        window: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate BER/EVM recovery around slip events.
        
        Args:
            x_true_seq: True states
            x_hat_seq: Estimated states
            slip_frames: Frames where slip occurred
            snr_db: SNR in dB
            window: Frames to analyze after slip
            
        Returns:
            Dictionary with time-aligned BER/EVM around slips
        """
        if not slip_frames:
            return {}
        
        n_frames = len(x_true_seq)
        
        # Collect BER/EVM for each slip event
        ber_aligned = []
        evm_aligned = []
        
        for sf in slip_frames:
            if sf + window > n_frames:
                continue
            
            ber_window = []
            evm_window = []
            
            for k in range(sf, min(sf + window, n_frames)):
                ber, evm = self.evaluate_frame(x_true_seq[k], x_hat_seq[k], snr_db)
                ber_window.append(ber)
                evm_window.append(evm)
            
            if len(ber_window) == window:
                ber_aligned.append(ber_window)
                evm_aligned.append(evm_window)
        
        if not ber_aligned:
            return {}
        
        return {
            'ber_aligned': np.array(ber_aligned),
            'evm_aligned': np.array(evm_aligned),
            'ber_mean': np.mean(ber_aligned, axis=0),
            'evm_mean': np.mean(evm_aligned, axis=0),
            'frames_relative': np.arange(window),
        }


# =========================================================================
# Convenience functions
# =========================================================================

def quick_ber_evm(
    x_true_seq: List[np.ndarray],
    x_hat_seq: List[np.ndarray],
    snr_db: float,
    seed: int = 42,
    model_cfg: 'THzISACConfig' = None,
) -> Tuple[float, float]:
    """Quick BER/EVM evaluation for a sequence.
    
    Args:
        x_true_seq: True state sequence
        x_hat_seq: Estimated state sequence
        snr_db: SNR in dB
        seed: Random seed
        model_cfg: THzISACConfig for physical parameters (optional, uses defaults if None)
        
    Returns:
        avg_ber: Average BER
        avg_evm: Average EVM
    """
    if model_cfg is not None:
        cfg = SystemMetricsConfig.from_model_cfg(model_cfg)
    else:
        cfg = SystemMetricsConfig()
    
    evaluator = SystemMetricsEvaluator(cfg)
    evaluator.set_seed(seed)
    _, _, avg_ber, avg_evm = evaluator.evaluate_sequence(x_true_seq, x_hat_seq, snr_db)
    return avg_ber, avg_evm
