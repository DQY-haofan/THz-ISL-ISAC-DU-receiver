# src/physics/thz_isac_model.py
"""
Unified THz-ISAC Physical Model

Per advisor restructure (2025-12-27) + P0.5 fixes:
- 2D pilot grid: frequency (f_i) x time (t_j) for Doppler observability
- Fast/slow variable coupling: φ fast (driven by ν), ν slow (AR process)
- State transition: φ_{k+1} = φ_k + 2π*ν*T_frame + w_φ
- AQNM quantization with equivalent Gaussian noise

P0.5 CRITICAL FIXES:
1. Measurement uses INTRA-FRAME time t_j only (not absolute t_abs = kT + t_j)
   - Cross-frame accumulation is ONLY in state transition
   - This avoids "double counting" of Doppler information
2. Phase wrapping utilities for circular manifold handling

This is the SINGLE SOURCE OF TRUTH for all algorithms (EKF, GN, DU, PCRB).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


# =========================================================================
# Module-level phase wrapping utilities
# =========================================================================

def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def wrap_angle_array(angles: np.ndarray) -> np.ndarray:
    """Wrap angles to [-π, π]."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def circular_error(phi_hat: float, phi_true: float) -> float:
    """
    Compute circular (wrapped) phase error.
    
    Returns error in [-π, π], the shortest angular distance.
    """
    return wrap_angle(phi_hat - phi_true)


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class THzISACConfig:
    """THz-ISAC system configuration."""
    
    # Pilot grid (frequency x time)
    n_f: int = 8                  # frequency pilots
    n_t: int = 2                  # time pilots within a frame (>=2 enables ν observability)
    bandwidth_hz: float = 100e6   # 100 MHz
    carrier_freq_hz: float = 300e9  # 300 GHz
    frame_duration_s: float = 100e-6  # 100 μs
    
    # SNR and quantization
    snr_db: float = 10.0
    adc_bits: int = 4
    apply_quantization: bool = True
    
    # Normalization scales (state is normalized)
    delay_scale: float = 1e-9     # 1 ns
    doppler_scale: float = 1e3    # 1 kHz
    phase_scale: float = 1.0      # 1 rad
    
    # Drift/transition (normalized-domain process noise)
    nu_ar: float = 0.99           # AR coefficient for Doppler (slow variable)
    q_std_norm: Tuple[float, float, float] = (0.02, 0.01, 0.05)  # (τ, ν, φ) in normalized coords
    
    # AQNM parameters (computed from adc_bits)
    alpha_aqnm: float = field(default=1.0, init=False)
    quant_var: float = field(default=0.0, init=False)


def _aqnm_params(bits: int) -> Tuple[float, float]:
    """
    Get AQNM parameters (alpha, sigma_q^2) for given ADC bits.
    
    Based on Gaussian input approximation (Orhan-Erkip-Rangan 2015).
    """
    table = {
        1: (0.6366, 0.3634),
        2: (0.8825, 0.1175),
        3: (0.9625, 0.0375),
        4: (0.9900, 0.0100),
        5: (0.9975, 0.0025),
        6: (0.9994, 0.0006),
        7: (0.9998, 0.0002),
        8: (0.9999, 0.0001),
    }
    if bits in table:
        return table[bits]
    # Conservative fallback for high resolution
    if bits >= 8:
        return (1.0, 0.0)
    # Extrapolate for low bits
    return (0.3634 * (4.0 ** (1 - bits)), 1.0 - 0.3634 * (4.0 ** (1 - bits)))


# =========================================================================
# Main Model Class
# =========================================================================

class THzISACModel:
    """
    2D pilot grid THz-ISAC observation model.
    
    Observation model:
        h(f_i, t_j) = exp(-j*2π*f_i*τ) * exp(+j*2π*ν*t_j) * exp(+j*φ)
    
    CRITICAL: Uses intra-frame time t_j only, NOT absolute time.
    Cross-frame accumulation is in state transition only.
    
    State is normalized: x = [τ/τ_s, ν/ν_s, φ/φ_s]
    
    Key feature: Pilot time diversity (n_t >= 2) enables single-frame Doppler observability
    """
    
    def __init__(self, cfg: THzISACConfig):
        self.cfg = cfg
        self._init_pilots()
        self._init_noise()
    
    def _init_pilots(self) -> None:
        """Initialize 2D pilot grid (frequency x time)."""
        cfg = self.cfg
        
        # Frequency grid
        self.f_grid = np.linspace(
            -cfg.bandwidth_hz / 2,
            cfg.bandwidth_hz / 2,
            cfg.n_f,
            endpoint=False
        )  # [n_f]
        
        # Intra-frame pilot times, spread across frame
        self.t_grid = np.linspace(
            0.0,
            cfg.frame_duration_s,
            cfg.n_t,
            endpoint=False
        )  # [n_t]
        
        # Vectorized measurement coordinates
        # f_vec repeats for each time slot; t_vec tiles over frequencies
        self.f_vec = np.repeat(self.f_grid, cfg.n_t)   # [m]
        self.t_vec = np.tile(self.t_grid, cfg.n_f)     # [m]
        self.m = cfg.n_f * cfg.n_t
    
    def _init_noise(self) -> None:
        """Initialize noise parameters (AWGN + AQNM)."""
        cfg = self.cfg
        
        # AWGN variance in complex baseband: E|n|^2 = sigma^2
        snr_lin = 10.0 ** (cfg.snr_db / 10.0)
        self.awgn_var = 1.0 / snr_lin
        
        # AQNM parameters
        alpha, sigma_q_sq = _aqnm_params(cfg.adc_bits)
        cfg.alpha_aqnm = alpha
        cfg.quant_var = sigma_q_sq
        self.alpha = alpha
        self.quant_var = sigma_q_sq
        
        # Effective noise after AQNM scaling
        # y_q = alpha*y + q, treat q as additional noise; move to equivalent noise on y:
        # y = y_q/alpha = y + q/alpha
        self.sigma_eff_sq = self.awgn_var + (self.quant_var / max(alpha**2, 1e-12))
    
    # =========================================================================
    # Normalization helpers
    # =========================================================================
    
    def denorm(self, x_norm: np.ndarray) -> Tuple[float, float, float]:
        """Convert normalized state to physical units."""
        cfg = self.cfg
        tau = float(x_norm[0] * cfg.delay_scale)   # seconds
        nu = float(x_norm[1] * cfg.doppler_scale)  # Hz
        phi = float(x_norm[2] * cfg.phase_scale)   # rad
        return tau, nu, phi
    
    def norm(self, tau: float, nu: float, phi: float) -> np.ndarray:
        """Convert physical units to normalized state."""
        cfg = self.cfg
        return np.array([
            tau / cfg.delay_scale,
            nu / cfg.doppler_scale,
            phi / cfg.phase_scale,
        ], dtype=float)
    
    def wrap_phase_norm(self, x_norm: np.ndarray) -> np.ndarray:
        """Keep phase in [-π, π] in physical domain, map back to normalized."""
        cfg = self.cfg
        phi = x_norm[2] * cfg.phase_scale
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        x_norm = x_norm.copy()
        x_norm[2] = phi / cfg.phase_scale
        return x_norm
    
    # =========================================================================
    # Dynamics (multi-frame fast/slow coupling)
    # =========================================================================
    
    def transition(self, x_norm: np.ndarray) -> np.ndarray:
        """
        State transition with fast/slow variable coupling.
        
        τ_{k+1} = τ_k + w_τ           (slow, nearly constant)
        ν_{k+1} = ρ*ν_k + w_ν         (slow, AR(1) process)
        φ_{k+1} = φ_k + 2π*ν*T + w_φ  (fast, driven by Doppler)
        
        This is where cross-frame phase accumulation happens.
        """
        cfg = self.cfg
        tau_n, nu_n, phi_n = x_norm
        
        # τ_{k+1} = τ_k (plus process noise added separately)
        tau_n_next = tau_n
        
        # ν_{k+1} = ρ*ν_k
        nu_n_next = cfg.nu_ar * nu_n
        
        # φ_{k+1} = φ_k + 2π*ν*T_frame
        # Convert ν_n to Hz: nu = nu_n * doppler_scale
        # Phase increment in rad: incr = 2π * nu * T
        incr = 2.0 * np.pi * (nu_n * cfg.doppler_scale) * cfg.frame_duration_s
        phi_n_next = phi_n + incr / cfg.phase_scale
        
        x_next = np.array([tau_n_next, nu_n_next, phi_n_next], dtype=float)
        return self.wrap_phase_norm(x_next)
    
    def F_jacobian(self) -> np.ndarray:
        """
        State transition Jacobian (linearization of transition).
        
        F = ∂f/∂x where x_next = f(x)
        """
        cfg = self.cfg
        F = np.eye(3, dtype=float)
        
        # ν coefficient
        F[1, 1] = cfg.nu_ar
        
        # φ depends on ν: ∂φ_next/∂ν = 2π*T*doppler_scale/phase_scale
        F[2, 1] = (2.0 * np.pi * cfg.doppler_scale * cfg.frame_duration_s) / cfg.phase_scale
        
        return F
    
    def Q_cov(self) -> np.ndarray:
        """Process noise covariance (normalized domain)."""
        q = np.array(self.cfg.q_std_norm, dtype=float)
        return np.diag(q * q)
    
    # =========================================================================
    # Measurement model
    # =========================================================================
    
    def h(self, x_norm: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Observation function h(x, t).
        
        h(f_i, t_j) = exp(-j*2π*f_i*τ) * exp(+j*2π*ν*t_j) * exp(+j*φ)
        
        CRITICAL FIX (P0.5): Use intra-frame time t_j only, NOT absolute time.
        Cross-frame phase accumulation is handled by state transition only.
        This avoids "double counting" of Doppler information that caused
        artificially low ν-PCRB.
        
        Args:
            x_norm: Normalized state [τ_n, ν_n, φ_n]
            frame_idx: Frame index (unused in measurement, for API consistency)
            
        Returns:
            h: Complex observation vector [m]
        """
        tau, nu, phi = self.denorm(x_norm)
        
        # FIXED: Use intra-frame time t_j only (not absolute time kT + t_j)
        t_intra = self.t_vec  # [m], intra-frame pilot times
        
        # Phase components
        phase = (-2.0 * np.pi * self.f_vec * tau) + (2.0 * np.pi * nu * t_intra) + phi
        
        return np.exp(1j * phase)  # [m] complex
    
    def jacobian(self, x_norm: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Analytical Jacobian ∂h/∂x (w.r.t. normalized state).
        
        For h = exp(j*ψ) where ψ = -2π*f*τ + 2π*ν*t_j + φ:
        
        ∂h/∂τ_n = ∂h/∂τ * ∂τ/∂τ_n = (-j*2π*f) * h * delay_scale
        ∂h/∂ν_n = ∂h/∂ν * ∂ν/∂ν_n = (+j*2π*t_j) * h * doppler_scale
        ∂h/∂φ_n = ∂h/∂φ * ∂φ/∂φ_n = (j) * h * phase_scale
        
        CRITICAL FIX (P0.5): Use intra-frame time t_j, not absolute time.
        This matches h() and avoids double-counting Doppler information.
        
        Args:
            x_norm: Normalized state [τ_n, ν_n, φ_n]
            frame_idx: Frame index (unused, for API consistency)
            
        Returns:
            J: Jacobian [m, 3] complex
        """
        cfg = self.cfg
        
        # FIXED: Use intra-frame time only (matches h())
        t_intra = self.t_vec  # [m]
        
        h = self.h(x_norm, frame_idx)  # [m]
        
        # Derivatives w.r.t normalized coords
        d_tau = (1j * (-2.0 * np.pi * self.f_vec)) * h * cfg.delay_scale
        d_nu = (1j * (2.0 * np.pi * t_intra)) * h * cfg.doppler_scale
        d_phi = (1j) * h * cfg.phase_scale
        
        J = np.stack([d_tau, d_nu, d_phi], axis=1)  # [m, 3] complex
        return J
    
    def R_cov(self) -> np.ndarray:
        """Observation noise covariance (complex-domain: σ_eff^2 * I)."""
        return self.sigma_eff_sq * np.eye(self.m, dtype=float)
    
    # =========================================================================
    # Observation generation (for simulation)
    # =========================================================================
    
    def observe(
        self,
        x_norm: np.ndarray,
        frame_idx: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Generate noisy observation (for simulation).
        
        y = h(x) + n, then AQNM quantization if enabled.
        
        Args:
            x_norm: True normalized state
            frame_idx: Frame index
            rng: Random generator (optional)
            
        Returns:
            y: Noisy observation [m] complex
        """
        if rng is None:
            rng = np.random.default_rng()
        
        h = self.h(x_norm, frame_idx)
        
        # AWGN (complex)
        n = np.sqrt(self.awgn_var / 2) * (
            rng.standard_normal(self.m) + 1j * rng.standard_normal(self.m)
        )
        y = h + n
        
        # AQNM quantization
        if self.cfg.apply_quantization and self.cfg.adc_bits < 12:
            q = np.sqrt(self.quant_var / 2) * (
                rng.standard_normal(self.m) + 1j * rng.standard_normal(self.m)
            )
            y = self.alpha * y + q
        
        return y
    
    # =========================================================================
    # Fisher Information (for PCRB)
    # =========================================================================
    
    def compute_fim_data(self, x_norm: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Compute data Fisher Information Matrix.
        
        J_data = (2/σ²) * Re(J^H J) for complex Gaussian likelihood.
        
        Args:
            x_norm: Linearization point
            frame_idx: Frame index
            
        Returns:
            J_data: [3, 3] real symmetric positive semi-definite
        """
        J = self.jacobian(x_norm, frame_idx)  # [m, 3] complex
        
        # For real-valued state with complex observations:
        # J_data = (2/σ²) * Re(J^H J)
        JHJ = J.conj().T @ J  # [3, 3] complex
        J_data = (2.0 / max(self.sigma_eff_sq, 1e-12)) * np.real(JHJ)
        
        return J_data


# =========================================================================
# Module-level factory function
# =========================================================================

def create_default_model(snr_db: float = 10.0, adc_bits: int = 4) -> THzISACModel:
    """Create model with default configuration."""
    cfg = THzISACConfig(snr_db=snr_db, adc_bits=adc_bits)
    return THzISACModel(cfg)
