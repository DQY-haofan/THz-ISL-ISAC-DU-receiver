# src/sim/slip.py
"""
Phase Slip Process Model

Per advisor P2 directive:
- Make slip a defensible physical process (cycle slip / PLL unlock)
- Support discrete/continuous/burst modes
- p_slip per frame, configurable amplitude distribution

Physical motivation:
- In THz/mmWave systems, phase slips occur due to:
  1. PLL cycle slip under low SNR
  2. Carrier frequency offset estimation errors
  3. Hardware impairments causing sudden phase jumps
- This is NOT Gaussian process noise; it's impulsive/mixture
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class SlipConfig:
    """Configuration for phase slip process."""
    
    # Slip probability per frame
    p_slip: float = 0.02
    
    # Mode: "discrete" | "gaussian" | "burst"
    mode: str = "discrete"
    
    # Discrete slip values (radians)
    values: Tuple[float, ...] = (
        -2*np.pi, -np.pi, -np.pi/2, 
        np.pi/2, np.pi, 2*np.pi
    )
    
    # Probabilities for each discrete value (must sum to 1)
    probs: Tuple[float, ...] = (0.10, 0.20, 0.20, 0.20, 0.20, 0.10)
    
    # Gaussian mode parameters
    gaussian_std: float = np.pi / 2
    
    # Burst mode: probability of ending burst each frame
    burst_p_end: float = 0.4
    
    def __post_init__(self):
        if self.mode == "discrete":
            assert len(self.values) == len(self.probs), \
                "values and probs must have same length"
            assert abs(sum(self.probs) - 1.0) < 1e-6, \
                "probs must sum to 1"


class PhaseSlipProcess:
    """
    Phase slip random process.
    
    Models impulsive phase disturbances that cause EKF to fail
    but can be recovered by iterative MAP methods (GN/DU).
    """
    
    def __init__(self, cfg: SlipConfig, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        self._burst_remaining = 0
        self._slip_history: List[Tuple[int, float]] = []
        # Initialize to -1 so first sample() call gives frame_idx=0
        # This aligns with generate_episode_with_impairments() which uses k=0,1,2,...
        self._frame_idx = -1
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset process state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._burst_remaining = 0
        self._slip_history = []
        self._frame_idx = -1  # Reset to -1 for consistency
    
    def sample(self) -> float:
        """
        Sample slip for current frame.
        
        Returns:
            Δφ_slip in radians (0 if no slip)
        """
        self._frame_idx += 1
        
        if self.cfg.mode == "burst":
            return self._sample_burst()
        elif self.cfg.mode == "gaussian":
            return self._sample_gaussian()
        else:  # discrete
            return self._sample_discrete()
    
    def _sample_discrete(self) -> float:
        """Sample from discrete slip distribution."""
        if self.rng.random() < self.cfg.p_slip:
            slip = float(self.rng.choice(self.cfg.values, p=self.cfg.probs))
            self._slip_history.append((self._frame_idx, slip))
            return slip
        return 0.0
    
    def _sample_gaussian(self) -> float:
        """Sample from Gaussian slip distribution."""
        if self.rng.random() < self.cfg.p_slip:
            slip = float(self.rng.normal(0, self.cfg.gaussian_std))
            self._slip_history.append((self._frame_idx, slip))
            return slip
        return 0.0
    
    def _sample_burst(self) -> float:
        """Sample with burst mode (consecutive slips)."""
        if self._burst_remaining > 0:
            # Continue burst
            self._burst_remaining -= 1
            slip = float(self.rng.choice(self.cfg.values, p=self.cfg.probs))
            self._slip_history.append((self._frame_idx, slip))
            return slip
        
        if self.rng.random() < self.cfg.p_slip:
            # Start new burst - geometric length
            burst_len = 1
            while self.rng.random() > self.cfg.burst_p_end:
                burst_len += 1
            self._burst_remaining = burst_len - 1
            
            slip = float(self.rng.choice(self.cfg.values, p=self.cfg.probs))
            self._slip_history.append((self._frame_idx, slip))
            return slip
        
        return 0.0
    
    def get_slip_frames(self) -> List[int]:
        """Get list of frames where slip occurred."""
        return [frame for frame, _ in self._slip_history]
    
    def get_slip_history(self) -> List[Tuple[int, float]]:
        """Get full slip history: [(frame, Δφ), ...]"""
        return self._slip_history.copy()


# =========================================================================
# Continuous Phase Noise (Wiener PN) Model
# =========================================================================

@dataclass
class PhaseNoiseConfig:
    """
    Configuration for continuous phase noise (Wiener process).
    
    Physical motivation:
    - Oscillator phase noise in THz/mmWave systems
    - Continuous random walk, distinct from discrete cycle slips
    - Causes phase drift and ICI in OFDM systems
    
    Model: φ_{k+1} = φ_k + 2πνT + w_φ, where w_φ ~ N(0, σ_φ²)
    """
    
    # Phase noise standard deviation per frame (rad)
    # Typical values: 0.01-0.1 rad for THz oscillators
    sigma_phi: float = 0.0
    
    # Enable/disable
    enabled: bool = False
    
    def __post_init__(self):
        if self.sigma_phi > 0:
            self.enabled = True


class WienerPhaseNoise:
    """
    Wiener process phase noise generator.
    
    Models continuous phase drift due to oscillator imperfections.
    This is DISTINCT from phase slip (discrete jumps).
    """
    
    def __init__(self, cfg: PhaseNoiseConfig, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        self._cumulative_pn = 0.0
        self._pn_history: List[float] = []
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset PN state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._cumulative_pn = 0.0
        self._pn_history = []
    
    def sample(self) -> float:
        """
        Sample phase noise increment for one frame.
        
        Returns:
            w_phi: Phase noise increment (rad)
        """
        if not self.cfg.enabled or self.cfg.sigma_phi <= 0:
            self._pn_history.append(0.0)
            return 0.0
        
        w_phi = self.cfg.sigma_phi * self.rng.standard_normal()
        self._cumulative_pn += w_phi
        self._pn_history.append(w_phi)
        
        return w_phi
    
    def get_cumulative(self) -> float:
        """Get cumulative phase drift."""
        return self._cumulative_pn
    
    def get_history(self) -> List[float]:
        """Get history of PN increments."""
        return self._pn_history.copy()


def get_pn_config(preset: str) -> PhaseNoiseConfig:
    """Get preset PN configuration."""
    presets = {
        'none': PhaseNoiseConfig(sigma_phi=0.0, enabled=False),
        'mild': PhaseNoiseConfig(sigma_phi=0.02, enabled=True),
        'moderate': PhaseNoiseConfig(sigma_phi=0.05, enabled=True),
        'severe': PhaseNoiseConfig(sigma_phi=0.10, enabled=True),
        'extreme': PhaseNoiseConfig(sigma_phi=0.20, enabled=True),
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown PN preset: {preset}. Available: {list(presets.keys())}")
    
    return presets[preset]


# =========================================================================
# Episode generation with slip AND/OR PN
# =========================================================================

def generate_episode_with_impairments(
    model,  # THzISACModel
    n_frames: int,
    x0: np.ndarray,
    slip_cfg: Optional[SlipConfig] = None,
    pn_cfg: Optional[PhaseNoiseConfig] = None,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[float]]:
    """
    Generate episode with phase impairments (slip AND/OR continuous PN).
    
    Args:
        model: THzISACModel instance
        n_frames: Number of frames
        x0: Initial state (normalized)
        slip_cfg: Slip configuration (None to disable)
        pn_cfg: Phase noise configuration (None to disable)
        seed: Random seed
        
    Returns:
        y_seq: Observations
        x_true_seq: True states
        slip_frames: Frames where slip occurred
        pn_increments: Per-frame PN increments
        
    Note (P1-2):
        If both pn_cfg.enabled=True AND model.cfg.q_std_norm[2] > 0, the phase
        will have double random walk (PN process + Q_cov phase noise).
        Recommendation: When using PN process, set q_std_norm[2] = 0 in model config.
    """
    import warnings
    
    rng = np.random.default_rng(seed)
    
    # P1-2: Check for potential phase noise double counting
    if pn_cfg is not None and pn_cfg.enabled and pn_cfg.sigma_phi > 0:
        q_phi = model.cfg.q_std_norm[2] if hasattr(model.cfg, 'q_std_norm') else 0
        if q_phi > 0.01:  # Non-trivial Q_cov phase noise
            warnings.warn(
                f"P1-2: Both PN process (sigma_phi={pn_cfg.sigma_phi}) and Q_cov phase noise "
                f"(q_std_norm[2]={q_phi}) are enabled. This may cause double counting of "
                f"phase random walk. Consider setting q_std_norm[2]=0 when using PN process.",
                UserWarning
            )
    
    # Initialize processes
    slip_process = None
    if slip_cfg is not None and slip_cfg.p_slip > 0:
        slip_process = PhaseSlipProcess(slip_cfg, rng=np.random.default_rng(seed + 1000))
    
    pn_process = None
    if pn_cfg is not None and pn_cfg.enabled:
        pn_process = WienerPhaseNoise(pn_cfg, rng=np.random.default_rng(seed + 2000))
    
    q_std = np.sqrt(np.diag(model.Q_cov()))
    
    x = x0.copy()
    y_seq = []
    x_true_seq = []
    pn_increments = []
    
    for k in range(n_frames):
        # 1. Sample slip (discrete jump)
        delta_phi_slip = 0.0
        if slip_process is not None:
            delta_phi_slip = slip_process.sample()
        
        # 2. Sample continuous PN (Wiener increment)
        w_phi_pn = 0.0
        if pn_process is not None:
            w_phi_pn = pn_process.sample()
        pn_increments.append(w_phi_pn)
        
        # 3. Inject both impairments into phase (in normalized domain)
        total_phase_impairment = delta_phi_slip + w_phi_pn
        if abs(total_phase_impairment) > 1e-10:
            x[2] += total_phase_impairment / model.cfg.phase_scale
            x = model.wrap_phase_norm(x)
        
        # Store true state
        x_true_seq.append(x.copy())
        
        # Generate observation
        y = model.observe(x, frame_idx=k, rng=rng)
        y_seq.append(y)
        
        # State transition with process noise
        x_next = model.transition(x)
        w = q_std * rng.standard_normal(3)
        x = model.wrap_phase_norm(x_next + w)
    
    slip_frames = slip_process.get_slip_frames() if slip_process else []
    
    return y_seq, x_true_seq, slip_frames, pn_increments


# =========================================================================
# Legacy function (backward compatible)
# =========================================================================

def generate_episode_with_slip(
    model,  # THzISACModel
    n_frames: int,
    x0: np.ndarray,
    slip_cfg: SlipConfig,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Generate episode with phase slip process.
    
    Args:
        model: THzISACModel instance
        n_frames: Number of frames
        x0: Initial state (normalized)
        slip_cfg: Slip configuration
        seed: Random seed
        
    Returns:
        y_seq: Observations
        x_true_seq: True states
        slip_frames: Frames where slip occurred
    """
    rng = np.random.default_rng(seed)
    slip_process = PhaseSlipProcess(slip_cfg, rng=np.random.default_rng(seed + 1000))
    
    q_std = np.sqrt(np.diag(model.Q_cov()))
    
    x = x0.copy()
    y_seq = []
    x_true_seq = []
    
    for k in range(n_frames):
        # Sample slip
        delta_phi_slip = slip_process.sample()
        
        # Inject slip into state (in normalized domain)
        if abs(delta_phi_slip) > 1e-10:
            x[2] += delta_phi_slip / model.cfg.phase_scale
            x = model.wrap_phase_norm(x)
        
        # Store true state
        x_true_seq.append(x.copy())
        
        # Generate observation
        y = model.observe(x, frame_idx=k, rng=rng)
        y_seq.append(y)
        
        # State transition with process noise
        x_next = model.transition(x)
        w = q_std * rng.standard_normal(3)
        x = model.wrap_phase_norm(x_next + w)
    
    slip_frames = slip_process.get_slip_frames()
    
    return y_seq, x_true_seq, slip_frames


# =========================================================================
# Preset configurations
# =========================================================================

def get_slip_config(preset: str) -> SlipConfig:
    """Get preset slip configuration."""
    presets = {
        'none': SlipConfig(p_slip=0.0),
        'mild': SlipConfig(p_slip=0.01, mode='discrete'),
        'moderate': SlipConfig(p_slip=0.03, mode='discrete'),
        'severe': SlipConfig(p_slip=0.05, mode='discrete'),
        'burst': SlipConfig(p_slip=0.02, mode='burst', burst_p_end=0.3),
        'pi_only': SlipConfig(
            p_slip=0.03,
            values=(-np.pi, np.pi),
            probs=(0.5, 0.5),
        ),
        '2pi_only': SlipConfig(
            p_slip=0.03,
            values=(-2*np.pi, 2*np.pi),
            probs=(0.5, 0.5),
        ),
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    return presets[preset]
