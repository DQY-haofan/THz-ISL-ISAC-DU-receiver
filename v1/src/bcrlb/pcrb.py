# src/bcrlb/pcrb.py
"""
Posterior Cramer-Rao Bound (PCRB) Recursion

Per advisor restructure (2025-12-27):
- Practical PCRB approximation (Tichavský-style information filter form)
- Can be used for: (1) evaluation anchor, (2) normalized training loss
- Handles single-frame weak observability via prior/dynamics coupling

Key equations:
    P_pred = F P F^T + Q
    J_pred = inv(P_pred)
    J_meas = (2/σ²) * Re(J^H J)
    J_post = J_pred + J_meas
    PCRB = diag(inv(J_post))
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np


def _ensure_spd(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Ensure matrix is symmetric positive definite."""
    M = (M + M.T) / 2  # Symmetrize
    return M + eps * np.eye(M.shape[0])


@dataclass
class PCRBState:
    """State of PCRB recursion."""
    J: np.ndarray   # Posterior information matrix [d, d]


class PCRBRecursion:
    """
    Posterior Cramer-Rao Bound recursion.
    
    Implements practical PCRB approximation in information filter form.
    This provides theoretical lower bound on estimation variance.
    
    Key insight: Even if single-frame Doppler observability is weak,
    the dynamics coupling (φ ← φ + 2π*ν*T) allows information to accumulate.
    """
    
    def __init__(
        self,
        d: int = 3,
        J0: Optional[np.ndarray] = None,
    ):
        """
        Initialize PCRB recursion.
        
        Args:
            d: State dimension
            J0: Initial information matrix (default: I)
        """
        if J0 is None:
            J0 = np.eye(d, dtype=float)
        self.state = PCRBState(J=_ensure_spd(J0))
        self.d = d
    
    def reset(self, J0: Optional[np.ndarray] = None) -> None:
        """Reset to initial state."""
        if J0 is None:
            J0 = np.eye(self.d, dtype=float)
        self.state = PCRBState(J=_ensure_spd(J0))
    
    def step(
        self,
        model,
        x_lin: np.ndarray,
        frame_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one PCRB recursion step.
        
        Args:
            model: THzISACModel instance
            x_lin: Linearization point for Jacobian
            frame_idx: Frame index
            
        Returns:
            pcrb_diag: Diagonal of PCRB matrix [d] (per-component bounds)
            J_post: Posterior information matrix [d, d]
        """
        # Get dynamics
        F = model.F_jacobian()
        Q = model.Q_cov()
        
        # Convert information to covariance
        P = np.linalg.inv(_ensure_spd(self.state.J))
        
        # Prediction step
        P_pred = F @ P @ F.T + Q
        J_pred = np.linalg.inv(_ensure_spd(P_pred))
        
        # Measurement information
        Jc = model.jacobian(x_lin, frame_idx)  # [m, d] complex
        
        # Real-stack Jacobian for FIM computation
        Jr = np.concatenate([Jc.real, Jc.imag], axis=0)  # [2m, d]
        
        # Weight matrix (isotropic)
        W = (1.0 / max(model.sigma_eff_sq, 1e-12)) * np.eye(Jr.shape[0], dtype=float)
        
        # Measurement FIM: J_meas = 2 * J^T W J
        J_meas = 2.0 * (Jr.T @ W @ Jr)  # [d, d]
        
        # Posterior information
        J_post = _ensure_spd(J_pred + J_meas)
        self.state.J = J_post
        
        # PCRB = inv(J_post)
        PCRB = np.linalg.inv(J_post)
        pcrb_diag = np.diag(PCRB)
        
        return pcrb_diag, J_post
    
    def run_sequence(
        self,
        model,
        x_true_seq: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run PCRB recursion over a sequence.
        
        Args:
            model: THzISACModel instance
            x_true_seq: True state sequence for linearization
            
        Returns:
            pcrb_seq: PCRB diagonal over sequence [n_frames, d]
            J_seq: Information matrices over sequence [n_frames, d, d]
        """
        n_frames = len(x_true_seq)
        pcrb_seq = np.zeros((n_frames, self.d), dtype=float)
        J_seq = np.zeros((n_frames, self.d, self.d), dtype=float)
        
        for k, x_lin in enumerate(x_true_seq):
            pcrb_diag, J_post = self.step(model, x_lin, k)
            pcrb_seq[k] = pcrb_diag
            J_seq[k] = J_post
        
        return pcrb_seq, J_seq


# =========================================================================
# Utility functions for PCRB analysis
# =========================================================================

def compute_efficiency(
    mse: np.ndarray,
    pcrb: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute estimation efficiency (MSE / PCRB).
    
    Efficiency >= 1 (equality means efficient estimator).
    
    Args:
        mse: Mean squared error [d] or [n_frames, d]
        pcrb: PCRB bounds [d] or [n_frames, d]
        eps: Small constant to avoid division by zero
        
    Returns:
        efficiency: Efficiency ratio (same shape as input)
    """
    return mse / np.maximum(pcrb, eps)


def pcrb_normalized_mse(
    error: np.ndarray,
    pcrb_diag: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute PCRB-normalized MSE (for training loss).
    
    L = mean((x_hat - x_true)² / PCRB)
    
    Args:
        error: Estimation error [d]
        pcrb_diag: PCRB diagonal [d]
        eps: Small constant for stability
        
    Returns:
        Normalized MSE (scalar)
    """
    denom = np.maximum(pcrb_diag, eps)
    return float(np.mean((error ** 2) / denom))


def check_pcrb_sanity(
    pcrb_seq: np.ndarray,
    component_names: List[str] = None,
) -> dict:
    """
    Sanity check PCRB values.
    
    Expected behaviors:
    - PCRB should decrease over time (information accumulates)
    - τ, φ should have lower PCRB than ν (better observability)
    - PCRB should be positive
    
    Args:
        pcrb_seq: PCRB over sequence [n_frames, d]
        component_names: Names for reporting
        
    Returns:
        Sanity check results
    """
    if component_names is None:
        component_names = ["τ", "ν", "φ"]
    
    n_frames, d = pcrb_seq.shape
    
    results = {
        "all_positive": bool(np.all(pcrb_seq > 0)),
        "final_pcrb": dict(zip(component_names, pcrb_seq[-1].tolist())),
        "initial_pcrb": dict(zip(component_names, pcrb_seq[0].tolist())),
    }
    
    # Check monotonic decrease (allowing small increases)
    for i, name in enumerate(component_names):
        decreases = np.diff(pcrb_seq[:, i]) < 0
        results[f"{name}_monotonic_decrease_rate"] = float(np.mean(decreases))
    
    # ν should have higher PCRB than τ and φ (less observable)
    if d >= 3:
        results["nu_less_observable"] = bool(
            np.mean(pcrb_seq[:, 1]) > np.mean(pcrb_seq[:, 0]) and
            np.mean(pcrb_seq[:, 1]) > np.mean(pcrb_seq[:, 2])
        )
    
    return results
