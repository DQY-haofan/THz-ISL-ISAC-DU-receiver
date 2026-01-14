# src/inference/energy.py
"""
MAP Energy Function for THz-ISAC State Estimation

Per advisor restructure (2025-12-27):
- Implements posterior energy: E(x) = ||y - h(x)||²_R + ||x - x_pred||²_P
- Provides GN gradient and Hessian for iterative optimization
- Uses real-stacking for complex-valued observations

This is the foundation for both GN-MAP baseline and DU-MAP unfolding.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


# =========================================================================
# Phase wrapping utility (for circular manifold)
# =========================================================================

def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def wrap_phase_residual(dx: np.ndarray, phase_idx: int = 2) -> np.ndarray:
    """
    Wrap the phase component of state residual to [-π, π].
    
    CRITICAL: Phase lives on S^1 (circle), not R^1.
    Using Euclidean residual for phase causes optimization to diverge
    when true error is small but crosses ±π boundary.
    
    Args:
        dx: State residual [d]
        phase_idx: Index of phase component (default: 2)
        
    Returns:
        dx_wrapped: Residual with wrapped phase
    """
    dx = dx.copy()
    dx[phase_idx] = wrap_angle(dx[phase_idx])
    return dx


# =========================================================================
# Complex-to-real conversion utilities
# =========================================================================

def complex_to_real_vec(z: np.ndarray) -> np.ndarray:
    """
    Convert complex vector to real-stacked vector.
    
    [z_1, z_2, ...] -> [Re(z_1), Re(z_2), ..., Im(z_1), Im(z_2), ...]
    """
    return np.concatenate([z.real, z.imag], axis=0)


def complex_jac_to_real(J: np.ndarray) -> np.ndarray:
    """
    Convert complex Jacobian to real-stacked form.
    
    For complex observation h(x) where x is real:
    [Re(h); Im(h)] ≈ [Re(J); Im(J)] * δx
    
    Args:
        J: Complex Jacobian [m, d]
        
    Returns:
        Jr: Real Jacobian [2m, d]
    """
    A = J.real  # [m, d]
    B = J.imag  # [m, d]
    Jr = np.concatenate([A, B], axis=0)  # [2m, d]
    return Jr


# =========================================================================
# MAP Terms Container
# =========================================================================

@dataclass
class MAPTerms:
    """
    Container for MAP optimization terms.
    
    Energy: E(x) = r^T W r + dx^T P^{-1} dx
    where r = y - h(x), dx = x - x_pred
    """
    r_real: np.ndarray     # Real-stacked residual [2m]
    J_real: np.ndarray     # Real-stacked Jacobian [2m, d]
    W: np.ndarray          # Measurement weight matrix [2m, 2m] (often scalar*I)
    Pinv: np.ndarray       # Prior precision matrix [d, d]
    dx: np.ndarray         # State deviation from prior [d]


def build_map_terms(
    y: np.ndarray,
    h: np.ndarray,
    J: np.ndarray,
    sigma_eff_sq: float,
    x: np.ndarray,
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    phase_idx: int = 2,
) -> MAPTerms:
    """
    Build MAP optimization terms from model outputs.
    
    Args:
        y: Observation [m] complex
        h: Predicted observation [m] complex
        J: Jacobian [m, d] complex
        sigma_eff_sq: Effective noise variance
        x: Current state estimate [d]
        x_pred: Prior state prediction [d]
        P_pred: Prior covariance [d, d]
        phase_idx: Index of phase component for wrapping
        
    Returns:
        MAPTerms container
    """
    # Residual (complex -> real-stacked)
    r = y - h
    r_real = complex_to_real_vec(r)
    
    # Jacobian (complex -> real-stacked)
    J_real = complex_jac_to_real(J)  # [2m, d]
    
    # Measurement weight matrix (isotropic for equivalent Gaussian)
    # NOTE: For CN(0, σ²), Re/Im each have variance σ²/2
    # So weight should be 2/σ² for real-stacked form
    m2 = r_real.shape[0]  # 2m
    W = (2.0 / max(sigma_eff_sq, 1e-12)) * np.eye(m2, dtype=float)
    
    # Prior precision
    d = P_pred.shape[0]
    P_reg = P_pred + 1e-12 * np.eye(d)
    Pinv = np.linalg.inv(P_reg)
    
    # State deviation from prior - WRAP PHASE!
    dx = x - x_pred
    dx = wrap_phase_residual(dx, phase_idx=phase_idx)
    
    return MAPTerms(r_real=r_real, J_real=J_real, W=W, Pinv=Pinv, dx=dx)


# =========================================================================
# Energy and Gradient/Hessian Computation
# =========================================================================

def energy_map(terms: MAPTerms) -> float:
    """
    Compute MAP energy.
    
    E(x) = (y - h(x))^H R^{-1} (y - h(x)) + (x - x_pred)^T P^{-1} (x - x_pred)
         = r^T W r + dx^T P^{-1} dx
    
    Args:
        terms: MAPTerms container
        
    Returns:
        E: Scalar energy value
    """
    e_meas = float(terms.r_real.T @ terms.W @ terms.r_real)
    e_prior = float(terms.dx.T @ terms.Pinv @ terms.dx)
    return e_meas + e_prior


def gn_grad_hess(
    terms: MAPTerms,
    damping: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Gauss-Newton gradient and Hessian approximation.
    
    Gradient: g = ∂E/∂x = -2 J^T W r + 2 P^{-1} dx
    Hessian:  H ≈ 2 J^T W J + 2 P^{-1}  (+ λI for damping)
    
    Args:
        terms: MAPTerms container
        damping: Levenberg-Marquardt damping parameter
        
    Returns:
        grad: Gradient vector [d]
        H: Hessian matrix [d, d]
    """
    # Gradient
    grad = (-2.0 * terms.J_real.T @ terms.W @ terms.r_real) + \
           (2.0 * terms.Pinv @ terms.dx)
    
    # Gauss-Newton Hessian approximation
    H = (2.0 * terms.J_real.T @ terms.W @ terms.J_real) + \
        (2.0 * terms.Pinv)
    
    # Levenberg-Marquardt damping
    if damping > 0.0:
        H = H + damping * np.eye(H.shape[0], dtype=float)
    
    return grad, H


def gn_step(
    terms: MAPTerms,
    damping: float = 0.0,
) -> np.ndarray:
    """
    Compute Gauss-Newton step.
    
    δx = -H^{-1} g
    
    Args:
        terms: MAPTerms container
        damping: LM damping parameter
        
    Returns:
        delta: Step direction [d]
    """
    grad, H = gn_grad_hess(terms, damping=damping)
    
    try:
        delta = np.linalg.solve(H, -grad)
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(H, -grad, rcond=None)[0]
    
    return delta


# =========================================================================
# Energy decrease check (for training loss)
# =========================================================================

def energy_decrease_penalty(
    E_before: float,
    E_after: float,
    margin: float = 0.0,
) -> float:
    """
    Compute penalty for energy increase (used in training loss).
    
    L = max(0, E_after - E_before + margin)
    
    Args:
        E_before: Energy before step
        E_after: Energy after step
        margin: Optional margin for robustness
        
    Returns:
        Penalty (0 if energy decreased)
    """
    return max(0.0, E_after - E_before + margin)
