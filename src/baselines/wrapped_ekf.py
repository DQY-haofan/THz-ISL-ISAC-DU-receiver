# src/baselines/wrapped_ekf.py
"""
Wrapped-EKF and Robust-EKF Baselines

Per advisor P2-C directive:
- Wrapped-EKF: circular residual for phase component
- Robust-EKF: NIS gating + R inflation for outlier rejection

This ensures fair baseline comparison - DU wins not because
"EKF forgot to wrap", but because of "unfolded MAP + curvature-aware".
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.stats import chi2


def wrap_to_pi(x: float) -> float:
    """Wrap angle to [-π, π]."""
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def wrap_state(x: np.ndarray, phase_idx: int = 2) -> np.ndarray:
    """Wrap phase component of state vector."""
    y = x.copy()
    y[phase_idx] = wrap_to_pi(y[phase_idx])
    return y


@dataclass
class EKFConfig:
    """Configuration for EKF variants."""
    
    # Phase wrapping
    wrap_phase: bool = True
    phase_idx: int = 2
    
    # Robustification
    use_gating: bool = False
    # P1-1 Fix: NIS threshold should depend on observation dimension (2*m), not state dim
    # Default uses chi2(0.99) which will be computed dynamically based on obs dim
    # Set to None for dynamic computation, or override with fixed value
    nis_threshold: float = None  # Will be computed as chi2.ppf(0.99, df=obs_dim)
    nis_confidence: float = 0.99  # Confidence level for chi2 threshold
    r_inflation: float = 10.0   # R multiplier when NIS exceeds threshold
    
    # Reject update entirely if NIS too high
    reject_threshold: float = 20.0


class WrappedEKF:
    """
    Extended Kalman Filter with phase wrapping.
    
    Key difference from standard EKF:
    - State update wraps phase to [-π, π]
    - Ensures circular manifold is respected
    """
    
    def __init__(self, cfg: EKFConfig):
        self.cfg = cfg
    
    def run_sequence(
        self,
        model,  # THzISACModel
        y_seq: List[np.ndarray],
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Run EKF over observation sequence.
        
        Args:
            model: THzISACModel instance
            y_seq: Observations
            x0: Initial state
            P0: Initial covariance
            
        Returns:
            x_hat_seq: Estimated states
            info: Diagnostic information
        """
        x = x0.copy()
        P = P0.copy()
        
        F = model.F_jacobian()
        Q = model.Q_cov()
        
        x_hat_seq = []
        nis_seq = []
        gated_seq = []
        
        for k, y in enumerate(y_seq):
            # Prediction
            x_pred = model.transition(x)
            if self.cfg.wrap_phase:
                x_pred = wrap_state(x_pred, self.cfg.phase_idx)
            P_pred = F @ P @ F.T + Q
            
            # Measurement prediction
            h_pred = model.h(x_pred, k)
            H = model.jacobian(x_pred, k)
            
            # Real-stacked innovation
            r = y - h_pred
            r_real = np.concatenate([r.real, r.imag])
            H_real = np.concatenate([H.real, H.imag])
            
            # Measurement covariance
            R = (model.sigma_eff_sq / 2.0) * np.eye(len(r_real))
            
            # Innovation covariance
            S = H_real @ P_pred @ H_real.T + R
            
            # Update
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
            
            K = P_pred @ H_real.T @ S_inv
            
            # State update
            dx = K @ r_real
            x = x_pred + dx
            
            # Wrap phase after update
            if self.cfg.wrap_phase:
                x = wrap_state(x, self.cfg.phase_idx)
            
            # Covariance update (Joseph form for stability)
            I_KH = np.eye(3) - K @ H_real
            P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
            
            x_hat_seq.append(x.copy())
            nis_seq.append(0.0)  # Not computed for basic wrapped EKF
            gated_seq.append(False)
        
        info = {
            'nis_seq': nis_seq,
            'gated_seq': gated_seq,
            'n_gated': sum(gated_seq),
        }
        
        return x_hat_seq, info


class RobustEKF:
    """
    Robust Extended Kalman Filter with gating and R-inflation.
    
    Key features:
    - Phase wrapping (same as WrappedEKF)
    - NIS (Normalized Innovation Squared) gating
    - R inflation when outlier detected
    - Optional update rejection for extreme outliers
    """
    
    def __init__(self, cfg: EKFConfig):
        self.cfg = cfg
    
    def run_sequence(
        self,
        model,  # THzISACModel
        y_seq: List[np.ndarray],
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Run Robust EKF over observation sequence.
        """
        x = x0.copy()
        P = P0.copy()
        
        F = model.F_jacobian()
        Q = model.Q_cov()
        
        x_hat_seq = []
        nis_seq = []
        gated_seq = []
        rejected_seq = []
        
        for k, y in enumerate(y_seq):
            # Prediction
            x_pred = model.transition(x)
            if self.cfg.wrap_phase:
                x_pred = wrap_state(x_pred, self.cfg.phase_idx)
            P_pred = F @ P @ F.T + Q
            
            # Measurement prediction
            h_pred = model.h(x_pred, k)
            H = model.jacobian(x_pred, k)
            
            # Real-stacked innovation
            r = y - h_pred
            r_real = np.concatenate([r.real, r.imag])
            H_real = np.concatenate([H.real, H.imag])
            
            # Base measurement covariance
            R_base = (model.sigma_eff_sq / 2.0) * np.eye(len(r_real))
            
            # Innovation covariance for NIS computation
            S = H_real @ P_pred @ H_real.T + R_base
            
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
            
            # Normalized Innovation Squared
            nis = float(r_real @ S_inv @ r_real)
            nis_seq.append(nis)
            
            # Gating decision
            gated = False
            rejected = False
            R = R_base.copy()
            
            if self.cfg.use_gating:
                # P1-1 Fix: Compute NIS threshold based on observation dimension
                obs_dim = len(r_real)  # This is 2*m (real representation of complex obs)
                
                if self.cfg.nis_threshold is None:
                    # Dynamic threshold based on chi2 distribution
                    nis_thresh = chi2.ppf(self.cfg.nis_confidence, df=obs_dim)
                else:
                    # Use fixed threshold (user override)
                    nis_thresh = self.cfg.nis_threshold
                
                # Reject threshold is typically higher (e.g., 0.999 confidence)
                reject_thresh = self.cfg.reject_threshold if self.cfg.reject_threshold else chi2.ppf(0.999, df=obs_dim)
                
                if nis > reject_thresh:
                    # Extreme outlier - reject update entirely
                    rejected = True
                    rejected_seq.append(True)
                    gated_seq.append(True)
                    
                    # Keep prediction as estimate
                    x = x_pred
                    # Don't update P (or inflate slightly)
                    P = P_pred * 1.1
                    
                    x_hat_seq.append(x.copy())
                    continue
                    
                elif nis > nis_thresh:
                    # Outlier - inflate R
                    gated = True
                    R = R_base * self.cfg.r_inflation
                    
                    # Recompute S with inflated R
                    S = H_real @ P_pred @ H_real.T + R
                    try:
                        S_inv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        S_inv = np.linalg.pinv(S)
            
            gated_seq.append(gated)
            rejected_seq.append(rejected)
            
            # Kalman gain with (possibly inflated) R
            K = P_pred @ H_real.T @ S_inv
            
            # State update
            dx = K @ r_real
            x = x_pred + dx
            
            # Wrap phase
            if self.cfg.wrap_phase:
                x = wrap_state(x, self.cfg.phase_idx)
            
            # Covariance update
            I_KH = np.eye(3) - K @ H_real
            P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
            
            x_hat_seq.append(x.copy())
        
        info = {
            'nis_seq': nis_seq,
            'gated_seq': gated_seq,
            'rejected_seq': rejected_seq,
            'n_gated': sum(gated_seq),
            'n_rejected': sum(rejected_seq),
            'mean_nis': float(np.mean(nis_seq)),
            'max_nis': float(np.max(nis_seq)),
        }
        
        return x_hat_seq, info


# =========================================================================
# Standard EKF (for comparison - no wrapping)
# =========================================================================

class StandardEKF:
    """Standard EKF without wrapping (for ablation)."""
    
    def run_sequence(
        self,
        model,
        y_seq: List[np.ndarray],
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Run standard EKF."""
        x = x0.copy()
        P = P0.copy()
        
        F = model.F_jacobian()
        Q = model.Q_cov()
        
        x_hat_seq = []
        
        for k, y in enumerate(y_seq):
            x_pred = model.transition(x)
            P_pred = F @ P @ F.T + Q
            
            h_pred = model.h(x_pred, k)
            H = model.jacobian(x_pred, k)
            
            r = y - h_pred
            r_real = np.concatenate([r.real, r.imag])
            H_real = np.concatenate([H.real, H.imag])
            R = (model.sigma_eff_sq / 2.0) * np.eye(len(r_real))
            
            S = H_real @ P_pred @ H_real.T + R
            K = P_pred @ H_real.T @ np.linalg.inv(S)
            
            x = x_pred + K @ r_real
            P = (np.eye(3) - K @ H_real) @ P_pred
            
            x_hat_seq.append(x.copy())
        
        return x_hat_seq, {}


# =========================================================================
# Iterated EKF (IEKF)
# =========================================================================

@dataclass
class IEKFConfig:
    """Configuration for Iterated EKF."""
    
    # Number of iterations per update step
    n_iters: int = 3
    
    # Phase wrapping
    wrap_phase: bool = True
    phase_idx: int = 2
    
    # Convergence threshold (stop if delta < tol)
    tol: float = 1e-6


class IteratedEKF:
    """
    Iterated Extended Kalman Filter (IEKF).
    
    Key difference from standard EKF:
    - Re-linearizes around current estimate within each update step
    - Typically 2-3 iterations per frame
    - Better handles nonlinearity than single-iteration EKF
    
    Reference: Bar-Shalom et al., "Estimation with Applications to 
    Tracking and Navigation", Chapter 5
    """
    
    def __init__(self, cfg: IEKFConfig = None):
        self.cfg = cfg if cfg is not None else IEKFConfig()
    
    def run_sequence(
        self,
        model,  # THzISACModel
        y_seq: List[np.ndarray],
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Run IEKF over observation sequence.
        
        Args:
            model: THzISACModel instance
            y_seq: Observations
            x0: Initial state
            P0: Initial covariance
            
        Returns:
            x_hat_seq: Estimated states
            info: Diagnostic information (iteration counts)
        """
        x = x0.copy()
        P = P0.copy()
        
        F = model.F_jacobian()
        Q = model.Q_cov()
        
        x_hat_seq = []
        iter_counts = []
        
        for k, y in enumerate(y_seq):
            # Prediction
            x_pred = model.transition(x)
            P_pred = F @ P @ F.T + Q
            
            # Wrap prediction if enabled
            if self.cfg.wrap_phase:
                x_pred = wrap_state(x_pred, self.cfg.phase_idx)
            
            # Iterated update
            x_iter = x_pred.copy()
            
            for i in range(self.cfg.n_iters):
                # Linearize around current iterate
                h_iter = model.h(x_iter, k)
                H = model.jacobian(x_iter, k)
                
                # Compute residual
                r = y - h_iter
                r_real = np.concatenate([r.real, r.imag])
                H_real = np.concatenate([H.real, H.imag])
                R = (model.sigma_eff_sq / 2.0) * np.eye(len(r_real))
                
                # EKF update equations
                S = H_real @ P_pred @ H_real.T + R
                K = P_pred @ H_real.T @ np.linalg.inv(S + 1e-10 * np.eye(S.shape[0]))
                
                # Update with re-linearization correction
                # x_new = x_pred + K @ (y - h(x_iter) - H @ (x_pred - x_iter))
                correction = r_real + H_real @ (x_iter - x_pred)
                x_new = x_pred + K @ correction
                
                # Wrap phase if enabled
                if self.cfg.wrap_phase:
                    x_new = wrap_state(x_new, self.cfg.phase_idx)
                
                # Check convergence
                delta = np.linalg.norm(x_new - x_iter)
                x_iter = x_new
                
                if delta < self.cfg.tol:
                    break
            
            iter_counts.append(i + 1)
            
            # Final covariance update (using final linearization point)
            H = model.jacobian(x_iter, k)
            H_real = np.concatenate([H.real, H.imag])
            R = (model.sigma_eff_sq / 2.0) * np.eye(len(H_real))
            S = H_real @ P_pred @ H_real.T + R
            K = P_pred @ H_real.T @ np.linalg.inv(S + 1e-10 * np.eye(S.shape[0]))
            P = (np.eye(3) - K @ H_real) @ P_pred
            
            x = x_iter.copy()
            x_hat_seq.append(x.copy())
        
        return x_hat_seq, {'iter_counts': iter_counts, 'avg_iters': np.mean(iter_counts)}


# =========================================================================
# Factory function
# =========================================================================

def create_ekf(variant: str = 'wrapped') -> Any:
    """
    Create EKF variant.
    
    Args:
        variant: 'standard' | 'wrapped' | 'robust' | 'iekf'
        
    Returns:
        EKF instance
    """
    if variant == 'standard':
        return StandardEKF()
    elif variant == 'wrapped':
        cfg = EKFConfig(wrap_phase=True, use_gating=False)
        return WrappedEKF(cfg)
    elif variant == 'robust':
        cfg = EKFConfig(wrap_phase=True, use_gating=True)
        return RobustEKF(cfg)
    elif variant == 'iekf':
        cfg = IEKFConfig(n_iters=3, wrap_phase=True)
        return IteratedEKF(cfg)
    else:
        raise ValueError(f"Unknown variant: {variant}")
