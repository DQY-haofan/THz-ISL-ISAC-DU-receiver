"""
Unscented Kalman Filter (UKF) Baseline

Per advisor suggestion: UKF/CKF 是低成本高性价比的基线，
能补齐"非线性滤波家族"对比。

Reference: Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation"
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np


def wrap_to_pi(x: float) -> float:
    """Wrap angle to [-π, π]."""
    return float((x + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class UKFConfig:
    """Configuration for UKF."""
    
    # Sigma point parameters
    alpha: float = 1e-3  # Spread of sigma points
    beta: float = 2.0    # Prior knowledge (2 for Gaussian)
    kappa: float = 0.0   # Secondary scaling parameter
    
    # Phase wrapping
    wrap_phase: bool = True
    phase_idx: int = 2


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for nonlinear state estimation.
    
    Uses sigma points to propagate mean and covariance through
    nonlinear functions without linearization.
    """
    
    def __init__(self, cfg: UKFConfig = None):
        self.cfg = cfg if cfg is not None else UKFConfig()
    
    def _compute_sigma_weights(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute weights for sigma points."""
        alpha = self.cfg.alpha
        beta = self.cfg.beta
        kappa = self.cfg.kappa
        
        lam = alpha**2 * (n + kappa) - n
        
        # Weights for mean
        Wm = np.full(2*n + 1, 1.0 / (2*(n + lam)))
        Wm[0] = lam / (n + lam)
        
        # Weights for covariance
        Wc = np.full(2*n + 1, 1.0 / (2*(n + lam)))
        Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)
        
        return Wm, Wc
    
    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points from mean and covariance."""
        n = len(x)
        alpha = self.cfg.alpha
        kappa = self.cfg.kappa
        lam = alpha**2 * (n + kappa) - n
        
        # Matrix square root
        try:
            sqrt_P = np.linalg.cholesky((n + lam) * P)
        except np.linalg.LinAlgError:
            # Fallback if not positive definite
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-10)
            sqrt_P = eigvecs @ np.diag(np.sqrt((n + lam) * eigvals)) @ eigvecs.T
        
        # Sigma points: [x, x + sqrt_P, x - sqrt_P]
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = x
        
        for i in range(n):
            sigma_points[i + 1] = x + sqrt_P[i]
            sigma_points[n + i + 1] = x - sqrt_P[i]
        
        return sigma_points
    
    def run_sequence(
        self,
        model,  # THzISACModel
        y_seq: List[np.ndarray],
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Run UKF over observation sequence.
        
        Args:
            model: THzISACModel instance
            y_seq: Observations
            x0: Initial state
            P0: Initial covariance
            
        Returns:
            x_hat_seq: Estimated states
            info: Diagnostic information
        """
        n = len(x0)
        x = x0.copy()
        P = P0.copy()
        
        F = model.F_jacobian()
        Q = model.Q_cov()
        
        Wm, Wc = self._compute_sigma_weights(n)
        
        x_hat_seq = []
        
        for k, y in enumerate(y_seq):
            # === PREDICTION ===
            # Generate sigma points
            sigma_pts = self._generate_sigma_points(x, P)
            
            # Propagate through transition
            sigma_pts_pred = np.zeros_like(sigma_pts)
            for i in range(2*n + 1):
                sigma_pts_pred[i] = model.transition(sigma_pts[i])
                if self.cfg.wrap_phase:
                    sigma_pts_pred[i, self.cfg.phase_idx] = wrap_to_pi(
                        sigma_pts_pred[i, self.cfg.phase_idx]
                    )
            
            # Predicted mean
            x_pred = np.sum(Wm[:, np.newaxis] * sigma_pts_pred, axis=0)
            if self.cfg.wrap_phase:
                x_pred[self.cfg.phase_idx] = wrap_to_pi(x_pred[self.cfg.phase_idx])
            
            # Predicted covariance
            P_pred = Q.copy()
            for i in range(2*n + 1):
                dx = sigma_pts_pred[i] - x_pred
                if self.cfg.wrap_phase:
                    dx[self.cfg.phase_idx] = wrap_to_pi(dx[self.cfg.phase_idx])
                P_pred += Wc[i] * np.outer(dx, dx)
            
            # === UPDATE ===
            # Generate new sigma points from predicted state
            sigma_pts = self._generate_sigma_points(x_pred, P_pred)
            
            # Propagate through measurement
            m = len(y)
            sigma_y = np.zeros((2*n + 1, m), dtype=complex)
            for i in range(2*n + 1):
                sigma_y[i] = model.h(sigma_pts[i], k)
            
            # Predicted measurement
            y_pred = np.sum(Wm[:, np.newaxis] * sigma_y, axis=0)
            
            # Innovation covariance (complex to real stacking)
            r = y - y_pred
            r_real = np.concatenate([r.real, r.imag])
            
            # Measurement covariance S
            R = (model.sigma_eff_sq / 2.0) * np.eye(2*m)
            S = R.copy()
            
            for i in range(2*n + 1):
                dy = sigma_y[i] - y_pred
                dy_real = np.concatenate([dy.real, dy.imag])
                S += Wc[i] * np.outer(dy_real, dy_real)
            
            # Cross-covariance
            Pxy = np.zeros((n, 2*m))
            for i in range(2*n + 1):
                dx = sigma_pts[i] - x_pred
                if self.cfg.wrap_phase:
                    dx[self.cfg.phase_idx] = wrap_to_pi(dx[self.cfg.phase_idx])
                dy = sigma_y[i] - y_pred
                dy_real = np.concatenate([dy.real, dy.imag])
                Pxy += Wc[i] * np.outer(dx, dy_real)
            
            # Kalman gain
            try:
                K = Pxy @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = Pxy @ np.linalg.pinv(S)
            
            # Update
            x = x_pred + K @ r_real
            if self.cfg.wrap_phase:
                x[self.cfg.phase_idx] = wrap_to_pi(x[self.cfg.phase_idx])
            
            P = P_pred - K @ S @ K.T
            
            x_hat_seq.append(x.copy())
        
        return x_hat_seq, {'method': 'ukf'}


# Factory function
def create_ukf() -> UnscentedKalmanFilter:
    """Create UKF with default configuration."""
    return UnscentedKalmanFilter(UKFConfig())
