# src/inference/gn_solver.py
"""
Pure Gauss-Newton MAP Solver (No Learning)

Per advisor restructure (2025-12-27):
- This is the BASELINE before deep unfolding
- Supports fast/slow variable handling (ν is slow)
- Must demonstrate energy decrease before DU can be trusted

Key features:
- Damped GN iterations
- Variable-specific step scaling (ν gets smaller steps)
- Update frequency control (ν updated less often)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

from src.inference.energy import build_map_terms, gn_grad_hess, energy_map


@dataclass
class GNSolverConfig:
    """Configuration for Gauss-Newton MAP solver."""
    
    # Iteration control
    max_iters: int = 8
    tol: float = 1e-6
    
    # Damping (Levenberg-Marquardt style)
    damping: float = 1e-2
    
    # Fast/slow variable control
    # [τ, ν, φ] - ν is slow variable
    step_scale: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.3, 1.0], dtype=float)
    )
    
    # Update mask: update variable every N iterations
    # ν updated every 2 iters (more conservative)
    update_mask_every: np.ndarray = field(
        default_factory=lambda: np.array([1, 2, 1], dtype=int)
    )
    
    # Preconditioning (information-geometric)
    use_preconditioner: bool = True
    precond_eps: float = 1e-6


class GaussNewtonMAP:
    """
    Pure Gauss-Newton MAP solver.
    
    This is the BASELINE for THz-ISAC state estimation.
    Must work before deep unfolding can be applied.
    """
    
    def __init__(self, cfg: Optional[GNSolverConfig] = None):
        self.cfg = cfg or GNSolverConfig()
    
    def solve(
        self,
        model,
        y: np.ndarray,
        frame_idx: int,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        x_init: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve MAP estimation problem.
        
        min_x ||y - h(x)||²_R + ||x - x_pred||²_P
        
        Args:
            model: THzISACModel instance
            y: Observation [m] complex
            frame_idx: Frame index
            x_pred: Prior state prediction [d]
            P_pred: Prior covariance [d, d]
            x_init: Initial guess (default: x_pred)
            
        Returns:
            x_hat: Estimated state [d]
            info: Solver information dict
        """
        cfg = self.cfg
        
        # Initialize
        x = x_pred.copy() if x_init is None else x_init.copy()
        
        E_hist: List[float] = []
        delta_norm_hist: List[float] = []
        
        for it in range(cfg.max_iters):
            # Compute model outputs
            h = model.h(x, frame_idx)
            J = model.jacobian(x, frame_idx)
            
            # Build MAP terms
            terms = build_map_terms(
                y=y,
                h=h,
                J=J,
                sigma_eff_sq=model.sigma_eff_sq,
                x=x,
                x_pred=x_pred,
                P_pred=P_pred,
            )
            
            # Compute energy
            E = energy_map(terms)
            E_hist.append(E)
            
            # Compute GN step
            grad, H = gn_grad_hess(terms, damping=cfg.damping)
            
            # Apply diagonal preconditioning (information-geometric)
            if cfg.use_preconditioner:
                D = np.sqrt(np.diag(H) + cfg.precond_eps)
                D_inv = 1.0 / D
                # Preconditioned system: (D^{-1} H D^{-1}) (D δ) = -D^{-1} g
                H_precond = np.diag(D_inv) @ H @ np.diag(D_inv)
                grad_precond = D_inv * grad
                
                try:
                    delta_precond = np.linalg.solve(H_precond, -grad_precond)
                except np.linalg.LinAlgError:
                    delta_precond = np.linalg.lstsq(H_precond, -grad_precond, rcond=None)[0]
                
                delta = D_inv * delta_precond
            else:
                try:
                    delta = np.linalg.solve(H, -grad)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(H, -grad, rcond=None)[0]
            
            # Apply fast/slow variable control
            delta = delta * cfg.step_scale
            
            for k in range(len(delta)):
                # Update variable only every N iterations
                if (it % int(cfg.update_mask_every[k])) != 0:
                    delta[k] = 0.0
            
            # Update state
            x = x + delta
            x = model.wrap_phase_norm(x)
            
            delta_norm = float(np.linalg.norm(delta))
            delta_norm_hist.append(delta_norm)
            
            # Check convergence
            if delta_norm < cfg.tol:
                break
        
        # Final energy
        h_final = model.h(x, frame_idx)
        J_final = model.jacobian(x, frame_idx)
        terms_final = build_map_terms(
            y=y, h=h_final, J=J_final,
            sigma_eff_sq=model.sigma_eff_sq,
            x=x, x_pred=x_pred, P_pred=P_pred,
        )
        E_final = energy_map(terms_final)
        E_hist.append(E_final)
        
        info = {
            "method": "gn_map",
            "iters": len(delta_norm_hist),
            "converged": delta_norm_hist[-1] < cfg.tol if delta_norm_hist else False,
            "E_hist": E_hist,
            "delta_norm_hist": delta_norm_hist,
            "E_initial": E_hist[0] if E_hist else None,
            "E_final": E_final,
            "energy_decrease": E_hist[0] - E_final if E_hist else 0.0,
        }
        
        return x, info
    
    def solve_sequence(
        self,
        model,
        y_seq: List[np.ndarray],
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Solve MAP estimation for a sequence of frames.
        
        Uses model dynamics for prediction between frames.
        
        Args:
            model: THzISACModel instance
            y_seq: List of observations [n_frames]
            x0: Initial state
            P0: Initial covariance
            
        Returns:
            x_hat_seq: List of estimated states
            info: Sequence-level information
        """
        x_hat_seq: List[np.ndarray] = []
        x = x0.copy()
        P = P0.copy()
        
        total_iters = 0
        total_energy_decrease = 0.0
        
        F = model.F_jacobian()
        Q = model.Q_cov()
        
        for k, y in enumerate(y_seq):
            # Prediction
            x_pred = model.transition(x)
            P_pred = F @ P @ F.T + Q
            
            # Solve
            x_hat, info = self.solve(
                model=model,
                y=y,
                frame_idx=k,
                x_pred=x_pred,
                P_pred=P_pred,
            )
            
            x_hat_seq.append(x_hat)
            total_iters += info["iters"]
            total_energy_decrease += info.get("energy_decrease", 0.0)
            
            # Update for next iteration
            x = x_hat.copy()
            
            # Approximate posterior covariance using Hessian at solution
            # P_post ≈ (H_gn)^{-1} = (2 J^T W J + 2 P_pred^{-1})^{-1}
            h = model.h(x_hat, k)
            J = model.jacobian(x_hat, k)
            Jr = np.concatenate([J.real, J.imag], axis=0)
            W = (2.0 / max(model.sigma_eff_sq, 1e-12)) * np.eye(Jr.shape[0])
            Pinv = np.linalg.inv(P_pred + 1e-12 * np.eye(3))
            H_gn = 2.0 * Jr.T @ W @ Jr + 2.0 * Pinv
            try:
                P = np.linalg.inv(H_gn + 1e-10 * np.eye(3)) * 2.0  # Factor of 2 for Hessian scaling
            except np.linalg.LinAlgError:
                P = P_pred * 0.5  # Fallback
        
        seq_info = {
            "n_frames": len(y_seq),
            "total_iters": total_iters,
            "avg_iters": total_iters / len(y_seq) if y_seq else 0,
            "total_energy_decrease": total_energy_decrease,
        }
        
        return x_hat_seq, seq_info
