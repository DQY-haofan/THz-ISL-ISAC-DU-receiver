# src/unfolding/du_map.py
"""
Deep Unfolded MAP (DU-MAP) Estimator

Per advisor restructure (2025-12-27):
- Unrolls L layers of Gauss-Newton iterations
- Learnable parameters: damping per layer (λ_l), step scales
- Fast/slow variable handling: ν updated less frequently

This is the MAIN CONTRIBUTION when combined with BCRLB-normalized training.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

from src.inference.energy import build_map_terms, gn_grad_hess, energy_map


@dataclass
class DUMAPConfig:
    """Configuration for Deep Unfolded MAP."""
    
    # Number of unfolded layers
    n_layers: int = 6
    
    # Per-layer damping (learnable in torch version)
    # Starts high (conservative) and decreases
    damping_per_layer: np.ndarray = field(default=None)
    
    # Fast/slow variable control
    step_scale: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.3, 1.0], dtype=float)
    )
    update_mask_every: np.ndarray = field(
        default_factory=lambda: np.array([1, 2, 1], dtype=int)
    )
    
    def __post_init__(self):
        if self.damping_per_layer is None:
            # Default: annealing from 0.1 to 0.001
            self.damping_per_layer = np.logspace(-1, -3, self.n_layers)


class DUMAP:
    """
    Deep Unfolded MAP estimator.
    
    Unrolls Gauss-Newton iterations into L layers:
        x^{l+1} = x^l + S_l * δ_l
        where (H_l + λ_l I) δ_l = -g_l
    
    Key features:
    - Per-layer learnable damping
    - Fast/slow variable handling (ν conservative)
    - Energy decrease tracking for training loss
    """
    
    def __init__(self, cfg: Optional[DUMAPConfig] = None):
        self.cfg = cfg or DUMAPConfig()
    
    def forward(
        self,
        model,
        y: np.ndarray,
        frame_idx: int,
        x_init: np.ndarray,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass through unfolded layers.
        
        Args:
            model: THzISACModel instance
            y: Observation [m] complex
            frame_idx: Frame index
            x_init: Initial state estimate
            x_pred: Prior prediction (from dynamics)
            P_pred: Prior covariance
            
        Returns:
            x_hat: Final state estimate [d]
            info: Layer-wise information for training/analysis
        """
        cfg = self.cfg
        x = x_init.copy()
        
        layer_hist: List[Dict[str, Any]] = []
        E_hist: List[float] = []
        
        for l in range(cfg.n_layers):
            # Compute model outputs at current estimate
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
            
            # Compute energy before step
            E_before = energy_map(terms)
            E_hist.append(E_before)
            
            # Get per-layer damping
            damping = float(cfg.damping_per_layer[l])
            
            # Compute GN step
            grad, H = gn_grad_hess(terms, damping=damping)
            
            try:
                delta = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, -grad, rcond=None)[0]
            
            # Apply fast/slow variable control
            delta = delta * cfg.step_scale
            
            for k in range(len(delta)):
                if (l % int(cfg.update_mask_every[k])) != 0:
                    delta[k] = 0.0
            
            # Update state
            x_new = x + delta
            x_new = model.wrap_phase_norm(x_new)
            
            # Compute energy after step (for training loss)
            h_new = model.h(x_new, frame_idx)
            J_new = model.jacobian(x_new, frame_idx)
            terms_new = build_map_terms(
                y=y, h=h_new, J=J_new,
                sigma_eff_sq=model.sigma_eff_sq,
                x=x_new, x_pred=x_pred, P_pred=P_pred,
            )
            E_after = energy_map(terms_new)
            
            # Record layer info
            layer_hist.append({
                "layer": l,
                "damping": damping,
                "delta_norm": float(np.linalg.norm(delta)),
                "E_before": E_before,
                "E_after": E_after,
                "energy_decreased": E_after < E_before,
            })
            
            x = x_new
        
        # Final energy
        E_hist.append(E_after)
        
        info = {
            "method": "du_map",
            "n_layers": cfg.n_layers,
            "layer_hist": layer_hist,
            "E_hist": E_hist,
            "E_initial": E_hist[0],
            "E_final": E_hist[-1],
            "total_energy_decrease": E_hist[0] - E_hist[-1],
            "all_layers_decreased": all(lh["energy_decreased"] for lh in layer_hist),
        }
        
        return x, info
    
    def forward_sequence(
        self,
        model,
        y_seq: List[np.ndarray],
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Forward pass over a sequence of frames.
        
        Args:
            model: THzISACModel instance
            y_seq: Observations [n_frames]
            x0: Initial state
            P0: Initial covariance
            
        Returns:
            x_hat_seq: Estimated states [n_frames]
            info: Sequence-level information
        """
        x_hat_seq: List[np.ndarray] = []
        x = x0.copy()
        P = P0.copy()
        
        F = model.F_jacobian()
        Q = model.Q_cov()
        
        total_energy_decrease = 0.0
        all_decreased = True
        
        for k, y in enumerate(y_seq):
            # Prediction
            x_pred = model.transition(x)
            P_pred = F @ P @ F.T + Q
            
            # DU-MAP forward
            x_hat, info = self.forward(
                model=model,
                y=y,
                frame_idx=k,
                x_init=x_pred,
                x_pred=x_pred,
                P_pred=P_pred,
            )
            
            x_hat_seq.append(x_hat)
            total_energy_decrease += info.get("total_energy_decrease", 0.0)
            all_decreased = all_decreased and info.get("all_layers_decreased", True)
            
            # Update for next frame
            x = x_hat.copy()
            
            # Approximate posterior covariance using Hessian at solution
            h = model.h(x_hat, k)
            J = model.jacobian(x_hat, k)
            Jr = np.concatenate([J.real, J.imag], axis=0)
            W = (2.0 / max(model.sigma_eff_sq, 1e-12)) * np.eye(Jr.shape[0])
            Pinv = np.linalg.inv(P_pred + 1e-12 * np.eye(3))
            H_gn = 2.0 * Jr.T @ W @ Jr + 2.0 * Pinv
            try:
                P = np.linalg.inv(H_gn + 1e-10 * np.eye(3)) * 2.0
            except np.linalg.LinAlgError:
                P = P_pred * 0.5
        
        seq_info = {
            "n_frames": len(y_seq),
            "total_energy_decrease": total_energy_decrease,
            "all_layers_decreased": all_decreased,
        }
        
        return x_hat_seq, seq_info
