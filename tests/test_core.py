"""核心模块测试"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.physics.thz_isac_model import THzISACConfig, THzISACModel, wrap_angle
from src.inference.gn_solver import GNSolverConfig, GaussNewtonMAP
from src.unfolding.du_map import DUMAP, DUMAPConfig
from src.baselines.wrapped_ekf import create_ekf
from src.sim.slip import generate_episode_with_impairments, get_slip_config


class TestTHzISACModel:
    def test_create(self):
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
        model = THzISACModel(cfg)
        assert model.m == 32  # 8 * 4

    def test_observation(self):
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=20, adc_bits=4)
        model = THzISACModel(cfg)
        x = np.array([1.0, 0.5, 0.0])
        y = model.observe(x, frame_idx=0)
        assert y.shape == (32,)
        assert np.iscomplexobj(y)

    def test_jacobian_shape(self):
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10)
        model = THzISACModel(cfg)
        x = np.array([1.0, 0.5, 0.0])
        J = model.jacobian(x, 0)
        assert J.shape == (32, 3)


class TestGNSolver:
    def test_single_frame(self):
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=20, adc_bits=4)
        model = THzISACModel(cfg)
        x0 = np.array([1.0, 0.5, 0.0])
        P0 = np.eye(3) * 0.1
        
        y = model.observe(x0, 0)
        gn = GaussNewtonMAP(GNSolverConfig(max_iters=5))
        x_hat, info = gn.solve(model, y, 0, x0, P0)
        
        assert x_hat.shape == (3,)
        assert info['iters'] <= 5


class TestDUMAP:
    def test_gn_du_equivalence(self):
        """当配置相同时，GN和DU应该产生相同结果"""
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
        model = THzISACModel(cfg)
        x0 = np.array([1.0, 0.5, 0.0])
        P0 = np.eye(3) * 0.1
        
        y_seq, x_true_seq, _, _ = generate_episode_with_impairments(
            model, 5, x0, slip_cfg=None, pn_cfg=None, seed=42
        )
        
        # 完全相同的配置
        gn_cfg = GNSolverConfig(max_iters=6, damping=0.01, 
                                step_scale=np.array([1.0, 0.3, 1.0]))
        du_cfg = DUMAPConfig(n_layers=6)
        du_cfg.damping_per_layer = np.array([0.01] * 6)
        du_cfg.step_scale = np.array([1.0, 0.3, 1.0])
        
        gn = GaussNewtonMAP(gn_cfg)
        du = DUMAP(du_cfg)
        
        gn_hat, _ = gn.solve_sequence(model, y_seq, x0, P0)
        du_hat, _ = du.forward_sequence(model, y_seq, x0, P0)
        
        for g, d in zip(gn_hat, du_hat):
            np.testing.assert_allclose(g, d, atol=1e-10)


class TestSlipGeneration:
    def test_slip_rate(self):
        """验证slip发生率"""
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
        model = THzISACModel(cfg)
        x0 = np.array([1.0, 0.5, 0.0])
        slip_cfg = get_slip_config('severe')  # p_slip = 0.05
        
        n_trials = 100
        n_frames = 100
        slip_counts = []
        
        for seed in range(n_trials):
            _, _, slip_frames, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed
            )
            slip_counts.append(len(slip_frames))
        
        avg_slips = np.mean(slip_counts)
        expected = n_frames * 0.05
        assert abs(avg_slips - expected) / expected < 0.2  # 20%误差内


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
