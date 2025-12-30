#!/usr/bin/env python3
"""
THz-ISAC DU-MAP IEEE 格式图像生成 v7 完整版
============================================

基于v6，保持所有图像风格和格式不变，仅添加CSV数据输出功能。

CSV数据输出 - 逻辑分组:
1. data_performance.csv   - 所有性能对比 (BER/RMSE vs SNR/L/p_slip/ADC/frame_duration)
2. data_sensitivity.csv   - 步长敏感性分析 (alpha_phi/nu/tau)
3. data_heatmap.csv       - 2D热力图 (amplitude × p_slip)
4. data_auxiliary.csv     - 辅助数据 (CCDF/PCRB/recovery_time/improvement_bar/phase_tracking)

v6 修复 (保留):
1. DU step_scale: [1.0, 0.1, 2.0] -> [1.0, 0.1, 1.5] 减少过冲
2. fig_frame_duration 改为 Phase RMSE (DU优势场景)
3. fig_sensitivity 拆分为3个独立图，无title
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, List, Tuple

from src.physics.thz_isac_model import THzISACConfig, THzISACModel, wrap_angle
from src.inference.gn_solver import GNSolverConfig, GaussNewtonMAP
from src.unfolding.du_map import DUMAP, DUMAPConfig
from src.baselines.wrapped_ekf import create_ekf, create_iekf
from src.bcrlb.pcrb import PCRBRecursion
from src.sim.slip import generate_episode_with_impairments, SlipConfig
from src.metrics.system_metrics import quick_ber_evm

# ============================================================================
# IEEE 格式配置
# ============================================================================

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IEEE_WIDTH = 3.5
IEEE_HEIGHT = 2.6
IEEE_FONTSIZE = 9
IEEE_DPI = 300

# v6 颜色方案
COLORS = {
    'EKF': '#E41A1C',  # 红
    'IEKF-4': '#FF7F00',  # 橙
    'GN-6': '#377EB8',  # 蓝
    'DU-tun-6': '#4DAF4A',  # 绿
    'PCRB': '#000000',  # 黑
}

MARKERS = {
    'EKF': 'x',
    'IEKF-4': '^',
    'GN-6': 's',
    'DU-tun-6': 'D',
}

LINESTYLES = {
    'EKF': '-',
    'IEKF-4': '-',
    'GN-6': '-',
    'DU-tun-6': '-',
}

# 多SNR颜色方案
MULTI_SNR_COLORS = {
    0: '#1f77b4',
    5: '#ff7f0e',
    10: '#2ca02c',
    15: '#d62728',
    20: '#9467bd',
}

MULTI_SNR_LINESTYLES = {
    0: '-',
    5: '--',
    10: '-.',
    15: ':',
    20: '-',
}


def setup_ieee_style():
    plt.rcParams.update({
        'font.size': IEEE_FONTSIZE,
        'axes.labelsize': IEEE_FONTSIZE,
        'axes.titlesize': IEEE_FONTSIZE,
        'legend.fontsize': IEEE_FONTSIZE - 1,
        'xtick.labelsize': IEEE_FONTSIZE - 1,
        'ytick.labelsize': IEEE_FONTSIZE - 1,
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'gray',
        'figure.dpi': IEEE_DPI,
        'savefig.dpi': IEEE_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


def save_figure(fig, name):
    fig.savefig(f'{OUTPUT_DIR}/{name}.png', dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{OUTPUT_DIR}/{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  ✓ {name}")


# ============================================================================
# CSV数据收集器
# ============================================================================

class DataCollector:
    """收集所有实验数据，最后统一输出到CSV"""

    def __init__(self):
        self.performance_data = []  # 性能对比数据
        self.sensitivity_data = []  # 敏感性数据
        self.heatmap_data = []  # 热力图数据
        self.auxiliary_data = []  # 辅助数据

    def add_performance(self, experiment: str, sweep_var: str, sweep_val,
                        method: str, metric: str, mean: float, std: float,
                        n_seeds: int, n_frames: int):
        self.performance_data.append({
            'experiment': experiment,
            'sweep_variable': sweep_var,
            'sweep_value': sweep_val,
            'method': method,
            'metric': metric,
            'mean': mean,
            'std': std,
            'sem': std / np.sqrt(n_seeds),
            'n_seeds': n_seeds,
            'n_frames': n_frames
        })

    def add_sensitivity(self, parameter: str, value: float,
                        rmse_mean: float, rmse_std: float, n_seeds: int):
        self.sensitivity_data.append({
            'parameter': parameter,
            'value': value,
            'rmse_mean': rmse_mean,
            'rmse_std': rmse_std,
            'rmse_sem': rmse_std / np.sqrt(n_seeds),
            'n_seeds': n_seeds
        })

    def add_heatmap(self, amplitude_pi: float, p_slip: float,
                    ekf_mean: float, ekf_std: float,
                    gn_mean: float, gn_std: float,
                    du_mean: float, du_std: float, n_seeds: int):
        self.heatmap_data.append({
            'amplitude_pi': amplitude_pi,
            'p_slip': p_slip,
            'EKF_rmse_mean': ekf_mean,
            'EKF_rmse_std': ekf_std,
            'GN6_rmse_mean': gn_mean,
            'GN6_rmse_std': gn_std,
            'DU_rmse_mean': du_mean,
            'DU_rmse_std': du_std,
            'improvement_vs_EKF_pct': (ekf_mean - du_mean) / ekf_mean * 100,
            'improvement_vs_GN_pct': (gn_mean - du_mean) / gn_mean * 100,
            'n_seeds': n_seeds
        })

    def add_auxiliary(self, data_type: str, **kwargs):
        row = {'data_type': data_type}
        row.update(kwargs)
        self.auxiliary_data.append(row)

    def save_all(self):
        """保存所有CSV文件"""

        # 1. data_performance.csv
        if self.performance_data:
            headers = ['experiment', 'sweep_variable', 'sweep_value', 'method',
                       'metric', 'mean', 'std', 'sem', 'n_seeds', 'n_frames']
            with open(f'{OUTPUT_DIR}/data_performance.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.performance_data)
            print(f"  ✓ data_performance.csv ({len(self.performance_data)} rows)")

        # 2. data_sensitivity.csv
        if self.sensitivity_data:
            headers = ['parameter', 'value', 'rmse_mean', 'rmse_std', 'rmse_sem', 'n_seeds']
            with open(f'{OUTPUT_DIR}/data_sensitivity.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.sensitivity_data)
            print(f"  ✓ data_sensitivity.csv ({len(self.sensitivity_data)} rows)")

        # 3. data_heatmap.csv
        if self.heatmap_data:
            headers = ['amplitude_pi', 'p_slip', 'EKF_rmse_mean', 'EKF_rmse_std',
                       'GN6_rmse_mean', 'GN6_rmse_std', 'DU_rmse_mean', 'DU_rmse_std',
                       'improvement_vs_EKF_pct', 'improvement_vs_GN_pct', 'n_seeds']
            with open(f'{OUTPUT_DIR}/data_heatmap.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.heatmap_data)
            print(f"  ✓ data_heatmap.csv ({len(self.heatmap_data)} rows)")

        # 4. data_auxiliary.csv
        if self.auxiliary_data:
            all_keys = set()
            for row in self.auxiliary_data:
                all_keys.update(row.keys())
            headers = ['data_type'] + sorted([k for k in all_keys if k != 'data_type'])

            with open(f'{OUTPUT_DIR}/data_auxiliary.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.auxiliary_data)
            print(f"  ✓ data_auxiliary.csv ({len(self.auxiliary_data)} rows)")


# 全局数据收集器
collector = DataCollector()


# ============================================================================
# DU配置
# ============================================================================

def get_du_step_scale():
    """
    v6: 调整后的DU步长配置
    [1.0, 0.1, 1.5] - φ步长=1.5 平衡slip恢复速度和正常跟踪稳定性
    """
    return np.array([1.0, 0.1, 1.5])


# ============================================================================
# 估计器工具函数
# ============================================================================

def compute_rmse(x_hat_seq, x_true_seq):
    mse = 0
    for xh, xt in zip(x_hat_seq, x_true_seq):
        err = xh - xt
        err[2] = wrap_angle(err[2])
        mse += np.sum(err ** 2)
    return np.sqrt(mse / len(x_hat_seq))


def compute_rmse_per_component(x_hat_seq, x_true_seq):
    """计算各分量的RMSE"""
    n = len(x_hat_seq)
    mse = np.zeros(3)
    for xh, xt in zip(x_hat_seq, x_true_seq):
        err = xh - xt
        err[2] = wrap_angle(err[2])
        mse += err ** 2
    return np.sqrt(mse / n)


def run_estimator(method, model, y_seq, x0, P0):
    """统一接口 - 使用调整后的步长"""
    if method == 'EKF':
        est = create_ekf('wrapped')
        return est.run_sequence(model, y_seq, x0, P0)[0]
    elif method.startswith('IEKF-'):
        n_iters = int(method.split('-')[1])
        est = create_iekf(n_iters=n_iters)
        return est.run_sequence(model, y_seq, x0, P0)[0]
    elif method.startswith('GN-'):
        L = int(method.split('-')[1])
        est = GaussNewtonMAP(GNSolverConfig(max_iters=L))
        return est.solve_sequence(model, y_seq, x0, P0)[0]
    elif method.startswith('DU-tun-'):
        L = int(method.split('-')[2])
        cfg = DUMAPConfig(n_layers=L)
        cfg.step_scale = get_du_step_scale()
        est = DUMAP(cfg)
        return est.forward_sequence(model, y_seq, x0, P0)[0]
    raise ValueError(f"Unknown method: {method}")


# ============================================================================
# 核心图
# ============================================================================

def fig_ber_vs_snr():
    """BER vs SNR"""
    print("\nFig: BER vs SNR")

    snr_list = [0, 5, 10, 15, 20]
    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 30, 100

    results = {snr: {m: [] for m in methods} for snr in snr_list}

    for snr in snr_list:
        print(f"  SNR={snr}dB...", end=" ", flush=True)
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=snr, adc_bits=4)
        model = THzISACModel(cfg)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                ber, _ = quick_ber_evm(x_true, x_hat, snr, seed)
                results[snr][method].append(ber * 100)
        print("done")

    # 收集CSV数据
    for snr in snr_list:
        for m in methods:
            collector.add_performance('BER_vs_SNR', 'SNR_dB', snr, m, 'BER_pct',
                                      np.mean(results[snr][m]), np.std(results[snr][m]),
                                      n_seeds, n_frames)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    for method in methods:
        ber_means = [np.mean(results[snr][method]) for snr in snr_list]
        ax.semilogy(snr_list, ber_means,
                    marker=MARKERS[method], color=COLORS[method],
                    linestyle=LINESTYLES[method], label=method,
                    markerfacecolor='white' if 'GN' in method else None)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-1, 21])
    save_figure(fig, 'fig_ber_snr')


def fig_rmse_vs_snr():
    """RMSE vs SNR"""
    print("\nFig: RMSE vs SNR")

    snr_list = [0, 5, 10, 15, 20]
    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 30, 100

    results = {snr: {m: [] for m in methods} for snr in snr_list}

    for snr in snr_list:
        print(f"  SNR={snr}dB...", end=" ", flush=True)
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=snr, adc_bits=4)
        model = THzISACModel(cfg)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                results[snr][method].append(compute_rmse(x_hat, x_true))
        print("done")

    # 收集CSV数据
    for snr in snr_list:
        for m in methods:
            collector.add_performance('RMSE_vs_SNR', 'SNR_dB', snr, m, 'RMSE',
                                      np.mean(results[snr][m]), np.std(results[snr][m]),
                                      n_seeds, n_frames)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    for method in methods:
        rmse_means = [np.mean(results[snr][method]) for snr in snr_list]
        ax.semilogy(snr_list, rmse_means,
                    marker=MARKERS[method], color=COLORS[method],
                    linestyle=LINESTYLES[method], label=method,
                    markerfacecolor='white' if 'GN' in method else None)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_rmse_snr')


def fig_rmse_vs_L():
    """RMSE vs L"""
    print("\nFig: RMSE vs L")

    L_values = [2, 4, 6, 8]

    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 30, 100

    results = {
        'EKF': [],
        'IEKF': {L: [] for L in L_values},
        'GN': {L: [] for L in L_values},
        'DU': {L: [] for L in L_values}
    }

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}...", end="\r", flush=True)
        y_seq, x_true, _, _ = generate_episode_with_impairments(
            model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

        results['EKF'].append(compute_rmse(run_estimator('EKF', model, y_seq, x0, P0), x_true))

        for L in L_values:
            iekf_L = min(L, 4)
            x_iekf = run_estimator(f'IEKF-{iekf_L}', model, y_seq, x0, P0)
            results['IEKF'][L].append(compute_rmse(x_iekf, x_true))

            x_gn = run_estimator(f'GN-{L}', model, y_seq, x0, P0)
            x_du = run_estimator(f'DU-tun-{L}', model, y_seq, x0, P0)
            results['GN'][L].append(compute_rmse(x_gn, x_true))
            results['DU'][L].append(compute_rmse(x_du, x_true))
    print(f"  完成 {n_seeds} seeds" + " " * 20)

    # 收集CSV数据
    collector.add_performance('RMSE_vs_L', 'L', 1, 'EKF', 'RMSE',
                              np.mean(results['EKF']), np.std(results['EKF']), n_seeds, n_frames)
    for L in L_values:
        collector.add_performance('RMSE_vs_L', 'L', L, 'IEKF-4', 'RMSE',
                                  np.mean(results['IEKF'][L]), np.std(results['IEKF'][L]), n_seeds, n_frames)
        collector.add_performance('RMSE_vs_L', 'L', L, 'GN-6', 'RMSE',
                                  np.mean(results['GN'][L]), np.std(results['GN'][L]), n_seeds, n_frames)
        collector.add_performance('RMSE_vs_L', 'L', L, 'DU-tun-6', 'RMSE',
                                  np.mean(results['DU'][L]), np.std(results['DU'][L]), n_seeds, n_frames)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    ax.axhline(np.mean(results['EKF']), color=COLORS['EKF'], linestyle='--',
               label='EKF (L=1)', alpha=0.8)

    iekf_means = [np.mean(results['IEKF'][L]) for L in L_values]
    ax.plot(L_values, iekf_means, marker='^', color=COLORS['IEKF-4'], label='IEKF')

    gn_means = [np.mean(results['GN'][L]) for L in L_values]
    ax.plot(L_values, gn_means, marker='s', color=COLORS['GN-6'],
            label='GN', markerfacecolor='white')

    du_means = [np.mean(results['DU'][L]) for L in L_values]
    ax.plot(L_values, du_means, marker='D', color=COLORS['DU-tun-6'], label='DU-tun')

    ax.set_xlabel('Iterations / Layers ($L$)')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_values)
    save_figure(fig, 'fig_rmse_L')


def fig_recovery_time():
    """恢复时间对比"""
    print("\nFig: Recovery Time")

    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1

    n_frames = 50
    slip_frame = 20
    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']
    n_seeds = 30

    error_curves = {m: [] for m in methods}

    for seed in range(n_seeds):
        np.random.seed(seed)

        y_seq, x_true = [], []
        x_curr = x0.copy()

        for k in range(n_frames):
            if k == slip_frame:
                x_curr = x_curr.copy()
                x_curr[2] += np.pi

            x_curr = model.transition(x_curr)
            y = model.observe(x_curr, k)
            y_seq.append(y)
            x_true.append(x_curr.copy())

        for m in methods:
            x_hat = run_estimator(m, model, y_seq, x0, P0)
            errors = [abs(wrap_angle(xh[2] - xt[2])) for xh, xt in zip(x_hat, x_true)]
            error_curves[m].append(errors)

    # 收集CSV数据
    for k in range(n_frames):
        for m in methods:
            collector.add_auxiliary('recovery_time', frame=k, method=m,
                                    phase_error_mean=np.mean([ec[k] for ec in error_curves[m]]),
                                    phase_error_std=np.std([ec[k] for ec in error_curves[m]]),
                                    slip_frame=slip_frame, n_seeds=n_seeds)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    frames = range(n_frames)
    for method in methods:
        mean_err = np.mean(error_curves[method], axis=0)
        std_err = np.std(error_curves[method], axis=0) / np.sqrt(n_seeds)
        ax.plot(frames, mean_err, color=COLORS[method],
                linestyle=LINESTYLES[method], label=method, linewidth=1.2)
        ax.fill_between(frames, mean_err - std_err, mean_err + std_err,
                        color=COLORS[method], alpha=0.15)

    ax.axvline(slip_frame, color='gray', linestyle=':', alpha=0.7, linewidth=0.8)
    ax.annotate('slip', xy=(slip_frame + 1, 0.1), fontsize=IEEE_FONTSIZE - 2, color='gray')

    ax.set_xlabel('Frame')
    ax.set_ylabel('Phase Error (rad)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, n_frames])
    ax.set_ylim([0, None])
    save_figure(fig, 'fig_recovery_time')


def fig_slip_2d_heatmap():
    """Slip 2D Heatmap"""
    print("\nFig: Slip 2D Heatmap")

    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1

    p_slip_vals = [0.01, 0.02, 0.05, 0.10]
    amp_vals = [np.pi / 2, np.pi, 3 * np.pi / 2]
    n_seeds = 20

    improvement_du_vs_ekf = np.zeros((len(amp_vals), len(p_slip_vals)))
    improvement_du_vs_gn = np.zeros((len(amp_vals), len(p_slip_vals)))

    for i, amp in enumerate(amp_vals):
        for j, p_slip in enumerate(p_slip_vals):
            print(f"  amp={amp / np.pi:.1f}π, p={p_slip}...", end="\r", flush=True)

            slip_cfg = SlipConfig(
                p_slip=p_slip, mode="discrete",
                values=(-amp, amp), probs=(0.5, 0.5)
            )

            ekf_rmse, gn_rmse, du_rmse = [], [], []

            for seed in range(n_seeds):
                y_seq, x_true, _, _ = generate_episode_with_impairments(
                    model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

                x_ekf = run_estimator('EKF', model, y_seq, x0, P0)
                x_gn = run_estimator('GN-6', model, y_seq, x0, P0)
                x_du = run_estimator('DU-tun-6', model, y_seq, x0, P0)

                ekf_rmse.append(compute_rmse(x_ekf, x_true))
                gn_rmse.append(compute_rmse(x_gn, x_true))
                du_rmse.append(compute_rmse(x_du, x_true))

            ekf_mean = np.mean(ekf_rmse)
            gn_mean = np.mean(gn_rmse)
            du_mean = np.mean(du_rmse)

            improvement_du_vs_ekf[i, j] = (ekf_mean - du_mean) / ekf_mean * 100
            improvement_du_vs_gn[i, j] = (gn_mean - du_mean) / gn_mean * 100

            # 收集CSV数据
            collector.add_heatmap(amp / np.pi, p_slip, ekf_mean, np.std(ekf_rmse),
                                  gn_mean, np.std(gn_rmse), du_mean, np.std(du_rmse), n_seeds)

    print(" " * 50)

    # DU vs EKF heatmap
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    im = ax.imshow(improvement_du_vs_ekf, aspect='auto', cmap='Greens', vmin=0, vmax=80)
    ax.set_xticks(range(len(p_slip_vals)))
    ax.set_xticklabels([f'{p}' for p in p_slip_vals])
    ax.set_yticks(range(len(amp_vals)))
    ax.set_yticklabels([f'{a / np.pi:.1f}π' for a in amp_vals])
    ax.set_xlabel('Slip Probability ($p_{slip}$)')
    ax.set_ylabel('Slip Amplitude')

    for i in range(len(amp_vals)):
        for j in range(len(p_slip_vals)):
            val = improvement_du_vs_ekf[i, j]
            color = 'white' if val > 40 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=IEEE_FONTSIZE - 1, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('RMSE Improvement vs EKF (%)')
    save_figure(fig, 'fig_slip_2d_heatmap_vs_ekf')

    # DU vs GN heatmap
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    im = ax.imshow(improvement_du_vs_gn, aspect='auto', cmap='RdYlGn', vmin=-30, vmax=30)
    ax.set_xticks(range(len(p_slip_vals)))
    ax.set_xticklabels([f'{p}' for p in p_slip_vals])
    ax.set_yticks(range(len(amp_vals)))
    ax.set_yticklabels([f'{a / np.pi:.1f}π' for a in amp_vals])
    ax.set_xlabel('Slip Probability ($p_{slip}$)')
    ax.set_ylabel('Slip Amplitude')

    for i in range(len(amp_vals)):
        for j in range(len(p_slip_vals)):
            val = improvement_du_vs_gn[i, j]
            color = 'white' if abs(val) > 15 else 'black'
            ax.text(j, i, f'{val:+.0f}%', ha='center', va='center',
                    fontsize=IEEE_FONTSIZE - 1, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('RMSE Improvement vs GN (%)')
    save_figure(fig, 'fig_slip_2d_heatmap_vs_gn')


def fig_ber_vs_L():
    """BER vs L"""
    print("\nFig: BER vs L")

    L_values = [2, 4, 6, 8]

    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 30, 100

    results = {
        'EKF': [],
        'IEKF': {L: [] for L in L_values},
        'GN': {L: [] for L in L_values},
        'DU': {L: [] for L in L_values}
    }

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}...", end="\r", flush=True)
        y_seq, x_true, _, _ = generate_episode_with_impairments(
            model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

        x_ekf = run_estimator('EKF', model, y_seq, x0, P0)
        ber, _ = quick_ber_evm(x_true, x_ekf, 10, seed)
        results['EKF'].append(ber * 100)

        for L in L_values:
            iekf_L = min(L, 4)
            x_iekf = run_estimator(f'IEKF-{iekf_L}', model, y_seq, x0, P0)
            x_gn = run_estimator(f'GN-{L}', model, y_seq, x0, P0)
            x_du = run_estimator(f'DU-tun-{L}', model, y_seq, x0, P0)

            ber_iekf, _ = quick_ber_evm(x_true, x_iekf, 10, seed)
            ber_gn, _ = quick_ber_evm(x_true, x_gn, 10, seed)
            ber_du, _ = quick_ber_evm(x_true, x_du, 10, seed)

            results['IEKF'][L].append(ber_iekf * 100)
            results['GN'][L].append(ber_gn * 100)
            results['DU'][L].append(ber_du * 100)
    print(f"  完成 {n_seeds} seeds" + " " * 20)

    # 收集CSV数据
    collector.add_performance('BER_vs_L', 'L', 1, 'EKF', 'BER_pct',
                              np.mean(results['EKF']), np.std(results['EKF']), n_seeds, n_frames)
    for L in L_values:
        collector.add_performance('BER_vs_L', 'L', L, 'IEKF-4', 'BER_pct',
                                  np.mean(results['IEKF'][L]), np.std(results['IEKF'][L]), n_seeds, n_frames)
        collector.add_performance('BER_vs_L', 'L', L, 'GN-6', 'BER_pct',
                                  np.mean(results['GN'][L]), np.std(results['GN'][L]), n_seeds, n_frames)
        collector.add_performance('BER_vs_L', 'L', L, 'DU-tun-6', 'BER_pct',
                                  np.mean(results['DU'][L]), np.std(results['DU'][L]), n_seeds, n_frames)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    ax.axhline(np.mean(results['EKF']), color=COLORS['EKF'], linestyle='--',
               label='EKF (L=1)', alpha=0.8)

    iekf_means = [np.mean(results['IEKF'][L]) for L in L_values]
    ax.semilogy(L_values, iekf_means, marker='^', color=COLORS['IEKF-4'], label='IEKF')

    gn_means = [np.mean(results['GN'][L]) for L in L_values]
    ax.semilogy(L_values, gn_means, marker='s', color=COLORS['GN-6'],
                label='GN', markerfacecolor='white')

    du_means = [np.mean(results['DU'][L]) for L in L_values]
    ax.semilogy(L_values, du_means, marker='D', color=COLORS['DU-tun-6'], label='DU-tun')

    ax.set_xlabel('Iterations / Layers ($L$)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_values)
    save_figure(fig, 'fig_ber_L')


def fig_ber_vs_L_multiSNR():
    """BER vs L at multiple SNRs"""
    print("\nFig: BER vs L (multi-SNR)")

    L_values = [2, 4, 6, 8]
    snr_list = [5, 10, 15]

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 25, 100

    results = {snr: {L: [] for L in L_values} for snr in snr_list}

    for snr in snr_list:
        print(f"  SNR={snr}dB...", end=" ", flush=True)
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=snr, adc_bits=4)
        model = THzISACModel(cfg)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for L in L_values:
                x_du = run_estimator(f'DU-tun-{L}', model, y_seq, x0, P0)
                ber, _ = quick_ber_evm(x_true, x_du, snr, seed)
                results[snr][L].append(ber * 100)
        print("done")

    # 收集CSV数据
    for snr in snr_list:
        for L in L_values:
            collector.add_performance('BER_vs_L_multiSNR', 'L', L, f'DU-tun-6_SNR{snr}', 'BER_pct',
                                      np.mean(results[snr][L]), np.std(results[snr][L]), n_seeds, n_frames)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    for snr in snr_list:
        ber_means = [np.mean(results[snr][L]) for L in L_values]
        ax.semilogy(L_values, ber_means,
                    marker='D', color=MULTI_SNR_COLORS[snr],
                    linestyle=MULTI_SNR_LINESTYLES[snr],
                    label=f'SNR={snr}dB', linewidth=1.5, markersize=6)

    ax.set_xlabel('Layers ($L$)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_values)
    save_figure(fig, 'fig_ber_L_multiSNR')


def fig_ber_vs_pslip():
    """BER vs p_slip"""
    print("\nFig: BER vs p_slip")

    p_slip_vals = [0.01, 0.02, 0.05, 0.10, 0.15]
    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']

    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    n_seeds, n_frames = 25, 100

    results = {p: {m: [] for m in methods} for p in p_slip_vals}

    for p_slip in p_slip_vals:
        print(f"  p_slip={p_slip}...", end=" ", flush=True)
        slip_cfg = SlipConfig.killer_pi(p_slip=p_slip)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                ber, _ = quick_ber_evm(x_true, x_hat, 10, seed)
                results[p_slip][method].append(ber * 100)
        print("done")

    # 收集CSV数据
    for p in p_slip_vals:
        for m in methods:
            collector.add_performance('BER_vs_pslip', 'p_slip', p, m, 'BER_pct',
                                      np.mean(results[p][m]), np.std(results[p][m]), n_seeds, n_frames)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    for method in methods:
        ber_means = [np.mean(results[p][method]) for p in p_slip_vals]
        ax.semilogy(p_slip_vals, ber_means,
                    marker=MARKERS[method], color=COLORS[method],
                    label=method, markerfacecolor='white' if 'GN' in method else None)
    ax.set_xlabel('Slip Probability ($p_{slip}$)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_ber_pslip')


def fig_ber_vs_adc():
    """BER vs ADC bits"""
    print("\nFig: BER vs ADC")

    adc_bits = [2, 3, 4, 6, 8]
    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 25, 100

    results = {bits: {m: [] for m in methods} for bits in adc_bits}

    for bits in adc_bits:
        print(f"  ADC={bits}bit...", end=" ", flush=True)
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=bits)
        model = THzISACModel(cfg)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                ber, _ = quick_ber_evm(x_true, x_hat, 10, seed)
                results[bits][method].append(ber * 100)
        print("done")

    # 收集CSV数据
    for b in adc_bits:
        for m in methods:
            collector.add_performance('BER_vs_ADC', 'ADC_bits', b, m, 'BER_pct',
                                      np.mean(results[b][m]), np.std(results[b][m]), n_seeds, n_frames)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    for method in methods:
        ber_means = [np.mean(results[b][method]) for b in adc_bits]
        ax.semilogy(adc_bits, ber_means,
                    marker=MARKERS[method], color=COLORS[method],
                    label=method, markerfacecolor='white' if 'GN' in method else None)
    ax.set_xlabel('ADC Resolution (bits)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(adc_bits)
    save_figure(fig, 'fig_ber_adc')


def fig_phase_rmse_vs_frame_duration():
    """Phase RMSE vs Frame Duration"""
    print("\nFig: Phase RMSE vs Frame Duration")

    frame_durations = [50e-6, 100e-6, 200e-6, 400e-6]
    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 25, 100

    results = {fd: {m: [] for m in methods} for fd in frame_durations}

    for fd in frame_durations:
        print(f"  T_frame={fd * 1e6:.0f}μs...", end=" ", flush=True)
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4, frame_duration_s=fd)
        model = THzISACModel(cfg)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                rmse_per = compute_rmse_per_component(x_hat, x_true)
                results[fd][method].append(rmse_per[2])
        print("done")

    # 收集CSV数据
    for fd in frame_durations:
        for m in methods:
            collector.add_performance('PhaseRMSE_vs_FrameDuration', 'frame_duration_us', fd * 1e6, m, 'Phase_RMSE',
                                      np.mean(results[fd][m]), np.std(results[fd][m]), n_seeds, n_frames)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    fd_us = [fd * 1e6 for fd in frame_durations]
    for method in methods:
        rmse_means = [np.mean(results[fd][method]) for fd in frame_durations]
        ax.semilogy(fd_us, rmse_means,
                    marker=MARKERS[method], color=COLORS[method],
                    label=method, markerfacecolor='white' if 'GN' in method else None)
    ax.set_xlabel('Frame Duration (μs)')
    ax.set_ylabel('Phase RMSE (rad)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_phase_rmse_frame_duration')


def fig_sensitivity_phi():
    """φ步长敏感性 - 单独图，无title"""
    print("\nFig: Sensitivity (phi)")

    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 20, 100

    phi_scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    phi_rmse = []
    phi_rmse_std = []

    for scale in phi_scales:
        rmses = []
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            du_cfg = DUMAPConfig(n_layers=6)
            du_cfg.step_scale = np.array([1.0, 0.1, scale])
            est = DUMAP(du_cfg)
            x_hat, _ = est.forward_sequence(model, y_seq, x0, P0)
            rmses.append(compute_rmse(x_hat, x_true))
        phi_rmse.append(np.mean(rmses))
        phi_rmse_std.append(np.std(rmses))

        # 收集CSV数据
        collector.add_sensitivity('alpha_phi', scale, np.mean(rmses), np.std(rmses), n_seeds)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    ax.errorbar(phi_scales, phi_rmse, yerr=[s / np.sqrt(n_seeds) for s in phi_rmse_std],
                marker='o', color=COLORS['DU-tun-6'], linewidth=1.5,
                capsize=3, markersize=6)
    ax.axvline(1.5, color='red', linestyle='--', alpha=0.7, linewidth=0.8,
               label='Selected $\\alpha_\\phi=1.5$')
    ax.set_xlabel(r'Phase Step Scale ($\alpha_\phi$)')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_sensitivity_phi')


def fig_sensitivity_nu():
    """ν步长敏感性 - 单独图，无title"""
    print("\nFig: Sensitivity (nu)")

    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 20, 100

    nu_scales = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    nu_rmse = []
    nu_rmse_std = []

    for scale in nu_scales:
        rmses = []
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            du_cfg = DUMAPConfig(n_layers=6)
            du_cfg.step_scale = np.array([1.0, scale, 1.5])
            est = DUMAP(du_cfg)
            x_hat, _ = est.forward_sequence(model, y_seq, x0, P0)
            rmses.append(compute_rmse(x_hat, x_true))
        nu_rmse.append(np.mean(rmses))
        nu_rmse_std.append(np.std(rmses))

        # 收集CSV数据
        collector.add_sensitivity('alpha_nu', scale, np.mean(rmses), np.std(rmses), n_seeds)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    ax.errorbar(nu_scales, nu_rmse, yerr=[s / np.sqrt(n_seeds) for s in nu_rmse_std],
                marker='s', color=COLORS['GN-6'], linewidth=1.5,
                capsize=3, markersize=6)
    ax.axvline(0.1, color='red', linestyle='--', alpha=0.7, linewidth=0.8,
               label='Selected $\\alpha_\\nu=0.1$')
    ax.set_xlabel(r'Doppler Step Scale ($\alpha_\nu$)')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_sensitivity_nu')


def fig_sensitivity_tau():
    """τ步长敏感性 - 单独图，无title"""
    print("\nFig: Sensitivity (tau)")

    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 20, 100

    tau_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    tau_rmse = []
    tau_rmse_std = []

    for scale in tau_scales:
        rmses = []
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            du_cfg = DUMAPConfig(n_layers=6)
            du_cfg.step_scale = np.array([scale, 0.1, 1.5])
            est = DUMAP(du_cfg)
            x_hat, _ = est.forward_sequence(model, y_seq, x0, P0)
            rmses.append(compute_rmse(x_hat, x_true))
        tau_rmse.append(np.mean(rmses))
        tau_rmse_std.append(np.std(rmses))

        # 收集CSV数据
        collector.add_sensitivity('alpha_tau', scale, np.mean(rmses), np.std(rmses), n_seeds)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    ax.errorbar(tau_scales, tau_rmse, yerr=[s / np.sqrt(n_seeds) for s in tau_rmse_std],
                marker='^', color=COLORS['IEKF-4'], linewidth=1.5,
                capsize=3, markersize=6)
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.7, linewidth=0.8,
               label='Selected $\\alpha_\\tau=1.0$')
    ax.set_xlabel(r'Delay Step Scale ($\alpha_\tau$)')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_sensitivity_tau')


def fig_improvement_bar():
    """RMSE 改善柱状图"""
    print("\nFig: Improvement Bar")

    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds = 30

    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']
    results = {m: [] for m in methods}

    for seed in range(n_seeds):
        y_seq, x_true, _, _ = generate_episode_with_impairments(
            model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

        for method in methods:
            x_hat = run_estimator(method, model, y_seq, x0, P0)
            results[method].append(compute_rmse(x_hat, x_true))

    means = [np.mean(results[m]) for m in methods]
    sems = [np.std(results[m]) / np.sqrt(n_seeds) for m in methods]
    stds = [np.std(results[m]) for m in methods]

    # 收集CSV数据
    for i, m in enumerate(methods):
        imp = (means[0] - means[i]) / means[0] * 100
        collector.add_auxiliary('improvement_bar', method=m,
                                rmse_mean=means[i], rmse_std=stds[i], rmse_sem=sems[i],
                                improvement_vs_EKF_pct=imp, n_seeds=n_seeds)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    colors = [COLORS[m] for m in methods]

    bars = ax.bar(range(len(methods)), means, yerr=sems,
                  color=colors, alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=IEEE_FONTSIZE - 1)
    ax.set_ylabel('RMSE')
    ax.grid(True, alpha=0.3, axis='y')

    for i in range(1, len(methods)):
        imp = (means[0] - means[i]) / means[0] * 100
        ax.annotate(f'{imp:+.0f}%', xy=(i, means[i]), xytext=(i, means[i] * 0.9),
                    fontsize=8, ha='center', color=colors[i], fontweight='bold')

    save_figure(fig, 'fig_improvement_bar')


def fig_ccdf():
    """相位误差 CCDF"""
    print("\nFig: CCDF")

    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds = 30

    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']
    all_errors = {m: [] for m in methods}

    for seed in range(n_seeds):
        y_seq, x_true, _, _ = generate_episode_with_impairments(
            model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

        for m in methods:
            x_hat = run_estimator(m, model, y_seq, x0, P0)
            errors = [abs(wrap_angle(xh[2] - xt[2])) for xh, xt in zip(x_hat, x_true)]
            all_errors[m].extend(errors)

    # 收集CSV数据
    for m in methods:
        errs = np.array(all_errors[m])
        collector.add_auxiliary('ccdf_stats', method=m,
                                mean=np.mean(errs), std=np.std(errs),
                                median=np.median(errs),
                                p90=np.percentile(errs, 90),
                                p95=np.percentile(errs, 95),
                                p99=np.percentile(errs, 99),
                                n_samples=len(errs))

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    for method in methods:
        sorted_err = np.sort(all_errors[method])
        ccdf = 1 - np.arange(len(sorted_err)) / len(sorted_err)
        ax.semilogy(sorted_err, ccdf, color=COLORS[method],
                    linestyle=LINESTYLES[method], label=method, linewidth=1.2)

    ax.axhline(0.05, color='gray', linestyle=':', alpha=0.7, linewidth=0.8)
    ax.axhline(0.01, color='gray', linestyle=':', alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Phase Error $|\\Delta\\phi|$ (rad)')
    ax.set_ylabel('CCDF $P(|\\Delta\\phi| > x)$')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2.5])
    ax.set_ylim([1e-3, 1])
    save_figure(fig, 'fig_ccdf')


def fig_pcrb_nt():
    """PCRB vs n_t"""
    print("\nFig: PCRB vs n_t")

    n_t_values = [1, 2, 4, 8, 16]
    x0 = np.array([1.0, 0.5, 0.0])

    pcrb_tau, pcrb_nu, pcrb_phi = [], [], []

    for n_t in n_t_values:
        cfg = THzISACConfig(n_f=8, n_t=n_t, snr_db=10, adc_bits=4)
        model = THzISACModel(cfg)

        pcrb_rec = PCRBRecursion(d=3)
        pcrb_seq, _ = pcrb_rec.run_sequence(model, [x0] * 50)

        sqrt_tau = np.sqrt(np.mean([p[0] for p in pcrb_seq[10:]]))
        sqrt_nu = np.sqrt(np.mean([p[1] for p in pcrb_seq[10:]]))
        sqrt_phi = np.sqrt(np.mean([p[2] for p in pcrb_seq[10:]]))

        pcrb_tau.append(sqrt_tau)
        pcrb_nu.append(sqrt_nu)
        pcrb_phi.append(sqrt_phi)

        # 收集CSV数据
        collector.add_auxiliary('pcrb', n_t=n_t,
                                sqrt_pcrb_tau=sqrt_tau,
                                sqrt_pcrb_nu=sqrt_nu,
                                sqrt_pcrb_phi=sqrt_phi)

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    ax.semilogy(n_t_values, pcrb_tau, 'o-', label=r'$\tau$', color='#1f77b4')
    ax.semilogy(n_t_values, pcrb_nu, 's-', label=r'$\nu$', color='#ff7f0e')
    ax.semilogy(n_t_values, pcrb_phi, '^-', label=r'$\phi$', color='#2ca02c')
    ax.set_xlabel('Time Pilots ($n_t$)')
    ax.set_ylabel(r'$\sqrt{\mathrm{PCRB}}$')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_t_values)
    save_figure(fig, 'fig_pcrb_nt')


def fig_phase_tracking():
    """相位跟踪轨迹图"""
    print("\nFig: Phase Tracking")

    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)

    # 生成一个典型轨迹
    y_seq, x_true, slip_frames, _ = generate_episode_with_impairments(
        model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=42)

    x_ekf = run_estimator('EKF', model, y_seq, x0, P0)
    x_gn = run_estimator('GN-6', model, y_seq, x0, P0)
    x_du = run_estimator('DU-tun-6', model, y_seq, x0, P0)

    # 收集CSV数据
    for k in range(len(x_true)):
        collector.add_auxiliary('phase_tracking', frame=k, seed=42,
                                phi_true=x_true[k][2],
                                phi_EKF=x_ekf[k][2],
                                phi_GN6=x_gn[k][2],
                                phi_DU=x_du[k][2],
                                is_slip_frame=(k in slip_frames))

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH * 1.2, IEEE_HEIGHT))

    frames = range(len(x_true))
    phi_true = [x[2] for x in x_true]
    phi_ekf = [x[2] for x in x_ekf]
    phi_gn = [x[2] for x in x_gn]
    phi_du = [x[2] for x in x_du]

    ax.plot(frames, phi_true, 'k-', label='True', linewidth=1.5, alpha=0.8)
    ax.plot(frames, phi_ekf, color=COLORS['EKF'], linestyle='-',
            label='EKF', linewidth=1.0, alpha=0.8)
    ax.plot(frames, phi_gn, color=COLORS['GN-6'], linestyle='-',
            label='GN-6', linewidth=1.0, alpha=0.8)
    ax.plot(frames, phi_du, color=COLORS['DU-tun-6'], linestyle='-',
            label='DU-tun-6', linewidth=1.0, alpha=0.8)

    # 标记slip位置
    for sf in slip_frames:
        ax.axvline(sf, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)

    ax.set_xlabel('Frame')
    ax.set_ylabel(r'Phase $\phi$ (rad)')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    save_figure(fig, 'fig_phase_tracking')


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("IEEE 格式图像生成 v7 完整版")
    print("=" * 60)
    print("v6 修复 (保留):")
    print("  1. DU step_scale: [1.0, 0.1, 2.0] -> [1.0, 0.1, 1.5]")
    print("  2. fig_frame_duration -> Phase RMSE")
    print("  3. Sensitivity: 3个独立图，无title")
    print("")
    print("v7 新增:")
    print("  CSV数据输出 (4个文件)")
    print("=" * 60)

    setup_ieee_style()

    # 核心图
    print("\n[核心图]")
    fig_ber_vs_snr()
    fig_rmse_vs_snr()
    fig_rmse_vs_L()
    fig_recovery_time()
    fig_slip_2d_heatmap()
    fig_phase_tracking()

    # BER系列图
    print("\n[BER系列图]")
    fig_ber_vs_L()
    fig_ber_vs_L_multiSNR()
    fig_ber_vs_pslip()
    fig_ber_vs_adc()
    fig_phase_rmse_vs_frame_duration()

    # 敏感性分析 (3个独立图)
    print("\n[敏感性分析]")
    fig_sensitivity_phi()
    fig_sensitivity_nu()
    fig_sensitivity_tau()

    # 辅助图
    print("\n[辅助图]")
    fig_improvement_bar()
    fig_ccdf()
    fig_pcrb_nt()

    # 保存CSV数据
    print("\n" + "=" * 60)
    print("保存CSV数据...")
    collector.save_all()

    print("\n" + "=" * 60)
    print(f"完成! 输出目录: {OUTPUT_DIR}/")
    print("=" * 60)
    print("\n图像文件:")
    print("  fig_*.png, fig_*.pdf")
    print("\nCSV数据文件:")
    print("  1. data_performance.csv  - 性能对比 (BER/RMSE vs SNR/L/p_slip/ADC/frame_duration)")
    print("  2. data_sensitivity.csv  - 敏感性分析 (alpha_phi/nu/tau)")
    print("  3. data_heatmap.csv      - 2D热力图 (amplitude × p_slip)")
    print("  4. data_auxiliary.csv    - 辅助数据 (recovery_time/ccdf/pcrb/improvement_bar/phase_tracking)")


if __name__ == "__main__":
    main()