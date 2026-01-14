#!/usr/bin/env python3
"""
THz-ISAC DU-MAP IEEE 格式图像生成 v8 - THz效应启用版
======================================================

基于v7，启用Doppler squint效应，使模型更贴近THz-ISL物理现实。

关键修改：
1. 所有THzISACConfig调用添加 enable_doppler_squint=True
2. 新增3张THz-specific主文图（来自generate_thz_specific_figures.py）
3. 保持所有其他设置不变

使用方法：
    python3 generate_ieee_figures_v8_thz.py

输出：outputs/ 目录下所有图像和CSV数据
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

COLORS = {
    'EKF': '#E41A1C',
    'IEKF-4': '#FF7F00',
    'GN-6': '#377EB8',
    'DU-tun-6': '#4DAF4A',
    'PCRB': '#000000',
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
# ★★★ 关键函数：创建THz配置（启用Doppler Squint）★★★
# ============================================================================

def create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4,
                      bandwidth_hz=100e6, frame_duration_s=100e-6,
                      pn_linewidth_hz=100.0):
    """
    创建THz-ISL配置 - V3完整物理模型版

    TWC主文配置：
    - P0: 连续PN Wiener模型（linewidth参数化）✅
    - P0: Doppler squint ✅
    - P1: Beam squint（pilot subband可忽略，默认关闭）
    - P2: Pointing jitter（stress-test用，默认关闭）
    """
    return THzISACConfig(
        n_f=n_f,
        n_t=n_t,
        snr_db=snr_db,
        adc_bits=adc_bits,
        bandwidth_hz=bandwidth_hz,
        frame_duration_s=frame_duration_s,
        # ★★★ V3 THz物理效应 ★★★
        # P0: 连续PN（主文启用）
        enable_continuous_pn=True,
        pn_linewidth_hz=pn_linewidth_hz,
        # P0: Doppler squint（主文启用）
        enable_doppler_squint=True,
        # P1: Beam squint（pilot subband可忽略）
        enable_beam_squint=False,
        # P2: Pointing jitter（stress-test时启用）
        enable_pointing_jitter=False,
    )


# ============================================================================
# CSV数据收集器
# ============================================================================

class DataCollector:
    def __init__(self):
        self.performance_data = []
        self.sensitivity_data = []
        self.heatmap_data = []
        self.auxiliary_data = []

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
        if self.performance_data:
            headers = ['experiment', 'sweep_variable', 'sweep_value', 'method',
                       'metric', 'mean', 'std', 'sem', 'n_seeds', 'n_frames']
            with open(f'{OUTPUT_DIR}/data_performance.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.performance_data)
            print(f"  ✓ data_performance.csv ({len(self.performance_data)} rows)")

        if self.sensitivity_data:
            headers = ['parameter', 'value', 'rmse_mean', 'rmse_std', 'rmse_sem', 'n_seeds']
            with open(f'{OUTPUT_DIR}/data_sensitivity.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.sensitivity_data)
            print(f"  ✓ data_sensitivity.csv ({len(self.sensitivity_data)} rows)")

        if self.heatmap_data:
            headers = ['amplitude_pi', 'p_slip', 'EKF_rmse_mean', 'EKF_rmse_std',
                       'GN6_rmse_mean', 'GN6_rmse_std', 'DU_rmse_mean', 'DU_rmse_std',
                       'improvement_vs_EKF_pct', 'improvement_vs_GN_pct', 'n_seeds']
            with open(f'{OUTPUT_DIR}/data_heatmap.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.heatmap_data)
            print(f"  ✓ data_heatmap.csv ({len(self.heatmap_data)} rows)")

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


collector = DataCollector()


# ============================================================================
# DU配置
# ============================================================================

def get_du_step_scale():
    return np.array([1.0, 0.1, 1.5])


def run_estimator(method, model, y_seq, x0, P0):
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


def compute_rmse(x_hat, x_true):
    errors = []
    for xh, xt in zip(x_hat, x_true):
        e = xh - xt
        e[2] = wrap_angle(e[2])
        errors.append(np.sqrt(np.sum(e ** 2)))
    return np.mean(errors), np.std(errors)


def compute_phase_rmse(x_hat, x_true):
    errors = []
    for xh, xt in zip(x_hat, x_true):
        e_phi = wrap_angle(xh[2] - xt[2])
        errors.append(e_phi ** 2)
    return np.sqrt(np.mean(errors))


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
        # ★ 使用THz配置 ★
        cfg = create_thz_config(n_f=8, n_t=4, snr_db=snr, adc_bits=4)
        model = THzISACModel(cfg)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                ber, _ = quick_ber_evm(x_true, x_hat, snr, seed)
                results[snr][method].append(ber * 100)

        for m in methods:
            collector.add_performance('ber_vs_snr', 'snr_db', snr, m, 'BER_pct',
                                      np.mean(results[snr][m]), np.std(results[snr][m]),
                                      n_seeds, n_frames)
        print("done")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    for method in methods:
        ber_means = [np.mean(results[snr][method]) for snr in snr_list]
        ber_stds = [np.std(results[snr][method]) for snr in snr_list]
        ax.errorbar(snr_list, ber_means, yerr=ber_stds, color=COLORS[method],
                    marker=MARKERS[method], linestyle=LINESTYLES[method],
                    label=method, capsize=3)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(snr_list)
    save_figure(fig, 'fig_ber_vs_snr')


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
    pcrb_results = []

    for snr in snr_list:
        print(f"  SNR={snr}dB...", end=" ", flush=True)
        cfg = create_thz_config(n_f=8, n_t=4, snr_db=snr, adc_bits=4)
        model = THzISACModel(cfg)

        pcrb_rec = PCRBRecursion(d=3)
        pcrb_seq, _ = pcrb_rec.run_sequence(model, [x0] * 50)
        pcrb_avg = np.sqrt(np.mean([np.sum(p) for p in pcrb_seq[10:]]))
        pcrb_results.append(pcrb_avg)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                rmse, _ = compute_rmse(x_hat, x_true)
                results[snr][method].append(rmse)

        for m in methods:
            collector.add_performance('rmse_vs_snr', 'snr_db', snr, m, 'RMSE',
                                      np.mean(results[snr][m]), np.std(results[snr][m]),
                                      n_seeds, n_frames)
        print("done")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    for method in methods:
        rmse_means = [np.mean(results[snr][method]) for snr in snr_list]
        rmse_stds = [np.std(results[snr][method]) for snr in snr_list]
        ax.errorbar(snr_list, rmse_means, yerr=rmse_stds, color=COLORS[method],
                    marker=MARKERS[method], linestyle=LINESTYLES[method],
                    label=method, capsize=3)
    ax.plot(snr_list, pcrb_results, 'k--', label='PCRB', linewidth=1.5)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(snr_list)
    save_figure(fig, 'fig_rmse_vs_snr')


def fig_rmse_vs_L():
    """RMSE vs 迭代层数 L - 包含EKF和IEKF基线"""
    print("\nFig: RMSE vs L")

    L_list = [1, 2, 4, 6, 8, 10]
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 30, 100

    cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)

    # EKF基线（L=1，单次）
    ekf_results = []
    # IEKF结果（不同迭代次数）
    iekf_results = {L: [] for L in L_list}
    # GN和DU结果
    gn_results = {L: [] for L in L_list}
    du_results = {L: [] for L in L_list}

    for L in L_list:
        print(f"  L={L}...", end=" ", flush=True)
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            # EKF (只在L=1时计算一次)
            if L == 1:
                x_ekf = run_estimator('EKF', model, y_seq, x0, P0)
                rmse_ekf, _ = compute_rmse(x_ekf, x_true)
                ekf_results.append(rmse_ekf)

            # IEKF-L
            x_iekf = run_estimator(f'IEKF-{L}', model, y_seq, x0, P0)
            rmse_iekf, _ = compute_rmse(x_iekf, x_true)
            iekf_results[L].append(rmse_iekf)

            # GN-L
            gn = GaussNewtonMAP(GNSolverConfig(max_iters=L))
            x_gn = gn.solve_sequence(model, y_seq, x0, P0)[0]
            rmse_gn, _ = compute_rmse(x_gn, x_true)
            gn_results[L].append(rmse_gn)

            # DU-L
            du_cfg = DUMAPConfig(n_layers=L)
            du_cfg.step_scale = get_du_step_scale()
            du = DUMAP(du_cfg)
            x_du = du.forward_sequence(model, y_seq, x0, P0)[0]
            rmse_du, _ = compute_rmse(x_du, x_true)
            du_results[L].append(rmse_du)

        # 记录数据
        if L == 1:
            collector.add_performance('rmse_vs_L', 'L', L, 'EKF',
                                      'RMSE', np.mean(ekf_results), np.std(ekf_results),
                                      n_seeds, n_frames)
        collector.add_performance('rmse_vs_L', 'L', L, 'IEKF',
                                  'RMSE', np.mean(iekf_results[L]), np.std(iekf_results[L]),
                                  n_seeds, n_frames)
        collector.add_performance('rmse_vs_L', 'L', L, 'GN',
                                  'RMSE', np.mean(gn_results[L]), np.std(gn_results[L]),
                                  n_seeds, n_frames)
        collector.add_performance('rmse_vs_L', 'L', L, 'DU-tun',
                                  'RMSE', np.mean(du_results[L]), np.std(du_results[L]),
                                  n_seeds, n_frames)
        print("done")

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    # EKF基线（水平虚线）
    ekf_mean = np.mean(ekf_results)
    ax.axhline(y=ekf_mean, color=COLORS['EKF'], linestyle='--',
               label='EKF (L=1)', linewidth=1.2, alpha=0.8)

    # IEKF
    iekf_means = [np.mean(iekf_results[L]) for L in L_list]
    ax.plot(L_list, iekf_means, marker='^', color=COLORS['IEKF-4'],
            label='IEKF', linewidth=1.2)

    # GN
    gn_means = [np.mean(gn_results[L]) for L in L_list]
    ax.plot(L_list, gn_means, marker='s', color=COLORS['GN-6'],
            label='GN', linewidth=1.2)

    # DU
    du_means = [np.mean(du_results[L]) for L in L_list]
    ax.plot(L_list, du_means, marker='D', color=COLORS['DU-tun-6'],
            label='DU-tun', linewidth=1.2)

    ax.set_xlabel('Number of Layers/Iterations $L$')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_list)
    save_figure(fig, 'fig_rmse_vs_L')


def fig_recovery_time():
    """Slip恢复时间"""
    print("\nFig: Recovery Time")

    cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds = 30

    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']
    recovery_times = {m: [] for m in methods}
    threshold = 0.3

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}...", end="\r", flush=True)
        y_seq, x_true, slip_frames, _ = generate_episode_with_impairments(
            model, 200, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

        for method in methods:
            x_hat = run_estimator(method, model, y_seq, x0, P0)

            for sf in slip_frames:
                if sf + 50 < len(x_hat):
                    for k in range(sf, min(sf + 50, len(x_hat))):
                        err = abs(wrap_angle(x_hat[k][2] - x_true[k][2]))
                        if err < threshold:
                            recovery_times[method].append(k - sf)
                            break
                    else:
                        recovery_times[method].append(50)

    for m in methods:
        if recovery_times[m]:
            collector.add_auxiliary('recovery_time', method=m,
                                    mean=np.mean(recovery_times[m]),
                                    std=np.std(recovery_times[m]),
                                    median=np.median(recovery_times[m]),
                                    n_samples=len(recovery_times[m]))

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    positions = range(len(methods))
    bp = ax.boxplot([recovery_times[m] for m in methods], positions=positions,
                    patch_artist=True, widths=0.6)

    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Recovery Time (frames)')
    ax.grid(True, alpha=0.3, axis='y')
    print()
    save_figure(fig, 'fig_recovery_time')


def fig_slip_2d_heatmap():
    """Slip严重度 2D热力图"""
    print("\nFig: Slip 2D Heatmap")

    amplitudes = [0.5, 1.0, 1.5, 2.0]
    p_slips = [0.01, 0.03, 0.05, 0.1]

    cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    n_seeds, n_frames = 20, 100

    ekf_grid = np.zeros((len(amplitudes), len(p_slips)))
    gn_grid = np.zeros((len(amplitudes), len(p_slips)))
    du_grid = np.zeros((len(amplitudes), len(p_slips)))

    for i, amp in enumerate(amplitudes):
        for j, p_slip in enumerate(p_slips):
            print(f"  amp={amp}π, p_slip={p_slip}...", end=" ", flush=True)
            # 使用values和probs参数指定slip幅度
            slip_cfg = SlipConfig(
                p_slip=p_slip,
                values=(amp * np.pi, -amp * np.pi),
                probs=(0.5, 0.5)
            )

            ekf_rmse, gn_rmse, du_rmse = [], [], []
            for seed in range(n_seeds):
                y_seq, x_true, _, _ = generate_episode_with_impairments(
                    model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

                x_ekf = run_estimator('EKF', model, y_seq, x0, P0)
                x_gn = run_estimator('GN-6', model, y_seq, x0, P0)
                x_du = run_estimator('DU-tun-6', model, y_seq, x0, P0)

                ekf_rmse.append(compute_rmse(x_ekf, x_true)[0])
                gn_rmse.append(compute_rmse(x_gn, x_true)[0])
                du_rmse.append(compute_rmse(x_du, x_true)[0])

            ekf_grid[i, j] = np.mean(ekf_rmse)
            gn_grid[i, j] = np.mean(gn_rmse)
            du_grid[i, j] = np.mean(du_rmse)

            collector.add_heatmap(amp, p_slip,
                                  np.mean(ekf_rmse), np.std(ekf_rmse),
                                  np.mean(gn_rmse), np.std(gn_rmse),
                                  np.mean(du_rmse), np.std(du_rmse), n_seeds)
            print("done")

    for name, grid in [('EKF', ekf_grid), ('GN6', gn_grid), ('DU', du_grid)]:
        fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
        im = ax.imshow(grid, cmap='YlOrRd', aspect='auto', origin='lower')
        ax.set_xticks(range(len(p_slips)))
        ax.set_xticklabels([f'{p}' for p in p_slips])
        ax.set_yticks(range(len(amplitudes)))
        ax.set_yticklabels([f'{a}π' for a in amplitudes])
        ax.set_xlabel('Slip Probability $p_{slip}$')
        ax.set_ylabel('Slip Amplitude')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('RMSE')
        save_figure(fig, f'fig_slip_2d_heatmap_{name}')


def fig_phase_tracking():
    """相位跟踪轨迹图"""
    print("\nFig: Phase Tracking")

    cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)

    y_seq, x_true, slip_frames, _ = generate_episode_with_impairments(
        model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=42)

    x_ekf = run_estimator('EKF', model, y_seq, x0, P0)
    x_gn = run_estimator('GN-6', model, y_seq, x0, P0)
    x_du = run_estimator('DU-tun-6', model, y_seq, x0, P0)

    for k in range(len(x_true)):
        collector.add_auxiliary('phase_tracking', frame=k, seed=42,
                                phi_true=x_true[k][2],
                                phi_EKF=x_ekf[k][2],
                                phi_GN6=x_gn[k][2],
                                phi_DU=x_du[k][2],
                                is_slip_frame=(k in slip_frames))

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

    for sf in slip_frames:
        ax.axvline(sf, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)

    ax.set_xlabel('Frame')
    ax.set_ylabel(r'Phase $\phi$ (rad)')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    save_figure(fig, 'fig_phase_tracking')


# ============================================================================
# BER系列图
# ============================================================================

def fig_ber_vs_L():
    """BER vs L - 包含EKF和IEKF基线"""
    print("\nFig: BER vs L")

    L_list = [1, 2, 4, 6, 8, 10]
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 30, 100

    cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)

    ekf_results = []
    iekf_results = {L: [] for L in L_list}
    gn_results = {L: [] for L in L_list}
    du_results = {L: [] for L in L_list}

    for L in L_list:
        print(f"  L={L}...", end=" ", flush=True)
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            # EKF
            if L == 1:
                x_ekf = run_estimator('EKF', model, y_seq, x0, P0)
                ber_ekf, _ = quick_ber_evm(x_true, x_ekf, 10, seed)
                ekf_results.append(ber_ekf * 100)

            # IEKF-L
            x_iekf = run_estimator(f'IEKF-{L}', model, y_seq, x0, P0)
            ber_iekf, _ = quick_ber_evm(x_true, x_iekf, 10, seed)
            iekf_results[L].append(ber_iekf * 100)

            # GN-L
            gn = GaussNewtonMAP(GNSolverConfig(max_iters=L))
            x_gn = gn.solve_sequence(model, y_seq, x0, P0)[0]
            ber_gn, _ = quick_ber_evm(x_true, x_gn, 10, seed)
            gn_results[L].append(ber_gn * 100)

            # DU-L
            du_cfg = DUMAPConfig(n_layers=L)
            du_cfg.step_scale = get_du_step_scale()
            du = DUMAP(du_cfg)
            x_du = du.forward_sequence(model, y_seq, x0, P0)[0]
            ber_du, _ = quick_ber_evm(x_true, x_du, 10, seed)
            du_results[L].append(ber_du * 100)

        if L == 1:
            collector.add_performance('ber_vs_L', 'L', L, 'EKF',
                                      'BER_pct', np.mean(ekf_results), np.std(ekf_results),
                                      n_seeds, n_frames)
        collector.add_performance('ber_vs_L', 'L', L, 'IEKF',
                                  'BER_pct', np.mean(iekf_results[L]), np.std(iekf_results[L]),
                                  n_seeds, n_frames)
        collector.add_performance('ber_vs_L', 'L', L, 'GN',
                                  'BER_pct', np.mean(gn_results[L]), np.std(gn_results[L]),
                                  n_seeds, n_frames)
        collector.add_performance('ber_vs_L', 'L', L, 'DU-tun',
                                  'BER_pct', np.mean(du_results[L]), np.std(du_results[L]),
                                  n_seeds, n_frames)
        print("done")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    # EKF基线
    ekf_mean = np.mean(ekf_results)
    ax.axhline(y=ekf_mean, color=COLORS['EKF'], linestyle='--',
               label='EKF (L=1)', linewidth=1.2, alpha=0.8)

    # IEKF
    iekf_means = [np.mean(iekf_results[L]) for L in L_list]
    ax.plot(L_list, iekf_means, '^-', color=COLORS['IEKF-4'], label='IEKF')

    # GN
    gn_means = [np.mean(gn_results[L]) for L in L_list]
    ax.plot(L_list, gn_means, 's-', color=COLORS['GN-6'], label='GN')

    # DU
    du_means = [np.mean(du_results[L]) for L in L_list]
    ax.plot(L_list, du_means, 'D-', color=COLORS['DU-tun-6'], label='DU-tun')

    ax.set_xlabel('Number of Layers/Iterations $L$')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_list)
    ax.set_yscale('log')
    save_figure(fig, 'fig_ber_vs_L')



def fig_ber_vs_L_multiSNR():
    """BER vs L (多SNR)"""
    print("\nFig: BER vs L (multi-SNR)")

    L_list = [1, 2, 4, 6, 8]
    snr_list = [0, 5, 10, 15, 20]
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 20, 100

    results = {snr: {L: [] for L in L_list} for snr in snr_list}

    for snr in snr_list:
        cfg = create_thz_config(n_f=8, n_t=4, snr_db=snr, adc_bits=4)
        model = THzISACModel(cfg)

        for L in L_list:
            print(f"  SNR={snr}dB, L={L}...", end=" ", flush=True)
            for seed in range(n_seeds):
                y_seq, x_true, _, _ = generate_episode_with_impairments(
                    model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

                du_cfg = DUMAPConfig(n_layers=L)
                du_cfg.step_scale = get_du_step_scale()
                du = DUMAP(du_cfg)
                x_du = du.forward_sequence(model, y_seq, x0, P0)[0]
                ber, _ = quick_ber_evm(x_true, x_du, snr, seed)
                results[snr][L].append(ber * 100)

            collector.add_performance('ber_vs_L_multiSNR', 'L', L, f'DU_SNR{snr}',
                                      'BER_pct', np.mean(results[snr][L]),
                                      np.std(results[snr][L]), n_seeds, n_frames)
            print("done")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    for snr in snr_list:
        means = [np.mean(results[snr][L]) for L in L_list]
        ax.plot(L_list, means, marker='o',
                color=MULTI_SNR_COLORS[snr],
                linestyle=MULTI_SNR_LINESTYLES[snr],
                label=f'SNR={snr}dB')
    ax.set_xlabel('Number of Layers $L$')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_list)
    save_figure(fig, 'fig_ber_vs_L_multiSNR')


def fig_ber_vs_pslip():
    """BER vs p_slip"""
    print("\nFig: BER vs p_slip")

    p_slip_list = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15]
    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']

    cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    n_seeds, n_frames = 30, 100

    results = {p: {m: [] for m in methods} for p in p_slip_list}

    for p_slip in p_slip_list:
        print(f"  p_slip={p_slip}...", end=" ", flush=True)
        slip_cfg = SlipConfig.killer_pi(p_slip=p_slip) if p_slip > 0 else None

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                ber, _ = quick_ber_evm(x_true, x_hat, 10, seed)
                results[p_slip][method].append(ber * 100)

        for m in methods:
            collector.add_performance('ber_vs_pslip', 'p_slip', p_slip, m,
                                      'BER_pct', np.mean(results[p_slip][m]),
                                      np.std(results[p_slip][m]), n_seeds, n_frames)
        print("done")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    for method in methods:
        means = [np.mean(results[p][method]) for p in p_slip_list]
        marker = 'x' if method == 'EKF' else 's' if method == 'GN-6' else 'D'
        ax.plot(p_slip_list, means, marker=marker, color=COLORS[method], label=method)
    ax.set_xlabel('Slip Probability $p_{slip}$')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_ber_vs_pslip')


def fig_ber_vs_adc():
    """BER vs ADC bits"""
    print("\nFig: BER vs ADC")

    adc_list = [2, 3, 4, 5, 6, 8]
    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 30, 100

    results = {b: {m: [] for m in methods} for b in adc_list}

    for bits in adc_list:
        print(f"  ADC={bits}bits...", end=" ", flush=True)
        cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=bits)
        model = THzISACModel(cfg)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                ber, _ = quick_ber_evm(x_true, x_hat, 10, seed)
                results[bits][method].append(ber * 100)

        for m in methods:
            collector.add_performance('ber_vs_adc', 'adc_bits', bits, m,
                                      'BER_pct', np.mean(results[bits][m]),
                                      np.std(results[bits][m]), n_seeds, n_frames)
        print("done")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    for method in methods:
        means = [np.mean(results[b][method]) for b in adc_list]
        marker = 'x' if method == 'EKF' else 's' if method == 'GN-6' else 'D'
        ax.plot(adc_list, means, marker=marker, color=COLORS[method], label=method)
    ax.set_xlabel('ADC Resolution (bits)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(adc_list)
    save_figure(fig, 'fig_ber_vs_adc')


def fig_phase_rmse_vs_frame_duration():
    """Phase RMSE vs frame duration"""
    print("\nFig: Phase RMSE vs Frame Duration")

    fd_list = [50e-6, 100e-6, 200e-6, 500e-6, 1e-3]
    methods = ['EKF', 'GN-6', 'DU-tun-6']

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 30, 100

    results = {fd: {m: [] for m in methods} for fd in fd_list}

    for fd in fd_list:
        print(f"  T_fr={fd * 1e6:.0f}μs...", end=" ", flush=True)
        cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4, frame_duration_s=fd)
        model = THzISACModel(cfg)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                rmse = compute_phase_rmse(x_hat, x_true)
                results[fd][method].append(rmse)

        for m in methods:
            collector.add_performance('phase_rmse_vs_fd', 'frame_duration_us',
                                      fd * 1e6, m, 'Phase_RMSE',
                                      np.mean(results[fd][m]), np.std(results[fd][m]),
                                      n_seeds, n_frames)
        print("done")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    fd_us = [fd * 1e6 for fd in fd_list]
    for method in methods:
        means = [np.mean(results[fd][method]) for fd in fd_list]
        marker = 'x' if method == 'EKF' else 's' if method == 'GN-6' else 'D'
        ax.semilogx(fd_us, means, marker=marker, color=COLORS[method], label=method)
    ax.set_xlabel('Frame Duration ($\\mu$s)')
    ax.set_ylabel('Phase RMSE (rad)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_phase_rmse_vs_frame_duration')


# ============================================================================
# 敏感性分析
# ============================================================================

def fig_sensitivity_phi():
    """步长敏感性 - φ"""
    print("\nFig: Sensitivity (phi)")

    alpha_phi_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 30, 100

    results = {a: [] for a in alpha_phi_list}

    for alpha in alpha_phi_list:
        print(f"  α_φ={alpha}...", end=" ", flush=True)
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            du_cfg = DUMAPConfig(n_layers=6)
            du_cfg.step_scale = np.array([1.0, 0.1, alpha])
            du = DUMAP(du_cfg)
            x_du = du.forward_sequence(model, y_seq, x0, P0)[0]
            rmse, _ = compute_rmse(x_du, x_true)
            results[alpha].append(rmse)

        collector.add_sensitivity('alpha_phi', alpha,
                                  np.mean(results[alpha]), np.std(results[alpha]), n_seeds)
        print("done")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    means = [np.mean(results[a]) for a in alpha_phi_list]
    stds = [np.std(results[a]) for a in alpha_phi_list]
    ax.errorbar(alpha_phi_list, means, yerr=stds, marker='o',
                color='#E41A1C', capsize=3, linewidth=1.5, markersize=7)
    ax.axvline(1.5, color='gray', linestyle='--', alpha=0.7, label='Default (1.5)')
    ax.set_xlabel(r'Phase Step Scale $\alpha_\phi$')
    ax.set_ylabel('RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_sensitivity_phi')


def fig_sensitivity_nu():
    """步长敏感性 - ν"""
    print("\nFig: Sensitivity (nu)")

    alpha_nu_list = [0.02, 0.05, 0.1, 0.2, 0.5]
    cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 30, 100

    results = {a: [] for a in alpha_nu_list}

    for alpha in alpha_nu_list:
        print(f"  α_ν={alpha}...", end=" ", flush=True)
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            du_cfg = DUMAPConfig(n_layers=6)
            du_cfg.step_scale = np.array([1.0, alpha, 1.5])
            du = DUMAP(du_cfg)
            x_du = du.forward_sequence(model, y_seq, x0, P0)[0]
            rmse, _ = compute_rmse(x_du, x_true)
            results[alpha].append(rmse)

        collector.add_sensitivity('alpha_nu', alpha,
                                  np.mean(results[alpha]), np.std(results[alpha]), n_seeds)
        print("done")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    means = [np.mean(results[a]) for a in alpha_nu_list]
    stds = [np.std(results[a]) for a in alpha_nu_list]
    ax.errorbar(alpha_nu_list, means, yerr=stds, marker='s',
                color='#377EB8', capsize=3, linewidth=1.5, markersize=7)
    ax.axvline(0.1, color='gray', linestyle='--', alpha=0.7, label='Default (0.1)')
    ax.set_xlabel(r'Doppler Step Scale $\alpha_\nu$')
    ax.set_ylabel('RMSE')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_sensitivity_nu')


def fig_sensitivity_tau():
    """步长敏感性 - τ"""
    print("\nFig: Sensitivity (tau)")

    alpha_tau_list = [0.2, 0.5, 1.0, 2.0, 5.0]
    cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 30, 100

    results = {a: [] for a in alpha_tau_list}

    for alpha in alpha_tau_list:
        print(f"  α_τ={alpha}...", end=" ", flush=True)
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            du_cfg = DUMAPConfig(n_layers=6)
            du_cfg.step_scale = np.array([alpha, 0.1, 1.5])
            du = DUMAP(du_cfg)
            x_du = du.forward_sequence(model, y_seq, x0, P0)[0]
            rmse, _ = compute_rmse(x_du, x_true)
            results[alpha].append(rmse)

        collector.add_sensitivity('alpha_tau', alpha,
                                  np.mean(results[alpha]), np.std(results[alpha]), n_seeds)
        print("done")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    means = [np.mean(results[a]) for a in alpha_tau_list]
    stds = [np.std(results[a]) for a in alpha_tau_list]
    ax.errorbar(alpha_tau_list, means, yerr=stds, marker='^',
                color='#4DAF4A', capsize=3, linewidth=1.5, markersize=7)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.7, label='Default (1.0)')
    ax.set_xlabel(r'Delay Step Scale $\alpha_\tau$')
    ax.set_ylabel('RMSE')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_sensitivity_tau')


# ============================================================================
# 辅助图
# ============================================================================

def fig_improvement_bar():
    """改进百分比柱状图"""
    print("\nFig: Improvement Bar")

    cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 30, 100

    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']
    all_rmse = {m: [] for m in methods}

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}...", end="\r", flush=True)
        y_seq, x_true, _, _ = generate_episode_with_impairments(
            model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

        for method in methods:
            x_hat = run_estimator(method, model, y_seq, x0, P0)
            rmse, _ = compute_rmse(x_hat, x_true)
            all_rmse[method].append(rmse)

    means = [np.mean(all_rmse[m]) for m in methods]
    sems = [np.std(all_rmse[m]) / np.sqrt(n_seeds) for m in methods]

    for m in methods:
        collector.add_auxiliary('improvement_bar', method=m,
                                mean=np.mean(all_rmse[m]),
                                std=np.std(all_rmse[m]),
                                n_seeds=n_seeds)

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
    print()
    save_figure(fig, 'fig_improvement_bar')


def fig_ccdf():
    """相位误差 CCDF"""
    print("\nFig: CCDF")

    cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4)
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

    for m in methods:
        errs = np.array(all_errors[m])
        collector.add_auxiliary('ccdf_stats', method=m,
                                mean=np.mean(errs), std=np.std(errs),
                                median=np.median(errs),
                                p90=np.percentile(errs, 90),
                                p95=np.percentile(errs, 95),
                                p99=np.percentile(errs, 99),
                                n_samples=len(errs))

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
        cfg = create_thz_config(n_f=8, n_t=n_t, snr_db=10, adc_bits=4)
        model = THzISACModel(cfg)

        pcrb_rec = PCRBRecursion(d=3)
        pcrb_seq, _ = pcrb_rec.run_sequence(model, [x0] * 50)

        sqrt_tau = np.sqrt(np.mean([p[0] for p in pcrb_seq[10:]]))
        sqrt_nu = np.sqrt(np.mean([p[1] for p in pcrb_seq[10:]]))
        sqrt_phi = np.sqrt(np.mean([p[2] for p in pcrb_seq[10:]]))

        pcrb_tau.append(sqrt_tau)
        pcrb_nu.append(sqrt_nu)
        pcrb_phi.append(sqrt_phi)

        collector.add_auxiliary('pcrb', n_t=n_t,
                                sqrt_pcrb_tau=sqrt_tau,
                                sqrt_pcrb_nu=sqrt_nu,
                                sqrt_pcrb_phi=sqrt_phi)

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


# ============================================================================
# ★★★ 新增：THz-Specific 主文图 ★★★
# ============================================================================

def fig_thz_phase_sensitivity():
    """
    THz-Specific图1: 相位敏感度随载频scaling

    通过调整slip概率模拟不同载频下的PN效应
    """
    print("\nFig: THz Phase Sensitivity Scaling")

    fc_list = [10, 60, 100, 300]  # GHz
    p_slip_base = 0.005

    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']

    results = {fc: {m: [] for m in methods} for fc in fc_list}

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    n_seeds, n_frames = 20, 100

    for fc in fc_list:
        pn_scaling = (fc / 10) ** 2
        p_slip = min(p_slip_base * np.sqrt(pn_scaling), 0.15)

        print(f"  f_c = {fc} GHz, p_slip = {p_slip:.3f}...", end=" ", flush=True)

        cfg = create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4)
        model = THzISACModel(cfg)
        slip_cfg = SlipConfig.killer_pi(p_slip=p_slip)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                rmse = compute_phase_rmse(x_hat, x_true)
                results[fc][method].append(rmse)

        for m in methods:
            collector.add_performance('thz_phase_sensitivity', 'fc_GHz', fc, m,
                                      'Phase_RMSE', np.mean(results[fc][m]),
                                      np.std(results[fc][m]), n_seeds, n_frames)
        print("done")

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    for method in methods:
        means = [np.mean(results[fc][method]) for fc in fc_list]
        stds = [np.std(results[fc][method]) for fc in fc_list]
        marker = 'x' if method == 'EKF' else 's' if method == 'GN-6' else 'D'
        ax.errorbar(fc_list, means, yerr=stds,
                    color=COLORS[method], marker=marker,
                    label=method, capsize=3, linewidth=1.2)

    ax.set_xlabel('Carrier Frequency $f_c$ (GHz)')
    ax.set_ylabel('Phase RMSE (rad)')
    ax.set_xscale('log')
    ax.set_xticks(fc_list)
    ax.set_xticklabels([str(fc) for fc in fc_list])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.annotate('PN $\\propto f_c^2$', xy=(100, 0.5), fontsize=8, style='italic')

    save_figure(fig, 'fig_thz_phase_sensitivity')


def fig_thz_doppler_squint():
    """
    THz-Specific图2: Doppler Squint影响验证
    """
    print("\nFig: THz Doppler Squint Validation")

    bandwidth_list = [100e6, 300e6, 500e6, 1e9]
    f_c = 300e9

    methods = ['GN-6', 'DU-tun-6']
    results_no_sq = {B: {m: [] for m in methods} for B in bandwidth_list}
    results_sq = {B: {m: [] for m in methods} for B in bandwidth_list}

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 15, 100

    for B in bandwidth_list:
        print(f"  B = {B / 1e6:.0f} MHz...", end=" ", flush=True)

        cfg_no = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4,
                               bandwidth_hz=B, enable_doppler_squint=False)
        model_no = THzISACModel(cfg_no)

        cfg_sq = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4,
                               bandwidth_hz=B, enable_doppler_squint=True)
        model_sq = THzISACModel(cfg_sq)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model_no, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            for method in methods:
                x_hat = run_estimator(method, model_no, y_seq, x0, P0)
                rmse = compute_phase_rmse(x_hat, x_true)
                results_no_sq[B][method].append(rmse)

            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model_sq, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            for method in methods:
                x_hat = run_estimator(method, model_sq, y_seq, x0, P0)
                rmse = compute_phase_rmse(x_hat, x_true)
                results_sq[B][method].append(rmse)

        for m in methods:
            no_sq = np.mean(results_no_sq[B][m])
            sq = np.mean(results_sq[B][m])
            impact = (sq - no_sq) / no_sq * 100
            print(f"{m}: {impact:+.2f}%", end=" ")
        print()

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    for method in methods:
        impacts = []
        for B in bandwidth_list:
            no_sq = np.mean(results_no_sq[B][method])
            sq = np.mean(results_sq[B][method])
            impact = (sq - no_sq) / no_sq * 100
            impacts.append(impact)

        marker = 's' if method == 'GN-6' else 'D'
        ax.plot([B / 1e6 for B in bandwidth_list], impacts,
                marker=marker, color=COLORS[method], label=method, linewidth=1.2)

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='1% threshold')
    ax.set_xlabel('Pilot Bandwidth $B$ (MHz)')
    ax.set_ylabel('RMSE Increase due to Squint (%)')
    ax.set_xscale('log')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 2])
    ax.axvspan(50, 200, alpha=0.1, color='green')
    ax.text(100, 1.5, 'Pilot subband\n(negligible)', fontsize=7, ha='center', color='green')

    save_figure(fig, 'fig_thz_doppler_squint')


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("IEEE 格式图像生成 - V3 完整物理模型版")
    print("=" * 70)
    print()
    print("★ V3物理效应：")
    print("  - P0: 连续PN Wiener模型（linewidth=100Hz）")
    print("  - P0: Doppler squint")
    print("  - P1/P2: Beam squint & Pointing jitter（stress-test）")
    print()
    print("=" * 70)

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

    # 敏感性分析
    print("\n[敏感性分析]")
    fig_sensitivity_phi()
    fig_sensitivity_nu()
    fig_sensitivity_tau()

    # 辅助图
    print("\n[辅助图]")
    fig_improvement_bar()
    fig_ccdf()
    fig_pcrb_nt()

    # ★★★ THz-Specific 主文图 ★★★
    print("\n[THz-Specific 主文图]")
    fig_thz_phase_sensitivity()
    fig_thz_doppler_squint()

    # 保存CSV数据
    print("\n" + "=" * 70)
    print("保存CSV数据...")
    collector.save_all()

    print("\n" + "=" * 70)
    print(f"完成! 输出目录: {OUTPUT_DIR}/")
    print("=" * 70)
    print("\n生成的图像 (共20+张):")
    print("  - 核心图: fig_ber_vs_snr, fig_rmse_vs_snr, ...")
    print("  - BER系列: fig_ber_vs_L, fig_ber_vs_pslip, ...")
    print("  - 敏感性: fig_sensitivity_phi/nu/tau")
    print("  - 辅助图: fig_ccdf, fig_pcrb_nt, ...")
    print("  - ★THz图: fig_thz_phase_sensitivity, fig_thz_doppler_squint")
    print("\nCSV数据文件:")
    print("  - data_performance.csv")
    print("  - data_sensitivity.csv")
    print("  - data_heatmap.csv")
    print("  - data_auxiliary.csv")


if __name__ == "__main__":
    main()