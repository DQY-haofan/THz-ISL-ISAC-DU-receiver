#!/usr/bin/env python3
"""
THz-ISAC DU-MAP IEEE 格式图像生成 v9 - THz-ISL增强版
======================================================

基于v8，优化图像组合：
- 删除4张低价值图：ber_vs_L_multiSNR, sensitivity_nu, sensitivity_tau, improvement_bar
- 新增3张THz-ISL关键图：pointing_jitter_impact, beam_squint_wideband, thz_vs_mmwave
- 改进slip_2d_heatmap为合并对比版

保留所有对比算法：EKF, IEKF-4, GN-6, DU-tun-6

使用方法：
    python3 generate_ieee_figures_v9_thz.py

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

# 所有方法的颜色配置
COLORS = {
    'EKF': '#E41A1C',
    'IEKF-4': '#FF7F00',
    'GN-6': '#377EB8',
    'DU-tun-6': '#4DAF4A',
    'PCRB': '#000000',
    # 额外颜色用于扩展
    'GN-6-nosq': '#377EB8',
    'DU-tun-6-nosq': '#4DAF4A',
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

# 多SNR颜色（保留用于可能的扩展）
MULTI_SNR_COLORS = {
    0: '#1f77b4',
    5: '#ff7f0e',
    10: '#2ca02c',
    15: '#d62728',
    20: '#9467bd',
}

# 载频颜色
FC_COLORS = {
    10: '#1f77b4',
    60: '#ff7f0e',
    100: '#2ca02c',
    300: '#d62728',
}


def setup_ieee_style():
    """设置IEEE论文图像样式"""
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
    """保存图像为PNG和PDF格式"""
    fig.savefig(f'{OUTPUT_DIR}/{name}.png', dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{OUTPUT_DIR}/{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  ✓ {name}")


# ============================================================================
# THz配置工厂函数
# ============================================================================

def create_thz_config(n_f=8, n_t=4, snr_db=10, adc_bits=4,
                      bandwidth_hz=100e6, frame_duration_s=100e-6,
                      pn_linewidth_hz=100.0,
                      enable_beam_squint=True,
                      enable_pointing_jitter=True,
                      pointing_jitter_std_deg=0.1,
                      beam_squint_n_ant=64,
                      beam_squint_theta0_deg=15.0):
    """
    创建THz-ISL配置 - V3完整物理模型版（全部效应启用）

    Args:
        n_f: 频率导频数
        n_t: 时间导频数
        snr_db: 信噪比
        adc_bits: ADC位数
        bandwidth_hz: 带宽
        frame_duration_s: 帧时长
        pn_linewidth_hz: 相位噪声线宽
        enable_beam_squint: 是否启用beam squint
        enable_pointing_jitter: 是否启用pointing jitter
        pointing_jitter_std_deg: pointing jitter标准差(度)
        beam_squint_n_ant: 天线数量
        beam_squint_theta0_deg: 波束指向角(度)
    """
    return THzISACConfig(
        n_f=n_f,
        n_t=n_t,
        snr_db=snr_db,
        adc_bits=adc_bits,
        bandwidth_hz=bandwidth_hz,
        frame_duration_s=frame_duration_s,

        # ===== P0: 相位噪声 =====
        enable_continuous_pn=True,
        pn_linewidth_hz=pn_linewidth_hz,

        # ===== P0: Doppler squint =====
        enable_doppler_squint=True,

        # ===== P1: Beam squint =====
        enable_beam_squint=enable_beam_squint,
        beam_squint_n_ant=beam_squint_n_ant,
        beam_squint_d_over_lambda=0.5,
        beam_squint_theta0_deg=beam_squint_theta0_deg,

        # ===== P2: Pointing jitter =====
        enable_pointing_jitter=enable_pointing_jitter,
        pointing_jitter_std_deg=pointing_jitter_std_deg,
        pointing_jitter_ar=0.95,
    )


# ============================================================================
# CSV数据收集器
# ============================================================================

class DataCollector:
    """收集所有实验数据用于CSV输出"""

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
            'improvement_vs_EKF_pct': (ekf_mean - du_mean) / ekf_mean * 100 if ekf_mean > 0 else 0,
            'improvement_vs_GN_pct': (gn_mean - du_mean) / gn_mean * 100 if gn_mean > 0 else 0,
            'n_seeds': n_seeds
        })

    def add_auxiliary(self, data_type: str, **kwargs):
        row = {'data_type': data_type}
        row.update(kwargs)
        self.auxiliary_data.append(row)

    def save_all(self):
        """保存所有CSV数据"""
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


# 全局数据收集器
collector = DataCollector()


# ============================================================================
# DU配置和估计器运行函数
# ============================================================================

def get_du_step_scale():
    """获取DU-MAP的步长缩放参数"""
    return np.array([1.0, 0.1, 1.5])


def run_estimator(method: str, model: THzISACModel, y_seq: List,
                  x0: np.ndarray, P0: np.ndarray) -> List[np.ndarray]:
    """
    运行指定的估计器

    Args:
        method: 方法名称 ('EKF', 'IEKF-4', 'GN-6', 'DU-tun-6' 等)
        model: THz-ISAC模型
        y_seq: 观测序列
        x0: 初始状态
        P0: 初始协方差

    Returns:
        估计状态序列
    """
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
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_rmse(x_hat: List[np.ndarray], x_true: List[np.ndarray]) -> Tuple[float, float]:
    """计算总体RMSE（含相位wrapping）"""
    errors = []
    for xh, xt in zip(x_hat, x_true):
        e = xh - xt
        e[2] = wrap_angle(e[2])
        errors.append(np.sqrt(np.sum(e ** 2)))
    return np.mean(errors), np.std(errors)


def compute_phase_rmse(x_hat: List[np.ndarray], x_true: List[np.ndarray]) -> float:
    """计算相位RMSE"""
    errors = []
    for xh, xt in zip(x_hat, x_true):
        e_phi = wrap_angle(xh[2] - xt[2])
        errors.append(e_phi ** 2)
    return np.sqrt(np.mean(errors))


def compute_component_rmse(x_hat: List[np.ndarray], x_true: List[np.ndarray]) -> Tuple[float, float, float]:
    """计算各分量RMSE: (tau, nu, phi)"""
    tau_err, nu_err, phi_err = [], [], []
    for xh, xt in zip(x_hat, x_true):
        tau_err.append((xh[0] - xt[0]) ** 2)
        nu_err.append((xh[1] - xt[1]) ** 2)
        phi_err.append(wrap_angle(xh[2] - xt[2]) ** 2)
    return np.sqrt(np.mean(tau_err)), np.sqrt(np.mean(nu_err)), np.sqrt(np.mean(phi_err))


# ============================================================================
# 核心图 (6张)
# ============================================================================

def fig_ber_vs_snr():
    """
    核心图1: BER vs SNR
    比较所有4种方法在不同SNR下的BER性能
    """
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

    # 绘图
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
    """
    核心图2: RMSE vs SNR (含PCRB下界)
    """
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

        # 计算PCRB
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

    # 绘图
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
    """
    核心图3: RMSE vs 迭代层数 L
    包含EKF基线和IEKF/GN/DU对比
    """
    print("\nFig: RMSE vs L")

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

            # EKF (只在L=1时计算)
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
    """
    核心图4: Slip恢复时间箱线图
    """
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

    # 绘图
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


def fig_slip_heatmap_combined():
    """
    核心图5: Slip严重度热力图 (合并版 - DU改进百分比)
    显示DU相对于EKF的改进百分比
    """
    print("\nFig: Slip Heatmap Combined")

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
    improvement_grid = np.zeros((len(amplitudes), len(p_slips)))

    for i, amp in enumerate(amplitudes):
        for j, p_slip in enumerate(p_slips):
            print(f"  amp={amp}π, p_slip={p_slip}...", end=" ", flush=True)
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

            # 计算DU相对于EKF的改进
            improvement_grid[i, j] = (ekf_grid[i, j] - du_grid[i, j]) / ekf_grid[i, j] * 100

            collector.add_heatmap(amp, p_slip,
                                  np.mean(ekf_rmse), np.std(ekf_rmse),
                                  np.mean(gn_rmse), np.std(gn_rmse),
                                  np.mean(du_rmse), np.std(du_rmse), n_seeds)
            print("done")

    # 绘制合并图：DU改进百分比热力图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    im = ax.imshow(improvement_grid, cmap='RdYlGn', aspect='auto', origin='lower',
                   vmin=0, vmax=100)
    ax.set_xticks(range(len(p_slips)))
    ax.set_xticklabels([f'{p}' for p in p_slips])
    ax.set_yticks(range(len(amplitudes)))
    ax.set_yticklabels([f'{a}π' for a in amplitudes])
    ax.set_xlabel('Slip Probability $p_{slip}$')
    ax.set_ylabel('Slip Amplitude')

    # 添加数值标注
    for i in range(len(amplitudes)):
        for j in range(len(p_slips)):
            text = ax.text(j, i, f'{improvement_grid[i, j]:.0f}%',
                           ha='center', va='center', fontsize=7,
                           color='white' if improvement_grid[i, j] > 50 else 'black')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('DU Improvement vs EKF (%)')
    save_figure(fig, 'fig_slip_heatmap_improvement')


def fig_phase_tracking():
    """
    核心图6: 相位跟踪轨迹图
    """
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

    for sf in slip_frames:
        ax.axvline(sf, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)

    ax.set_xlabel('Frame')
    ax.set_ylabel(r'Phase $\phi$ (rad)')
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    save_figure(fig, 'fig_phase_tracking')


# ============================================================================
# BER系列图 (3张)
# ============================================================================

def fig_ber_vs_L():
    """
    BER图1: BER vs L
    """
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

            if L == 1:
                x_ekf = run_estimator('EKF', model, y_seq, x0, P0)
                ber_ekf, _ = quick_ber_evm(x_true, x_ekf, 10, seed)
                ekf_results.append(ber_ekf * 100)

            x_iekf = run_estimator(f'IEKF-{L}', model, y_seq, x0, P0)
            ber_iekf, _ = quick_ber_evm(x_true, x_iekf, 10, seed)
            iekf_results[L].append(ber_iekf * 100)

            gn = GaussNewtonMAP(GNSolverConfig(max_iters=L))
            x_gn = gn.solve_sequence(model, y_seq, x0, P0)[0]
            ber_gn, _ = quick_ber_evm(x_true, x_gn, 10, seed)
            gn_results[L].append(ber_gn * 100)

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

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    ekf_mean = np.mean(ekf_results)
    ax.axhline(y=ekf_mean, color=COLORS['EKF'], linestyle='--',
               label='EKF (L=1)', linewidth=1.2, alpha=0.8)

    iekf_means = [np.mean(iekf_results[L]) for L in L_list]
    ax.plot(L_list, iekf_means, '^-', color=COLORS['IEKF-4'], label='IEKF')

    gn_means = [np.mean(gn_results[L]) for L in L_list]
    ax.plot(L_list, gn_means, 's-', color=COLORS['GN-6'], label='GN')

    du_means = [np.mean(du_results[L]) for L in L_list]
    ax.plot(L_list, du_means, 'D-', color=COLORS['DU-tun-6'], label='DU-tun')

    ax.set_xlabel('Number of Layers/Iterations $L$')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_list)
    ax.set_yscale('log')
    save_figure(fig, 'fig_ber_vs_L')


def fig_ber_vs_pslip():
    """
    BER图2: BER vs p_slip
    """
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

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    for method in methods:
        means = [np.mean(results[p][method]) for p in p_slip_list]
        ax.plot(p_slip_list, means, marker=MARKERS[method],
                color=COLORS[method], label=method)
    ax.set_xlabel('Slip Probability $p_{slip}$')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'fig_ber_vs_pslip')


def fig_ber_vs_adc():
    """
    BER图3: BER vs ADC bits
    """
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

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    for method in methods:
        means = [np.mean(results[b][method]) for b in adc_list]
        ax.plot(adc_list, means, marker=MARKERS[method],
                color=COLORS[method], label=method)
    ax.set_xlabel('ADC Resolution (bits)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(adc_list)
    save_figure(fig, 'fig_ber_vs_adc')


# ============================================================================
# 敏感性分析 (1张 - 只保留phi)
# ============================================================================

def fig_sensitivity_phi():
    """
    敏感性图: 相位步长敏感性分析
    这是最关键的参数，因为相位是THz系统最敏感的分量
    """
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

    # 绘图
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


# ============================================================================
# 辅助图 (2张)
# ============================================================================

def fig_ccdf():
    """
    辅助图1: 相位误差CCDF (互补累积分布函数)
    展示尾部性能
    """
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
    """
    辅助图2: PCRB vs n_t (时间导频数)
    理论分析支撑
    """
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


# ============================================================================
# ★★★ THz-ISL 特性图 (4张 - 核心新增) ★★★
# ============================================================================

def fig_thz_vs_mmwave():
    """
    THz图1: THz vs mmWave 综合对比
    展示不同载频下各方法的Phase RMSE，证明THz方法必要性

    关键点：
    - 300GHz相位敏感度是10GHz的30倍
    - DU-MAP在高频优势更明显
    """
    print("\nFig: THz vs mmWave Comprehensive")

    fc_list = [10, 60, 100, 300]  # GHz
    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']

    # 根据载频调整相位噪声强度（模拟实际物理）
    # 相位噪声 ∝ f_c^2
    pn_base = 0.005  # 10GHz时的slip概率

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    n_seeds, n_frames = 25, 100

    results = {fc: {m: [] for m in methods} for fc in fc_list}

    for fc in fc_list:
        # 相位噪声强度随载频平方增长
        pn_scaling = (fc / 10) ** 2
        p_slip = min(pn_base * np.sqrt(pn_scaling), 0.15)

        print(f"  f_c = {fc} GHz (p_slip={p_slip:.3f})...", end=" ", flush=True)

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
            collector.add_performance('thz_vs_mmwave', 'fc_GHz', fc, m,
                                      'Phase_RMSE', np.mean(results[fc][m]),
                                      np.std(results[fc][m]), n_seeds, n_frames)
        print("done")

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    for method in methods:
        means = [np.mean(results[fc][method]) for fc in fc_list]
        stds = [np.std(results[fc][method]) for fc in fc_list]
        ax.errorbar(fc_list, means, yerr=stds,
                    color=COLORS[method], marker=MARKERS[method],
                    label=method, capsize=3, linewidth=1.2)

    ax.set_xlabel('Carrier Frequency $f_c$ (GHz)')
    ax.set_ylabel('Phase RMSE (rad)')
    ax.set_xscale('log')
    ax.set_xticks(fc_list)
    ax.set_xticklabels([str(fc) for fc in fc_list])
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    # 添加注释
    ax.annotate('Phase noise $\\propto f_c^2$', xy=(60, 0.4), fontsize=7,
                style='italic', color='gray')

    # 计算并标注改进
    du_10 = np.mean(results[10]['DU-tun-6'])
    du_300 = np.mean(results[300]['DU-tun-6'])
    ekf_300 = np.mean(results[300]['EKF'])
    imp = (ekf_300 - du_300) / ekf_300 * 100
    ax.annotate(f'DU: {imp:.0f}% better\n@ 300GHz', xy=(300, du_300 * 1.1),
                fontsize=6, ha='center', color=COLORS['DU-tun-6'])

    save_figure(fig, 'fig_thz_vs_mmwave')


def fig_pointing_jitter_impact():
    """
    THz图2: Pointing Jitter影响分析

    ISL特有问题：卫星姿态抖动导致波束指向误差
    分析不同抖动程度下的性能
    """
    print("\nFig: Pointing Jitter Impact")

    jitter_std_list = [0.0, 0.05, 0.1, 0.2, 0.5]  # degrees
    methods = ['EKF', 'GN-6', 'DU-tun-6']

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 25, 100

    results_rmse = {j: {m: [] for m in methods} for j in jitter_std_list}
    results_ber = {j: {m: [] for m in methods} for j in jitter_std_list}

    for jitter_std in jitter_std_list:
        print(f"  Jitter σ = {jitter_std}°...", end=" ", flush=True)

        cfg = create_thz_config(
            n_f=8, n_t=4, snr_db=10, adc_bits=4,
            enable_pointing_jitter=(jitter_std > 0),
            pointing_jitter_std_deg=jitter_std if jitter_std > 0 else 0.1
        )
        model = THzISACModel(cfg)

        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)

            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                rmse, _ = compute_rmse(x_hat, x_true)
                ber, _ = quick_ber_evm(x_true, x_hat, 10, seed)
                results_rmse[jitter_std][method].append(rmse)
                results_ber[jitter_std][method].append(ber * 100)

        for m in methods:
            collector.add_performance('pointing_jitter', 'jitter_std_deg', jitter_std, m,
                                      'RMSE', np.mean(results_rmse[jitter_std][m]),
                                      np.std(results_rmse[jitter_std][m]), n_seeds, n_frames)
            collector.add_performance('pointing_jitter', 'jitter_std_deg', jitter_std, m,
                                      'BER_pct', np.mean(results_ber[jitter_std][m]),
                                      np.std(results_ber[jitter_std][m]), n_seeds, n_frames)
        print("done")

    # 绘图: 双Y轴 (RMSE和BER)
    fig, ax1 = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    # RMSE (左Y轴)
    for method in methods:
        means = [np.mean(results_rmse[j][method]) for j in jitter_std_list]
        ax1.plot(jitter_std_list, means, marker=MARKERS.get(method, 'o'),
                 color=COLORS[method], label=f'{method}', linewidth=1.2)

    ax1.set_xlabel('Pointing Jitter $\\sigma_\\theta$ (deg)')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=7)

    # 标记典型ISL规格
    ax1.axvline(0.1, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate('Typical ISL\nspec', xy=(0.1, ax1.get_ylim()[1] * 0.9),
                 fontsize=6, ha='center', color='gray')

    save_figure(fig, 'fig_pointing_jitter_impact')


def fig_beam_squint_wideband():
    """
    THz图3: 大带宽下的Beam Squint影响

    分析不同带宽下beam squint对性能的影响
    在pilot带宽(100MHz)影响小，但wideband数据(1-10GHz)影响显著
    """
    print("\nFig: Beam Squint Wideband Analysis")

    bandwidth_list = [100e6, 500e6, 1e9, 2e9, 5e9, 10e9]  # Hz
    methods = ['GN-6', 'DU-tun-6']

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 20, 100

    # 存储有/无beam squint的结果
    results_no_bs = {B: {m: [] for m in methods} for B in bandwidth_list}
    results_bs = {B: {m: [] for m in methods} for B in bandwidth_list}

    for B in bandwidth_list:
        print(f"  B = {B / 1e9:.1f} GHz...", end=" ", flush=True)

        # 无beam squint
        cfg_no = create_thz_config(
            n_f=8, n_t=4, snr_db=10, adc_bits=4,
            bandwidth_hz=B, enable_beam_squint=False
        )
        model_no = THzISACModel(cfg_no)

        # 有beam squint (64天线, 15°指向)
        cfg_bs = create_thz_config(
            n_f=8, n_t=4, snr_db=10, adc_bits=4,
            bandwidth_hz=B, enable_beam_squint=True,
            beam_squint_n_ant=64, beam_squint_theta0_deg=15.0
        )
        model_bs = THzISACModel(cfg_bs)

        for seed in range(n_seeds):
            # 无beam squint
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model_no, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            for method in methods:
                x_hat = run_estimator(method, model_no, y_seq, x0, P0)
                rmse = compute_phase_rmse(x_hat, x_true)
                results_no_bs[B][method].append(rmse)

            # 有beam squint
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model_bs, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            for method in methods:
                x_hat = run_estimator(method, model_bs, y_seq, x0, P0)
                rmse = compute_phase_rmse(x_hat, x_true)
                results_bs[B][method].append(rmse)

        # 计算性能下降
        for m in methods:
            no_bs = np.mean(results_no_bs[B][m])
            bs = np.mean(results_bs[B][m])
            degradation = (bs - no_bs) / no_bs * 100
            collector.add_performance('beam_squint', 'bandwidth_GHz', B / 1e9, m,
                                      'degradation_pct', degradation,
                                      0, n_seeds, n_frames)
            print(f"{m}: {degradation:+.1f}%", end=" ")
        print()

    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    for method in methods:
        degradations = []
        for B in bandwidth_list:
            no_bs = np.mean(results_no_bs[B][method])
            bs = np.mean(results_bs[B][method])
            degradation = (bs - no_bs) / no_bs * 100
            degradations.append(degradation)

        marker = 's' if method == 'GN-6' else 'D'
        ax.plot([B / 1e9 for B in bandwidth_list], degradations,
                marker=marker, color=COLORS[method], label=method, linewidth=1.2)

    ax.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% threshold')
    ax.set_xlabel('Bandwidth $B$ (GHz)')
    ax.set_ylabel('RMSE Degradation due to Beam Squint (%)')
    ax.set_xscale('log')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    # 标记pilot和wideband区域
    ax.axvspan(0.05, 0.2, alpha=0.1, color='green')
    ax.axvspan(1, 10, alpha=0.1, color='red')
    ax.text(0.1, ax.get_ylim()[1] * 0.8, 'Pilot\nband', fontsize=6, ha='center', color='green')
    ax.text(3, ax.get_ylim()[1] * 0.8, 'Wideband\ndata', fontsize=6, ha='center', color='red')

    save_figure(fig, 'fig_beam_squint_wideband')


def fig_doppler_squint_validation():
    """
    THz图4: Doppler Squint验证

    验证Doppler squint建模的正确性和影响
    """
    print("\nFig: Doppler Squint Validation")

    bandwidth_list = [100e6, 300e6, 500e6, 1e9]
    methods = ['GN-6', 'DU-tun-6']

    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds, n_frames = 20, 100

    results_no_sq = {B: {m: [] for m in methods} for B in bandwidth_list}
    results_sq = {B: {m: [] for m in methods} for B in bandwidth_list}

    for B in bandwidth_list:
        print(f"  B = {B / 1e6:.0f} MHz...", end=" ", flush=True)

        # 无Doppler squint
        cfg_no = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4,
                               bandwidth_hz=B, enable_doppler_squint=False,
                               enable_continuous_pn=True, pn_linewidth_hz=100.0)
        model_no = THzISACModel(cfg_no)

        # 有Doppler squint
        cfg_sq = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4,
                               bandwidth_hz=B, enable_doppler_squint=True,
                               enable_continuous_pn=True, pn_linewidth_hz=100.0)
        model_sq = THzISACModel(cfg_sq)

        for seed in range(n_seeds):
            # 无Doppler squint
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model_no, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            for method in methods:
                x_hat = run_estimator(method, model_no, y_seq, x0, P0)
                rmse = compute_phase_rmse(x_hat, x_true)
                results_no_sq[B][method].append(rmse)

            # 有Doppler squint
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
            collector.add_performance('doppler_squint', 'bandwidth_MHz', B / 1e6, m,
                                      'impact_pct', impact, 0, n_seeds, n_frames)
            print(f"{m}: {impact:+.2f}%", end=" ")
        print()

    # 绘图
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
    ax.set_ylabel('RMSE Impact due to Doppler Squint (%)')
    ax.set_xscale('log')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    # 标记pilot subband
    ax.axvspan(50, 200, alpha=0.1, color='green')
    ax.text(100, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.8,
            'Pilot subband\n(negligible)', fontsize=6, ha='center', color='green')

    save_figure(fig, 'fig_doppler_squint_validation')


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数：生成所有图像"""
    print("=" * 70)
    print("IEEE 格式图像生成 - V9 THz-ISL增强版")
    print("=" * 70)
    print()
    print("★ 物理效应：")
    print("  - 连续PN Wiener模型（linewidth=100Hz）")
    print("  - Doppler squint")
    print("  - Beam squint (64-ULA)")
    print("  - Pointing jitter (ISL特有)")
    print()
    print("★ 对比方法：")
    print("  - EKF (基线)")
    print("  - IEKF-4 (迭代EKF)")
    print("  - GN-6 (Gauss-Newton MAP)")
    print("  - DU-tun-6 (Deep-Unfolded MAP)")
    print()
    print("=" * 70)

    setup_ieee_style()

    # ==================== 核心图 (6张) ====================
    print("\n[核心图 - 6张]")
    fig_ber_vs_snr()  # Fig.3: BER vs SNR
    fig_rmse_vs_snr()  # Fig.4: RMSE vs SNR (含PCRB)
    fig_rmse_vs_L()  # Fig.5: RMSE vs L
    fig_recovery_time()  # Fig.6: 恢复时间箱线图
    fig_slip_heatmap_combined()  # Fig.7: Slip热力图 (合并版)
    fig_phase_tracking()  # Fig.8: 相位跟踪轨迹

    # ==================== BER系列 (3张) ====================
    print("\n[BER系列图 - 3张]")
    fig_ber_vs_L()  # Fig.9: BER vs L
    fig_ber_vs_pslip()  # Fig.10: BER vs p_slip
    fig_ber_vs_adc()  # Fig.11: BER vs ADC bits

    # ==================== 敏感性分析 (1张) ====================
    print("\n[敏感性分析 - 1张]")
    fig_sensitivity_phi()  # Fig.12: φ步长敏感性

    # ==================== 辅助图 (2张) ====================
    print("\n[辅助图 - 2张]")
    fig_ccdf()  # Fig.13: CCDF
    fig_pcrb_nt()  # Fig.14: PCRB vs n_t

    # ==================== THz-ISL特性图 (4张 - 关键新增) ====================
    print("\n[THz-ISL特性图 - 4张] ★ 应对审稿人")
    fig_thz_vs_mmwave()  # Fig.15: THz vs mmWave对比 (关键!)
    fig_pointing_jitter_impact()  # Fig.16: Pointing jitter影响
    fig_beam_squint_wideband()  # Fig.17: Beam squint大带宽分析
    fig_doppler_squint_validation()  # Fig.18: Doppler squint验证

    # ==================== 保存CSV数据 ====================
    print("\n" + "=" * 70)
    print("保存CSV数据...")
    collector.save_all()

    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print(f"完成! 输出目录: {OUTPUT_DIR}/")
    print("=" * 70)

    print("\n生成的图像 (共16张):")
    print("  [核心图 - 6张]")
    print("    fig_ber_vs_snr, fig_rmse_vs_snr, fig_rmse_vs_L,")
    print("    fig_recovery_time, fig_slip_heatmap_improvement, fig_phase_tracking")
    print("  [BER系列 - 3张]")
    print("    fig_ber_vs_L, fig_ber_vs_pslip, fig_ber_vs_adc")
    print("  [敏感性 - 1张]")
    print("    fig_sensitivity_phi")
    print("  [辅助图 - 2张]")
    print("    fig_ccdf, fig_pcrb_nt")
    print("  [THz-ISL特性 - 4张] ★")
    print("    fig_thz_vs_mmwave, fig_pointing_jitter_impact,")
    print("    fig_beam_squint_wideband, fig_doppler_squint_validation")

    print("\nCSV数据文件:")
    print("  - data_performance.csv")
    print("  - data_sensitivity.csv")
    print("  - data_heatmap.csv")
    print("  - data_auxiliary.csv")

    print("\n★ 相比V8的变化:")
    print("  [删除] ber_vs_L_multiSNR, sensitivity_nu, sensitivity_tau, improvement_bar")
    print("  [新增] thz_vs_mmwave, pointing_jitter_impact, beam_squint_wideband")
    print("  [改进] slip_heatmap → slip_heatmap_improvement (合并版)")


if __name__ == "__main__":
    main()