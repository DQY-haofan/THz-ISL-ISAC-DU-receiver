#!/usr/bin/env python3
"""
THz-ISAC DU-MAP IEEE 格式图像生成
==================================

格式规范：
- IEEE 单栏: 3.5 inch 宽度
- 字体: 9pt
- 无 title (caption 在论文中)
- 统一颜色/线型/marker
- 输出: PNG (300dpi) + PDF + CSV
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
from typing import Dict, List

# 核心模块
from src.physics.thz_isac_model import THzISACConfig, THzISACModel, wrap_angle
from src.inference.gn_solver import GNSolverConfig, GaussNewtonMAP
from src.unfolding.du_map import DUMAP, DUMAPConfig
from src.baselines.wrapped_ekf import create_ekf
from src.baselines.ukf import create_ukf
from src.bcrlb.pcrb import PCRBRecursion
from src.sim.slip import generate_episode_with_impairments, SlipConfig, PhaseNoiseConfig
from src.metrics.system_metrics import quick_ber_evm

# ============================================================================
# IEEE 格式配置
# ============================================================================

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IEEE 单栏格式
IEEE_WIDTH = 3.5  # inches
IEEE_HEIGHT = 2.6  # inches (golden ratio approx)
IEEE_WIDTH_DOUBLE = 7.0  # 双栏或2panel
IEEE_FONTSIZE = 9
IEEE_DPI = 300

# 统一颜色方案
COLORS = {
    'EKF': '#E41A1C',      # 红
    'UKF': '#984EA3',      # 紫
    'GN-2': '#377EB8',     # 浅蓝
    'GN-4': '#377EB8',
    'GN-6': '#377EB8',     # 蓝
    'GN-8': '#377EB8',
    'DU-tun-2': '#4DAF4A', # 浅绿
    'DU-tun-4': '#4DAF4A',
    'DU-tun-6': '#4DAF4A', # 绿
    'DU-tun-8': '#4DAF4A',
    'PCRB': '#000000',     # 黑
}

MARKERS = {
    'EKF': 'x',
    'UKF': '+',
    'GN-2': 'o',
    'GN-4': 'o',
    'GN-6': 's',
    'GN-8': 's',
    'DU-tun-2': '^',
    'DU-tun-4': '^',
    'DU-tun-6': 'D',
    'DU-tun-8': 'D',
    'PCRB': 'None',
}

LINESTYLES = {
    'EKF': '-',
    'UKF': '-',
    'GN-2': '--',
    'GN-4': '--',
    'GN-6': '-',
    'GN-8': '-',
    'DU-tun-2': '--',
    'DU-tun-4': '--',
    'DU-tun-6': '-',
    'DU-tun-8': '-',
    'PCRB': '--',
}

def setup_ieee_style():
    """设置 IEEE 图像风格"""
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
    """保存图像为 PNG 和 PDF"""
    fig.savefig(f'{OUTPUT_DIR}/{name}.png', dpi=IEEE_DPI, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{OUTPUT_DIR}/{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  ✓ {name}")

def save_csv(data: Dict, name: str, headers: List[str]):
    """保存数据为 CSV"""
    with open(f'{OUTPUT_DIR}/{name}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        keys = list(data.keys())
        n_rows = len(data[keys[0]])
        for i in range(n_rows):
            row = [data[k][i] if isinstance(data[k], list) else data[k] for k in keys]
            writer.writerow(row)

# ============================================================================
# 工具函数
# ============================================================================

def compute_rmse(x_hat_seq, x_true_seq):
    mse = 0
    for xh, xt in zip(x_hat_seq, x_true_seq):
        err = xh - xt
        err[2] = wrap_angle(err[2])
        mse += np.sum(err**2)
    return np.sqrt(mse / len(x_hat_seq))

def compute_phase_rmse(x_hat_seq, x_true_seq):
    return np.sqrt(np.mean([wrap_angle(xh[2] - xt[2])**2 for xh, xt in zip(x_hat_seq, x_true_seq)]))

def run_estimator(method, model, y_seq, x0, P0):
    """统一接口"""
    if method == 'EKF':
        est = create_ekf('wrapped')
        return est.run_sequence(model, y_seq, x0, P0)[0]
    elif method == 'UKF':
        est = create_ukf()
        return est.run_sequence(model, y_seq, x0, P0)[0]
    elif method.startswith('GN-'):
        L = int(method.split('-')[1])
        est = GaussNewtonMAP(GNSolverConfig(max_iters=L))
        return est.solve_sequence(model, y_seq, x0, P0)[0]
    elif method.startswith('DU-tun-'):
        L = int(method.split('-')[2])
        cfg = DUMAPConfig(n_layers=L)
        cfg.step_scale = np.array([1.0, 0.1, 2.0])
        est = DUMAP(cfg)
        return est.forward_sequence(model, y_seq, x0, P0)[0]
    raise ValueError(f"Unknown method: {method}")

# ============================================================================
# Fig 1: BER vs SNR (核心通信曲线)
# ============================================================================

def fig_ber_vs_snr():
    """BER vs SNR - 标准通信系统曲线"""
    print("\nFig: BER vs SNR")
    
    snr_list = [0, 5, 10, 15, 20]
    methods = ['EKF', 'GN-6', 'DU-tun-6']
    
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig(p_slip=0.05, mode="discrete")
    n_seeds, n_frames = 25, 100
    
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
    
    # 绘图
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    for method in methods:
        ber_means = [np.mean(results[snr][method]) for snr in snr_list]
        ber_stds = [np.std(results[snr][method]) / np.sqrt(n_seeds) for snr in snr_list]
        ax.semilogy(snr_list, ber_means, 
                   marker=MARKERS[method], color=COLORS[method],
                   linestyle=LINESTYLES[method], label=method,
                   markerfacecolor='white' if method.startswith('GN') else COLORS[method])
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-1, 21])
    
    save_figure(fig, 'fig_ber_snr')
    
    # CSV
    csv_data = {'SNR': snr_list}
    for m in methods:
        csv_data[f'{m}_mean'] = [np.mean(results[snr][m]) for snr in snr_list]
        csv_data[f'{m}_std'] = [np.std(results[snr][m]) for snr in snr_list]
    save_csv(csv_data, 'fig_ber_snr', list(csv_data.keys()))

# ============================================================================
# Fig 2: RMSE vs SNR
# ============================================================================

def fig_rmse_vs_snr():
    """RMSE vs SNR"""
    print("\nFig: RMSE vs SNR")
    
    snr_list = [0, 5, 10, 15, 20]
    methods = ['EKF', 'GN-6', 'DU-tun-6']
    
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig(p_slip=0.05, mode="discrete")
    n_seeds, n_frames = 25, 100
    
    results = {snr: {m: [] for m in methods} for snr in snr_list}
    pcrb_vals = []
    
    for snr in snr_list:
        print(f"  SNR={snr}dB...", end=" ", flush=True)
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=snr, adc_bits=4)
        model = THzISACModel(cfg)
        
        # PCRB
        pcrb_rec = PCRBRecursion(d=3)
        pcrb_seq, _ = pcrb_rec.run_sequence(model, [x0]*50)
        pcrb_vals.append(np.sqrt(np.mean([np.sum(p) for p in pcrb_seq[10:]])))
        
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            
            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                results[snr][method].append(compute_rmse(x_hat, x_true))
        print("done")
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    ax.semilogy(snr_list, pcrb_vals, 'k--', label='√PCRB', linewidth=1.5)
    
    for method in methods:
        rmse_means = [np.mean(results[snr][method]) for snr in snr_list]
        ax.semilogy(snr_list, rmse_means,
                   marker=MARKERS[method], color=COLORS[method],
                   linestyle=LINESTYLES[method], label=method,
                   markerfacecolor='white' if method.startswith('GN') else COLORS[method])
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-1, 21])
    
    save_figure(fig, 'fig_rmse_snr')
    
    csv_data = {'SNR': snr_list, 'PCRB': pcrb_vals}
    for m in methods:
        csv_data[f'{m}_mean'] = [np.mean(results[snr][m]) for snr in snr_list]
    save_csv(csv_data, 'fig_rmse_snr', list(csv_data.keys()))

# ============================================================================
# Fig 3: BER vs p_slip
# ============================================================================

def fig_ber_vs_pslip():
    """BER vs slip probability"""
    print("\nFig: BER vs p_slip")
    
    p_slip_list = [0.0, 0.01, 0.02, 0.05, 0.10]
    methods = ['EKF', 'GN-6', 'DU-tun-6']
    
    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    n_seeds, n_frames = 25, 100
    
    results = {p: {m: [] for m in methods} for p in p_slip_list}
    
    for p_slip in p_slip_list:
        print(f"  p_slip={p_slip}...", end=" ", flush=True)
        slip_cfg = SlipConfig(p_slip=p_slip, mode="discrete") if p_slip > 0 else None
        
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            
            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                ber, _ = quick_ber_evm(x_true, x_hat, 10, seed)
                results[p_slip][method].append(ber * 100)
        print("done")
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    for method in methods:
        ber_means = [np.mean(results[p][method]) for p in p_slip_list]
        ber_stds = [np.std(results[p][method]) / np.sqrt(n_seeds) for p in p_slip_list]
        ax.errorbar(p_slip_list, ber_means, yerr=ber_stds,
                   marker=MARKERS[method], color=COLORS[method],
                   linestyle=LINESTYLES[method], label=method, capsize=2,
                   markerfacecolor='white' if method.startswith('GN') else COLORS[method])
    
    ax.set_xlabel('Slip Probability ($p_{slip}$)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'fig_ber_pslip')
    
    csv_data = {'p_slip': p_slip_list}
    for m in methods:
        csv_data[f'{m}_mean'] = [np.mean(results[p][m]) for p in p_slip_list]
    save_csv(csv_data, 'fig_ber_pslip', list(csv_data.keys()))

# ============================================================================
# Fig 4: RMSE vs L (Compute-Performance Pareto)
# ============================================================================

def fig_rmse_vs_L():
    """RMSE vs Iterations/Layers - Pareto curve"""
    print("\nFig: RMSE vs L")
    
    L_values = [2, 4, 6, 8]
    
    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig(p_slip=0.05, mode="discrete")
    n_seeds, n_frames = 30, 100
    
    results = {'EKF': [], 'GN': {L: [] for L in L_values}, 'DU': {L: [] for L in L_values}}
    
    for seed in range(n_seeds):
        print(f"  Seed {seed+1}/{n_seeds}...", end="\r", flush=True)
        y_seq, x_true, _, _ = generate_episode_with_impairments(
            model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
        
        # EKF
        x_hat = run_estimator('EKF', model, y_seq, x0, P0)
        results['EKF'].append(compute_rmse(x_hat, x_true))
        
        for L in L_values:
            x_gn = run_estimator(f'GN-{L}', model, y_seq, x0, P0)
            x_du = run_estimator(f'DU-tun-{L}', model, y_seq, x0, P0)
            results['GN'][L].append(compute_rmse(x_gn, x_true))
            results['DU'][L].append(compute_rmse(x_du, x_true))
    
    print(f"  完成 {n_seeds} seeds" + " "*20)
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    # EKF baseline
    ekf_mean = np.mean(results['EKF'])
    ax.axhline(ekf_mean, color=COLORS['EKF'], linestyle='--', label='EKF')
    
    # GN curve
    gn_means = [np.mean(results['GN'][L]) for L in L_values]
    gn_stds = [np.std(results['GN'][L]) / np.sqrt(n_seeds) for L in L_values]
    ax.errorbar(L_values, gn_means, yerr=gn_stds,
               marker='s', color=COLORS['GN-6'], linestyle='-', 
               label='GN', capsize=2, markerfacecolor='white')
    
    # DU curve
    du_means = [np.mean(results['DU'][L]) for L in L_values]
    du_stds = [np.std(results['DU'][L]) / np.sqrt(n_seeds) for L in L_values]
    ax.errorbar(L_values, du_means, yerr=du_stds,
               marker='D', color=COLORS['DU-tun-6'], linestyle='-',
               label='DU-tun', capsize=2)
    
    ax.set_xlabel('Iterations / Layers ($L$)')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_values)
    
    save_figure(fig, 'fig_rmse_L')
    
    csv_data = {'L': L_values,
                'GN_mean': gn_means, 'GN_std': gn_stds,
                'DU_mean': du_means, 'DU_std': du_stds}
    save_csv(csv_data, 'fig_rmse_L', list(csv_data.keys()))

# ============================================================================
# Fig 5: BER vs L
# ============================================================================

def fig_ber_vs_L():
    """BER vs Iterations/Layers"""
    print("\nFig: BER vs L")
    
    L_values = [2, 4, 6, 8]
    
    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig(p_slip=0.05, mode="discrete")
    n_seeds, n_frames = 30, 100
    
    results = {'EKF': [], 'GN': {L: [] for L in L_values}, 'DU': {L: [] for L in L_values}}
    
    for seed in range(n_seeds):
        print(f"  Seed {seed+1}/{n_seeds}...", end="\r", flush=True)
        y_seq, x_true, _, _ = generate_episode_with_impairments(
            model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
        
        x_hat = run_estimator('EKF', model, y_seq, x0, P0)
        ber, _ = quick_ber_evm(x_true, x_hat, 10, seed)
        results['EKF'].append(ber * 100)
        
        for L in L_values:
            x_gn = run_estimator(f'GN-{L}', model, y_seq, x0, P0)
            x_du = run_estimator(f'DU-tun-{L}', model, y_seq, x0, P0)
            ber_gn, _ = quick_ber_evm(x_true, x_gn, 10, seed)
            ber_du, _ = quick_ber_evm(x_true, x_du, 10, seed)
            results['GN'][L].append(ber_gn * 100)
            results['DU'][L].append(ber_du * 100)
    
    print(f"  完成 {n_seeds} seeds" + " "*20)
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    ekf_mean = np.mean(results['EKF'])
    ax.axhline(ekf_mean, color=COLORS['EKF'], linestyle='--', label='EKF')
    
    gn_means = [np.mean(results['GN'][L]) for L in L_values]
    ax.semilogy(L_values, gn_means, marker='s', color=COLORS['GN-6'], 
               linestyle='-', label='GN', markerfacecolor='white')
    
    du_means = [np.mean(results['DU'][L]) for L in L_values]
    ax.semilogy(L_values, du_means, marker='D', color=COLORS['DU-tun-6'],
               linestyle='-', label='DU-tun')
    
    ax.set_xlabel('Iterations / Layers ($L$)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_values)
    
    save_figure(fig, 'fig_ber_L')
    
    csv_data = {'L': L_values,
                'GN_mean': gn_means,
                'DU_mean': du_means,
                'EKF': [ekf_mean] * len(L_values)}
    save_csv(csv_data, 'fig_ber_L', list(csv_data.keys()))

# ============================================================================
# Fig 6: ADC Robustness
# ============================================================================

def fig_ber_vs_adc():
    """BER vs ADC bits"""
    print("\nFig: BER vs ADC")
    
    adc_bits = [2, 3, 4, 6, 8]
    methods = ['EKF', 'GN-6', 'DU-tun-6']
    
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig(p_slip=0.05, mode="discrete")
    n_seeds = 20
    
    results = {bits: {m: [] for m in methods} for bits in adc_bits}
    
    for bits in adc_bits:
        print(f"  ADC={bits}bit...", end=" ", flush=True)
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=bits)
        model = THzISACModel(cfg)
        
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            
            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                ber, _ = quick_ber_evm(x_true, x_hat, 10, seed)
                results[bits][method].append(ber * 100)
        print("done")
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    for method in methods:
        ber_means = [np.mean(results[b][method]) for b in adc_bits]
        ax.semilogy(adc_bits, ber_means,
                   marker=MARKERS[method], color=COLORS[method],
                   linestyle=LINESTYLES[method], label=method,
                   markerfacecolor='white' if method.startswith('GN') else COLORS[method])
    
    ax.set_xlabel('ADC Resolution (bits)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(adc_bits)
    
    save_figure(fig, 'fig_ber_adc')
    
    csv_data = {'ADC_bits': adc_bits}
    for m in methods:
        csv_data[f'{m}_mean'] = [np.mean(results[b][m]) for b in adc_bits]
    save_csv(csv_data, 'fig_ber_adc', list(csv_data.keys()))

# ============================================================================
# Fig 7: Observability (PCRB vs n_t)
# ============================================================================

def fig_pcrb_vs_nt():
    """PCRB vs time pilots"""
    print("\nFig: PCRB vs n_t")
    
    n_t_values = [1, 2, 4, 8, 16]
    x0 = np.array([1.0, 0.5, 0.0])
    
    pcrb_tau, pcrb_nu, pcrb_phi = [], [], []
    
    for n_t in n_t_values:
        cfg = THzISACConfig(n_f=8, n_t=n_t, snr_db=10, adc_bits=4)
        model = THzISACModel(cfg)
        
        pcrb_rec = PCRBRecursion(d=3)
        pcrb_seq, _ = pcrb_rec.run_sequence(model, [x0]*50)
        pcrb_tau.append(np.sqrt(np.mean([p[0] for p in pcrb_seq[10:]])))
        pcrb_nu.append(np.sqrt(np.mean([p[1] for p in pcrb_seq[10:]])))
        pcrb_phi.append(np.sqrt(np.mean([p[2] for p in pcrb_seq[10:]])))
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    ax.semilogy(n_t_values, pcrb_tau, 'o-', label=r'$\sqrt{\mathrm{PCRB}(\tau)}$', color='#1f77b4')
    ax.semilogy(n_t_values, pcrb_nu, 's-', label=r'$\sqrt{\mathrm{PCRB}(\nu)}$', color='#ff7f0e')
    ax.semilogy(n_t_values, pcrb_phi, '^-', label=r'$\sqrt{\mathrm{PCRB}(\phi)}$', color='#2ca02c')
    
    ax.set_xlabel('Time Pilots ($n_t$)')
    ax.set_ylabel('PCRB (normalized)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_t_values)
    
    save_figure(fig, 'fig_pcrb_nt')
    
    csv_data = {'n_t': n_t_values, 'PCRB_tau': pcrb_tau, 'PCRB_nu': pcrb_nu, 'PCRB_phi': pcrb_phi}
    save_csv(csv_data, 'fig_pcrb_nt', list(csv_data.keys()))

# ============================================================================
# Fig 8: Phase Tracking (Single Episode)
# ============================================================================

def fig_phase_tracking():
    """Phase tracking over single episode"""
    print("\nFig: Phase Tracking")
    
    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig(p_slip=0.05, mode="discrete")
    
    y_seq, x_true, slips, _ = generate_episode_with_impairments(
        model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=42)
    
    methods = ['EKF', 'GN-6', 'DU-tun-6']
    results = {m: run_estimator(m, model, y_seq, x0, P0) for m in methods}
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    frames = range(len(x_true))
    ax.plot(frames, [x[2] for x in x_true], 'k-', linewidth=1.5, label='True', alpha=0.7)
    
    for method in methods:
        ax.plot(frames, [x[2] for x in results[method]], 
               color=COLORS[method], linestyle=LINESTYLES[method],
               linewidth=1.0, label=method)
    
    for sf in slips:
        ax.axvline(sf, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Phase $\\phi$ (rad)')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    
    save_figure(fig, 'fig_phase_tracking')
    
    csv_data = {'frame': list(frames), 'true': [x[2] for x in x_true]}
    for m in methods:
        csv_data[m] = [x[2] for x in results[m]]
    save_csv(csv_data, 'fig_phase_tracking', list(csv_data.keys()))

# ============================================================================
# Fig 9: Phase Error (Single Episode)
# ============================================================================

def fig_phase_error():
    """Phase error over single episode"""
    print("\nFig: Phase Error")
    
    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig(p_slip=0.05, mode="discrete")
    
    y_seq, x_true, slips, _ = generate_episode_with_impairments(
        model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=42)
    
    methods = ['EKF', 'GN-6', 'DU-tun-6']
    results = {m: run_estimator(m, model, y_seq, x0, P0) for m in methods}
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    frames = range(len(x_true))
    
    for method in methods:
        errors = [abs(wrap_angle(results[method][k][2] - x_true[k][2])) for k in frames]
        ax.plot(frames, errors, color=COLORS[method], 
               linestyle=LINESTYLES[method], linewidth=1.0, label=method)
    
    for sf in slips:
        ax.axvline(sf, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Phase Error $|\\Delta\\phi|$ (rad)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, None])
    
    save_figure(fig, 'fig_phase_error')

# ============================================================================
# Fig 10: Improvement Bar Chart (L=6)
# ============================================================================

def fig_improvement_bar():
    """Improvement bar chart at L=6"""
    print("\nFig: Improvement Bar")
    
    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig(p_slip=0.05, mode="discrete")
    n_seeds = 30
    
    results = {'EKF': [], 'GN-6': [], 'DU-tun-6': []}
    
    for seed in range(n_seeds):
        y_seq, x_true, _, _ = generate_episode_with_impairments(
            model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
        
        for method in results.keys():
            x_hat = run_estimator(method, model, y_seq, x0, P0)
            results[method].append(compute_rmse(x_hat, x_true))
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    methods = ['EKF', 'GN-6', 'DU-tun-6']
    means = [np.mean(results[m]) for m in methods]
    stds = [np.std(results[m]) / np.sqrt(n_seeds) for m in methods]
    colors = [COLORS[m] for m in methods]
    
    bars = ax.bar(range(len(methods)), means, yerr=stds, 
                 color=colors, alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel('RMSE')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 标注改善
    imp = (means[1] - means[2]) / means[1] * 100
    ax.annotate(f'$-${imp:.0f}%', xy=(2, means[2]), xytext=(2, means[2]*0.7),
               fontsize=10, ha='center', color=COLORS['DU-tun-6'], fontweight='bold')
    
    save_figure(fig, 'fig_improvement_bar')
    
    csv_data = {'method': methods, 'mean': means, 'std': stds}
    save_csv(csv_data, 'fig_improvement_bar', list(csv_data.keys()))

# ============================================================================
# Fig 11: Sensitivity Analysis
# ============================================================================

def fig_sensitivity():
    """Step scale sensitivity"""
    print("\nFig: Sensitivity")
    
    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig(p_slip=0.05, mode="discrete")
    n_seeds = 15
    
    phi_scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    phi_results = []
    
    for phi_s in phi_scales:
        rmse_list = []
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            
            du_cfg = DUMAPConfig(n_layers=6)
            du_cfg.step_scale = np.array([1.0, 0.1, phi_s])
            du = DUMAP(du_cfg)
            x_hat, _ = du.forward_sequence(model, y_seq, x0, P0)
            rmse_list.append(compute_rmse(x_hat, x_true))
        phi_results.append((np.mean(rmse_list), np.std(rmse_list)/np.sqrt(n_seeds)))
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    means = [r[0] for r in phi_results]
    stds = [r[1] for r in phi_results]
    
    ax.errorbar(phi_scales, means, yerr=stds, marker='o', color=COLORS['DU-tun-6'],
               capsize=2, linestyle='-')
    ax.axvline(2.0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Phase Step Scale ($\\alpha_\\phi$)')
    ax.set_ylabel('RMSE')
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'fig_sensitivity')
    
    csv_data = {'phi_scale': phi_scales, 'mean': means, 'std': stds}
    save_csv(csv_data, 'fig_sensitivity', list(csv_data.keys()))

# ============================================================================
# Fig 12: CCDF
# ============================================================================

def fig_ccdf():
    """Phase error CCDF"""
    print("\nFig: CCDF")
    
    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig(p_slip=0.05, mode="discrete")
    n_seeds = 30
    
    methods = ['EKF', 'GN-6', 'DU-tun-6']
    all_errors = {m: [] for m in methods}
    
    for seed in range(n_seeds):
        y_seq, x_true, _, _ = generate_episode_with_impairments(
            model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
        
        for m in methods:
            x_hat = run_estimator(m, model, y_seq, x0, P0)
            errors = [abs(wrap_angle(xh[2] - xt[2])) for xh, xt in zip(x_hat, x_true)]
            all_errors[m].extend(errors)
    
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

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*60)
    print("IEEE 格式图像生成")
    print("="*60)
    
    setup_ieee_style()
    
    # 核心通信曲线 (类似 scan2 效果好的)
    fig_ber_vs_snr()      # BER vs SNR
    fig_rmse_vs_snr()     # RMSE vs SNR  
    fig_ber_vs_pslip()    # BER vs p_slip
    fig_rmse_vs_L()       # RMSE vs L (Pareto)
    fig_ber_vs_L()        # BER vs L
    fig_ber_vs_adc()      # BER vs ADC
    
    # 理论/可观测性
    fig_pcrb_vs_nt()      # PCRB vs n_t
    
    # 时序/Episode
    fig_phase_tracking()  # Phase tracking
    fig_phase_error()     # Phase error
    
    # 其他
    fig_improvement_bar() # Bar chart
    fig_sensitivity()     # Sensitivity
    fig_ccdf()            # CCDF
    
    print("\n" + "="*60)
    print("完成! 12 张 IEEE 格式图像")
    print("="*60)
    print(f"\n输出目录: {OUTPUT_DIR}/")
    print("  - PNG (300 dpi)")
    print("  - PDF (vector)")
    print("  - CSV (data)")

if __name__ == "__main__":
    main()

# ============================================================================
# 补充图: 导师建议的高价值扫描
# ============================================================================

def fig_slip_2d_heatmap():
    """p_slip × amplitude 2D heatmap - Domain-specificity 证据"""
    print("\nFig: Slip 2D Heatmap")
    
    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    
    p_slip_vals = [0.01, 0.02, 0.05, 0.10]
    amp_vals = [np.pi/2, np.pi, 3*np.pi/2]  # 避免2π (wrap后等于0)
    n_seeds = 15
    
    improvement = np.zeros((len(amp_vals), len(p_slip_vals)))
    gn_rmse_mat = np.zeros((len(amp_vals), len(p_slip_vals)))
    du_rmse_mat = np.zeros((len(amp_vals), len(p_slip_vals)))
    
    for i, amp in enumerate(amp_vals):
        for j, p_slip in enumerate(p_slip_vals):
            print(f"  amp={amp/np.pi:.1f}π, p={p_slip}...", end="\r", flush=True)
            
            slip_cfg = SlipConfig(
                p_slip=p_slip, mode="discrete",
                values=(-amp, amp), probs=(0.5, 0.5)
            )
            
            gn_rmse, du_rmse = [], []
            for seed in range(n_seeds):
                y_seq, x_true, _, _ = generate_episode_with_impairments(
                    model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
                
                x_gn = run_estimator('GN-6', model, y_seq, x0, P0)
                x_du = run_estimator('DU-tun-6', model, y_seq, x0, P0)
                
                gn_rmse.append(compute_rmse(x_gn, x_true))
                du_rmse.append(compute_rmse(x_du, x_true))
            
            gn_mean, du_mean = np.mean(gn_rmse), np.mean(du_rmse)
            gn_rmse_mat[i, j] = gn_mean
            du_rmse_mat[i, j] = du_mean
            improvement[i, j] = (gn_mean - du_mean) / gn_mean * 100
    
    print(" " * 50)
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    im = ax.imshow(improvement, aspect='auto', cmap='RdYlGn', vmin=-20, vmax=30)
    ax.set_xticks(range(len(p_slip_vals)))
    ax.set_xticklabels([f'{p}' for p in p_slip_vals])
    ax.set_yticks(range(len(amp_vals)))
    ax.set_yticklabels([f'{a/np.pi:.1f}π' for a in amp_vals])
    ax.set_xlabel('Slip Probability ($p_{slip}$)')
    ax.set_ylabel('Slip Amplitude')
    
    # 数值标注
    for i in range(len(amp_vals)):
        for j in range(len(p_slip_vals)):
            val = improvement[i, j]
            color = 'white' if abs(val) > 15 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', 
                   fontsize=IEEE_FONTSIZE-1, color=color, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (%)')
    
    save_figure(fig, 'fig_slip_2d_heatmap')
    
    # CSV
    csv_data = {'amplitude': [f'{a/np.pi:.1f}pi' for a in amp_vals]}
    for j, p in enumerate(p_slip_vals):
        csv_data[f'p={p}'] = [improvement[i, j] for i in range(len(amp_vals))]
    save_csv(csv_data, 'fig_slip_2d_heatmap', list(csv_data.keys()))


def fig_frame_duration():
    """Frame duration sweep - Observability 理论证据"""
    print("\nFig: Frame Duration")
    
    T_factors = [0.5, 1.0, 2.0, 4.0]  # 相对于默认 100μs
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    n_seeds = 15
    
    pcrb_nu = []
    gn_nu_rmse = []
    du_nu_rmse = []
    
    for T_factor in T_factors:
        print(f"  T_factor={T_factor}...", end=" ", flush=True)
        
        T = 100e-6 * T_factor
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4, frame_duration_s=T)
        model = THzISACModel(cfg)
        
        # PCRB
        pcrb_rec = PCRBRecursion(d=3)
        pcrb_seq, _ = pcrb_rec.run_sequence(model, [x0]*50)
        pcrb_nu.append(np.sqrt(np.mean([p[1] for p in pcrb_seq[10:]])))
        
        # 实测 RMSE
        gn_list, du_list = [], []
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, 100, x0, slip_cfg=None, pn_cfg=None, seed=seed)
            
            x_gn = run_estimator('GN-6', model, y_seq, x0, P0)
            x_du = run_estimator('DU-tun-6', model, y_seq, x0, P0)
            
            # Doppler RMSE only
            gn_list.append(np.sqrt(np.mean([(xh[1] - xt[1])**2 for xh, xt in zip(x_gn, x_true)])))
            du_list.append(np.sqrt(np.mean([(xh[1] - xt[1])**2 for xh, xt in zip(x_du, x_true)])))
        
        gn_nu_rmse.append(np.mean(gn_list))
        du_nu_rmse.append(np.mean(du_list))
        print("done")
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    T_us = [100 * f for f in T_factors]
    ax.semilogy(T_us, pcrb_nu, 'k--', lw=1.5, label=r'$\sqrt{\mathrm{PCRB}(\nu)}$')
    ax.semilogy(T_us, gn_nu_rmse, 's-', color=COLORS['GN-6'], 
               label='GN-6', markerfacecolor='white')
    ax.semilogy(T_us, du_nu_rmse, 'D-', color=COLORS['DU-tun-6'], label='DU-tun-6')
    
    ax.set_xlabel('Frame Duration $T$ (μs)')
    ax.set_ylabel('Doppler RMSE')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'fig_frame_duration')
    
    csv_data = {'T_us': T_us, 'PCRB': pcrb_nu, 'GN': gn_nu_rmse, 'DU': du_nu_rmse}
    save_csv(csv_data, 'fig_frame_duration', list(csv_data.keys()))


def fig_recovery_time():
    """Recovery time comparison - Slip恢复速度"""
    print("\nFig: Recovery Time")
    
    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    
    # 在固定帧注入slip
    n_frames = 50
    slip_frame = 20
    methods = ['EKF', 'GN-6', 'DU-tun-6']
    n_seeds = 20
    
    # 收集每个方法在slip后的误差曲线
    error_curves = {m: [] for m in methods}
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        
        # 生成带固定slip的序列
        y_seq, x_true = [], []
        x_curr = x0.copy()
        
        for k in range(n_frames):
            if k == slip_frame:
                x_curr = x_curr.copy()
                x_curr[2] += np.pi  # 注入 π slip

            x_curr = model.transition(x_curr)
            y = model.observe(x_curr, k)
            y_seq.append(y)
            x_true.append(x_curr.copy())
        
        for m in methods:
            x_hat = run_estimator(m, model, y_seq, x0, P0)
            errors = [abs(wrap_angle(xh[2] - xt[2])) for xh, xt in zip(x_hat, x_true)]
            error_curves[m].append(errors)
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    frames = range(n_frames)
    for method in methods:
        mean_err = np.mean(error_curves[method], axis=0)
        ax.plot(frames, mean_err, color=COLORS[method], 
               linestyle=LINESTYLES[method], label=method, linewidth=1.2)
    
    ax.axvline(slip_frame, color='gray', linestyle=':', alpha=0.7, linewidth=0.8)
    ax.annotate('slip', xy=(slip_frame, 0.1), fontsize=IEEE_FONTSIZE-2, color='gray')
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Phase Error (rad)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, n_frames])
    ax.set_ylim([0, None])
    
    save_figure(fig, 'fig_recovery_time')
    
    # 计算恢复时间 (误差降到0.3以下的帧数)
    recovery = {}
    for m in methods:
        mean_err = np.mean(error_curves[m], axis=0)
        post_slip = mean_err[slip_frame:]
        try:
            rec_frames = np.where(post_slip < 0.3)[0][0]
        except:
            rec_frames = len(post_slip)
        recovery[m] = rec_frames
    
    print(f"  Recovery times: {recovery}")


def fig_ber_vs_L_multiSNR():
    """BER vs L at multiple SNR - 展示不同SNR下的Pareto"""
    print("\nFig: BER vs L (multi-SNR)")
    
    L_values = [2, 4, 6, 8]
    snr_list = [5, 10, 15]
    
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig(p_slip=0.05, mode="discrete")
    n_seeds = 20
    
    results = {snr: {'GN': {L: [] for L in L_values}, 
                     'DU': {L: [] for L in L_values}} for snr in snr_list}
    
    for snr in snr_list:
        print(f"  SNR={snr}dB...", end=" ", flush=True)
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=snr, adc_bits=4)
        model = THzISACModel(cfg)
        
        for seed in range(n_seeds):
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, 100, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            
            for L in L_values:
                x_gn = run_estimator(f'GN-{L}', model, y_seq, x0, P0)
                x_du = run_estimator(f'DU-tun-{L}', model, y_seq, x0, P0)
                
                ber_gn, _ = quick_ber_evm(x_true, x_gn, snr, seed)
                ber_du, _ = quick_ber_evm(x_true, x_du, snr, seed)
                
                results[snr]['GN'][L].append(ber_gn * 100)
                results[snr]['DU'][L].append(ber_du * 100)
        print("done")
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    line_styles = {5: '--', 10: '-', 15: ':'}
    
    for snr in snr_list:
        gn_means = [np.mean(results[snr]['GN'][L]) for L in L_values]
        du_means = [np.mean(results[snr]['DU'][L]) for L in L_values]
        
        ax.semilogy(L_values, gn_means, 's', color=COLORS['GN-6'], 
                   linestyle=line_styles[snr], markerfacecolor='white',
                   label=f'GN ({snr}dB)')
        ax.semilogy(L_values, du_means, 'D', color=COLORS['DU-tun-6'],
                   linestyle=line_styles[snr],
                   label=f'DU ({snr}dB)')
    
    ax.set_xlabel('Iterations / Layers ($L$)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right', ncol=2, fontsize=IEEE_FONTSIZE-2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_values)
    
    save_figure(fig, 'fig_ber_L_multiSNR')


def fig_pilot_geometry_heatmap():
    """Pilot geometry (n_f × n_t) heatmap - 理论图"""
    print("\nFig: Pilot Geometry Heatmap")
    
    n_f_vals = [4, 8, 16, 32]
    n_t_vals = [1, 2, 4, 8]
    x0 = np.array([1.0, 0.5, 0.0])
    
    pcrb_matrix = np.zeros((len(n_t_vals), len(n_f_vals)))
    
    for i, n_t in enumerate(n_t_vals):
        for j, n_f in enumerate(n_f_vals):
            cfg = THzISACConfig(n_f=n_f, n_t=n_t, snr_db=10, adc_bits=4)
            model = THzISACModel(cfg)
            
            pcrb_rec = PCRBRecursion(d=3)
            pcrb_seq, _ = pcrb_rec.run_sequence(model, [x0]*50)
            pcrb_matrix[i, j] = np.sqrt(np.mean([p[1] for p in pcrb_seq[10:]]))  # Doppler PCRB
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    pcrb_log = np.log10(np.maximum(pcrb_matrix, 1e-10))
    
    im = ax.imshow(pcrb_log, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(n_f_vals)))
    ax.set_xticklabels([str(n) for n in n_f_vals])
    ax.set_yticks(range(len(n_t_vals)))
    ax.set_yticklabels([str(n) for n in n_t_vals])
    ax.set_xlabel('Frequency Pilots ($n_f$)')
    ax.set_ylabel('Time Pilots ($n_t$)')
    
    # 数值标注
    for i in range(len(n_t_vals)):
        for j in range(len(n_f_vals)):
            val = pcrb_matrix[i, j]
            color = 'white' if pcrb_log[i, j] < -1.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=IEEE_FONTSIZE-2, color=color)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\log_{10}\sqrt{\mathrm{PCRB}(\nu)}$')
    
    save_figure(fig, 'fig_pilot_heatmap')
    
    csv_data = {'n_t': n_t_vals}
    for j, n_f in enumerate(n_f_vals):
        csv_data[f'n_f={n_f}'] = [pcrb_matrix[i, j] for i in range(len(n_t_vals))]
    save_csv(csv_data, 'fig_pilot_heatmap', list(csv_data.keys()))


# 运行补充图
if __name__ == "__main__":
    print("\n" + "="*60)
    print("补充图像生成")
    print("="*60)
    
    setup_ieee_style()
    
    fig_slip_2d_heatmap()    # Domain-specificity 热力图
    fig_frame_duration()      # 帧长 vs Doppler可观测性
    fig_recovery_time()       # 恢复时间对比
    fig_ber_vs_L_multiSNR()   # 多SNR的BER vs L
    fig_pilot_geometry_heatmap()  # Pilot几何热力图
    
    print("\n补充完成! 新增 5 张图")
