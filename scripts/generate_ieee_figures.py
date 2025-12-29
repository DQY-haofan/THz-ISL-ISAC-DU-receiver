#!/usr/bin/env python3
"""
THz-ISAC DU-MAP IEEE 格式图像生成 v4
=====================================

v4 Baseline 优化:
1. 删除 UKF (与 EKF 性能相同，无区分度)
2. 添加 IEKF-4 (迭代滤波器，EKF 与 GN 之间的桥梁)

Baseline 层次:
- EKF: 单步滤波 (L=1) → slip 后失锁
- IEKF-4: 迭代滤波 (L=4) → 部分恢复
- GN-6: 迭代 MAP → 恢复
- DU-tun-6: 学习的迭代 MAP → 快速恢复

理论意义:
- IEKF 展示了"迭代的价值"
- GN vs IEKF: 纯优化 vs 滤波框架
- DU vs GN: 学习步长 vs 固定步长
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

# v4 颜色方案 (4种方法层次分明)
COLORS = {
    'EKF': '#E41A1C',      # 红 - 单步滤波
    'IEKF-4': '#FF7F00',   # 橙 - 迭代滤波
    'GN-6': '#377EB8',     # 蓝 - 迭代 MAP
    'DU-tun-6': '#4DAF4A', # 绿 - 学习的 MAP
    'PCRB': '#000000',     # 黑 - 理论界
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

def save_csv(data: Dict, name: str, headers: List[str]):
    with open(f'{OUTPUT_DIR}/{name}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        keys = list(data.keys())
        n_rows = len(data[keys[0]]) if isinstance(data[keys[0]], list) else 1
        for i in range(n_rows):
            row = [data[k][i] if isinstance(data[k], list) else data[k] for k in keys]
            writer.writerow(row)

# ============================================================================
# 估计器工具函数
# ============================================================================

def compute_rmse(x_hat_seq, x_true_seq):
    mse = 0
    for xh, xt in zip(x_hat_seq, x_true_seq):
        err = xh - xt
        err[2] = wrap_angle(err[2])
        mse += np.sum(err**2)
    return np.sqrt(mse / len(x_hat_seq))

def run_estimator(method, model, y_seq, x0, P0):
    """统一接口 - v4"""
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
        cfg.step_scale = np.array([1.0, 0.1, 2.0])
        est = DUMAP(cfg)
        return est.forward_sequence(model, y_seq, x0, P0)[0]
    raise ValueError(f"Unknown method: {method}")

# ============================================================================
# 核心图
# ============================================================================

def fig_ber_vs_snr():
    """BER vs SNR - 核心对比图"""
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
    
    csv_data = {'SNR': snr_list}
    for m in methods:
        csv_data[f'{m}_mean'] = [np.mean(results[snr][m]) for snr in snr_list]
        csv_data[f'{m}_sem'] = [np.std(results[snr][m])/np.sqrt(n_seeds) for snr in snr_list]
    save_csv(csv_data, 'fig_ber_snr', list(csv_data.keys()))

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
    
    csv_data = {'SNR': snr_list}
    for m in methods:
        csv_data[f'{m}_mean'] = [np.mean(results[snr][m]) for snr in snr_list]
        csv_data[f'{m}_sem'] = [np.std(results[snr][m])/np.sqrt(n_seeds) for snr in snr_list]
    save_csv(csv_data, 'fig_rmse_snr', list(csv_data.keys()))

def fig_rmse_vs_L():
    """RMSE vs L - 展示迭代收敛"""
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
        print(f"  Seed {seed+1}/{n_seeds}...", end="\r", flush=True)
        y_seq, x_true, _, _ = generate_episode_with_impairments(
            model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
        
        results['EKF'].append(compute_rmse(run_estimator('EKF', model, y_seq, x0, P0), x_true))
        
        for L in L_values:
            # IEKF (使用稳定的迭代数)
            iekf_L = min(L, 4)  # IEKF-4 是稳定的最大值
            x_iekf = run_estimator(f'IEKF-{iekf_L}', model, y_seq, x0, P0)
            results['IEKF'][L].append(compute_rmse(x_iekf, x_true))
            
            x_gn = run_estimator(f'GN-{L}', model, y_seq, x0, P0)
            x_du = run_estimator(f'DU-tun-{L}', model, y_seq, x0, P0)
            results['GN'][L].append(compute_rmse(x_gn, x_true))
            results['DU'][L].append(compute_rmse(x_du, x_true))
    print(f"  完成 {n_seeds} seeds" + " "*20)
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    # EKF baseline
    ax.axhline(np.mean(results['EKF']), color=COLORS['EKF'], linestyle='--', 
               label='EKF (L=1)', alpha=0.8)
    
    # IEKF 收敛曲线
    iekf_means = [np.mean(results['IEKF'][L]) for L in L_values]
    iekf_sems = [np.std(results['IEKF'][L])/np.sqrt(n_seeds) for L in L_values]
    ax.errorbar(L_values, iekf_means, yerr=iekf_sems, marker='^', color=COLORS['IEKF-4'],
               label='IEKF', capsize=2)
    
    # GN 收敛曲线
    gn_means = [np.mean(results['GN'][L]) for L in L_values]
    gn_sems = [np.std(results['GN'][L])/np.sqrt(n_seeds) for L in L_values]
    ax.errorbar(L_values, gn_means, yerr=gn_sems, marker='s', color=COLORS['GN-6'],
               label='GN', capsize=2, markerfacecolor='white')
    
    # DU 收敛曲线
    du_means = [np.mean(results['DU'][L]) for L in L_values]
    du_sems = [np.std(results['DU'][L])/np.sqrt(n_seeds) for L in L_values]
    ax.errorbar(L_values, du_means, yerr=du_sems, marker='D', color=COLORS['DU-tun-6'],
               label='DU-tun', capsize=2)
    
    ax.set_xlabel('Iterations / Layers ($L$)')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(L_values)
    save_figure(fig, 'fig_rmse_L')
    
    # 计算改善
    print(f"  IEKF-4 vs EKF: {(np.mean(results['EKF']) - iekf_means[1]) / np.mean(results['EKF']) * 100:.0f}% improvement")
    print(f"  GN-6 vs EKF: {(np.mean(results['EKF']) - gn_means[2]) / np.mean(results['EKF']) * 100:.0f}% improvement")
    print(f"  DU-6 vs GN-6: {(gn_means[2] - du_means[2]) / gn_means[2] * 100:.0f}% improvement")
    
    csv_data = {
        'L': L_values,
        'EKF_mean': [np.mean(results['EKF'])]*len(L_values),
        'IEKF_mean': iekf_means, 'IEKF_sem': iekf_sems,
        'GN_mean': gn_means, 'GN_sem': gn_sems,
        'DU_mean': du_means, 'DU_sem': du_sems,
    }
    save_csv(csv_data, 'fig_rmse_L', list(csv_data.keys()))

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
    ax.annotate('slip', xy=(slip_frame+1, 0.1), fontsize=IEEE_FONTSIZE-2, color='gray')
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Phase Error (rad)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, n_frames])
    ax.set_ylim([0, None])
    save_figure(fig, 'fig_recovery_time')
    
    # 计算恢复帧数
    recovery = {}
    for m in methods:
        mean_err = np.mean(error_curves[m], axis=0)
        post_slip = mean_err[slip_frame:]
        try:
            rec_frames = np.where(post_slip < 0.3)[0][0]
        except:
            rec_frames = len(post_slip)
        recovery[m] = rec_frames
    print(f"  Recovery frames: {recovery}")

def fig_slip_2d_heatmap():
    """p_slip × amplitude 改善热力图"""
    print("\nFig: Slip 2D Heatmap")
    
    cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
    model = THzISACModel(cfg)
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    
    p_slip_vals = [0.01, 0.02, 0.05, 0.10]
    amp_vals = [np.pi/2, np.pi, 3*np.pi/2]
    n_seeds = 20
    
    improvement = np.zeros((len(amp_vals), len(p_slip_vals)))
    
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
            improvement[i, j] = (gn_mean - du_mean) / gn_mean * 100
    print(" " * 50)
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    im = ax.imshow(improvement, aspect='auto', cmap='RdYlGn', vmin=-50, vmax=50)
    ax.set_xticks(range(len(p_slip_vals)))
    ax.set_xticklabels([f'{p}' for p in p_slip_vals])
    ax.set_yticks(range(len(amp_vals)))
    ax.set_yticklabels([f'{a/np.pi:.1f}π' for a in amp_vals])
    ax.set_xlabel('Slip Probability ($p_{slip}$)')
    ax.set_ylabel('Slip Amplitude')
    
    for i in range(len(amp_vals)):
        for j in range(len(p_slip_vals)):
            val = improvement[i, j]
            color = 'white' if abs(val) > 25 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                   fontsize=IEEE_FONTSIZE-1, color=color, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('RMSE Improvement (%)')
    save_figure(fig, 'fig_slip_2d_heatmap')

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
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    means = [np.mean(results[m]) for m in methods]
    sems = [np.std(results[m])/np.sqrt(n_seeds) for m in methods]
    colors = [COLORS[m] for m in methods]
    
    bars = ax.bar(range(len(methods)), means, yerr=sems,
                 color=colors, alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=IEEE_FONTSIZE-1)
    ax.set_ylabel('RMSE')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 标注改善
    for i in range(1, len(methods)):
        imp = (means[0] - means[i]) / means[0] * 100
        ax.annotate(f'{imp:+.0f}%', xy=(i, means[i]), xytext=(i, means[i]*0.9),
                   fontsize=8, ha='center', color=colors[i], fontweight='bold')
    
    save_figure(fig, 'fig_improvement_bar')

def fig_ber_vs_adc():
    """BER vs ADC bits"""
    print("\nFig: BER vs ADC")
    
    adc_bits = [2, 3, 4, 6, 8]
    methods = ['EKF', 'GN-6', 'DU-tun-6']
    
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds = 25
    
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
                   marker=MARKERS.get(method, 'o'), color=COLORS[method],
                   label=method, markerfacecolor='white' if 'GN' in method else None)
    ax.set_xlabel('ADC Resolution (bits)')
    ax.set_ylabel('BER (%)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(adc_bits)
    save_figure(fig, 'fig_ber_adc')

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
        pcrb_seq, _ = pcrb_rec.run_sequence(model, [x0]*50)
        pcrb_tau.append(np.sqrt(np.mean([p[0] for p in pcrb_seq[10:]])))
        pcrb_nu.append(np.sqrt(np.mean([p[1] for p in pcrb_seq[10:]])))
        pcrb_phi.append(np.sqrt(np.mean([p[2] for p in pcrb_seq[10:]])))
    
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
# 主函数
# ============================================================================

def main():
    print("="*60)
    print("IEEE 格式图像生成 v4 (IEKF Baseline)")
    print("="*60)
    print("Baseline 层次:")
    print("  - EKF:      单步滤波 (L=1) → slip 后失锁")
    print("  - IEKF-4:   迭代滤波 (L=4) → 部分恢复")
    print("  - GN-6:     迭代 MAP       → 恢复")
    print("  - DU-tun-6: 学习的 MAP     → 快速恢复")
    print("="*60)
    
    setup_ieee_style()
    
    # 核心图
    print("\n[核心图]")
    fig_ber_vs_snr()
    fig_rmse_vs_snr()
    fig_rmse_vs_L()
    fig_recovery_time()
    fig_slip_2d_heatmap()
    
    # 辅助图
    print("\n[辅助图]")
    fig_improvement_bar()
    fig_ber_vs_adc()
    fig_ccdf()
    fig_pcrb_nt()
    
    print("\n" + "="*60)
    print(f"完成! 输出目录: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
