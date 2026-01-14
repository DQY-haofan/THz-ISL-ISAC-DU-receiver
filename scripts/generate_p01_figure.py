#!/usr/bin/env python3
"""
P0-1 补充实验: 分量级 RMSE vs PCRB (随SNR)
==========================================

目的: 把 Section II/III 的 PCRB 推导变成 Section V 的硬证据
- 展示各方法离理论极限的距离
- 验证 DU-MAP 是否"近似有效"

输出:
- fig_rmse_pcrb_tau.png/pdf  (时延分量)
- fig_rmse_pcrb_nu.png/pdf   (多普勒分量)
- fig_rmse_pcrb_phi.png/pdf  (相位分量)
- data_p01_component_rmse_pcrb.csv (原始数据)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Dict, List

from src.physics.thz_isac_model import THzISACConfig, THzISACModel, wrap_angle
from src.inference.gn_solver import GNSolverConfig, GaussNewtonMAP
from src.unfolding.du_map import DUMAP, DUMAPConfig
from src.baselines.wrapped_ekf import create_ekf, create_iekf
from src.bcrlb.pcrb import PCRBRecursion
from src.sim.slip import generate_episode_with_impairments, SlipConfig

# ============================================================================
# IEEE 格式配置 (与 generate_ieee_figures.py 完全一致)
# ============================================================================

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IEEE_WIDTH = 3.5
IEEE_HEIGHT = 2.6
IEEE_FONTSIZE = 9
IEEE_DPI = 300

# 颜色方案 (4种方法层次分明)
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


# ============================================================================
# 估计器接口
# ============================================================================

def run_estimator(method, model, y_seq, x0, P0):
    """统一估计器接口"""
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
        cfg.step_scale = np.array([1.0, 0.1, 1.5])  # 调优参数
        est = DUMAP(cfg)
        return est.forward_sequence(model, y_seq, x0, P0)[0]
    raise ValueError(f"Unknown method: {method}")


def compute_component_mse(x_hat_seq, x_true_seq):
    """
    计算分量级MSE
    
    Returns:
        mse_tau, mse_nu, mse_phi: 各分量的MSE
    """
    n = len(x_hat_seq)
    mse = np.zeros(3)
    
    for xh, xt in zip(x_hat_seq, x_true_seq):
        err = xh - xt
        err[2] = wrap_angle(err[2])  # phase wrap
        mse += err**2
    
    mse /= n
    return mse[0], mse[1], mse[2]


def compute_pcrb_sequence(model, x_true_seq):
    """
    计算PCRB序列
    
    Returns:
        pcrb_mean: 平均PCRB [3] (τ, ν, φ)
    """
    pcrb_rec = PCRBRecursion(d=3)
    pcrb_seq, _ = pcrb_rec.run_sequence(model, x_true_seq)
    
    # 取后半段平均（稳态）
    n_frames = len(x_true_seq)
    start_idx = n_frames // 2
    pcrb_mean = np.mean(pcrb_seq[start_idx:], axis=0)
    
    return pcrb_mean


# ============================================================================
# 主实验
# ============================================================================

def run_p01_experiment():
    """
    P0-1 主实验: 分量级 RMSE vs PCRB vs SNR
    """
    print("=" * 60)
    print("P0-1: Component-wise RMSE vs PCRB vs SNR")
    print("=" * 60)
    
    # 实验参数
    snr_list = [0, 5, 10, 15, 20]
    methods = ['EKF', 'IEKF-4', 'GN-6', 'DU-tun-6']
    
    x0 = np.array([1.0, 0.5, 0.0])
    P0 = np.eye(3) * 0.1
    slip_cfg = SlipConfig.killer_pi(p_slip=0.05)
    n_seeds = 30
    n_frames = 100
    
    # 存储结果
    results = {snr: {m: {'tau': [], 'nu': [], 'phi': []} for m in methods} for snr in snr_list}
    pcrb_results = {snr: {'tau': [], 'nu': [], 'phi': []} for snr in snr_list}
    
    for snr in snr_list:
        print(f"\nSNR = {snr} dB")
        cfg = THzISACConfig(n_f=8, n_t=4, snr_db=snr, adc_bits=4)
        model = THzISACModel(cfg)
        
        for seed in range(n_seeds):
            print(f"  Seed {seed+1}/{n_seeds}...", end="\r", flush=True)
            
            # 生成数据
            y_seq, x_true, _, _ = generate_episode_with_impairments(
                model, n_frames, x0, slip_cfg=slip_cfg, pn_cfg=None, seed=seed)
            
            # 计算PCRB
            pcrb = compute_pcrb_sequence(model, x_true)
            pcrb_results[snr]['tau'].append(pcrb[0])
            pcrb_results[snr]['nu'].append(pcrb[1])
            pcrb_results[snr]['phi'].append(pcrb[2])
            
            # 运行各方法
            for method in methods:
                x_hat = run_estimator(method, model, y_seq, x0, P0)
                mse_tau, mse_nu, mse_phi = compute_component_mse(x_hat, x_true)
                results[snr][method]['tau'].append(mse_tau)
                results[snr][method]['nu'].append(mse_nu)
                results[snr][method]['phi'].append(mse_phi)
        
        print(f"  Completed {n_seeds} seeds" + " " * 20)
    
    return snr_list, methods, results, pcrb_results


# ============================================================================
# 绘图函数 (三个独立图)
# ============================================================================

def fig_rmse_pcrb_tau(snr_list, methods, results, pcrb_results):
    """RMSE vs PCRB - 时延分量 τ"""
    print("\nFig: RMSE vs PCRB (τ)")
    setup_ieee_style()
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    # PCRB (虚线)
    pcrb_sqrt = [np.sqrt(np.mean(pcrb_results[snr]['tau'])) for snr in snr_list]
    ax.semilogy(snr_list, pcrb_sqrt, 'k--', linewidth=1.5, label=r'$\sqrt{\mathrm{PCRB}}$')
    
    # 各方法RMSE
    for method in methods:
        rmse = [np.sqrt(np.mean(results[snr][method]['tau'])) for snr in snr_list]
        ax.semilogy(snr_list, rmse, 
                   marker=MARKERS[method], 
                   color=COLORS[method],
                   linestyle=LINESTYLES[method],
                   label=method,
                   markerfacecolor='white' if 'GN' in method else None)
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(r'RMSE ($\tau$)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-1, 21])
    
    save_figure(fig, 'fig_rmse_pcrb_tau')


def fig_rmse_pcrb_nu(snr_list, methods, results, pcrb_results):
    """RMSE vs PCRB - 多普勒分量 ν"""
    print("\nFig: RMSE vs PCRB (ν)")
    setup_ieee_style()
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    # PCRB (虚线)
    pcrb_sqrt = [np.sqrt(np.mean(pcrb_results[snr]['nu'])) for snr in snr_list]
    ax.semilogy(snr_list, pcrb_sqrt, 'k--', linewidth=1.5, label=r'$\sqrt{\mathrm{PCRB}}$')
    
    # 各方法RMSE
    for method in methods:
        rmse = [np.sqrt(np.mean(results[snr][method]['nu'])) for snr in snr_list]
        ax.semilogy(snr_list, rmse, 
                   marker=MARKERS[method], 
                   color=COLORS[method],
                   linestyle=LINESTYLES[method],
                   label=method,
                   markerfacecolor='white' if 'GN' in method else None)
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(r'RMSE ($\nu$)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-1, 21])
    
    save_figure(fig, 'fig_rmse_pcrb_nu')


def fig_rmse_pcrb_phi(snr_list, methods, results, pcrb_results):
    """RMSE vs PCRB - 相位分量 φ"""
    print("\nFig: RMSE vs PCRB (φ)")
    setup_ieee_style()
    
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    
    # PCRB (虚线)
    pcrb_sqrt = [np.sqrt(np.mean(pcrb_results[snr]['phi'])) for snr in snr_list]
    ax.semilogy(snr_list, pcrb_sqrt, 'k--', linewidth=1.5, label=r'$\sqrt{\mathrm{PCRB}}$')
    
    # 各方法RMSE
    for method in methods:
        rmse = [np.sqrt(np.mean(results[snr][method]['phi'])) for snr in snr_list]
        ax.semilogy(snr_list, rmse, 
                   marker=MARKERS[method], 
                   color=COLORS[method],
                   linestyle=LINESTYLES[method],
                   label=method,
                   markerfacecolor='white' if 'GN' in method else None)
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(r'RMSE ($\phi$) [rad]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-1, 21])
    
    save_figure(fig, 'fig_rmse_pcrb_phi')


# ============================================================================
# CSV 保存
# ============================================================================

def save_p01_csv(snr_list, methods, results, pcrb_results):
    """保存P0-1数据到CSV"""
    
    with open(f'{OUTPUT_DIR}/data_p01_component_rmse_pcrb.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['SNR_dB', 
                  'sqrt_PCRB_tau', 'sqrt_PCRB_nu', 'sqrt_PCRB_phi']
        for method in methods:
            header.extend([f'RMSE_{method}_tau', f'RMSE_{method}_nu', f'RMSE_{method}_phi'])
        header.extend(['efficiency_DU_tau', 'efficiency_DU_nu', 'efficiency_DU_phi'])
        writer.writerow(header)
        
        # Data
        for snr in snr_list:
            row = [snr]
            
            # PCRB
            pcrb_tau = np.sqrt(np.mean(pcrb_results[snr]['tau']))
            pcrb_nu = np.sqrt(np.mean(pcrb_results[snr]['nu']))
            pcrb_phi = np.sqrt(np.mean(pcrb_results[snr]['phi']))
            row.extend([f'{pcrb_tau:.6f}', f'{pcrb_nu:.6f}', f'{pcrb_phi:.6f}'])
            
            # RMSE per method
            for method in methods:
                rmse_tau = np.sqrt(np.mean(results[snr][method]['tau']))
                rmse_nu = np.sqrt(np.mean(results[snr][method]['nu']))
                rmse_phi = np.sqrt(np.mean(results[snr][method]['phi']))
                row.extend([f'{rmse_tau:.6f}', f'{rmse_nu:.6f}', f'{rmse_phi:.6f}'])
            
            # Efficiency (MSE/PCRB) for DU
            mse_du_tau = np.mean(results[snr]['DU-tun-6']['tau'])
            mse_du_nu = np.mean(results[snr]['DU-tun-6']['nu'])
            mse_du_phi = np.mean(results[snr]['DU-tun-6']['phi'])
            eff_tau = mse_du_tau / np.mean(pcrb_results[snr]['tau'])
            eff_nu = mse_du_nu / np.mean(pcrb_results[snr]['nu'])
            eff_phi = mse_du_phi / np.mean(pcrb_results[snr]['phi'])
            row.extend([f'{eff_tau:.3f}', f'{eff_nu:.3f}', f'{eff_phi:.3f}'])
            
            writer.writerow(row)
    
    print(f"  ✓ data_p01_component_rmse_pcrb.csv")


# ============================================================================
# 汇总打印
# ============================================================================

def print_efficiency_summary(snr_list, methods, results, pcrb_results):
    """打印效率汇总表"""
    
    print("\n" + "=" * 70)
    print("EFFICIENCY SUMMARY: MSE / PCRB (closer to 1.0 = more efficient)")
    print("=" * 70)
    
    print(f"\n{'SNR':>6} | {'Method':>10} | {'η_τ':>8} | {'η_ν':>8} | {'η_φ':>8} | {'η_avg':>8}")
    print("-" * 70)
    
    for snr in snr_list:
        pcrb_tau = np.mean(pcrb_results[snr]['tau'])
        pcrb_nu = np.mean(pcrb_results[snr]['nu'])
        pcrb_phi = np.mean(pcrb_results[snr]['phi'])
        
        for method in methods:
            mse_tau = np.mean(results[snr][method]['tau'])
            mse_nu = np.mean(results[snr][method]['nu'])
            mse_phi = np.mean(results[snr][method]['phi'])
            
            eff_tau = mse_tau / pcrb_tau
            eff_nu = mse_nu / pcrb_nu
            eff_phi = mse_phi / pcrb_phi
            eff_avg = (eff_tau + eff_nu + eff_phi) / 3
            
            print(f"{snr:>6} | {method:>10} | {eff_tau:>8.2f} | {eff_nu:>8.2f} | {eff_phi:>8.2f} | {eff_avg:>8.2f}")
        print("-" * 70)
    
    # DU vs GN improvement
    print("\n" + "=" * 70)
    print("DU-tun-6 vs GN-6 IMPROVEMENT (RMSE reduction %)")
    print("=" * 70)
    print(f"\n{'SNR':>6} | {'Δτ%':>8} | {'Δν%':>8} | {'Δφ%':>8} | {'Δavg%':>8}")
    print("-" * 50)
    
    for snr in snr_list:
        du = results[snr]['DU-tun-6']
        gn = results[snr]['GN-6']
        
        imp_tau = (np.sqrt(np.mean(gn['tau'])) - np.sqrt(np.mean(du['tau']))) / np.sqrt(np.mean(gn['tau'])) * 100
        imp_nu = (np.sqrt(np.mean(gn['nu'])) - np.sqrt(np.mean(du['nu']))) / np.sqrt(np.mean(gn['nu'])) * 100
        imp_phi = (np.sqrt(np.mean(gn['phi'])) - np.sqrt(np.mean(du['phi']))) / np.sqrt(np.mean(gn['phi'])) * 100
        imp_avg = (imp_tau + imp_nu + imp_phi) / 3
        
        print(f"{snr:>6} | {imp_tau:>+8.1f} | {imp_nu:>+8.1f} | {imp_phi:>+8.1f} | {imp_avg:>+8.1f}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    # 运行实验
    snr_list, methods, results, pcrb_results = run_p01_experiment()
    
    # 绘制三个独立图
    fig_rmse_pcrb_tau(snr_list, methods, results, pcrb_results)
    fig_rmse_pcrb_nu(snr_list, methods, results, pcrb_results)
    fig_rmse_pcrb_phi(snr_list, methods, results, pcrb_results)
    
    # 保存CSV
    save_p01_csv(snr_list, methods, results, pcrb_results)
    
    # 打印汇总
    print_efficiency_summary(snr_list, methods, results, pcrb_results)
    
    print("\n" + "=" * 60)
    print("P0-1 实验完成!")
    print("输出文件:")
    print("  - fig_rmse_pcrb_tau.png/pdf")
    print("  - fig_rmse_pcrb_nu.png/pdf")
    print("  - fig_rmse_pcrb_phi.png/pdf")
    print("  - data_p01_component_rmse_pcrb.csv")
    print("=" * 60)


if __name__ == '__main__':
    main()
