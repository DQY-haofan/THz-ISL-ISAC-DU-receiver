# THz-ISAC DU-MAP: Deep Unfolding for Phase Slip Recovery

## 概述

本项目实现了基于 Deep Unfolding 的 THz-ISAC 信道估计方法，针对相位滑移 (Phase Slip) 场景优化。

## 核心贡献

1. **DU-tun**: 学习的迭代 MAP 估计器，在 killer-π slip 场景下比 GN 提升 22% RMSE
2. **IEKF**: 迭代扩展卡尔曼滤波器实现，填补 EKF 与 GN 之间的空白
3. **完整实验框架**: 支持 BER/RMSE/恢复时间等多维度评估

## 方法层次

```
EKF (L=1)    RMSE=1.09  → 单步滤波，slip 后失锁
    ↓ +27%
IEKF-4       RMSE=0.80  → 迭代滤波，部分恢复
    ↓ +31%
GN-6         RMSE=0.55  → 迭代 MAP，恢复
    ↓ +22%
DU-tun-6     RMSE=0.42  → 学习的 MAP，快速恢复
```

## 快速开始

```bash
# 安装依赖
pip install numpy scipy matplotlib

# 生成 IEEE 格式图像
python scripts/generate_ieee_figures.py

# 输出在 outputs_v4/ 目录
```

## 目录结构

```
sba_du_clean/
├── src/
│   ├── physics/          # 物理模型 (THzISACModel)
│   ├── inference/        # GN 求解器
│   ├── unfolding/        # DU-MAP 实现
│   ├── baselines/        # EKF, IEKF, UKF
│   ├── bcrlb/            # PCRB 理论界
│   ├── sim/              # 仿真 (slip, phase noise)
│   └── metrics/          # BER/EVM 计算
├── scripts/
│   └── generate_ieee_figures.py  # 图像生成脚本
├── outputs/              # 生成的图像和数据
└── tests/                # 单元测试
```

## 核心参数

```python
# DU-tun 步长配置 (核心贡献)
du_cfg.step_scale = np.array([1.0, 0.1, 2.0])
#                             τ    ν    φ
# τ: 标准步长
# ν: 保守步长 (避免 Doppler 过拟合)  
# φ: 激进步长 (加速相位恢复) ← 关键
```

## 实验配置

- **Slip 场景**: `SlipConfig.killer_pi(p_slip=0.05)` - ±π 相位跳变
- **SNR 范围**: 0-20 dB
- **ADC 分辨率**: 2-8 bits
- **迭代次数**: L = 2, 4, 6, 8

## 输出图像

| 图像 | 描述 |
|------|------|
| `fig_ber_snr.png` | BER vs SNR 曲线 |
| `fig_rmse_snr.png` | RMSE vs SNR 曲线 |
| `fig_rmse_L.png` | RMSE vs 迭代次数 |
| `fig_recovery_time.png` | 相位恢复动态 |
| `fig_slip_2d_heatmap.png` | 幅度×概率 改善热力图 |
| `fig_improvement_bar.png` | RMSE 改善柱状图 |
| `fig_ccdf.png` | 相位误差 CCDF |
| `fig_ber_adc.png` | BER vs ADC 分辨率 |
| `fig_pcrb_nt.png` | PCRB 理论界 |

## 依赖

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

## 许可

MIT License
