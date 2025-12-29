# THz-ISAC DU-MAP 代码包 (IEEE Format)

## 📁 目录结构

```
sba_du_clean/
├── src/                           # 核心源代码 (9个模块)
├── scripts/
│   └── generate_ieee_figures.py   # IEEE 格式图像生成
├── outputs/                       # 图像输出 (PNG+PDF+CSV)
└── README.md
```

## 🚀 运行

```bash
python scripts/generate_ieee_figures.py  # ~3分钟
```

## 📊 图像清单 (12张)

### 核心通信曲线
| 文件名 | 内容 | X轴 | Y轴 |
|--------|------|-----|-----|
| `fig_ber_snr` | BER vs SNR | SNR (dB) | BER (%) |
| `fig_rmse_snr` | RMSE vs SNR | SNR (dB) | RMSE |
| `fig_ber_pslip` | BER vs slip概率 | p_slip | BER (%) |
| `fig_rmse_L` | RMSE vs 计算量 | L | RMSE |
| `fig_ber_L` | BER vs 计算量 | L | BER (%) |
| `fig_ber_adc` | BER vs ADC分辨率 | bits | BER (%) |

### 理论/可观测性
| 文件名 | 内容 |
|--------|------|
| `fig_pcrb_nt` | PCRB vs 时间导频数 |

### 时序/过程
| 文件名 | 内容 |
|--------|------|
| `fig_phase_tracking` | 单episode相位追踪 |
| `fig_phase_error` | 单episode相位误差 |

### 其他
| 文件名 | 内容 |
|--------|------|
| `fig_improvement_bar` | L=6 对比柱状图 |
| `fig_sensitivity` | step_scale敏感性 |
| `fig_ccdf` | 相位误差CCDF |

## 📐 IEEE 格式规范

- 单栏宽度: 3.5 inch
- 字体: 9pt
- 分辨率: 300 dpi
- 无标题 (caption在论文中写)
- 统一颜色: EKF(红), GN(蓝), DU-tun(绿)
- 统一marker: EKF(×), GN(□), DU-tun(◇)

## 🔑 关键数值

| 方法 | RMSE | BER% |
|------|------|------|
| EKF | 0.750 | 10.3 |
| GN-6 | 0.384 | 3.9 |
| **DU-tun-6** | **0.333** | **3.3** |

**改善**: -13% RMSE, -15% BER
