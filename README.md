# THz-ISAC DU-MAP 代码包 (修复版)

## 🔧 修复记录

详见 `FIXES_APPLIED.md`，主要修复：

| 问题 | 修复 |
|------|------|
| P0-1 AQNM一致性 | observe() 返回等效观测 y/alpha |
| P0-2 UKF sigma points | 改为按列取向量 |
| P0-3 BER/EVM信道模型 | 使用一致的物理参数 |
| P0-4 slip帧索引 | _frame_idx初始化为-1 |

---

## 📁 目录结构

```
sba_du_clean/
├── src/
│   ├── physics/thz_isac_model.py   # THz信道模型
│   ├── inference/gn_solver.py      # GN求解器
│   ├── unfolding/du_map.py         # DU-MAP (核心)
│   ├── baselines/                  # EKF/UKF
│   ├── bcrlb/pcrb.py              # PCRB理论界
│   ├── sim/slip.py                # Slip仿真
│   └── metrics/system_metrics.py   # BER/EVM
├── scripts/
│   └── generate_ieee_figures.py    # 图像生成
├── FIXES_APPLIED.md               # 修复记录
└── README.md
```

---

## 🚀 快速开始

```bash
# 解压
tar -xzf sba_du_clean_fixed.tar.gz
cd sba_du_clean

# 安装依赖
pip install numpy scipy matplotlib

# 生成图像
python scripts/generate_ieee_figures.py
```

---

## ⚙️ 核心参数

```python
# DU-tun 关键参数
du_cfg.step_scale = np.array([1.0, 0.1, 2.0])
#                             τ    ν    φ
#                            标准 保守 激进
```

---

## 📊 验证测试

```python
import sys
sys.path.insert(0, '.')
import numpy as np

from src.physics.thz_isac_model import THzISACConfig, THzISACModel

# 测试 AQNM 一致性
cfg = THzISACConfig(n_f=8, n_t=4, snr_db=10, adc_bits=4)
model = THzISACModel(cfg)
x0 = np.array([1.0, 0.5, 0.0])

y = model.observe(x0, 0)
h = model.h(x0, 0)
print(f"|y - h|: {np.linalg.norm(y - h):.4f}")  # 应该是噪声水平
print(f"sigma_eff: {np.sqrt(model.sigma_eff_sq):.4f}")
```

---

## ⚠️ 设计说明与已知限制

### P1-2: 相位噪声与Q_cov的关系

当启用相位噪声 (PN) 时：
- `PhaseNoiseProcess` 会添加 Wiener 相位噪声到真值
- `Q_cov` 的 φ 分量 (`q_std_norm[2]`) 也包含相位过程噪声

**建议配置**：
```python
# 如果 PN 开启，Q_cov 的 phi 分量应设为 0 或很小
if pn_cfg is not None:
    cfg.q_std_norm = (0.02, 0.01, 0.0)  # phi 噪声由 PN 提供
```

### P1-3: GN vs DU 公平性

当前设计中：
- GN 可选 `use_preconditioner`、阻尼策略
- DU 使用固定层数、logspace damping

**公平比较原则**：
- 关闭 GN 的额外 trick，或在 DU 中使用相同策略
- 论文中明确说明配置

### P1-4: AQNM vs Hard Quantize

当前使用 AQNM (Additive Quantization Noise Model) 连续近似，而非离散量化。

**限制**：
- AQNM 是 Bussgang 定理的近似，在低比特 (2-3 bit) 下可能有偏差
- 顶刊审稿人可能质疑此近似

**未来工作**：
- 实现 hard quantizer 模块
- 对比 AQNM 仿真 vs hard quantize 仿真
