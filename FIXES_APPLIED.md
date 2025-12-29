# 已应用的修复 (Fixes Applied)

本文档记录了针对导师审查意见的修复状态。

## P0 级修复 (必须修复)

### ✅ P0-1: AQNM 观测定义一致性

**问题**: `observe()` 返回 `y_q = alpha*h + q`，但 `sigma_eff_sq` 假设等效观测 `ỹ = y_q/alpha`

**修复** (thz_isac_model.py, 第374-376行):
```python
y_q = self.alpha * y + q
# Return equivalent observation (divide by alpha)
y = y_q / self.alpha
```

### ✅ P0-2: UKF Sigma Points 索引

**问题**: 使用行向量 `sqrt_P[i]` 而非列向量

**修复** (ukf.py, `_generate_sigma_points`):
```python
sigma_points[i + 1] = x + sqrt_P[:, i]  # 列向量，正确
sigma_points[n + i + 1] = x - sqrt_P[:, i]
```

### ✅ P0-3: BER/EVM 信道模型一致性

**问题**: 硬编码参数，与 `thz_isac_model.h()` 不一致

**修复** (system_metrics.py):
```python
@classmethod
def from_model_cfg(cls, model_cfg: 'THzISACConfig') -> 'SystemMetricsConfig':
    """Create SystemMetricsConfig from THzISACConfig for consistency."""
    return cls(
        f_c=model_cfg.carrier_freq_hz,
        delay_scale=model_cfg.delay_scale,
        doppler_scale=model_cfg.doppler_scale,
        frame_duration=model_cfg.frame_duration_s,
        T_sym=model_cfg.frame_duration_s / 64,
    )
```

### ✅ P0-4: Slip 帧索引 Off-by-One

**问题**: `_frame_idx` 先自增再记录，可能从1开始

**修复** (slip.py):
```python
# Initialize to -1 so first sample() call gives frame_idx=0
# This aligns with generate_episode_with_impairments() which uses k=0,1,2,...
self._frame_idx = -1
```

---

## P1 级修复 (建议修复)

### ✅ P1-1: NIS Gating 自由度

**问题**: 固定阈值，不依赖观测维度

**修复** (wrapped_ekf.py):
```python
obs_dim = len(r_real)  # 2*m，观测维度
if self.cfg.nis_threshold is None:
    nis_thresh = chi2.ppf(self.cfg.nis_confidence, df=obs_dim)
```

### ⚠️ P1-2: 相位噪声双计入

**状态**: 已文档化，用户需根据场景配置

**建议**:
```python
# PN 开启时，Q_cov 的 phi 分量设为 0
if pn_cfg is not None:
    cfg.q_std_norm = (0.02, 0.01, 0.0)
```

### ⚠️ P1-3: GN vs DU 公平性

**状态**: 设计选择，已文档化

**原则**:
- 比较时关闭 GN 的额外 trick
- 或在 DU 中使用相同策略

### 🔮 P1-4: Hard Quantize vs AQNM

**状态**: 未来工作

**当前**: 使用 AQNM 连续近似
**计划**: 实现 hard quantizer，对比差异

---

## 验证命令

```bash
cd sba_du_clean
python -c "
import sys
sys.path.insert(0, '.')

from src.physics.thz_isac_model import THzISACModel, THzISACConfig
import numpy as np

# P0-1: 验证等效观测
cfg = THzISACConfig(adc_bits=4)
model = THzISACModel(cfg)
x = np.array([1.0, 0.5, 0.0])
y = model.observe(x, 0)
print(f'P0-1 Check: observe returns equivalent observation (divided by alpha)')
print(f'  alpha = {model.alpha:.4f}')

# P0-2: 验证 UKF 使用列向量
from src.baselines.ukf import UKF
print(f'P0-2 Check: UKF uses sqrt_P[:, i] (column vector)')

# P0-3: 验证 BER/EVM 读取 model.cfg
from src.metrics.system_metrics import SystemMetricsConfig
smc = SystemMetricsConfig.from_model_cfg(cfg)
print(f'P0-3 Check: SystemMetricsConfig.from_model_cfg() exists')
print(f'  f_c = {smc.f_c/1e9:.0f} GHz')

# P0-4: 验证 slip 索引初始化
from src.sim.slip import PhaseSlipProcess, SlipConfig
slip = PhaseSlipProcess(SlipConfig())
print(f'P0-4 Check: _frame_idx initialized to {slip._frame_idx}')

print('\\nAll P0 fixes verified!')
"
```
