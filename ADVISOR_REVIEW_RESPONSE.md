# 导师审阅意见回复

## 已修复问题

### ✅ P0-1: AQNM 观测定义不一致
**状态**: 代码已正确实现

**分析**: 检查 `thz_isac_model.py` 第367-376行：
```python
# AQNM quantization
# y_q = alpha*y + q, then return equivalent observation ỹ = y_q/alpha
# This makes sigma_eff_sq = awgn_var + quant_var/alpha^2 consistent
if self.cfg.apply_quantization and self.cfg.adc_bits < 12:
    q = np.sqrt(self.quant_var / 2) * (...)
    y_q = self.alpha * y + q
    # Return equivalent observation (divide by alpha)
    y = y_q / self.alpha
```

代码已经实现**方案A**（等效去增益观测）：
- `observe()` 返回 `y_q / alpha`
- `sigma_eff_sq = awgn_var + quant_var / alpha^2` (第176行)
- 与PCRB假设一致

### ✅ P0-2: UKF Sigma Points 索引错误
**状态**: 代码已正确实现

**分析**: 检查 `ukf.py` 第81-87行：
```python
# NOTE: sqrt_P[:, i] is the i-th column (correct for Cholesky lower triangular)
sigma_points = np.zeros((2*n + 1, n))
sigma_points[0] = x
for i in range(n):
    sigma_points[i + 1] = x + sqrt_P[:, i]      # 按列取 ✓
    sigma_points[n + i + 1] = x - sqrt_P[:, i]  # 按列取 ✓
```

已使用 `sqrt_P[:, i]` 按列取向量，符合标准UKF。

### ✅ P0-3: system_metrics 信道模型不一致
**状态**: 已修复

**修改内容**:
1. `SystemMetricsConfig` 添加 `from_model_cfg()` 方法，从 `THzISACConfig` 读取参数
2. `apply_channel_distortion()` 和 `compensate_channel()` 现在接受 `cfg` 参数
3. 移除所有硬编码常数 (`T_frame=100e-6` 等)

### ✅ P0-4: slip 帧索引偏移
**状态**: 代码已正确实现

**分析**: 检查 `slip.py` 第70-72行和89行：
```python
# Initialize to -1 so first sample() call gives frame_idx=0
# This aligns with generate_episode_with_impairments() which uses k=0,1,2,...
self._frame_idx = -1

def sample(self):
    self._frame_idx += 1  # 先自增，第一次调用后 frame_idx=0
    ...
```

帧索引与episode的k=0对齐。

---

## 待讨论问题 (P1级别)

### ❓ P1-1: NIS gating 自由度设置
**当前状态**: `use_gating=False` (默认关闭)

**问题**: 若启用gating，阈值按state_dim=3计算，但innovation维度是2*m

**建议处理**:
1. 若论文不涉及"鲁棒门控"，保持关闭即可
2. 若需要，应按 `chi2.ppf(p, df=2*m)` 动态计算阈值

**请导师决定**: 是否需要修复？或在论文中不提及此功能？

### ❓ P1-2: 相位噪声双计入风险
**当前状态**: PN默认关闭 (`pn_cfg=None`)

**问题**: 若PN和Q_cov的phi噪声同时存在，可能双计入相位扩散

**建议处理**:
- 明确原则: "PN on → Q_cov的phi噪声=0"
- 或在config中添加互斥检查

**请导师决定**: 是否需要添加此检查？当前实验是否使用PN？

### ❓ P1-3: GN vs DU 公平性
**当前状态**: GN有可选的preconditioner和阻尼策略

**问题**: 
- GN的`use_preconditioner`分支可能影响对比公平性
- DU使用固定的per-layer damping

**建议处理**:
- 对比实验中关闭GN的额外tricks（当前默认关闭）
- 或将同样tricks加入DU

**请导师决定**: 当前配置是否已足够公平？

### ❓ P1-4: Hard Quantization vs AQNM
**当前状态**: 评测和训练都使用AQNM连续输出

**问题**: DR计划建议评测用hard quantize，代理用AQNM

**建议处理**:
1. 当前AQNM仿真对于初步验证是合理的
2. 下一轮可补充hard quantizer模块，对比差异边界

**请导师决定**: 是否需要在本轮添加hard quantizer？或留待后续？

---

## 验证结果

### 数值验证 (SNR=10dB, p_slip=0.05, amplitude=π)
| 方法 | RMSE | BER% |
|------|------|------|
| EKF | 0.750 | ~10% |
| GN-6 | 0.384 | ~3.9% |
| **DU-tun-6** | **0.333** | **~3.3%** |

**改善**: -13% RMSE, -15% BER

### 图像验证
- 共生成16张图像
- 2D热力图显示domain-specificity：amplitude=π时改善+8%~+32%
- Recovery time显示DU峰值误差最低

---

## 文件变更清单

| 文件 | 修改内容 |
|------|----------|
| `src/metrics/system_metrics.py` | P0-3修复: 从cfg读取参数 |
| `scripts/generate_ieee_figures.py` | 修复model.f→transition, model.sample→observe |

其他文件经检查无需修改（P0-1/2/4已正确实现）。
