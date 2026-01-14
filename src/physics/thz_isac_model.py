#!/usr/bin/env python3
"""
THz-ISAC Physical Model V3 - 完整物理建模版
============================================

基于专家建议的完整实现：

P0（必须做）：
- 连续相位噪声 Wiener 模型（linewidth 参数化）
- Discrete slip 保留

P1（建议做）：
- 真正的 Beam squint：ULA 阵列的频率-角度耦合增益
- 不是简单的 g(f) proxy

P2（可选）：
- Pointing jitter：帧级增益抖动（作为 mismatch 注入）

保留：
- Doppler squint：ν_eff = ν(1 + f/fc)
- AQNM 量化模型
- Two-timescale state-space

所有新效应默认关闭，确保向后兼容。

参考文献：
- Wiener PN: Petrovic et al., "Effects of Phase Noise on OFDM Systems"
- Beam squint: 标准 ULA 阵列响应
- ISL 无大气吸收: ITU "Terahertz Wireless Communications in Space"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


# =========================================================================
# Phase wrapping utilities
# =========================================================================

def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def wrap_angle_array(angles: np.ndarray) -> np.ndarray:
    """Wrap angles to [-π, π]."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def circular_error(phi_hat: float, phi_true: float) -> float:
    """Compute circular (wrapped) phase error."""
    return wrap_angle(phi_hat - phi_true)


# =========================================================================
# Configuration - V3 完整版
# =========================================================================

@dataclass
class THzISACConfig:
    """
    THz-ISAC V3 配置 - 包含完整物理效应

    新增效应（相比V2）：
    - 连续PN Wiener模型（linewidth参数化）
    - 物理Beam squint（ULA阵列模型）
    - Pointing jitter（帧级增益抖动）
    """

    # =========================================================================
    # 基础参数（与V2相同）
    # =========================================================================
    n_f: int = 8
    n_t: int = 4
    bandwidth_hz: float = 100e6
    carrier_freq_hz: float = 300e9
    frame_duration_s: float = 100e-6

    snr_db: float = 10.0
    adc_bits: int = 4
    apply_quantization: bool = True

    delay_scale: float = 1e-9
    doppler_scale: float = 1e3
    phase_scale: float = 1.0

    nu_ar: float = 0.99

    # =========================================================================
    # P0: 连续相位噪声 Wiener 模型（新增）
    # =========================================================================
    enable_continuous_pn: bool = False
    """启用连续相位噪声 Wiener 模型"""

    pn_linewidth_hz: float = 100.0
    """
    振荡器3dB线宽 (Hz)

    典型值：
    - 高质量PLL @ 300GHz: 10-100 Hz
    - 中等质量: 100-1000 Hz
    - 自由运行VCO: 1-10 kHz

    Wiener过程方差: σ²_φ = 2π * linewidth * T_frame
    """

    # 旧的固定q_std（当disable continuous PN时使用）
    q_std_norm_tau: float = 0.02
    q_std_norm_nu: float = 0.01
    q_std_norm_phi: float = 0.05  # 仅当 enable_continuous_pn=False 时使用

    # =========================================================================
    # P0: Doppler squint（保留自V2）
    # =========================================================================
    enable_doppler_squint: bool = False
    """Doppler squint: ν_eff(f) = ν * (1 + f/f_c)"""

    # =========================================================================
    # P1: 物理 Beam squint - ULA 阵列模型（新增）
    # =========================================================================
    enable_beam_squint: bool = False
    """启用物理 Beam squint（ULA阵列频率-角度耦合）"""

    beam_squint_n_ant: int = 16
    """阵列天线数量"""

    beam_squint_d_over_lambda: float = 0.5
    """天线间距（波长倍数），0.5 = 半波长间距"""

    beam_squint_theta0_deg: float = 0.0
    """波束指向角度（度），0 = 正前方"""

    # =========================================================================
    # P2: Pointing jitter（新增）
    # =========================================================================
    enable_pointing_jitter: bool = False
    """启用指向抖动（帧级增益波动）"""

    pointing_jitter_std_deg: float = 0.1
    """指向抖动标准差（度）"""

    pointing_jitter_ar: float = 0.95
    """指向抖动的AR(1)系数（慢变特性）"""

    # =========================================================================
    # AQNM（与V2相同）
    # =========================================================================
    alpha_aqnm: float = field(default=1.0, init=False)
    quant_var: float = field(default=0.0, init=False)

    # =========================================================================
    # V2遗留（向后兼容，但建议用新的）
    # =========================================================================
    enable_beam_squint_proxy: bool = False
    """旧版 beam squint proxy（仅向后兼容）"""
    beam_squint_strength: float = 0.0


def _aqnm_params(bits: int) -> Tuple[float, float]:
    """AQNM 参数表"""
    table = {
        1: (0.6366, 0.3634),
        2: (0.8825, 0.1175),
        3: (0.9625, 0.0375),
        4: (0.9900, 0.0100),
        5: (0.9975, 0.0025),
        6: (0.9994, 0.0006),
        7: (0.9998, 0.0002),
        8: (0.9999, 0.0001),
    }
    if bits in table:
        return table[bits]
    if bits >= 8:
        return (1.0, 0.0)
    return (0.5, 0.5)


# =========================================================================
# Main Model Class - V3
# =========================================================================

class THzISACModel:
    """
    THz-ISAC 观测模型 V3 - 完整物理建模

    观测模型：
        h(f_i, t_j) = g_bs(f_i) * g_pt * exp(-j2πf_i*τ + j2πν_eff*t_j + jφ)

    其中：
        - g_bs(f_i): Beam squint 增益（ULA阵列物理模型）
        - g_pt: Pointing jitter 增益
        - ν_eff: 可选 Doppler squint

    状态转移：
        φ_{k+1} = φ_k + 2πν_k*T + w_φ,k

        w_φ,k ~ N(0, σ²_φ) 其中 σ²_φ = 2π * linewidth * T_frame（Wiener模型）
    """

    def __init__(self, cfg: THzISACConfig):
        self.cfg = cfg
        self._init_pilots()
        self._init_noise()
        self._init_beam_squint()
        self._init_pointing_jitter()

    def _init_pilots(self) -> None:
        """初始化2D导频网格"""
        cfg = self.cfg

        self.f_grid = np.linspace(
            -cfg.bandwidth_hz / 2,
            cfg.bandwidth_hz / 2,
            cfg.n_f,
            endpoint=False
        )

        self.t_grid = np.linspace(
            0.0,
            cfg.frame_duration_s,
            cfg.n_t,
            endpoint=False
        )

        self.f_vec = np.repeat(self.f_grid, cfg.n_t)
        self.t_vec = np.tile(self.t_grid, cfg.n_f)
        self.m = cfg.n_f * cfg.n_t

    def _init_noise(self) -> None:
        """初始化噪声参数"""
        cfg = self.cfg

        snr_lin = 10.0 ** (cfg.snr_db / 10.0)
        self.awgn_var = 1.0 / snr_lin

        alpha, sigma_q_sq = _aqnm_params(cfg.adc_bits)
        cfg.alpha_aqnm = alpha
        cfg.quant_var = sigma_q_sq
        self.alpha = alpha
        self.quant_var = sigma_q_sq

        self.sigma_eff_sq = self.awgn_var + (self.quant_var / max(alpha ** 2, 1e-12))

    # =========================================================================
    # P1: 物理 Beam squint - ULA 阵列模型
    # =========================================================================

    def _init_beam_squint(self) -> None:
        """
        初始化物理 Beam squint 增益

        ULA + analog phase shifter（在 f_c 设计）：

        阵列响应: a_n(f,θ) = exp(j2π * f/c * d * (n-1) * sinθ)
        波束权重: w_n = exp(-j2π * f_c/c * d * (n-1) * sinθ_0)

        Beamforming增益:
        g_bs(f) = (1/N) * |Σ exp(j2π * d/c * (n-1) * (f*sinθ - f_c*sinθ_0))|

        当 θ = θ_0（指向正确，仅频率失配）:
        g_bs(f) = |sin(Nπξ) / (N*sin(πξ))| 其中 ξ = d*(f-f_c)*sinθ_0/c
        """
        cfg = self.cfg

        if not cfg.enable_beam_squint:
            self._beam_squint_gain = np.ones(self.m, dtype=float)
            return

        N = cfg.beam_squint_n_ant
        d = cfg.beam_squint_d_over_lambda * (3e8 / cfg.carrier_freq_hz)  # 物理间距
        theta0 = np.radians(cfg.beam_squint_theta0_deg)
        c = 3e8

        # 计算每个频率点的beam squint增益
        g_bs = np.zeros(self.m, dtype=float)

        for i, f in enumerate(self.f_vec):
            # 实际频率 = f_c + f（f是相对基带偏移）
            f_abs = cfg.carrier_freq_hz + f

            # 相位差参数
            # ξ = d * (f_abs - f_c) * sin(θ_0) / c = d * f * sin(θ_0) / c
            xi = d * f * np.sin(theta0) / c

            # Dirichlet kernel (array factor)
            if abs(xi) < 1e-12:
                g_bs[i] = 1.0  # 无频率偏移时增益为1
            else:
                # |sin(Nπξ) / (N*sin(πξ))|
                numerator = np.sin(N * np.pi * xi)
                denominator = N * np.sin(np.pi * xi)
                if abs(denominator) < 1e-12:
                    g_bs[i] = 1.0
                else:
                    g_bs[i] = abs(numerator / denominator)

        self._beam_squint_gain = g_bs

        # 打印beam squint范围（调试用）
        # print(f"Beam squint gain range: [{g_bs.min():.4f}, {g_bs.max():.4f}]")

    # =========================================================================
    # P2: Pointing jitter 状态
    # =========================================================================

    def _init_pointing_jitter(self) -> None:
        """初始化 pointing jitter 状态"""
        self._pointing_jitter_state = 0.0  # 当前角度偏移（rad）
        self._pointing_gain = 1.0  # 当前帧增益

    def _update_pointing_jitter(self, rng: np.random.Generator) -> float:
        """
        更新 pointing jitter 并返回帧增益

        模型：
        - δθ_k = ρ * δθ_{k-1} + w_θ,k  （AR(1)过程）
        - g_pt = exp(-0.5 * (δθ/θ_3dB)²) （高斯波束近似）

        其中 θ_3dB ≈ λ/(N*d) 是阵列半功率波束宽度
        """
        cfg = self.cfg

        if not cfg.enable_pointing_jitter:
            return 1.0

        # AR(1) 更新
        sigma_theta = np.radians(cfg.pointing_jitter_std_deg)
        innovation_std = sigma_theta * np.sqrt(1 - cfg.pointing_jitter_ar ** 2)
        w_theta = rng.normal(0, innovation_std)
        self._pointing_jitter_state = cfg.pointing_jitter_ar * self._pointing_jitter_state + w_theta

        # 计算3dB波束宽度（ULA近似）
        if cfg.enable_beam_squint:
            N = cfg.beam_squint_n_ant
            d_over_lambda = cfg.beam_squint_d_over_lambda
            theta_3dB = 0.886 / (N * d_over_lambda)  # 近似公式
        else:
            theta_3dB = np.radians(1.0)  # 默认1度

        # 高斯波束增益
        delta_theta = self._pointing_jitter_state
        self._pointing_gain = np.exp(-0.5 * (delta_theta / theta_3dB) ** 2)

        return self._pointing_gain

    # =========================================================================
    # 归一化辅助函数
    # =========================================================================

    def denorm(self, x_norm: np.ndarray) -> Tuple[float, float, float]:
        """归一化状态 → 物理单位"""
        cfg = self.cfg
        tau = float(x_norm[0] * cfg.delay_scale)
        nu = float(x_norm[1] * cfg.doppler_scale)
        phi = float(x_norm[2] * cfg.phase_scale)
        return tau, nu, phi

    def norm(self, tau: float, nu: float, phi: float) -> np.ndarray:
        """物理单位 → 归一化状态"""
        cfg = self.cfg
        return np.array([
            tau / cfg.delay_scale,
            nu / cfg.doppler_scale,
            phi / cfg.phase_scale,
        ], dtype=float)

    def wrap_phase_norm(self, x_norm: np.ndarray) -> np.ndarray:
        """相位wrap到[-π,π]"""
        cfg = self.cfg
        phi = x_norm[2] * cfg.phase_scale
        phi = wrap_angle(phi)
        x_norm = x_norm.copy()
        x_norm[2] = phi / cfg.phase_scale
        return x_norm

    # =========================================================================
    # 状态转移（含连续PN Wiener模型）
    # =========================================================================

    def transition(self, x_norm: np.ndarray) -> np.ndarray:
        """
        状态转移（确定性部分）

        τ_{k+1} = τ_k
        ν_{k+1} = ρ * ν_k
        φ_{k+1} = φ_k + 2π * ν * T_frame
        """
        cfg = self.cfg
        tau_n, nu_n, phi_n = x_norm

        tau_n_next = tau_n
        nu_n_next = cfg.nu_ar * nu_n

        incr = 2.0 * np.pi * (nu_n * cfg.doppler_scale) * cfg.frame_duration_s
        phi_n_next = phi_n + incr / cfg.phase_scale

        x_next = np.array([tau_n_next, nu_n_next, phi_n_next], dtype=float)
        return self.wrap_phase_norm(x_next)

    def F_jacobian(self) -> np.ndarray:
        """状态转移 Jacobian"""
        cfg = self.cfg
        F = np.eye(3, dtype=float)
        F[1, 1] = cfg.nu_ar
        F[2, 1] = (2.0 * np.pi * cfg.doppler_scale * cfg.frame_duration_s) / cfg.phase_scale
        return F

    def Q_cov(self) -> np.ndarray:
        """
        过程噪声协方差

        P0 关键改动：当 enable_continuous_pn=True 时，
        φ 的过程噪声由 linewidth 物理参数化：

        σ²_φ = 2π * Δf_3dB * T_frame  (Wiener过程)

        这是 Lorentzian 线宽的标准离散化模型。
        """
        cfg = self.cfg

        q_tau = cfg.q_std_norm_tau
        q_nu = cfg.q_std_norm_nu

        if cfg.enable_continuous_pn:
            # Wiener 模型：方差 = 2π * linewidth * T_frame
            sigma_phi_sq = 2 * np.pi * cfg.pn_linewidth_hz * cfg.frame_duration_s
            q_phi = np.sqrt(sigma_phi_sq) / cfg.phase_scale
        else:
            # 旧的固定参数
            q_phi = cfg.q_std_norm_phi

        q = np.array([q_tau, q_nu, q_phi], dtype=float)
        return np.diag(q * q)

    # =========================================================================
    # 观测模型（含所有物理效应）
    # =========================================================================

    def h(self, x_norm: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        观测函数 h(x)

        完整模型：
        h(f_i, t_j) = g_bs(f_i) * exp(-j2πf_i*τ + j2πν_eff*t_j + jφ)

        其中：
        - g_bs(f_i): Beam squint 增益（ULA物理模型）
        - ν_eff = ν*(1+f/f_c) 如果启用 Doppler squint

        注意：Pointing jitter 增益在 observe() 中应用，不在 h() 中
        （因为它是 model mismatch，估计器不应该知道）
        """
        cfg = self.cfg
        tau, nu, phi = self.denorm(x_norm)

        t_intra = self.t_vec

        # Doppler squint
        if cfg.enable_doppler_squint:
            nu_eff = nu * (1.0 + self.f_vec / cfg.carrier_freq_hz)
        else:
            nu_eff = nu

        # 基础相位
        phase = (-2.0 * np.pi * self.f_vec * tau) + (2.0 * np.pi * nu_eff * t_intra) + phi
        h = np.exp(1j * phase)

        # Beam squint 增益
        if cfg.enable_beam_squint:
            h = self._beam_squint_gain * h
        elif cfg.enable_beam_squint_proxy and cfg.beam_squint_strength > 0:
            # 旧版 proxy（向后兼容）
            f_edge = cfg.bandwidth_hz / 2
            f_norm = self.f_vec / max(f_edge, 1e-12)
            proxy_gain = np.exp(-0.5 * (cfg.beam_squint_strength * f_norm) ** 2)
            h = proxy_gain * h

        return h

    def jacobian(self, x_norm: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        解析 Jacobian ∂h/∂x

        考虑 beam squint 和 Doppler squint 的影响
        """
        cfg = self.cfg
        t_intra = self.t_vec
        h = self.h(x_norm, frame_idx)

        # ∂h/∂τ_n
        d_tau = (1j * (-2.0 * np.pi * self.f_vec)) * h * cfg.delay_scale

        # ∂h/∂ν_n
        if cfg.enable_doppler_squint:
            squint_factor = 1.0 + self.f_vec / cfg.carrier_freq_hz
            d_nu = (1j * (2.0 * np.pi * t_intra * squint_factor)) * h * cfg.doppler_scale
        else:
            d_nu = (1j * (2.0 * np.pi * t_intra)) * h * cfg.doppler_scale

        # ∂h/∂φ_n
        d_phi = (1j) * h * cfg.phase_scale

        J = np.stack([d_tau, d_nu, d_phi], axis=1)
        return J

    def R_cov(self) -> np.ndarray:
        """观测噪声协方差"""
        return self.sigma_eff_sq * np.eye(self.m, dtype=float)

    # =========================================================================
    # 观测生成（含 pointing jitter）
    # =========================================================================

    def observe(
            self,
            x_norm: np.ndarray,
            frame_idx: int,
            rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        生成带噪声的观测

        y = g_pt * h(x) + n + AQNM

        其中 g_pt 是 pointing jitter 增益（model mismatch）
        """
        if rng is None:
            rng = np.random.default_rng()

        h = self.h(x_norm, frame_idx)

        # P2: Pointing jitter（作为 mismatch 注入）
        if self.cfg.enable_pointing_jitter:
            g_pt = self._update_pointing_jitter(rng)
            h = g_pt * h

        # AWGN
        n = np.sqrt(self.awgn_var / 2) * (
                rng.standard_normal(self.m) + 1j * rng.standard_normal(self.m)
        )
        y = h + n

        # AQNM
        if self.cfg.apply_quantization and self.cfg.adc_bits < 12:
            q = np.sqrt(self.quant_var / 2) * (
                    rng.standard_normal(self.m) + 1j * rng.standard_normal(self.m)
            )
            y_q = self.alpha * y + q
            y = y_q / self.alpha

        return y

    # =========================================================================
    # Fisher Information
    # =========================================================================

    def compute_fim_data(self, x_norm: np.ndarray, frame_idx: int) -> np.ndarray:
        """计算数据 Fisher Information Matrix"""
        J = self.jacobian(x_norm, frame_idx)
        JHJ = J.conj().T @ J
        J_data = (2.0 / max(self.sigma_eff_sq, 1e-12)) * np.real(JHJ)
        return J_data


# =========================================================================
# 工厂函数
# =========================================================================

def create_default_model(snr_db: float = 10.0, adc_bits: int = 4) -> THzISACModel:
    """创建默认模型（所有THz效应关闭，向后兼容）"""
    cfg = THzISACConfig(snr_db=snr_db, adc_bits=adc_bits)
    return THzISACModel(cfg)


def create_model_v3_full(
        snr_db: float = 10.0,
        adc_bits: int = 4,
        # P0: 连续 PN
        enable_continuous_pn: bool = True,
        pn_linewidth_hz: float = 100.0,
        # P0: Doppler squint
        enable_doppler_squint: bool = True,
        # P1: Beam squint
        enable_beam_squint: bool = True,
        n_ant: int = 16,
        theta0_deg: float = 10.0,
        # P2: Pointing jitter
        enable_pointing_jitter: bool = False,
        pointing_std_deg: float = 0.1,
) -> THzISACModel:
    """
    创建完整 V3 模型（所有物理效应可选）

    推荐的 TWC 主文配置：
    - enable_continuous_pn=True, pn_linewidth_hz=100
    - enable_doppler_squint=True
    - enable_beam_squint=True (附录 stress-test)
    - enable_pointing_jitter=False (或作为 stress-test)
    """
    cfg = THzISACConfig(
        snr_db=snr_db,
        adc_bits=adc_bits,
        # P0
        enable_continuous_pn=enable_continuous_pn,
        pn_linewidth_hz=pn_linewidth_hz,
        enable_doppler_squint=enable_doppler_squint,
        # P1
        enable_beam_squint=enable_beam_squint,
        beam_squint_n_ant=n_ant,
        beam_squint_theta0_deg=theta0_deg,
        # P2
        enable_pointing_jitter=enable_pointing_jitter,
        pointing_jitter_std_deg=pointing_std_deg,
    )
    return THzISACModel(cfg)


# =========================================================================
# 向后兼容别名
# =========================================================================

create_model = create_default_model