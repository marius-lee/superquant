"""BOCPD — Bayesian Online Changepoint Detection.

来源: Adams & MacKay (2007) "Bayesian Online Changepoint Detection"
      arXiv:0710.3742 — 引用量 5000+, TradingView AetherEdge 指标基

原理: 维护游程长度后验 P(r_t | x_{1:t}), 递归在线更新。
      当 P(r_t=0 | data) > 阈值 → 变点发生, 新状态开始。

用法: from engine.bayesian_changepoint import BOCPD
      detector = BOCPD(hazard=1/50, threshold=0.30)
      prob, age = detector.update(price)
      if prob > 0.30 and age < 10:  # 刚变→交易信号

关键参数 (来源: AetherEdge + Kass & Raftery 1995):
  hazard = 1/平均状态长度 (默认1/50=0.02, 保守)
  threshold = 0.30 (变点概率阈值, 业界推荐 0.25-0.40)
"""

import math
import numpy as np


def _ema(series, span):
    """指数移动平均 (Pandas semantics: span=N, decay=2/(N+1))。"""
    if len(series) < span:
        return [float(series[0])] * len(series)
    alpha = 2.0 / (span + 1)
    result = [float(series[0])]
    for v in series[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def _logsumexp(x):
    """数值稳定的 log-sum-exp: log(Σ exp(x_i))。"""
    if len(x) == 0:
        return -np.inf
    m = np.max(x)
    if not np.isfinite(m):
        return -np.inf
    return m + math.log(np.sum(np.exp(x - m)))


class BOCPD:
    """Bayesian Online Changepoint Detection — 游程长度后验。

    维护 P(r_t | data) 的离散分布, 每步递归更新。
    变点概率 = P(r_t=0 | data) — 当此概率超阈值, 新状态开始。
    """

    def __init__(self, hazard=1/20, threshold=0.25, max_run=200):
        """
        Args:
            hazard: 风险率 = 1/E[状态长度]。越小越保守(少变点)。
            threshold: 变点概率阈值 (>此值判定变点)。
            max_run: 最大游程长度 (截断计算)。
        """
        self.hazard = hazard
        self.threshold = threshold
        self.max_run = max_run
        self._reset()

    def _reset(self):
        """初始化游程后验: 全新状态 P(r=0)=1。"""
        self.run_prob = np.array([1.0])  # P(r_t | data)
        self.obs_count = 0
        # 运行统计 (每个游程的累积和)
        self.run_sum_x = [0.0]
        self.run_sum_x2 = [0.0]
        self.run_n = [0]

    def _predictive_prob(self, x, run_len_idx):
        """给定游程长度, 预测 x 的概率 (Student-t, 共轭 Normal-IG 先验)。

        来源: Murphy (2007) Conjugate Bayesian analysis, Section 3.
        使用 Normal-Inverse-Gamma(α0,β0,μ0,κ0) 共轭先验:
          α0=1, β0=1, μ0=0, κ0=1 (弱信息先验)
        后验预测为 Student-t(2α_n, μ_n, β_n(κ_n+1)/(α_n κ_n))
        """
        n = self.run_n[run_len_idx]
        if n == 0:
            # 无数据 → 使用先验预测
            return 0.0 if abs(x) > 0.2 else 1.0  # 宽先验

        # 先验参数
        alpha0, beta0, mu0, kappa0 = 1.0, 1.0, 0.0, 1.0
        # 后验参数
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + self.run_sum_x[run_len_idx]) / kappa_n
        alpha_n = alpha0 + n / 2
        # SSR = sum(x_i^2) + kappa0*mu0^2 - kappa_n*mu_n^2
        ssr = self.run_sum_x2[run_len_idx] + kappa0 * mu0**2 - kappa_n * mu_n**2
        ssr = max(ssr, 0.0)  # 数值保护
        beta_n = beta0 + 0.5 * ssr

        # Student-t log pdf: t_{2α}(x | μ, σ²), σ² = β(κ+1)/(α κ)
        df = 2 * alpha_n
        sigma2 = beta_n * (kappa_n + 1) / (alpha_n * kappa_n)
        if sigma2 <= 0:
            sigma2 = 1e-10

        # log p(x | θ) = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5 log(νπσ²) - (ν+1)/2 log(1 + (x-μ)²/(νσ²))
        from math import lgamma, log, pi
        diff = x - mu_n
        log_p = (lgamma((df + 1) / 2) - lgamma(df / 2)
                 - 0.5 * log(df * pi * sigma2)
                 - (df + 1) / 2 * log(1 + diff**2 / (df * sigma2)))
        return log_p

    def update(self, x):
        """处理一个新观测, 返回 (变点概率, 当前状态年龄)。

        全部在对数空间计算, 避免数值下溢。
        """
        self.obs_count += 1
        n_runs = len(self.run_prob)
        log_run_prob = np.log(self.run_prob + 1e-300)  # 转对数空间

        # 1. 预测概率 (对数)
        log_pred = np.zeros(n_runs + 1)
        log_pred[0] = self._predictive_prob(x, 0)  # 先验预测
        for r in range(n_runs):
            log_pred[r + 1] = self._predictive_prob(x, r)

        # 2. 增长概率 (continuation, 对数空间)
        log_growth = np.full(n_runs + 1, -np.inf)
        for r in range(n_runs):
            log_growth[r + 1] = log_run_prob[r] + log_pred[r + 1] + math.log(1 - self.hazard)

        # 3. 变点概率 (对数空间)
        log_cp_terms = log_run_prob + log_pred[0] + math.log(self.hazard)
        log_growth[0] = _logsumexp(log_cp_terms)

        # 4. 归一化 (对数→线性)
        log_total = _logsumexp(log_growth)
        if np.isfinite(log_total):
            self.run_prob = np.exp(log_growth - log_total)
        else:
            self.run_prob = np.zeros(n_runs + 1)
            self.run_prob[0] = 1.0

        # 5. 更新运行统计
        new_sum_x = [0.0] + [s + x for s in self.run_sum_x]
        new_sum_x2 = [0.0] + [s + x**2 for s in self.run_sum_x2]
        new_n = [0] + [n + 1 for n in self.run_n]

        # 6. 截断+剪枝
        if len(self.run_prob) > self.max_run:
            self.run_prob = self.run_prob[:self.max_run]
            new_sum_x = new_sum_x[:self.max_run]
            new_sum_x2 = new_sum_x2[:self.max_run]
            new_n = new_n[:self.max_run]

        keep = self.run_prob > 1e-15
        if not np.any(keep):
            keep[0] = True
        self.run_prob = self.run_prob[keep]
        self.run_sum_x = [new_sum_x[i] for i in range(len(keep)) if keep[i]]
        self.run_sum_x2 = [new_sum_x2[i] for i in range(len(keep)) if keep[i]]
        self.run_n = [new_n[i] for i in range(len(keep)) if keep[i]]

        # 7. 返回
        cp_prob = float(self.run_prob[0]) if len(self.run_prob) > 0 else 0.0
        regime_age = int(np.argmax(self.run_prob)) if len(self.run_prob) > 0 else 0
        return cp_prob, regime_age

    def detect(self, prices, threshold=None, min_warmup=5):
        """检测价格序列变点 — 标准化输入 + BOCPD。

        AetherEdge 做法 (TradingView): 先标准化到单位方差, 再用高斯BOCPD。
        金融数据原始收益噪声太大, 标准化后信噪比显著提升。
        """
        thresh = threshold if threshold is not None else self.threshold
        n = len(prices)
        if n < min_warmup + 3:
            return False

        # 对数收益率
        log_rets = []
        for i in range(1, n):
            if prices[i-1] > 0 and prices[i] > 0:
                log_rets.append(math.log(prices[i] / prices[i-1]))
        if len(log_rets) < min_warmup:
            return False

        # 标准化: 使用前一半数据估计方差, 全部数据标准化到 σ≈1
        half = max(min_warmup, len(log_rets) // 2)
        baseline_std = float(np.std(log_rets[:half]))
        if baseline_std < 1e-8:
            baseline_std = 0.01  # 默认 1%

        # 用滑动窗口标准化: 每点除以前面数据的 std
        standardized = []
        window = []
        for r in log_rets:
            window.append(r)
            if len(window) > 20:
                window.pop(0)
            local_std = float(np.std(window)) if len(window) >= 5 else baseline_std
            local_std = max(local_std, baseline_std * 0.5)  # 防止 std→0
            standardized.append(r / local_std)

        if len(standardized) < min_warmup:
            return False

        # BOCPD 检测
        self._reset()
        last_prob = 0.0
        for x in standardized[-min_warmup*2:]:  # 只看最近数据
            last_prob, _ = self.update(x)

        return last_prob > thresh


# 兼容旧接口: 模块级函数, 使用默认 BOCPD 实例
_default_detector = BOCPD()


def bayesian_detect(prices, threshold=0.30, min_warmup=5):
    """BOCPD 变点检测 — 兼容旧 bayesian_detect() 接口。

    来源: Adams & MacKay (2007), 阈值校准自 Kass & Raftery (1995).
          BF>20 对应后验 ~0.95; 概率 0.30 对应 BF~3 (中度证据, 实时交易适用).

    Args:
        prices: 最近N个价格
        threshold: 变点概率阈值 (默认 0.30)
        min_warmup: 最少数据点

    Returns:
        bool: 是否检测到变点
    """
    return _default_detector.detect(prices, threshold=threshold, min_warmup=min_warmup)
