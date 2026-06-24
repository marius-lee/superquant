"""贝叶斯变点检测 — 实时检测价格行为突变。

原理: 比较两个假设
  H0: 所有数据来自同一分布 N(μ₀, σ₀²)
  H1: 存在变点 τ, 之后数据来自不同分布 N(μ₁, σ₁²)

检测: 若 P(变点 | 数据) > 阈值 → 返回 True

用法: engine/bayesian_changepoint.py  (作为模块导入)
      from engine.bayesian_changepoint import bayesian_detect
      if bayesian_detect(price_list):
          buy()

数据: 分钟级价格序列 (从 history_cache 获取)
阈值: 贝叶斯因子 > 3 (对应 ~75% 后验概率)
参考: Adams & MacKay (2007) Bayesian Online Changepoint Detection
"""

import math


def bayesian_detect(prices, threshold=3.0, min_warmup=5):
    """检测价格序列中是否存在显著变点。

    Args:
        prices: 最近N个价格 (list of float), [-1]是最新
        threshold: 贝叶斯因子阈值 (>1表示有变点), 默认3.0
        min_warmup: 最少需要多少数据点

    Returns:
        bool: 是否检测到变点
    """
    n = len(prices)
    if n < min_warmup + 3:
        return False

    # 计算收益率序列
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, n) if prices[i-1] > 0]
    if len(returns) < min_warmup:
        return False

    # H0: 全序列的均值和方差
    mu0 = sum(returns) / len(returns)
    var0 = sum((r - mu0) ** 2 for r in returns) / (len(returns) - 1) if len(returns) > 1 else 1e-10
    var0 = max(var0, 1e-10)

    # H1: 在 τ 处变点 (τ扫描最近5-20个点)
    best_bf = 0.0
    for tau in range(max(min_warmup, len(returns) - 20), len(returns) - 2):
        before = returns[:tau]
        after = returns[tau:]

        if len(before) < 3 or len(after) < 2:
            continue

        mu1 = sum(before) / len(before)
        mu2 = sum(after) / len(after)

        # 仅检测"拉升": 变点后均值显著大于变点前
        if mu2 <= mu1:
            continue

        var1 = max(sum((r - mu1) ** 2 for r in before) / (len(before) - 1), 1e-10)
        var2 = max(sum((r - mu2) ** 2 for r in after) / (len(after) - 1), 1e-10)

        # 贝叶斯因子: log(P(data|H1) / P(data|H0))
        # 简化: 用after段均值偏离before段均值的程度
        log_bf_h1 = sum(-0.5 * math.log(2 * math.pi * var2) - 0.5 * (r - mu2) ** 2 / var2 for r in after)
        log_bf_h0 = sum(-0.5 * math.log(2 * math.pi * var0) - 0.5 * (r - mu0) ** 2 / var0 for r in after)

        bf = math.exp(log_bf_h1 - log_bf_h0)

        # 额外验证: 变点后的波动率是否更大 (拉升常伴随放量)
        if var2 > var1 * 1.5:
            bf *= 1.5  # 加权

        if bf > best_bf:
            best_bf = bf

    return best_bf > threshold
