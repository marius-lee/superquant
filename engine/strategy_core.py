"""止损止盈核心 — 波动率自适应, 无外部依赖。

用法:
  from engine.strategy_core import calc_adaptive_stop, calc_take_profit, generate_daily_returns

来源:
  calc_adaptive_stop: A股残差波动率年化30%中位 (Grinold 表3-1)
  calc_take_profit: 陈小群peak回撤5%移动止盈
"""

import math


def calc_adaptive_stop(entry_price, daily_returns, params=None):
    """波动率自适应止损价。

    公式: stop = entry × (1 - adaptive_pct)
          adaptive_pct = base × (stock_down_vol / 0.30)
          floor: 2%, ceiling: 8%
    """
    if params is None:
        params = {}
    base = params.get('adaptive_stop_base', 0.05)
    floor = params.get('adaptive_stop_floor', 0.02)
    ceiling = params.get('adaptive_stop_ceiling', 0.08)

    down_rets = [r for r in daily_returns if r < 0]
    if len(down_rets) < 5:
        return entry_price * (1 - base)

    daily_down_vol = _std(down_rets)
    annual_down_vol = daily_down_vol * math.sqrt(252)
    adaptive_pct = base * (annual_down_vol / 0.30)
    adaptive_pct = max(floor, min(ceiling, adaptive_pct))

    return entry_price * (1 - adaptive_pct)


def calc_take_profit(entry_price, peak_price, current_price, params=None):
    """移动止盈: 从最高点回撤 5% 触发。"""
    if params is None:
        params = {}
    trail_pct = params.get('trail_stop_pct', 0.05)
    peak = max(entry_price, peak_price or entry_price)
    trigger = peak * (1 - trail_pct)
    return trigger if current_price <= trigger else None


def _std(values):
    """无依赖标准差。"""
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def generate_daily_returns(prices):
    """从价格序列生成日收益率。prices[-1] 是最新。"""
    return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
