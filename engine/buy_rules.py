"""买入规则 — 纯函数, 可测试。

阶段一 (攻击期, capital < ¥50,000):
  1. 候选按评分降序
  2. 排第一: 全仓买入 (capital/price → 100股整数倍)
  3. 剩余资金: 检查下一位 → 够100股就全仓
  4. 持仓 ≤ max_positions

阶段二 (稳健期, capital ≥ ¥50,000):
  Kelly 仓位管理 (strategy_core.calc_position_size)

用法:
  from engine.buy_rules import calculate_buys
  orders = calculate_buys(capital, candidates)
"""


def calculate_buys(capital, candidates, max_positions=3, attack_threshold=50000.0):
    """计算买入订单。

    Args:
        capital: 可用资金
        candidates: [(symbol, price, score), ...] 按评分降序
        max_positions: 最大持仓数
        attack_threshold: 攻击期上限 (默认 ¥50,000)

    Returns:
        [(symbol, shares, price, phase), ...]  phase='attack' or 'kelly'
    """
    if capital <= 0 or not candidates:
        return []

    if capital < attack_threshold:
        return _attack_phase(capital, candidates, max_positions)
    else:
        return _kelly_phase(capital, candidates, max_positions)


def _attack_phase(capital, candidates, max_positions):
    """攻击期: 全仓集中。来源: 北极星 ¥5,000→¥50,000 阶段策略。"""
    orders = []
    remaining = capital

    for sym, price, score in candidates:
        if len(orders) >= max_positions:
            break
        if remaining < 0.01:
            break

        # 计算能买多少股 (100股整数倍), 确保不超资金
        max_shares = int(remaining / price / 100) * 100
        if max_shares < 100:
            continue
        # 向下调整直到含佣金不超过剩余资金
        while max_shares >= 100:
            cost = max_shares * price * (1 + 0.0003)
            if cost <= remaining:
                break
            max_shares -= 100
        if max_shares < 100:
            continue

        orders.append((sym, max_shares, price, 'attack'))
        remaining -= max_shares * price * (1 + 0.0003)

    return orders


def _kelly_phase(capital, candidates, max_positions):
    """稳健期: Kelly 分散。来源: Kelly 1956, Thorp 2006。"""
    from engine.strategy_core import calc_position_size

    orders = []
    remaining = capital
    for sym, price, score in candidates:
        if len(orders) >= max_positions:
            break
        risk_per_share = price * 0.05  # 简化: 5%止损间距
        shares = calc_position_size(remaining, price, risk_per_share)
        if shares < 100:
            continue
        cost = shares * price * (1 + 0.0003)
        if cost > remaining:
            continue
        orders.append((sym, shares, price, 'kelly'))
        remaining -= cost
    return orders
