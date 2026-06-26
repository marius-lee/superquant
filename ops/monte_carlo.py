"""蒙特卡洛置换检验 — 验证信号统计显著性。

来源: Timothy Masters, "Evidence-Based Technical Analysis" (2006)
     梁文峰 — "每个参数都经过蒙特卡洛验证"

方法: 对信号触发时的收益序列做随机置换, 真实收益 vs 置换分布。
      p < 0.05 → 信号有效; p > 0.50 → 不如随机 → 删除。

用法: python ops/monte_carlo.py
      from ops.monte_carlo import test_signal, test_strategy
"""

import sqlite3, os, sys, numpy as np
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRADE_DB = os.path.join(os.path.expanduser("~/project/quant"), "data", "trades.db")
MKT_DB = os.path.join(os.path.expanduser("~/project/quant"), "data", "market.db")
PERMUTATIONS = 1000
ALPHA = 0.05  # 显著性水平


def test_signal(signal_code, n_permutations=PERMUTATIONS):
    """对单个信号做置换检验。

    Args:
        signal_code: 'A', 'B', 或 'E'
        n_permutations: 置换次数

    Returns:
        {signal, n_trades, mean_return, p_value, is_significant, percentile}
    """
    conn = sqlite3.connect(TRADE_DB)
    # 获取该信号触发的所有交易的收益
    rows = conn.execute("""
        SELECT s.pnl_pct FROM sim_trades s
        JOIN signal_events e ON s.symbol = e.symbol AND s.date = e.date
        WHERE s.side = 'sell' AND e.signal LIKE ?
    """, (f'%{signal_code}%',)).fetchall()
    conn.close()

    returns = np.array([r[0] for r in rows if r[0] is not None])
    n = len(returns)

    if n < 5:
        return {'signal': signal_code, 'n_trades': n, 'mean_return': float(np.mean(returns)) if n > 0 else 0,
                'p_value': 1.0, 'is_significant': False, 'percentile': 50.0,
                'verdict': '数据不足 (n<5)'}

    true_mean = float(np.mean(returns))

    # 置换检验: 随机打乱收益符号和顺序
    perm_means = np.zeros(n_permutations)
    for i in range(n_permutations):
        # 随机翻转符号 + 随机打乱
        signs = np.random.choice([-1, 1], size=n)
        shuffled = returns * signs
        np.random.shuffle(shuffled)
        perm_means[i] = np.mean(shuffled)

    # 单侧检验: 真实收益是否显著 > 置换分布
    p_value = np.mean(perm_means >= true_mean)
    percentile = np.mean(perm_means < true_mean) * 100

    is_sig = p_value < ALPHA
    if p_value < ALPHA:
        verdict = '✅ 显著 (p<0.05)'
    elif p_value < 0.10:
        verdict = '⚠️ 边际显著 (p<0.10)'
    elif percentile < 50:
        verdict = '❌ 不如随机, 建议删除'
    else:
        verdict = '⚠️ 不显著, 需要更多数据'

    return {
        'signal': signal_code, 'n_trades': n, 'mean_return': round(true_mean, 4),
        'p_value': round(p_value, 4), 'is_significant': is_sig,
        'percentile': round(percentile, 1), 'verdict': verdict,
        'perm_mean': round(float(np.mean(perm_means)), 4),
        'perm_std': round(float(np.std(perm_means)), 4),
    }


def test_strategy(n_permutations=PERMUTATIONS):
    """对整个策略做置换检验 — 真实收益 vs 随机排列触发时间。

    如果 p>0.05, 整个策略可能是噪声。
    """
    conn = sqlite3.connect(TRADE_DB)
    rows = conn.execute("""
        SELECT date, symbol, pnl_pct FROM sim_trades
        WHERE side = 'sell' AND pnl_pct IS NOT NULL
        ORDER BY date, id
    """).fetchall()
    conn.close()

    returns = np.array([r[2] for r in rows])
    n = len(returns)

    if n < 10:
        return {'n_trades': n, 'verdict': '数据不足 (n<10)'}

    true_total = np.sum(returns)
    true_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    perm_totals = np.zeros(n_permutations)
    perm_sharpes = np.zeros(n_permutations)
    for i in range(n_permutations):
        signs = np.random.choice([-1, 1], size=n)
        shuffled = returns * signs
        np.random.shuffle(shuffled)
        perm_totals[i] = np.sum(shuffled)
        perm_sharpes[i] = (np.mean(shuffled) / np.std(shuffled) * np.sqrt(252)
                          if np.std(shuffled) > 0 else 0)

    p_total = np.mean(perm_totals >= true_total)
    p_sharpe = np.mean(perm_sharpes >= true_sharpe)

    return {
        'n_trades': n, 'true_total_pnl': round(true_total, 2),
        'true_sharpe': round(true_sharpe, 2),
        'p_total': round(p_total, 4), 'p_sharpe': round(p_sharpe, 4),
        'is_significant': p_total < ALPHA,
        'verdict': '✅ 策略显著' if p_total < ALPHA else ('⚠️ 边际' if p_total < 0.10 else '❌ 可能是噪声'),
        'perm_total_mean': round(float(np.mean(perm_totals)), 2),
        'perm_total_std': round(float(np.std(perm_totals)), 2),
    }


if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 50)
    print("蒙特卡洛置换检验 — 信号/策略显著性")
    print("=" * 50)

    for code in ['A', 'B', 'E']:
        r = test_signal(code)
        print(f"\n信号 {code}:")
        print(f"  交易数: {r['n_trades']}  均值收益: {r['mean_return']*100:+.2f}%")
        print(f"  p值: {r['p_value']:.4f}  分位: {r['percentile']:.0f}%")
        print(f"  判定: {r['verdict']}")

    print(f"\n--- 策略整体 ---")
    s = test_strategy()
    if 'n_trades' in s:
        print(f"  交易数: {s['n_trades']}  总PnL: ¥{s.get('true_total_pnl',0):,.0f}")
        print(f"  p(总收益): {s.get('p_total',1):.4f}  p(夏普): {s.get('p_sharpe',1):.4f}")
        print(f"  判定: {s['verdict']}")
    else:
        print(f"  {s['verdict']}")
