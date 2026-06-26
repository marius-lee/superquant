"""蒙特卡洛验证 — Renaissance 级别统计纪律。

来源: Timothy Masters (2006), Harvey & Liu (2014), Bailey & Lopez de Prado (2014)
     梁文峰 — "每个参数都经过蒙特卡洛验证"
     Renaissance Technologies — p < 0.01, 每个信号必须过MCP才上线

功能:
  1. 置换检验 — p < 0.01 (Renaissance标准)
  2. Deflated Sharpe — 校正多重检验
  3. PBO (CSCV) — 过拟合概率
  4. 合成数据鲁棒性 — 随机路径 vs 真实路径
  5. 预部署检验 — 所有信号上线前必须通过

用法: python ops/monte_carlo.py
      from ops.monte_carlo import pre_deploy_check, test_signal, test_strategy
"""

import sqlite3, os, sys, numpy as np, math
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRADE_DB = os.path.join(os.path.expanduser("~/project/quant"), "data", "trades.db")
ALPHA = 0.01  # Renaissance标准 (学界0.05, 业界0.01)
PERMUTATIONS = 1000


# ══════════════════════════════════════════
# 1. 置换检验 — p < 0.01
# ══════════════════════════════════════════

def test_signal(signal_code, n_permutations=PERMUTATIONS):
    """对单个信号做置换检验。"""
    conn = sqlite3.connect(TRADE_DB)
    rows = conn.execute("""
        SELECT s.pnl_pct FROM sim_trades s
        JOIN signal_events e ON s.symbol = e.symbol AND s.date = e.date
        WHERE s.side = 'sell' AND e.signal LIKE ?
    """, (f'%{signal_code}%',)).fetchall()
    conn.close()

    returns = np.array([r[0] for r in rows if r[0] is not None])
    n = len(returns)
    if n < 5:
        return {'signal': signal_code, 'n_trades': n, 'verdict': '数据不足 (n<5)', 'p_value': 1.0,
                'is_significant': False, 'percentile': 50.0, 'mean_return': 0}

    true_mean = float(np.mean(returns))
    perm_means = np.zeros(n_permutations)
    for i in range(n_permutations):
        signs = np.random.choice([-1, 1], size=n)
        shuffled = returns * signs
        np.random.shuffle(shuffled)
        perm_means[i] = np.mean(shuffled)

    p_value = np.mean(perm_means >= true_mean)
    percentile = np.mean(perm_means < true_mean) * 100
    is_sig = p_value < ALPHA

    return {'signal': signal_code, 'n_trades': n, 'mean_return': round(true_mean, 4),
            'p_value': round(p_value, 4), 'is_significant': is_sig,
            'percentile': round(percentile, 1),
            'verdict': '✅ 显著' if is_sig else ('⚠️ 边际' if p_value < 0.05 else '❌ 不显著'),
            'perm_mean': round(float(np.mean(perm_means)), 4),
            'perm_std': round(float(np.std(perm_means)), 4)}


def test_strategy(n_permutations=PERMUTATIONS):
    """策略整体置换检验。"""
    conn = sqlite3.connect(TRADE_DB)
    rows = conn.execute("SELECT date, symbol, pnl_pct FROM sim_trades WHERE side='sell' AND pnl_pct IS NOT NULL ORDER BY date, id").fetchall()
    conn.close()
    returns = np.array([r[2] for r in rows])
    n = len(returns)
    if n < 10:
        return {'n_trades': n, 'verdict': '数据不足 (n<10)'}

    true_total = np.sum(returns)
    true_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    perm_totals = np.zeros(n_permutations)
    for i in range(n_permutations):
        signs = np.random.choice([-1, 1], size=n)
        shuffled = returns * signs
        np.random.shuffle(shuffled)
        perm_totals[i] = np.sum(shuffled)
    p_total = np.mean(perm_totals >= true_total)
    return {'n_trades': n, 'true_total_pnl': round(true_total, 2), 'true_sharpe': round(true_sharpe, 2),
            'p_total': round(p_total, 4), 'is_significant': p_total < ALPHA,
            'verdict': '✅ 显著' if p_total < ALPHA else ('⚠️ 边际' if p_total < 0.05 else '❌ 不显著')}


# ══════════════════════════════════════════
# 2. Deflated Sharpe Ratio
# ══════════════════════════════════════════

def deflated_sharpe(sharpe_observed, n_trials, n_obs, skew=0, kurt=3):
    """Deflated Sharpe Ratio — 校正多重检验偏误。

    来源: Bailey & Lopez de Prado (2014), Journal of Portfolio Management
    输入: 观测夏普, 尝试的策略数, 观测数, 偏度, 峰度
    返回: (deflated_sharpe, p_value, verdict)
    """
    # Expected maximum Sharpe under null (all strategies worthless)
    # E[max] ≈ σ_SR × E[max of N standard normals]
    sigma_sr = math.sqrt((1 + 0.5 * sharpe_observed**2 - skew * sharpe_observed
                          + ((kurt - 3) / 4) * sharpe_observed**2) / n_obs)
    # Expected maximum of N standard normals (approx via extreme value theory)
    gamma = 0.5772156649  # Euler-Mascheroni constant
    e_max = math.sqrt(2 * math.log(n_trials)) - (math.log(math.log(n_trials)) + math.log(4 * math.pi)) / (2 * math.sqrt(2 * math.log(n_trials))) + gamma / math.sqrt(2 * math.log(n_trials))
    sr_star = sigma_sr * e_max  # expected max under H0

    dsr = (sharpe_observed - sr_star) / sigma_sr if sigma_sr > 0 else 0
    dsr = max(dsr, 0)  # floor at 0
    # p-value from normal CDF approximation
    p_dsr = 1 - _norm_cdf(dsr) if dsr > 0 else 1.0

    return {'sharpe_observed': round(sharpe_observed, 4), 'sharpe_deflated': round(dsr, 4),
            'expected_max_null': round(sr_star, 4), 'p_value': round(p_dsr, 4),
            'verdict': '✅ 通过' if p_dsr < ALPHA else ('⚠️ 边际' if p_dsr < 0.05 else '❌ 可能过拟合')}


# ══════════════════════════════════════════
# 3. PBO — 过拟合概率 (CSCV)
# ══════════════════════════════════════════

def compute_pbo(returns_matrix, n_splits=16):
    """Probability of Backtest Overfitting — 组合对称交叉验证。

    来源: Bailey, Borwein, Lopez de Prado & Zhu (2017)
    方法: S个策略 × T个时间点 → 分成n_splits组 → 所有组合的样本内外对比

    Args:
        returns_matrix: (S, T) array — S个策略(或参数组合) × T个时间点收益
        n_splits: 分组数 (偶数)
    Returns:
        PBO: 样本内最优在样本外表现低于中位数的概率
    """
    S, T = returns_matrix.shape
    if S < 2 or T < n_splits * 2:
        return {'PBO': -1, 'verdict': '数据不足', 'n_strategies': S, 'n_periods': T}

    half = n_splits // 2
    # 随机分成 half 个 IS 组和 half 个 OOS 组
    indices = np.arange(n_splits)
    np.random.shuffle(indices)
    is_groups = indices[:half]
    oos_groups = indices[half:]

    # 分割时间轴
    chunk_size = T // n_splits
    perf_is = np.zeros((S, half))
    perf_oos = np.zeros((S, half))

    for i, g in enumerate(is_groups):
        start, end = g * chunk_size, (g + 1) * chunk_size
        perf_is[:, i] = np.sum(returns_matrix[:, start:end], axis=1)
    for i, g in enumerate(oos_groups):
        start, end = g * chunk_size, (g + 1) * chunk_size
        perf_oos[:, i] = np.sum(returns_matrix[:, start:end], axis=1)

    # 每列: 样本内最优在样本外的排名
    ranks = np.zeros(S)
    for col in range(half):
        best_is = np.argmax(perf_is[:, col])
        # 该策略在OOS中的排名 (分数, 越高越好)
        oos_perf = perf_oos[best_is, :]
        oos_rank = np.sum(oos_perf >= np.median(perf_oos, axis=0))
        ranks[best_is] = oos_rank / half

    # PBO = Prob(IS最优在OOS低于中位数)
    pbo = np.mean([1.0 for r in ranks if r > 0 and r < 0.5])

    return {'PBO': round(pbo, 4), 'verdict': '✅ 稳健' if pbo < 0.1 else ('⚠️ 有过拟合风险' if pbo < 0.3 else '❌ 严重过拟合'),
            'n_strategies': S, 'n_periods': T}


# ══════════════════════════════════════════
# 4. 合成数据鲁棒性测试
# ══════════════════════════════════════════

def synthetic_robustness(returns, n_synthetic=200):
    """合成数据鲁棒性测试 — 在不同市场路径上验证策略。

    方法: 保持真实收益的分布特征(均值/方差/偏度/峰度),
          生成n条合成路径, 策略必须在80%+路径上盈利。

    Returns: {pass_rate, verdict, synthetic_sharpes, ...}
    """
    n = len(returns)
    if n < 10:
        return {'pass_rate': 0, 'verdict': '数据不足'}

    mu, sigma = np.mean(returns), np.std(returns)
    skew = np.mean((returns - mu)**3) / (sigma**3 + 1e-10)
    kurt = np.mean((returns - mu)**4) / (sigma**4 + 1e-10)

    total_true = np.sum(returns)
    synthetic_totals = np.zeros(n_synthetic)

    for i in range(n_synthetic):
        # 从真实分布采样 (保持统计特征)
        syn = np.random.choice(returns, size=n, replace=True)
        # 添加随机扰动模拟不同市场路径
        syn += np.random.normal(0, sigma * 0.1, size=n)
        synthetic_totals[i] = np.sum(syn)

    pass_count = np.sum(synthetic_totals > 0)
    pass_rate = pass_count / n_synthetic

    return {'pass_rate': round(pass_rate, 2), 'n_synthetic': n_synthetic,
            'verdict': '✅ 鲁棒' if pass_rate >= 0.8 else ('⚠️ 边际' if pass_rate >= 0.6 else '❌ 脆弱'),
            'true_total': round(total_true, 2),
            'syn_total_mean': round(float(np.mean(synthetic_totals)), 2),
            'syn_total_std': round(float(np.std(synthetic_totals)), 2)}


# ══════════════════════════════════════════
# 5. 预部署检验 — 所有信号上线前必须通过
# ══════════════════════════════════════════

def pre_deploy_check(signal_code):
    """预部署检验 — Renaissance标准: 信号必须通过全部检验才能上线。

    检验项:
      1. 置换检验 p < 0.01
      2. 合成数据鲁棒性 > 80%
      3. 收益 > 0
    """
    print(f"\n{'='*50}")
    print(f"预部署检验: 信号 {signal_code}")
    print(f"{'='*50}")

    # 1. 置换检验
    r = test_signal(signal_code)
    n = r.get('n_trades', 0)
    if n < 5:
        print(f"  ⏸️ 跳过 (n={n})")
        return False
    p1 = r['p_value'] < ALPHA
    print(f"  1. 置换检验: p={r['p_value']:.4f} {'✅' if p1 else '❌'} {r['verdict']}")

    # 2. 合成数据鲁棒性
    conn = sqlite3.connect(TRADE_DB)
    rows = conn.execute("""
        SELECT s.pnl_pct FROM sim_trades s
        JOIN signal_events e ON s.symbol = e.symbol AND s.date = e.date
        WHERE s.side = 'sell' AND e.signal LIKE ?
    """, (f'%{signal_code}%',)).fetchall()
    conn.close()
    returns = np.array([r[0] for r in rows if r[0] is not None])
    syn = synthetic_robustness(returns)
    p2 = syn['pass_rate'] >= 0.8
    print(f"  2. 鲁棒性: pass={syn['pass_rate']:.0%} {'✅' if p2 else '❌'} {syn['verdict']}")

    # 3. 总收益
    total_pnl = np.sum(returns) if len(returns) > 0 else 0
    p3 = total_pnl > 0
    print(f"  3. 盈亏: ¥{total_pnl:+,.0f} {'✅' if p3 else '❌'}")

    passed = p1 and p2 and p3
    print(f"\n  {'✅ 全部通过 — 可上线' if passed else '❌ 未通过 — 禁止上线'}")
    return passed


# ══════════════════════════════════════════
# 工具
# ══════════════════════════════════════════

def _norm_cdf(x):
    """标准正态CDF近似。"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _logsumexp(x):
    m = np.max(x)
    return m + math.log(np.sum(np.exp(x - m)))


if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 60)
    print("蒙特卡洛验证套件 — Renaissance标准 p<0.01")
    print("=" * 60)

    print("\n── 信号置换检验 ──")
    for code in ['A', 'B', 'E']:
        r = test_signal(code)
        print(f"  {code}: n={r['n_trades']} mean={r['mean_return']*100:+.2f}% p={r['p_value']:.4f} {r['verdict']}")

    print("\n── 策略整体 ──")
    s = test_strategy()
    if 'n_trades' in s:
        print(f"  n={s['n_trades']} PnL=¥{s.get('true_total_pnl',0):,.0f} p={s.get('p_total',1):.4f} {s['verdict']}")

    print("\n── 预部署检查 ──")
    for code in ['A', 'B', 'E']:
        pre_deploy_check(code)

    print("\n── Deflated Sharpe (模拟) ──")
    dsr = deflated_sharpe(2.5, 200, 252)
    print(f"  SR=2.5, N_trials=200 → DSR={dsr['sharpe_deflated']:.4f} {dsr['verdict']}")
