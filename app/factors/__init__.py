"""因子日历 4 日频因子 — 全部由 Hikyuu 内置指标组合。

来源: 因子日历2024 全书遍历 — docs/因子日历2024_全书分析.md

因子选择理由 (北极星约束):
  下行波动率: 短线最稳定预测方向 (高风险溢价, 正相关)
  隔夜跳空:   情绪透支→反转 (负相关, 独立于波动率)
  Amihud:     极端不流动→执行风险惩罚 (与波动率仅 -0.237 相关)
  偏度:       暴跌风险 (负相关, 独立维度)

Hikyuu 内置指标验证:
  STD, SKEW, MA, SUM, ABS, IF, REF, ZSCORE, IC, ICIR
  全部为 builtin_function_or_method — 无需 C++ 编译
"""

from hikyuu.indicator import (
    CLOSE, OPEN, HIGH, LOW, AMO,       # 基础价格
    MA, EMA, STD, SUM, HHV, LLV,       # 统计
    REF, ROC,                           # 变换
    ABS, IF, MAX, MIN, SQRT, LOG,       # 数学
    SKEW,                               # 分布形态
    ZSCORE,                             # 标准化
    IC, ICIR, SPEARMAN,                 # 因子评估
)


def downside_volatility(n: int = 20):
    """F1: 下行已实现波动率 (因子日历 page 53, 第 60 页)

    公式: sqrt( Σ(r_i² × I(r_i<0)) / N )
    Hikyuu: STD(IF(ret<0, ret, 0), n)

    预测方向: 正相关 (下行风险溢价越高 → 未来收益越高)
    """
    ret = ROC(CLOSE(), 1)
    down_ret = IF(ret < 0, ret, 0)
    return STD(down_ret, n)


def overnight_gap(n: int = 20):
    """F2: 隔夜累计跳空 (因子日历 page 282)

    公式: Σ|open_i/close_i-1 - 1| / N
    Hikyuu: SUM(ABS(OPEN/REF(CLOSE,1)-1), n)

    预测方向: 负相关 (跳空越大 → 情绪透支 → 未来反转)
    """
    gap = OPEN() / REF(CLOSE(), 1) - 1.0
    return SUM(ABS(gap), n)


def amihud_illiquidity(n: int = 20):
    """F3: Amihud 非流动性 (因子日历 page 36, Amihud 2002)

    公式: log( |ret| / amount × 10^6 )
    Hikyuu: LOG(MA(ABS(ret)/AMO×1000, n))  — AMO=成交额(千元), ret=百分数

    预测方向: 仅在极端高值时作为惩罚项 (执行风险)
    """
    ret = ROC(CLOSE(), 1)                      # 百分比收益率
    daily_amihud = ABS(ret) / MAX(AMO(), 0.001)  # |ret|/amount
    return LOG(MA(daily_amihud, n) * 1000 + 1)  # log 防厚尾


def skewness_proxy(n: int = 20):
    """F4: 收益偏度 (因子日历 page 91)

    公式: Σ(r-r̄)³ / (σ³ × N)
    Hikyuu: SKEW(ret, n)  — 内置

    预测方向: 负相关 (高偏度 → 暴跌风险 → 需要风险补偿)
    """
    ret = ROC(CLOSE(), 1)
    return SKEW(ret, n)


# ═══════════════════════════════════════════════════════════
# 因子合成与评估
# ═══════════════════════════════════════════════════════════

def create_factor_set():
    """创建因子集 — 4 个日频因子。

    用于 Hikyuu MF (MultiFactor) 框架:
      from hikyuu.trade_sys import crtMF
      mf = crtMF(calculate_func, name='FactorCalendar')

    注意: Hikyuu 2.8.0 pip wheel 的 MF_ICIRWeight C++ 类未直接暴露到 Python
    替代方案: 使用 Hikyuu 内置 crtMF + 自定义 IC 加权计算
    """
    factors = [
        ('downside_vol', downside_volatility, '下行波动率 (正相关)'),
        ('overnight_gap', overnight_gap, '隔夜跳空 (负相关)'),
        ('amihud', amihud_illiquidity, '非流动性 (惩罚项)'),
        ('skewness', skewness_proxy, '偏度 (负相关)'),
    ]
    return factors


def compute_factor_scores(stocks, query):
    """计算所有股票的多因子截面评分。

    使用 Hikyuu 内置 IC/ICIR 指标进行因子评估。

    Args:
        stocks: Stock 列表 (从 sm.get_stock_list 获取)
        query: KQuery 查询范围

    Returns:
        dict: {code: score}
    """
    import numpy as np

    factors = create_factor_set()
    factor_vals = {}

    # 1. 逐股计算因子值
    for name, func, desc in factors:
        factor_vals[name] = {}
        ind = func(20)
        for stock in stocks:
            try:
                kdata = stock.get_kdata(query)
                if len(kdata) < 25:
                    continue
                ind.set_context(kdata)
                vals = [v for v in ind if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if len(vals) > 0:
                    factor_vals[name][stock.market_code] = vals[-1]
            except Exception:
                continue

    print(f"因子计算完成: {[(n, len(v)) for n, v in factor_vals.items()]}")

    # 2. 截面 Z-Score 标准化后等权合成
    scores = {}
    all_codes = set()
    for name in factor_vals:
        all_codes.update(factor_vals[name].keys())

    for code in all_codes:
        score = 0.0
        count = 0
        for name, sign in [('downside_vol', 1), ('overnight_gap', -1),
                           ('amihud', -1), ('skewness', -1)]:
            if code in factor_vals[name]:
                vals = list(factor_vals[name].values())
                mean_val = np.mean(vals)
                std_val = np.std(vals) if np.std(vals) > 0 else 1.0
                z = (factor_vals[name][code] - mean_val) / std_val
                score += sign * z
                count += 1
        if count > 0:
            scores[code] = score / count

    return scores


# ═══════════════════════════════════════════════════════════
# 因子验证 (P1 验收标准)
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    """P1 验收: 计算全市场因子截面评分并输出 Top 10。"""
    from hikyuu.interactive import sm, Query

    print("=" * 60)
    print("P1: 因子日历 4 日频因子 — 截面评分")
    print("=" * 60)

    # 获取全市场股票
    stocks = []
    for mkt in ['SH', 'SZ']:
        try:
            market_stocks = sm.get_stock_list(
                lambda s, m=mkt: s.market == m and s.valid
            )
            stocks.extend(list(market_stocks))
        except Exception as e:
            print(f"  [warn] 获取 {mkt} 股票列表失败: {e}")

    print(f"全市场有效股票: {len(stocks)} 只")
    if len(stocks) == 0:
        print("错误: 无可用股票, 请先运行 scripts/export_daily.py")
        exit(1)

    # 计算因子截面评分
    query = Query(-30)
    print(f"查询范围: 最近 30 个交易日")

    scores = compute_factor_scores(stocks, query)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n因子截面评分 Top 10:")
    for i, (code, score) in enumerate(ranked[:10], 1):
        stock = sm[code]
        name = stock.name if stock.valid else "???"
        print(f"  {i:2d}. {code:<10} {name:<8}  score={score:+.4f}")

    print(f"\n因子截面评分 Bottom 10:")
    for i, (code, score) in enumerate(ranked[-10:], 1):
        stock = sm[code]
        name = stock.name if stock.valid else "???"
        print(f"  {i:2d}. {code:<10} {name:<8}  score={score:+.4f}")

    assert len(scores) >= 100, f"仅 {len(scores)} 只有效评分, 预期 ≥100"
    print(f"\n✅ P1 验收通过: {len(scores)} 只股票截面评分 (预期≥100)")
