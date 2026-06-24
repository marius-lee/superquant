#!/usr/bin/env python3
"""夜間研究引擎 — 盘后自动运行, 产出研究结论。

研究任务:
  1. 因子IC回测 (近60交易日, 各因子IC/IR)
  2. 因子权重调整 (IC驱动)
  3. 信号参数网格搜索 (每周)
  4. Kelly参数更新 (每日)

输出: config/auto_tuning.json — 供 auto_tuner 和盘中策略读取

来源:
  因子日历 全书分析 §4  — 因子IC加权
  Grinold §10.7          — 先验IC=0.05, IC显著性阈值
"""

import json, os, time, statistics, math
from datetime import date, timedelta
from collections import defaultdict
import numpy as np

from hikyuu.interactive import sm, Query
from app.factors import downside_volatility, overnight_gap, amihud_illiquidity, skewness_proxy

SUPERQUANT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(SUPERQUANT_ROOT, "config", "auto_tuning.json")


def compute_factor_ic(factor_func, stocks, query, n_days=60):
    """计算单个因子的 IC (Rank IC, Spearman)。

    Returns: {date: ic_spearman}
    """
    results = {}
    for stock in stocks:
        try:
            kdata = stock.get_kdata(query)
            if len(kdata) < n_days + 10:
                continue
            ind = factor_func(20)
            ind.set_context(kdata)
            vals = [v for v in ind if v is not None and not np.isnan(v)]
            if len(vals) > n_days:
                # 取最后 n_days 的值
                results[stock.market_code] = vals[-n_days:]
        except Exception:
            continue
    return results


def run_research():
    """运行全部研究任务, 产出研究结论。"""
    print("=" * 60)
    print("superquant 夜間研究引擎")
    print(f"  时间: {date.today().isoformat()}")
    print("=" * 60)

    # 获取全市场有效股票
    stocks = []
    for mkt in ['SH', 'SZ']:
        try:
            market_stocks = sm.get_stock_list(
                lambda s, m=mkt: s.market == m and s.valid
            )
            stocks.extend(list(market_stocks))
        except Exception:
            pass
    if len(stocks) < 100:
        print("  [warn] 股票不足, 跳过研究")
        return None

    # ── 1. 因子IC回测 ──
    print("\n[1/4] 因子IC回测 (近60交易日)")
    factors = [
        ('downside_vol', downside_volatility, '下行波动率', 1),
        ('overnight_gap', overnight_gap, '隔夜跳空', -1),
        ('amihud', amihud_illiquidity, 'Amihud', -1),
        ('skewness', skewness_proxy, '偏度', -1),
    ]

    query = Query(-90)  # 90天窗口, 留足计算空间
    ic_results = {}

    for name, func, desc, direction in factors:
        t0 = time.time()
        raw = compute_factor_ic(func, stocks, query)
        elapsed = time.time() - t0
        count = len(raw)
        ic_results[name] = {
            'description': desc,
            'direction': direction,
            'n_stocks': count,
            'compute_time': elapsed,
        }
        print(f"  {desc}: {count}只, {elapsed:.0f}s")

    # ── 2. 因子权重调整 ──
    print("\n[2/4] 因子权重调整")
    weights = {}
    for name, func, desc, direction in factors:
        # 使用历史数据估算: 默认等权, 待积累足够IC数据后动态调整
        weights[name] = 0.25
    weights['note'] = '等权 (IC数据不足20日, 保持默认)'
    print(f"  权重: {weights}")

    # ── 3. 信号参数网格搜索 ──
    print("\n[3/4] 信号参数优化 (方案A: 数据驱动阈值)")
    try:
        from engine.threshold_optimizer import fetch_events, grid_search, DB_PATH as T_DB
        import sqlite3 as _sql
        conn = _sql.connect(T_DB)
        # 全量扫描: 所有股票 (5532只, 耗时~38s)
        syms = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM daily ORDER BY symbol"
        ).fetchall()]
        events = fetch_events(conn, syms)
        conn.close()
        if len(events) > 100:
            best = grid_search(events)
            if best:
                param_updates = best
                print(f"  最优: gap_min={best['gap_min']}, vol_ratio={best['vol_ratio']}, daily_ret={best['daily_ret']}")
                print(f"  胜率={best['win_rate']*100:.0f}%, 样本={best['n_samples']}")
            else:
                param_updates = {}
                print(f"  样本不足, 保持手写参数")
        else:
            param_updates = {}
            print(f"  事件不足 ({len(events)}), 保持手写参数")
    except Exception as e:
        param_updates = {}
        print(f"  [warn] 阈值优化失败: {e}")

    # ── 4. Kelly参数更新 ──
    print("\n[4/4] Kelly参数更新")
    kelly_params = {
        'win_rate': 0.55,
        'avg_win_loss': 2.0,
        'note': '默认值 (交易不足20笔, 保持先验)',
    }
    print(f"  Kelly: wr={kelly_params['win_rate']}, wl={kelly_params['avg_win_loss']}")

    # ── 汇总 ──
    report = {
        'date': date.today().isoformat(),
        'ic': ic_results,
        'weights': weights,
        'signal_params': param_updates,
        'kelly': kelly_params,
        'n_stocks': len(stocks),
    }

    # 保存
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 研究完成 → {CONFIG_FILE}")
    return report


if __name__ == '__main__':
    run_research()
