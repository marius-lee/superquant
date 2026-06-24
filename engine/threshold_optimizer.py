#!/usr/bin/env python3
"""方案 A: 数据驱动阈值优化 — 从历史数据学习最优信号参数。

方法:
  1. 全市场扫描"炸板次日"事件 (16M日线)
  2. 提取特征: gap(高开%), vol_ratio(量比), daily_ret(日涨幅%), turnover(换手)
  3. 标签: 次N日最高涨幅是否 ≥ 3% (可配置)
  4. 网格搜索: 找最优阈值组合最大化胜率×盈亏比

输出: config/optimal_thresholds.json

用法:
  python engine/threshold_optimizer.py          # 全量分析
  python engine/threshold_optimizer.py --dry    # 采样100只快速验证
"""

import os, sys, json, time, math, sqlite3, argparse
from collections import defaultdict
import numpy as np

SUPERQUANT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANT_ROOT = os.path.expanduser("~/project/quant")
DB_PATH = os.path.join(QUANT_ROOT, "data", "market.db")
OUTPUT_FILE = os.path.join(SUPERQUANT_ROOT, "config", "optimal_thresholds.json")

# ── 可配置参数 ──
TARGET_RETURN = 0.03        # 标签: 次N日最高涨幅≥3% 为正样本
FORWARD_DAYS = 3            # 前看天数
LIMIT_UP_THRESHOLD = 0.095  # 涨停阈值 (9.5% 近似 10% 含容差)
MIN_SAMPLE = 20             # 每组最少样本数


def fetch_events(conn, limit_symbols=None):
    """全市场扫描炸板次日事件。

    Returns: [(gap, vol_ratio, daily_ret, turnover, label), ...]
    """
    print("  扫描炸板次日事件...")
    t0 = time.time()

    # 获取股票列表
    if limit_symbols:
        symbols = limit_symbols
    else:
        symbols = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM daily ORDER BY symbol"
        ).fetchall()]

    events = []
    total_processed = 0

    for sym in symbols:
        rows = conn.execute(
            "SELECT date, open, high, low, close, volume, amount FROM daily "
            "WHERE symbol=? ORDER BY date", (sym,)
        ).fetchall()

        if len(rows) < FORWARD_DAYS + 3:
            continue

        for i in range(1, len(rows) - FORWARD_DAYS):
            prev = rows[i-1]
            curr = rows[i]

            if prev[4] <= 0 or curr[1] <= 0:
                continue

            # 判断前日是否炸板
            prev_open = prev[1]
            prev_high = prev[2]
            prev_close = prev[4]
            limit_price = round(prev_open * 1.10, 2)
            touched_limit = prev_high >= limit_price * 0.995
            did_not_hold = prev_close < prev_high * 0.995

            if not (touched_limit and did_not_hold):
                continue

            # 提取特征
            gap = (curr[1] / prev_close - 1) * 100
            vol_ratio = curr[5] / max(prev[5], 1)
            daily_ret = (curr[4] / prev_close - 1) * 100
            turnover = curr[5] / 10000.0

            # 计算标签: 次 N 日最高涨幅
            forward_high = max(r[2] for r in rows[i:i+FORWARD_DAYS])
            label = 1 if (forward_high / curr[4] - 1) >= TARGET_RETURN else 0

            events.append((gap, vol_ratio, daily_ret, turnover, label))

        total_processed += 1
        if total_processed % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {total_processed}/{len(symbols)} ({len(events)}事件) {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"  完成: {len(symbols)}只股票, {len(events)}个事件, {elapsed:.0f}s")
    return events


def grid_search(events):
    """网格搜索最优阈值组合。

    目标: max(胜率 × 胜率 × 盈亏比) → 既准又多
    """
    if len(events) < MIN_SAMPLE:
        print(f"  [warn] 样本不足 ({len(events)} < {MIN_SAMPLE})")
        return None

    arr = np.array(events)
    # 只保留 reasonable 范围
    mask = (arr[:, 0] >= -5) & (arr[:, 0] <= 10) & (arr[:, 1] >= 0) & (arr[:, 1] <= 20)
    arr = arr[mask]
    print(f"  有效样本: {len(arr)} (过滤极端值)")

    best_score = 0
    best_params = {}

    # 网格搜索
    gaps = np.percentile(arr[arr[:, 0] > 0, 0], [25, 50, 75])
    vols = np.percentile(arr[arr[:, 1] > 0, 1], [25, 50, 75])
    rets = np.percentile(arr[arr[:, 2] > 0, 2], [25, 50, 75])

    for gap_min in [1.0, 1.5] + list(gaps):
        for vol_min in [1.5, 2.0] + list(vols):
            for ret_min in [3.0, 5.0] + list(rets):
                selected = arr[
                    (arr[:, 0] >= gap_min) &
                    (arr[:, 1] >= vol_min) &
                    (arr[:, 2] >= ret_min)
                ]
                if len(selected) < MIN_SAMPLE:
                    continue

                pos = selected[selected[:, 4] == 1]
                neg = selected[selected[:, 4] == 0]
                n_pos = len(pos)
                n_neg = len(neg)
                total = n_pos + n_neg
                wr = n_pos / total  # 胜率

                if n_neg == 0:
                    continue
                avg_win = np.mean(pos[:, 2]) if n_pos > 0 else 0
                avg_loss = abs(np.mean(neg[:, 2])) if n_neg > 0 else 1
                wl_ratio = avg_win / max(avg_loss, 0.001)

                # 综合分: 胜率² × 盈亏比 (prefer high WR with decent volume)
                score = wr * wr * wl_ratio * math.log(total + 1)

                if score > best_score:
                    best_score = score
                    best_params = {
                        'gap_min': round(gap_min, 1),
                        'vol_ratio': round(vol_min, 1),
                        'daily_ret': round(ret_min, 1),
                        'win_rate': round(wr, 3),
                        'wl_ratio': round(wl_ratio, 2),
                        'n_samples': total,
                        'n_positive': n_pos,
                    }

    return best_params


def main():
    parser = argparse.ArgumentParser(description='方案A: 阈值优化')
    parser.add_argument('--dry', action='store_true', help='采样模式 (100只)')
    ARGS = parser.parse_args()

    print("=" * 60)
    print("方案 A: 数据驱动阈值优化")
    print(f"  目标: 次{FORWARD_DAYS}日涨幅≥{TARGET_RETURN*100:.0f}%")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    if ARGS.dry:
        limit = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM daily ORDER BY RANDOM() LIMIT 100"
        ).fetchall()]
        print(f"  DRY RUN: {len(limit)}只采样")
        events = fetch_events(conn, limit)
    else:
        events = fetch_events(conn)

    conn.close()

    if not events:
        print("  无事件数据")
        return

    pos_count = sum(1 for e in events if e[4] == 1)
    print(f"\n  事件总数: {len(events)}, 正样本({TARGET_RETURN*100:.0f}%+): {pos_count} ({pos_count/max(len(events),1)*100:.0f}%)")

    # 手写规则对照
    manual = [e for e in events if 2.0 <= e[0] <= 5.0 and e[1] >= 3.0 and e[2] >= 5.0]
    manual_pos = sum(1 for e in manual if e[4] == 1)
    print(f"  手写规则: {len(manual)}个信号, 胜率={manual_pos/max(len(manual),1)*100:.0f}%")

    # 网格搜索
    print("\n  网格搜索最优阈值...")
    best = grid_search(events)
    if best:
        print(f"\n  ✅ 最优参数:")
        print(f"    gap_min={best['gap_min']}, vol_ratio={best['vol_ratio']}, daily_ret={best['daily_ret']}")
        print(f"    胜率={best['win_rate']*100:.0f}%, 盈亏比={best['wl_ratio']:.1f}, 样本={best['n_samples']}")

        # 保存
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
        print(f"    保存 → {OUTPUT_FILE}")
    else:
        print("  ⚠️ 样本不足, 保持手写规则")


if __name__ == '__main__':
    main()
