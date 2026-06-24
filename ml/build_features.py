#!/usr/bin/env python3
"""批量预计算 L1-L5 特征 → market.db daily_features 表。

速度设计:
  L1: 零成本 (日线已有)
  L2: HDF5批量读 + NumPy向量化 → 预估 ~5分钟 (6.6M条5分钟线)
  L3: easy-tdx get_history_fund_flow 一次取整年 → 预估 ~1小时 (一次性)
  L4: AKShare 逐日龙虎榜 → 预估 ~3分钟 (250天)
  L5: SQL聚合 → 预估 ~5秒

策略:
  - L3是最慢的, 一次性取完存表, 后续训练直接读表
  - L2次之, HDF5批量处理, 用h5py的高效索引
  - 所有结果写入 market.db daily_features 表

用法:
  python ml/build_features.py --l2     # 仅L2技术特征
  python ml/build_features.py --l3     # 仅L3资金流 (需1h)
  python ml/build_features.py --l4     # 仅L4龙虎榜
  python ml/build_features.py --all    # 全部
"""

import os, sys, time, sqlite3, argparse
from datetime import date, timedelta
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUANT_ROOT = os.path.expanduser("~/project/quant")
DB_PATH = os.path.join(QUANT_ROOT, "data", "market.db")
HDF5_DIR = os.path.expanduser("~/stock")


def init_db():
    """创建 daily_features 表。"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS daily_features (
        symbol TEXT, date TEXT,
        ksft REAL, slope REAL, ptc REAL, vol_5min REAL, max_ret REAL, min_ret REAL,
        main_net_in REAL, main_net_ratio REAL, super_large_in REAL, large_in REAL,
        lhb_net_buy REAL, lhb_buy_ratio REAL, lhb_count INTEGER, lhb_exists INTEGER,
        sector_limit_count INTEGER, sector_rank INTEGER,
        PRIMARY KEY (symbol, date)
    )""")
    conn.commit()
    conn.close()
    print("✅ daily_features 表已创建")


def build_l2_batch(conn, date_str=None):
    """L2: HDF5 5分钟线批量计算 KSFT/SLOPE/PTC。

    策略: 逐只股票读取HDF5 → NumPy向量化计算 → 批量INSERT
    速度: ~5分钟 (5532只 × 48条/天 × ~1ms向量化)
    """
    import h5py
    print("\n=== L2: 技术特征 (HDF5 5分钟线) ===")
    
    # 取所有交易日期
    dates = set()
    if date_str:
        dates.add(date_str)
    else:
        rows = conn.execute("SELECT DISTINCT date FROM daily WHERE date>='2025-06-01' ORDER BY date").fetchall()
        dates = {r[0] for r in rows}

    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM daily ORDER BY symbol").fetchall()]
    total = len(symbols)
    inserted = 0
    t0 = time.time()

    for i, sym in enumerate(symbols):
        mkt = 'sh' if sym.startswith(('6','5','9')) else 'sz'
        path = os.path.join(HDF5_DIR, f'{mkt}_5min.h5')
        if not os.path.exists(path):
            continue

        try:
            with h5py.File(path, 'r') as h5:
                ds_name = f"{'SH' if mkt=='sh' else 'SZ'}{sym}"
                if ds_name not in h5['data']:
                    continue
                ds = h5['data'][ds_name]
                n = len(ds)
                if n < 10:
                    continue

                # 批量读取全部数据 → NumPy
                all_dt = np.array([r['datetime'] for r in ds], dtype=np.uint64)
                all_open = np.array([r['openPrice'] for r in ds], dtype=np.float64) / 1000.0
                all_high = np.array([r['highPrice'] for r in ds], dtype=np.float64) / 1000.0
                all_low = np.array([r['lowPrice'] for r in ds], dtype=np.float64) / 1000.0
                all_close = np.array([r['closePrice'] for r in ds], dtype=np.float64) / 1000.0
                all_vol = np.array([r['transCount'] for r in ds], dtype=np.float64)

                # 按日期分组计算特征
                batch_insert = []
                for d in dates:
                    # 取当天数据 (datetime格式: 202606231045 → 前8位是日期)
                    day_mask = np.array([str(dt)[:8] == d.replace('-','') for dt in all_dt])
                    if day_mask.sum() < 10:
                        continue
                    idx = np.where(day_mask)[0]
                    closes_d = all_close[idx]
                    highs_d = all_high[idx]
                    lows_d = all_low[idx]
                    opens_d = all_open[idx]
                    vols_d = all_vol[idx]

                    # KSFT: (高-低)/开盘跌幅
                    ksft_val = float(np.mean((highs_d - lows_d) / np.maximum(np.abs(opens_d - closes_d[0]), 0.01)))

                    # SLOPE: 最近10根K线斜率
                    x = np.arange(min(10, len(closes_d)))
                    y = closes_d[-len(x):]
                    slope_val = float(np.polyfit(x, y, 1)[0]) if len(x) >= 5 else 0.0

                    # PTC: 价格-成交额相关性
                    amounts_d = closes_d * vols_d
                    if len(closes_d) >= 20:
                        c = np.corrcoef(closes_d[-20:], amounts_d[-20:])[0, 1]
                        ptc_val = float(0 if np.isnan(c) else c)
                    else:
                        ptc_val = 0.0

                    # 波动/最大最小
                    rets = np.diff(closes_d) / closes_d[:-1]
                    vol_val = float(np.std(rets)) if len(rets) > 3 else 0.0
                    max_val = float(np.max(rets)) if len(rets) > 0 else 0.0
                    min_val = float(np.min(rets)) if len(rets) > 0 else 0.0

                    batch_insert.append((sym, d, ksft_val, slope_val, ptc_val, vol_val, max_val, min_val))

                if batch_insert:
                    conn.executemany(
                        """INSERT OR REPLACE INTO daily_features
                           (symbol,date,ksft,slope,ptc,vol_5min,max_ret,min_ret)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        [(s, d, k, sl, p, v, mx, mn) for s, d, k, sl, p, v, mx, mn in batch_insert]
                    )
                    inserted += len(batch_insert)

        except Exception as e:
            if i < 5: print(f"  [warn] {sym}: {e}")
            continue

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{total} (写入{inserted}) {elapsed:.0f}s")

    conn.commit()
    elapsed = time.time() - t0
    print(f"✅ L2完成: {inserted}条, {elapsed:.0f}s")


def build_l3_batch(conn):
    """L3: easy-tdx 资金流 → 一次性拉取全年历史。

    策略: 每只股票调用 get_history_fund_flow (一次取365天)
    速度: 5532只 × ~0.7s = ~1小时。一次性完成, 存表后训练秒读。
    """
    print("\n=== L3: 资金流 (easy-tdx) ===")
    print("  ⚠️ 预计耗时 ~1小时 (5532只×0.7s), 只运行一次")
    print("  后续训练直接从DB读取, 秒级")

    try:
        import easy_tdx as et
    except ImportError:
        print("  ❌ easy-tdx 未安装")
        return

    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM daily ORDER BY symbol").fetchall()]
    client = et.TdxClient()
    try:
        client.connect()
    except Exception as e:
        print(f"  ❌ easy-tdx连接失败: {e}")
        return

    inserted = 0
    t0 = time.time()
    for i, sym in enumerate(symbols):
        try:
            mkt = et.Market.SZ if sym.startswith(('0','3')) else et.Market.SH
            df = client.get_history_fund_flow(mkt, sym, start='2025-06-01', count=250)
            if df is not None and len(df) > 0:
                batch = []
                for _, row in df.iterrows():
                    d = str(row.get('date', ''))[:10]
                    si = float(row.get('super_in', 0) or 0)
                    li = float(row.get('large_in', 0) or 0)
                    so = float(row.get('super_out', 0) or 0)
                    lo = float(row.get('large_out', 0) or 0)
                    main_net = (si + li) - (so + lo)
                    total = abs(si) + abs(li) + abs(so) + abs(lo) + 1
                    main_ratio = main_net / total
                    batch.append((sym, d, main_net, main_ratio, si, li))
                if batch:
                    conn.executemany(
                        """INSERT OR REPLACE INTO daily_features
                           (symbol,date,main_net_in,main_net_ratio,super_large_in,large_in)
                           VALUES(?,?,?,?,?,?)""",
                        batch
                    )
                    inserted += len(batch)
        except Exception:
            pass  # skip errors silently
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(symbols)} (写入{inserted}) {elapsed:.0f}s")
        time.sleep(0.05)  # 避免过快

    client.close()
    conn.commit()
    elapsed = time.time() - t0
    print(f"✅ L3完成: {inserted}条, {elapsed:.0f}s")


def build_l4_batch(conn):
    """L4: AKShare 龙虎榜 → 逐日拉取。

    速度: 250天 × ~0.8s = ~3分钟。
    """
    print("\n=== L4: 龙虎榜 (AKShare) ===")
    try:
        import akshare as ak
    except ImportError:
        print("  ❌ akshare 未安装")
        return

    t0 = time.time()
    start = date(2025, 6, 1)
    end = date(2026, 6, 24)
    d = start
    inserted = 0
    while d <= end:
        ds = d.strftime('%Y%m%d')
        try:
            df = ak.stock_lhb_detail_daily_sina(date=ds)
            if df is not None and len(df) > 0:
                batch = []
                for _, row in df.iterrows():
                    sym = row.get('股票代码', '')
                    if not sym: continue
                    buy = float(row.get('成交额', 0) or 0)
                    batch.append((sym, d.isoformat(), buy, 0.5, 1, 1))
                if batch:
                    conn.executemany(
                        """INSERT OR REPLACE INTO daily_features
                           (symbol,date,lhb_net_buy,lhb_buy_ratio,lhb_count,lhb_exists)
                           VALUES(?,?,?,?,?,?)""",
                        batch
                    )
                    inserted += len(batch)
        except Exception:
            pass
        d += timedelta(days=1)
    conn.commit()
    elapsed = time.time() - t0
    print(f"✅ L4完成: {inserted}条, {elapsed:.0f}s")


def build_l5_batch(conn):
    """L5: 板块涨停统计 → SQL聚合。

    速度: ~5秒。
    """
    print("\n=== L5: 板块共振 (SQL聚合) ===")
    t0 = time.time()
    # 统计每天每个板块的涨停数
    rows = conn.execute("""
        SELECT d.date, s.market, COUNT(*) as cnt
        FROM daily d JOIN stocks s ON d.symbol=s.symbol
        WHERE d.close>=d.open*1.095 AND d.date>='2025-06-01'
        GROUP BY d.date, s.market
    """).fetchall()
    # 写入 daily_features
    batch = []
    for date_str, mkt, cnt in rows:
        # 取该板块所有股票
        syms = conn.execute(
            "SELECT symbol FROM stocks WHERE market=?", (mkt,)
        ).fetchall()
        for (sym,) in syms:
            batch.append((sym, date_str, cnt, 0))
    if batch:
        conn.executemany(
            """INSERT OR REPLACE INTO daily_features
               (symbol,date,sector_limit_count,sector_rank)
               VALUES(?,?,?,?)""",
            batch
        )
    conn.commit()
    elapsed = time.time() - t0
    print(f"✅ L5完成: {len(batch)}条, {elapsed:.0f}s")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='批量预计算特征')
    p.add_argument('--l2', action='store_true', help='L2技术特征')
    p.add_argument('--l3', action='store_true', help='L3资金流')
    p.add_argument('--l4', action='store_true', help='L4龙虎榜')
    p.add_argument('--l5', action='store_true', help='L5板块')
    p.add_argument('--all', action='store_true', help='全部')
    ARGS = p.parse_args()

    init_db()
    conn = sqlite3.connect(DB_PATH)

    if ARGS.all or ARGS.l2:
        build_l2_batch(conn)
    if ARGS.all or ARGS.l3:
        build_l3_batch(conn)
    if ARGS.all or ARGS.l4:
        build_l4_batch(conn)
    if ARGS.all or ARGS.l5:
        build_l5_batch(conn)

    conn.close()
    print("\n✅ 特征预计算完成")
