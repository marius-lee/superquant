#!/usr/bin/env python3
"""P0a: 从 quant market.db 导出日线到 Hikyuu HDF5 格式。

来源:
  Hikyuu H5Record: hikyuu_cpp/hikyuu/data_driver/kdata/hdf5/H5Record.h:20-28
    struct H5Record {
        uint64_t datetime;     // YYYYMMDDhhmm 格式, 如 202606240000
        uint32_t openPrice;    // 价格 × 1000 (如 10.65 → 10650)
        uint32_t highPrice;
        uint32_t lowPrice;
        uint32_t closePrice;
        uint64_t transAmount;  // 成交额 (元)
        uint64_t transCount;   // 成交量 (股)
    };

  Hikyuu HDF5 布局:  hikyuu_cpp/.../H5KDataDriver.cpp:291
    sh_day.h5/data/SH600000  (dataset per stock, market+code naming)
    sz_day.h5/data/SZ000001

  market.db 单位:  store.py:216
    volume: 手 (1手=100股)
    amount: 千元
    open/high/low/close: 元
    date: 文本 "YYYY-MM-DD"

用法:
    python scripts/export_daily.py                    # 导出全部历史
    python scripts/export_daily.py --start 2025-01-01 # 从指定日期开始
    python scripts/export_daily.py --dry-run          # 不写文件, 仅打印统计
"""

import sys, os, argparse, time, sqlite3
import numpy as np

# 量化项目根目录
QUANT_ROOT = os.path.expanduser("~/project/quant")
# Hikyuu 数据目录 (对应 ~/.hikyuu/hikyuu.ini 中的 datadir)
HKU_DATA = os.path.expanduser("~/stock")

# ═══════════════════════════════════════════════════════════
# HDF5 写入 (使用 h5py, 与 Hikyuu C++ 格式兼容)
# ═══════════════════════════════════════════════════════════
# 来源: Hikyuu H5Record.h — 8字节对齐的复合类型
H5_DTYPE = np.dtype([
    ('datetime',    np.uint64),   # YYYYMMDDhhmm
    ('openPrice',   np.uint32),   # price × 1000
    ('highPrice',   np.uint32),
    ('lowPrice',    np.uint32),
    ('closePrice',  np.uint32),
    ('transAmount', np.uint64),   # yuan
    ('transCount',  np.uint64),   # shares
])


def to_h5_datetime(date_str: str) -> int:
    """'2026-05-29' → 202605290000 (uint64)"""
    return int(date_str.replace('-', '') + '0000')


def export_market(market: str, min_data: int = 15):
    """导出单个市场的全部股票日线到 HDF5。

    Args:
        market: 'SH' 或 'SZ'
        min_data: 最少需要多少条日线才导出 (Hikyuu 默认15)
    """
    import h5py

    db_path = os.path.join(QUANT_ROOT, 'data', 'market.db')
    if not os.path.exists(db_path):
        print(f"错误: market.db 不存在: {db_path}")
        return 0

    conn = sqlite3.connect(db_path)

    # 1. 获取该市场的所有股票代码
    if market == 'SH':
        codes = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM daily WHERE symbol LIKE '6%' OR symbol LIKE '5%' OR symbol LIKE '9%' ORDER BY symbol"
        ).fetchall()]
        h5_path = os.path.join(HKU_DATA, 'sh_day.h5')
    else:  # SZ
        codes = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM daily WHERE symbol LIKE '0%' OR symbol LIKE '3%' ORDER BY symbol"
        ).fetchall()]
        h5_path = os.path.join(HKU_DATA, 'sz_day.h5')

    print(f"[{market}] {len(codes)} 只股票, 输出: {h5_path}")

    # 2. 创建 HDF5 文件
    dry_run = ARGS.dry_run
    if not dry_run:
        os.makedirs(HKU_DATA, exist_ok=True)
        h5 = h5py.File(h5_path, 'w')
        data_group = h5.create_group('data')
    else:
        h5 = None
        data_group = None

    exported = 0
    skipped = 0
    t0 = time.time()

    for i, code in enumerate(codes):
        rows = conn.execute(
            "SELECT date, open, high, low, close, volume, amount FROM daily "
            "WHERE symbol=? "
            "ORDER BY date",
            (code,)
        ).fetchall()

        if len(rows) < min_data:
            skipped += 1
            continue

        # 3. 构建 H5Record 数组
        records = np.zeros(len(rows), dtype=H5_DTYPE)
        for j, (date_str, op, hi, lo, cl, vol, amt) in enumerate(rows):
            records[j]['datetime'] = to_h5_datetime(date_str)
            records[j]['openPrice'] = np.uint32(round(op * 1000))
            records[j]['highPrice'] = np.uint32(round(hi * 1000))
            records[j]['lowPrice'] = np.uint32(round(lo * 1000))
            records[j]['closePrice'] = np.uint32(round(cl * 1000))
            # store.py:216 — amount 存 千元, 需 ×1000 → 元
            records[j]['transAmount'] = np.uint64(round(amt * 1000))
            # store.py:216 — volume 存 手, 需 ×100 → 股
            records[j]['transCount'] = np.uint64(round(vol * 100))

        # 4. 写入 HDF5 dataset (命名: 市场代码 — 如 SH600000)
        if not dry_run:
            dataset_name = f"{market}{code}"
            data_group.create_dataset(dataset_name, data=records, compression='gzip',
                                      compression_opts=4, shuffle=True)

        exported += 1
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{market}] 进度: {i+1}/{len(codes)} ({exported}导出, {skipped}跳过) {elapsed:.0f}s")

    if not dry_run:
        h5.close()

    elapsed = time.time() - t0
    size_mb = os.path.getsize(h5_path) / 1024 / 1024 if not dry_run else 0
    print(f"[{market}] 完成: {exported}只导出, {skipped}只跳过, {elapsed:.0f}s, {size_mb:.1f}MB")

    conn.close()
    return exported


def export_stock_metadata():
    """导出股票元数据到 Hikyuu stock.db。

    精确匹配 Hikyuu 官方 schema:
      market 表: StockTable.h:23  — marketid,market,name,description,code,lastDate,openTime1,closeTime1,openTime2,closeTime2
      stock 表:  StockTable.h:79 — stockid(自增),marketid,code,name,type,valid,startDate,endDate
    """
    import sqlite3 as _sql

    src_db = os.path.join(QUANT_ROOT, 'data', 'market.db')
    dst_db = os.path.join(HKU_DATA, 'stock.db')

    if not os.path.exists(src_db):
        print("market.db 不存在, 跳过元数据导出")
        return

    src = _sql.connect(src_db)
    dst = _sql.connect(dst_db)

    # 删除旧表并重建 (匹配 Hikyuu 官方 schema)
    dst.execute("DROP TABLE IF EXISTS market")
    dst.execute("""CREATE TABLE market (
        marketid INTEGER PRIMARY KEY, market TEXT, name TEXT, description TEXT,
        code TEXT, lastDate INTEGER, openTime1 INTEGER, closeTime1 INTEGER,
        openTime2 INTEGER, closeTime2 INTEGER)""")
    dst.execute("INSERT INTO market VALUES(1,'SH','上海A股','上海证券交易所','SH',20260624,930,1130,1300,1500)")
    dst.execute("INSERT INTO market VALUES(2,'SZ','深圳A股','深圳证券交易所','SZ',20260624,930,1130,1300,1500)")

    # 股票表 (匹配 StockTable.h 官方 schema)
    dst.execute("DROP TABLE IF EXISTS stock")
    dst.execute("""CREATE TABLE stock (
        stockid INTEGER PRIMARY KEY AUTOINCREMENT, marketid INTEGER, code TEXT,
        name TEXT, type INTEGER, valid INTEGER, startDate INTEGER, endDate INTEGER)""")

    # 导出股票列表 (从 market.db stocks 表)
    stocks = src.execute("SELECT symbol, name, market, list_date FROM stocks").fetchall()
    for symbol, name, mkt, list_date in stocks:
        mkt_id = 1 if mkt in ('SH', 'SSE') else 2
        # 上市日期: "19910403" → 19910403 (uint64)
        start_date = int(list_date) if list_date and list_date.isdigit() else 19900101
        valid = 1  # 默认有效 (实际应根据退市状态判断)
        dst.execute(
            "INSERT INTO stock(marketid, code, name, type, valid, startDate, endDate) VALUES(?,?,?,1,?,?,99999999)",
            (mkt_id, symbol, name, valid, start_date))

    dst.commit()
    print(f"元数据: {len(stocks)}只股票 → {dst_db}")
    src.close()
    dst.close()


# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='P0a: SQLite → Hikyuu HDF5 数据导出')
    parser.add_argument('--start', type=str, default=None, help='起始日期 YYYY-MM-DD')
    parser.add_argument('--dry-run', action='store_true', help='不写文件, 仅统计')
    parser.add_argument('--market', type=str, default='all', choices=['SH', 'SZ', 'all'])
    ARGS = parser.parse_args()

    print("=" * 50)
    print("P0a: market.db (SQLite) → Hikyuu HDF5")
    print(f"  源: {QUANT_ROOT}/data/market.db")
    print(f"  目标: {HKU_DATA}/sh_day.h5 + sz_day.h5")
    if ARGS.dry_run:
        print("  *** DRY RUN — 不写入文件 ***")
    print("=" * 50)

    t_start = time.time()

    if ARGS.market in ('SH', 'all'):
        export_market('SH')
    if ARGS.market in ('SZ', 'all'):
        export_market('SZ')

    if not ARGS.dry_run:
        export_stock_metadata()

    t_elapsed = time.time() - t_start
    print(f"\n导出完成: {t_elapsed:.0f}s")
    if not ARGS.dry_run:
        print(f"验证: PYTHONPATH=. python -c \"from hikyuu.interactive import *; print(sm['sh600000'])\"")
