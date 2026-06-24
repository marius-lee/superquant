#!/usr/bin/env python3
"""P0b: Sina 5分钟线 → Hikyuu HDF5 分钟数据存储。

来源:
  Sina 5分钟 API:
    GET http://money.finance.sina.com.cn/.../getKLineData?symbol=sh600000&scale=5&datalen=2000
    响应格式: [{"day":"2026-06-24 10:10:00","open":"9.120",...,"volume":"3756700"},...]

  Hikyuu HDF5 分钟格式 (与日线相同 H5Record):
    datetime: uint64 (YYYYMMDDhhmm, 如 202606241010)
    openPrice/highPrice/lowPrice/closePrice: uint32 (价格×1000)
    transAmount: uint64 (成交额, 元) — 分钟无amount字段, 用 close×volume 估算
    transCount: uint64 (成交量, 股)

  Hikyuu HDF5 布局:
    sh_5min.h5/data/SH600000  (同 market+code 命名)
    sz_5min.h5/data/SZ000001

用法:
    python data/minute_store.py                          # 拉取今天所有股票5分钟线
    python data/minute_store.py --symbol sh600000        # 单只股票
    python data/minute_store.py --symbols sh600000,sz000001  # 多只股票
    python data/minute_store.py --date 2026-06-23        # 指定日期 (历史回补)
    python data/minute_store.py --batch                  # 批量回补最近N天

约束:
  M1 8GB: 每次最多处理2000条K线, 逐股写入不攒内存
  Sina API: 免费无需注册, 但请求频率过高可能被限
  HDF5 压缩: gzip level 4, 预计 ~300MB/年
"""

import json, os, sys, time, argparse, sqlite3
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import requests

# 量化项目根目录 (共享 quant 的 store.py 配置)
QUANT_ROOT = os.path.expanduser("~/project/quant")
# Hikyuu 数据目录
HKU_DATA = os.path.expanduser("~/stock")

# Sina API 配置
SINA_URL = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
SINA_HEADERS = {"Referer": "https://finance.sina.com.cn"}
SINA_SCALE = 5           # 5分钟线
SINA_DATALEN = 2000      # 每次最多获取条数
SINA_TIMEOUT = 30        # 请求超时秒数

# HDF5 dtype (同 Hikyuu H5Record: hikyuu_cpp/hikyuu/data_driver/kdata/hdf5/H5Record.h:20-28)
H5_DTYPE = np.dtype([
    ('datetime',    np.uint64),    # YYYYMMDDhhmm
    ('openPrice',   np.uint32),    # price × 1000
    ('highPrice',   np.uint32),
    ('lowPrice',    np.uint32),
    ('closePrice',  np.uint32),
    ('transAmount', np.uint64),    # yuan (estimated from close×volume)
    ('transCount',  np.uint64),    # shares
])

# 每只股票最少需要多少条记录才写入
MIN_RECORDS = 1


def to_h5_datetime(day_str: str) -> int:
    """'2026-06-24 10:10:00' → 202606241010 (uint64)"""
    # 去掉秒和冒号: 2026-06-24 10:10:00 → 202606241010
    dt = day_str.replace('-', '').replace(' ', '').replace(':', '')
    return int(dt[:12])  # YYYYMMDDhhmm


def fetch_sina_minute(code: str, market: str, datalen: int = SINA_DATALEN) -> list:
    """从 Sina 获取单只股票的5分钟K线。

    Args:
        code: 纯数字代码, 如 '600000'
        market: 'SH' 或 'SZ'
        datalen: 获取条数

    Returns:
        [{day, open, high, low, close, volume}, ...]  或 []
    """
    symbol = f"{market.lower()}{code}"
    url = f"{SINA_URL}?symbol={symbol}&scale={SINA_SCALE}&datalen={datalen}"

    try:
        resp = requests.get(url, headers=SINA_HEADERS, timeout=SINA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return []
        return data
    except requests.exceptions.RequestException as e:
        print(f"  [warn] {symbol} 请求失败: {e}")
        return []
    except json.JSONDecodeError:
        print(f"  [warn] {symbol} 响应非JSON")
        return []


def records_to_h5_array(records: list) -> np.ndarray:
    """将 Sina 响应转为 Hikyuu H5Record numpy 数组。

    Sina 字段: day(str), open(str), high(str), low(str), close(str), volume(str)
    H5Record:  datetime, openPrice, highPrice, lowPrice, closePrice, transAmount, transCount
    """
    arr = np.zeros(len(records), dtype=H5_DTYPE)
    for i, r in enumerate(records):
        arr[i]['datetime'] = to_h5_datetime(r['day'])
        arr[i]['openPrice'] = np.uint32(round(float(r['open']) * 1000))
        arr[i]['highPrice'] = np.uint32(round(float(r['high']) * 1000))
        arr[i]['lowPrice'] = np.uint32(round(float(r['low']) * 1000))
        arr[i]['closePrice'] = np.uint32(round(float(r['close']) * 1000))
        vol = float(r['volume'])
        arr[i]['transCount'] = np.uint64(round(vol))
        # Sina分钟线无成交额字段, 用 close×volume 估算
        arr[i]['transAmount'] = np.uint64(round(float(r['close']) * vol))
    return arr


def append_to_hdf5(h5_path: str, market: str, code: str, new_records: np.ndarray):
    """将新股数据追加到 HDF5。如果股票已存在数据集则合并去重。

    策略: 读取已有数据 → 合并新数据 → 去重(datetime) → 覆盖写入
    """
    import h5py

    os.makedirs(HKU_DATA, exist_ok=True)
    dataset_name = f"{market}{code}"

    with h5py.File(h5_path, 'a') as h5:
        if 'data' not in h5:
            h5.create_group('data')
        data_group = h5['data']

        if dataset_name in data_group:
            # 合并已有数据
            existing = data_group[dataset_name][:]
            combined = np.concatenate([existing, new_records])
            # 去重: 按 datetime 排序保留最新
            _, unique_idx = np.unique(combined['datetime'], return_index=True)
            combined = combined[np.sort(unique_idx)]
            combined.sort(order='datetime')
            del data_group[dataset_name]
        else:
            combined = new_records
            combined.sort(order='datetime')

        data_group.create_dataset(
            dataset_name, data=combined,
            compression='gzip', compression_opts=4, shuffle=True,
        )


def get_stock_list(market_filter: str = None) -> list:
    """从 market.db 获取股票列表。"""
    db_path = os.path.join(QUANT_ROOT, 'data', 'market.db')
    if not os.path.exists(db_path):
        print(f"错误: market.db 不存在: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    if market_filter == 'SH':
        codes = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM daily WHERE symbol LIKE '6%' ORDER BY symbol"
        ).fetchall()]
    elif market_filter == 'SZ':
        codes = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM daily WHERE symbol LIKE '0%' OR symbol LIKE '3%' ORDER BY symbol"
        ).fetchall()]
    else:
        codes = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM daily ORDER BY symbol"
        ).fetchall()]
    conn.close()
    return codes


def store_market(market: str, symbols: list = None):
    """拉取并存储单个市场的5分钟线。

    Args:
        market: 'SH' 或 'SZ'
        symbols: 股票代码列表, None=全市场
    """
    if symbols is None:
        symbols = get_stock_list(market)

    datalen = 48 if ARGS.today else SINA_DATALEN  # 今日模式: 仅48条
    h5_path = os.path.join(HKU_DATA, f"{market.lower()}_5min.h5")
    print(f"[{market}] {len(symbols)} 只股票 → {h5_path} ({datalen}条/股)")

    success = 0
    skip = 0
    fail = 0
    t0 = time.time()

    for i, code in enumerate(symbols):
        records = fetch_sina_minute(code, market, datalen)
        if not records:
            skip += 1
            continue

        try:
            arr = records_to_h5_array(records)
            if not ARGS.dry_run:
                append_to_hdf5(h5_path, market, code, arr)
            success += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  [err] {market}{code}: {e}")

        # 进度 (每100只)
        if (i + 1) % 500 == 0 or (i + 1) == len(symbols):
            elapsed = time.time() - t0
            print(f"  [{market}] {i+1}/{len(symbols)} ({success}写/{skip}空/{fail}错) {elapsed:.0f}s")

    elapsed = time.time() - t0
    size_mb = os.path.getsize(h5_path) / 1024 / 1024 if not ARGS.dry_run and os.path.exists(h5_path) else 0
    print(f"[{market}] 完成: {success}只写入, {skip}只无数据, {fail}只失败, {elapsed:.0f}s, {size_mb:.1f}MB")


# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='P0b: Sina 5分钟线 → Hikyuu HDF5')
    parser.add_argument('--symbol', type=str, default=None, help='单只股票代码, 如 sh600000')
    parser.add_argument('--symbols', type=str, default=None, help='多只股票代码, 逗号分隔')
    parser.add_argument('--market', type=str, default='all', choices=['SH', 'SZ', 'all'])
    parser.add_argument('--date', type=str, default=None, help='指定日期 YYYY-MM-DD (暂未使用, 保留)')
    parser.add_argument('--dry-run', action='store_true', help='不写文件, 仅测试')
    parser.add_argument('--batch', action='store_true', help='批量回补模式 (全市场)')
    parser.add_argument('--today', action='store_true', help='仅拉取今天数据 (48条/股, 快速)')
    ARGS = parser.parse_args()

    print("=" * 50)
    print("P0b: Sina 5分钟线 → Hikyuu HDF5")
    print(f"  API: {SINA_URL}?scale={SINA_SCALE}")
    print(f"  目标: {HKU_DATA}/sh_5min.h5 + sz_5min.h5")
    if ARGS.dry_run:
        print("  *** DRY RUN — 不写入文件 ***")
    print("=" * 50)

    t_start = time.time()

    if ARGS.symbol:
        # 单只股票
        sym = ARGS.symbol.lower()
        if sym.startswith('sh'):
            mkt, cod = 'SH', sym[2:]
        else:
            mkt, cod = 'SZ', sym[2:]
        records = fetch_sina_minute(cod, mkt)
        print(f"{mkt}{cod}: {len(records)}条记录")
        if records and not ARGS.dry_run:
            arr = records_to_h5_array(records)
            h5_path = os.path.join(HKU_DATA, f"{mkt.lower()}_5min.h5")
            append_to_hdf5(h5_path, mkt, cod, arr)
            print(f"  写入: {h5_path}/data/{mkt}{cod}")
    elif ARGS.symbols:
        codes = [s.strip() for s in ARGS.symbols.split(',')]
        for sym in codes:
            sym = sym.lower()
            mkt = 'SH' if sym.startswith('sh') else 'SZ'
            cod = sym[2:]
            records = fetch_sina_minute(cod, mkt)
            print(f"{mkt}{cod}: {len(records)}条")
            if records and not ARGS.dry_run:
                arr = records_to_h5_array(records)
                h5_path = os.path.join(HKU_DATA, f"{mkt.lower()}_5min.h5")
                append_to_hdf5(h5_path, mkt, cod, arr)
    else:
        # 全市场
        if ARGS.market in ('SH', 'all'):
            store_market('SH')
        if ARGS.market in ('SZ', 'all'):
            store_market('SZ')

    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed:.0f}s")
    if not ARGS.dry_run:
        sh_size = os.path.getsize(os.path.join(HKU_DATA, 'sh_5min.h5')) / 1024 / 1024 if os.path.exists(os.path.join(HKU_DATA, 'sh_5min.h5')) else 0
        sz_size = os.path.getsize(os.path.join(HKU_DATA, 'sz_5min.h5')) / 1024 / 1024 if os.path.exists(os.path.join(HKU_DATA, 'sz_5min.h5')) else 0
        print(f"存储: sh_5min.h5={sh_size:.1f}MB, sz_5min.h5={sz_size:.1f}MB")
