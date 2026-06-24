"""特征工程: 五层免费数据 → 特征向量。

来源:
  L1竞价:   market.db 日线 (开盘数据代理)
  L2技术:   HDF5 5分钟线
  L3资金:   easy-tdx get_fund_flow
  L4龙虎:   akshare stock_lhb_detail_daily_sina
  L5板块:   market.db 日线 (板块涨停统计)

输出: (N, 28) 特征矩阵
"""

import os, sys, sqlite3, time
from collections import defaultdict
import numpy as np

QUANT_ROOT = os.path.expanduser("~/project/quant")
DB_PATH = os.path.join(QUANT_ROOT, "data", "market.db")
HDF5_DIR = os.path.expanduser("~/stock")

FEATURE_NAMES = [
    # ── 日线基础 (10个) ──
    'ret_1d', 'ret_5d', 'vol_ratio', 'vol_5d', 'gap', 'turnover',
    'amt_log', 'hl_ratio', 'close_pos', 'ma_dev',
    # ── L2 技术形态 (6个) ──
    'ksft', 'slope_5min', 'ptc_5min', 'vol_5min', 'max_ret_5min', 'min_ret_5min',
    # ── L3 资金流 (4个) ──
    'main_net_in', 'main_net_ratio', 'super_large_in', 'large_in',
    # ── L4 龙虎榜 (4个) ──
    'lhb_net_buy', 'lhb_buy_ratio', 'lhb_count', 'lhb_exists',
    # ── L5 板块 (4个) ──
    'sector_limit_count', 'sector_turnover_ratio', 'sector_rank', 'is_leader',
]


def build_daily_features(rows_sym, i):
    """日线特征 (market.db)。"""
    closes = np.array([float(r[5]) for r in rows_sym])
    volumes = np.array([float(r[6]) for r in rows_sym])
    opens = np.array([float(r[2]) for r in rows_sym])
    highs = np.array([float(r[3]) for r in rows_sym])
    lows = np.array([float(r[4]) for r in rows_sym])
    amounts = np.array([float(r[7]) for r in rows_sym])

    if closes[i-1] <= 0:
        return None

    ret_1d = closes[i] / closes[i-1] - 1
    ret_5d = closes[i] / closes[i-5] - 1 if i >= 5 and closes[i-5] > 0 else 0
    vol_5d_avg = np.mean(volumes[i-5:i]) if i >= 5 else volumes[i]
    vol_ratio = volumes[i] / max(vol_5d_avg, 1)
    rets_5d = [(closes[j]/closes[j-1]-1) for j in range(i-4, i+1) if closes[j-1] > 0]
    vol_5d = np.std(rets_5d) if len(rets_5d) > 2 else 0
    gap = opens[i] / closes[i-1] - 1
    turnover = float(rows_sym[i][8]) if rows_sym[i][8] else volumes[i] / 10000.0
    amt_log = np.log(max(amounts[i], 1))
    hl_ratio = highs[i] / max(lows[i], 0.01) - 1
    close_pos = (closes[i] - lows[i]) / max(highs[i] - lows[i], 0.01)
    ma20 = np.mean(closes[max(0, i-20):i+1])
    ma_dev = closes[i] / ma20 - 1 if ma20 > 0 else 0

    return [ret_1d, ret_5d, vol_ratio, vol_5d, gap, turnover, amt_log, hl_ratio, close_pos, ma_dev]


def build_technical_features(symbol, date_str):
    """L2 技术形态: HDF5 5分钟线 (P0b)。"""
    try:
        import h5py
        mkt = 'sh' if symbol.startswith(('6','5','9')) else 'sz'
        path = os.path.join(HDF5_DIR, f'{mkt}_5min.h5')
        if not os.path.exists(path):
            return [0] * 6
        with h5py.File(path, 'r') as h5:
            ds_name = f"{'SH' if mkt=='sh' else 'SZ'}{symbol}"
            if ds_name not in h5['data']:
                return [0] * 6
            ds = h5['data'][ds_name]
            # 取最近 48 条 (一天)
            records = ds[-48:] if len(ds) >= 48 else ds[:]
            if len(records) < 10:
                return [0] * 6
            closes = np.array([r['closePrice']/1000.0 for r in records])
            highs_arr = np.array([r['highPrice']/1000.0 for r in records])
            lows_arr = np.array([r['lowPrice']/1000.0 for r in records])
            volumes_arr = np.array([r['transCount'] for r in records], dtype=float)

            # KSFT: K线位移 (高-低)/开盘跌幅
            opens_arr = np.array([r['openPrice']/1000.0 for r in records])
            ksft = np.mean((highs_arr - lows_arr) / np.maximum(np.abs(opens_arr - closes[0]), 0.01))

            # SLOPE: 最近 10 根 5分钟线的线性回归斜率
            x = np.arange(min(10, len(closes)))
            y = closes[-len(x):]
            slope = np.polyfit(x, y, 1)[0] if len(x) >= 5 else 0

            # PTC: 价格-成交额相关性
            amounts_5min = closes * volumes_arr
            ptc = np.corrcoef(closes[-20:], amounts_5min[-20:])[0,1] if len(closes) >= 20 else 0
            ptc = 0 if np.isnan(ptc) else ptc

            # 分钟波动率/最大最小收益
            rets_5min = np.diff(closes[-48:]) / closes[-49:-1]
            vol_5min = np.std(rets_5min) if len(rets_5min) > 3 else 0
            max_ret_5min = np.max(rets_5min) if len(rets_5min) > 0 else 0
            min_ret_5min = np.min(rets_5min) if len(rets_5min) > 0 else 0

            return [ksft, slope, ptc, vol_5min, max_ret_5min, min_ret_5min]
    except Exception:
        return [0] * 6


def build_fundflow_features(symbol, date_str):
    """L3 资金流: easy-tdx。"""
    try:
        import easy_tdx as et
        client = et.TdxClient()
        client.connect()
        mkt = et.Market.SZ if symbol.startswith(('0','3')) else et.Market.SH
        df = client.get_history_fund_flow(mkt, symbol, start=date_str, count=1)
        client.close()
        if df is not None and len(df) > 0:
            row = df.iloc[-1]
            super_in = float(row.get('super_in', 0) or 0)
            large_in = float(row.get('large_in', 0) or 0)
            super_out = float(row.get('super_out', 0) or 0)
            large_out = float(row.get('large_out', 0) or 0)
            main_net_in = (super_in + large_in) - (super_out + large_out)
            total_flow = abs(super_in) + abs(large_in) + abs(super_out) + abs(large_out) + 1
            main_net_ratio = main_net_in / total_flow
            return [main_net_in, main_net_ratio, super_in, large_in]
        return [0, 0, 0, 0]
    except Exception:
        return [0, 0, 0, 0]


def build_lhb_features(symbol, date_str):
    """L4 龙虎榜: AKShare。"""
    try:
        import akshare as ak
        # 取前一天的龙虎榜数据
        from datetime import datetime, timedelta
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        prev = (dt - timedelta(days=1)).strftime('%Y%m%d')
        df = ak.stock_lhb_detail_daily_sina(date=prev)
        if df is None or len(df) == 0:
            return [0, 0, 0, 0]
        stock_rows = df[df['股票代码'] == symbol]
        if len(stock_rows) == 0:
            return [0, 0, 0, 0]
        buy = stock_rows['成交额'].astype(float).sum()  # simplified
        return [buy, 0.5, len(stock_rows), 1]  # net_buy, ratio, count, exists
    except Exception:
        return [0, 0, 0, 0]


def build_sector_features(symbol, date_str, limit_counts):
    """L5 板块: 同板块涨停数。limit_counts = {sector: count}。"""
    # 简化: 先返回默认值, 后续可从 AData 获取板块归属
    try:
        # 从 market.db 间接计算: 同行业股票的涨停数
        conn = sqlite3.connect(DB_PATH)
        # 取股票的行业代码 (从 stocks 表)
        row = conn.execute("SELECT market FROM stocks WHERE symbol=?", (symbol,)).fetchone()
        conn.close()
        if row:
            mkt = row[0]
            sector_count = limit_counts.get(mkt, 0)
            return [sector_count, 0, 0, 0]
    except Exception:
        pass
    return [0, 0, 0, 0]


def build_all_features(daily_data, limit_counts):
    """构建完整特征矩阵。

    Args:
        daily_data: {symbol: [(date,open,high,low,close,volume,amount,turnover),...]}
        limit_counts: {sector_name: count}

    Returns:
        (X, y, symbols_dates) 或 (None, None, None)
    """
    print("  计算日线特征...")
    X_list, y_list, meta = [], [], []
    n = 0
    for sym, rs in daily_data.items():
        if len(rs) < 32:
            continue
        closes = np.array([float(r[4]) for r in rs])  # close=index4 in (date,o,h,l,c,v,a,t)
        n += 1
        if n % 500 == 0:
            print(f"    {n}只...")

        for i in range(25, len(rs) - 1):
            date_str = rs[i][0]  # date
            if closes[i-1] <= 0:
                continue

            # 日线特征 (基础)
            base = _build_daily_from_tuple(rs, i)
            if base is None:
                continue

            # 标签
            next_ret = closes[i+1] / closes[i] - 1 if closes[i] > 0 else 0
            label = 1 if next_ret >= 0.095 else 0

            feat = base  # 先用日线, L2-L4 每个样本单独计算太慢
            if any(abs(v) > 100 for v in feat):
                continue
            X_list.append(feat)
            y_list.append(label)
            meta.append((sym, date_str))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"  完成: {X.shape}, 正样本: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    return X, y, meta


def _build_daily_from_tuple(rs, i):
    """从 market.db 行元组构建日线特征。rs=(date,open,high,low,close,volume,amount,turnover)"""
    closes = np.array([float(r[4]) for r in rs])
    volumes = np.array([float(r[5]) for r in rs])
    opens = np.array([float(r[1]) for r in rs])
    highs = np.array([float(r[2]) for r in rs])
    lows = np.array([float(r[3]) for r in rs])
    amounts = np.array([float(r[6]) for r in rs])

    if closes[i-1] <= 0:
        return None

    ret_1d = closes[i]/closes[i-1]-1
    ret_5d = closes[i]/closes[i-5]-1 if i>=5 and closes[i-5]>0 else 0
    vol_5d_avg = np.mean(volumes[i-5:i]) if i>=5 else volumes[i]
    vol_ratio = volumes[i]/max(vol_5d_avg,1)
    rets_5d = [(closes[j]/closes[j-1]-1) for j in range(i-4,i+1) if closes[j-1]>0]
    vol_5d = np.std(rets_5d) if len(rets_5d)>2 else 0
    gap = opens[i]/closes[i-1]-1
    turnover = float(rs[i][7]) if rs[i][7] else volumes[i]/10000.0
    amt_log = np.log(max(amounts[i],1))
    hl_ratio = highs[i]/max(lows[i],0.01)-1
    close_pos = (closes[i]-lows[i])/max(highs[i]-lows[i],0.01)
    ma20 = np.mean(closes[max(0,i-20):i+1])
    ma_dev = closes[i]/ma20-1 if ma20>0 else 0

    return [ret_1d,ret_5d,vol_ratio,vol_5d,gap,turnover,amt_log,hl_ratio,close_pos,ma_dev]


def get_l2_features(symbol, date_str):
    """单只股票的 L2 技术形态特征。"""
    return build_technical_features(symbol, date_str)


def get_l3_features(symbol, date_str):
    """单只股票的 L3 资金流特征。"""
    return build_fundflow_features(symbol, date_str)


def get_l4_features(symbol, date_str):
    """单只股票的 L4 龙虎榜特征。"""
    return build_lhb_features(symbol, date_str)


if __name__ == '__main__':
    # 测试
    print("测试 L2 技术特征: SH600000")
    f2 = get_l2_features('600000', '2026-06-23')
    print(f"  {f2}")
    print("测试 L3 资金流: 000001")
    f3 = get_l3_features('000001', '2026-06-23')
    print(f"  {f3}")
    print("测试 L4 龙虎榜: 000001")
    f4 = get_l4_features('000001', '2026-06-23')
    print(f"  {f4}")
