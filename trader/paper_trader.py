#!/usr/bin/env python3
"""模拟交易引擎 — 实时行情驱动, strategy_core 共享逻辑。

数据流:
  Sina 实时行情 → 内存缓存 → strategy_core.detect_signals()
    → strategy_core.compute_factor_multiplier()
    → strategy_core.calc_position_size()
    → 模拟成交 → 记录

与回测使用同一份 strategy_core, 信号/因子/仓位计算完全一致。
"""

import os, sys, json, time, sqlite3, argparse, math
from datetime import date, datetime
import requests
import numpy as np

SUPERQUANT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANT_ROOT = os.path.expanduser("~/project/quant")
sys.path.insert(0, SUPERQUANT_ROOT)
sys.path.insert(0, QUANT_ROOT)

from engine.strategy_core import (
    detect_signals, compute_factor_multiplier, calc_position_size,
    calc_adaptive_stop, calc_take_profit, generate_daily_returns,
    is_broken_board, count_boards,
)

TRADE_DB = os.path.join(QUANT_ROOT, "data", "trades.db")
ACTIVE_PARAMS = os.path.join(SUPERQUANT_ROOT, "config", "active_params.json")
INITIAL_CAPITAL = 5000.0
COMMISSION = 0.0003
STAMP_TAX = 0.001
SINA_URL = "http://hq.sinajs.cn/list="

# ── 配置 ──
SIGNAL_SCORES = {'弱转强': 0.90, '首阴反包': 0.85, '连板接力': 0.70, '首板试探': 0.30}
ENTRY_THRESHOLD = 0.30  # 信号分×因子乘数 最低阈值
TRACKED_COUNT = 200     # 因子候选池大小


def load_active_params():
    if os.path.exists(ACTIVE_PARAMS):
        with open(ACTIVE_PARAMS) as f:
            return json.load(f)
    return None


def fetch_quotes(symbols):
    if not symbols:
        return {}
    codes = [f"{'sh' if s.startswith('6') else 'sz'}{s}" for s in symbols]
    url = SINA_URL + ",".join(codes[:200])
    try:
        resp = requests.get(url, timeout=15, headers={"Referer": "https://finance.sina.com.cn"})
        resp.encoding = 'gb2312'
        results = {}
        for line in resp.text.strip().split('\n'):
            if '=' not in line: continue
            var, data = line.split('=', 1)
            code = var.split('_')[-1]
            fields = data.strip('";').split(',')
            if len(fields) < 32 or not fields[1]: continue
            sym = code[2:]
            results[sym] = {
                'open': float(fields[1]), 'prev_close': float(fields[2]),
                'price': float(fields[3]), 'high': float(fields[4]),
                'low': float(fields[5]), 'volume': float(fields[8]),
                'amount': float(fields[9]),
            }
        return results
    except Exception as e:
        print(f"  [warn] 行情获取: {e}")
        return {}


def init_account(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS paper_account (
        id INTEGER PRIMARY KEY, date TEXT, cash REAL, equity REAL, updated_at TEXT)""")
    today = date.today().isoformat()
    row = conn.execute("SELECT cash FROM paper_account WHERE date=? ORDER BY id DESC LIMIT 1",
                       (today,)).fetchone()
    if not row:
        conn.execute("INSERT INTO paper_account(date, cash, equity, updated_at) VALUES(?,?,?,?)",
                     (today, INITIAL_CAPITAL, INITIAL_CAPITAL, datetime.now().isoformat()))
        conn.commit()
        return INITIAL_CAPITAL
    return row[0]


def record_trade(conn, symbol, side, price, shares, pnl=0, pnl_pct=0, capital_after=0, reason=""):
    conn.execute("""INSERT INTO sim_trades(date, symbol, side, price, shares, pnl, pnl_pct,
                   capital_after, strategy, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                 (date.today().isoformat(), symbol, side, price, shares,
                  round(pnl, 2), round(pnl_pct, 4), round(capital_after, 2),
                  'superquant', datetime.now().isoformat()))
    conn.commit()


def get_tracked_symbols():
    """因子 Top N 候选池。"""
    try:
        from hikyuu.interactive import sm, Query
        from app.factors import compute_factor_scores
        stocks = []
        for mkt in ['SH', 'SZ']:
            try:
                market_stocks = sm.get_stock_list(lambda s, m=mkt: s.market == m and s.valid)
                stocks.extend(list(market_stocks))
            except: pass
        scores = compute_factor_scores(stocks, Query(-30))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [code for code, _ in ranked[:TRACKED_COUNT]]
    except Exception as e:
        print(f"  [warn] 因子池: {e}")
        return ['SH600000', 'SZ000001', 'SH600036', 'SZ000002', 'SH600519']


def build_memory_kdata(history, new_quote):
    """从历史缓存+最新报价构建伪K线用于 detect_signals。"""
    # history = [(open, high, low, close, volume), ...]
    op, hi, lo, cl, vol = new_quote['open'], new_quote['high'], new_quote['low'], new_quote['price'], new_quote['volume']
    dt = datetime.now()  # 占位
    records = []
    for h in history:
        records.append((dt, h[0], h[1], h[2], h[3], h[4]))
    records.append((dt, op, hi, lo, cl, vol))
    return records


def run_scan(conn, capital, positions, tracked, history_cache):
    """单次扫描 — 全部通过 strategy_core。"""
    quotes = fetch_quotes(tracked)
    if not quotes:
        return capital, positions

    params = load_active_params()
    factor_params = params.get('factor_weights', {}) if params else {}
    kelly_params = params.get('kelly', {}) if params else {}
    stop_params = params.get('stops', {}) if params else {}
    signal_params = params.get('signal', {}).get('weak_to_strong', {}) if params else {}
    today_positions = {p['symbol'] for p in positions}
    signals_triggered = []

    # 1) 信号检测
    for sym, q in quotes.items():
        if q['open'] <= 0 or q['prev_close'] <= 0: continue
        # 涨停不买
        if q['price'] >= q['prev_close'] * 1.095: continue
        # 跌停不卖
        if q['price'] <= q['prev_close'] * 0.905: continue

        # 构建伪K线
        history = history_cache.get(sym, [])
        records = build_memory_kdata(history, q)
        signals = detect_signals(records, signal_params)
        for dt, sig_type, score in signals[-1:]:  # 只看最新
            factor_score = get_factor_score(sym)
            multiplier = compute_factor_multiplier(factor_score)
            final_score = score * multiplier
            if final_score >= ENTRY_THRESHOLD:
                signals_triggered.append({
                    'symbol': sym, 'price': q['price'], 'type': sig_type,
                    'signal_score': score, 'factor_score': factor_score,
                    'multiplier': multiplier, 'final_score': final_score,
                })

    # 2) 买入: 信号排序 → Kelly 仓位
    signals_triggered.sort(key=lambda s: s['final_score'], reverse=True)
    for s in signals_triggered:
        sym = s['symbol']
        if sym in today_positions: continue
        price = s['price']
        stop_price = calc_adaptive_stop(price, [], stop_params)
        risk_per_share = max(price - stop_price, 0.01)
        shares = calc_position_size(capital, price, risk_per_share, kelly_params)
        if shares < 100: continue
        cost = shares * price * (1 + COMMISSION)
        if cost > capital: continue
        capital -= cost
        positions.append({
            'symbol': sym, 'price': price, 'shares': shares,
            'date': date.today().isoformat(), 'mode': s['type'],
            'factor': s['factor_score'], 'peak': price,
        })
        record_trade(conn, sym, 'buy', price, shares, capital_after=capital,
                     reason=f"{s['type']}+因子{s['factor_score']:+.2f}")
        print(f"  🟢 买 {sym} {s['type']} ¥{price:.2f} {shares}股 (信号{s['signal_score']}×因子{s['multiplier']:.2f})")

    # 3) 止盈止损
    to_sell = []
    for i, pos in enumerate(positions):
        sym = pos['symbol']
        if sym not in quotes: continue
        q = quotes[sym]
        pnl_pct = (q['price'] / pos['price'] - 1) * 100

        # 更新 peak
        pos['peak'] = max(pos.get('peak', pos['price']), q['high'])

        # 止损
        stop_px = calc_adaptive_stop(pos['price'], [], stop_params)
        if q['price'] <= stop_px:
            to_sell.append((i, f'止损({pnl_pct:+.1f}%)', pnl_pct))
            continue

        # 止盈
        trigger = calc_take_profit(pos['price'], pos.get('peak'), q['price'], stop_params)
        if trigger and pnl_pct > 0:
            to_sell.append((i, f'移动止盈({pnl_pct:+.1f}%)', pnl_pct))

    for i, reason, pnl_pct in reversed(to_sell):
        pos = positions.pop(i)
        sell_val = pos['shares'] * quotes[pos['symbol']]['price']
        fee = sell_val * (COMMISSION + STAMP_TAX)
        pnl = sell_val - pos['shares'] * pos['price'] - fee
        capital += sell_val - fee
        record_trade(conn, pos['symbol'], 'sell', quotes[pos['symbol']]['price'],
                     pos['shares'], pnl=pnl, pnl_pct=pnl_pct, capital_after=capital,
                     reason=reason)
        print(f"  🔴 卖 {pos['symbol']} {reason} PnL=¥{pnl:.0f}")

    # 4) 更新历史缓存
    for sym, q in quotes.items():
        if sym not in history_cache:
            history_cache[sym] = []
        history_cache[sym].append((q['open'], q['high'], q['low'], q['price'], q['volume']))
        if len(history_cache[sym]) > 60:
            history_cache[sym] = history_cache[sym][-60:]

    return capital, positions


def get_factor_score(symbol):
    try:
        from hikyuu.interactive import sm, Query
        from app.factors import compute_factor_scores
        scores = compute_factor_scores([sm[symbol]], Query(-30))
        return scores.get(symbol, 0.0)
    except:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description='superquant 模拟交易')
    parser.add_argument('--live', action='store_true', help='持续运行')
    parser.add_argument('--interval', type=int, default=60, help='扫描间隔(秒)')
    ARGS = parser.parse_args()

    print("=" * 60)
    print("superquant 模拟交易 (strategy_core 驱动)")
    print(f"  启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  初始资金: ¥{INITIAL_CAPITAL:,.0f}")
    print("=" * 60)

    conn = sqlite3.connect(TRADE_DB)
    capital = init_account(conn)
    positions = []
    history_cache = {}

    print("  加载因子候选池...")
    tracked = get_tracked_symbols()
    print(f"  监控池: {len(tracked)}只 (因子 Top{TRACKED_COUNT})")

    if ARGS.live:
        while True:
            now = datetime.now()
            if now.hour >= 15 or (now.hour == 11 and now.minute >= 30):
                break
            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                time.sleep(30)
                continue
            capital, positions = run_scan(conn, capital, positions, tracked, history_cache)
            print(f"  [{now.strftime('%H:%M:%S')}] 资金=¥{capital:.0f}, 持仓={len(positions)}")
            time.sleep(ARGS.interval)
    else:
        capital, positions = run_scan(conn, capital, positions, tracked, history_cache)

    equity = capital + sum(p['shares'] * p['price'] * 0.99 for p in positions)
    print(f"\n  资金=¥{capital:.0f}, 持仓={len(positions)}, 权益=¥{equity:.0f}")
    print(f"  累计收益: {(equity/INITIAL_CAPITAL-1)*100:+.1f}%")
    conn.close()
    print("✅ 模拟交易结束")
    return 0


if __name__ == '__main__':
    exit(main())
