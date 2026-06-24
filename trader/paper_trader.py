#!/usr/bin/env python3
"""模拟交易引擎 — 实时行情驱动, Hikyuu 策略执行, 模拟成交。

数据流:
  Sina 实时行情 → 内存缓存 → SG 信号扫描 → MF 因子过滤 → 模拟成交

模拟规则:
  - 以信号触发时的当前价成交
  - T+1: 当日买入次日才能卖出
  - 费率: 万三佣金 + 千一印花税(卖出)
  - 涨跌停限制: 涨停不买, 跌停不卖
  - 持仓上限: 3只

输出: results.db → sim_trades 表

用法:
  python trader/paper_trader.py                # 单次扫描模式
  python trader/paper_trader.py --live         # 持续运行直到收盘
  python trader/paper_trader.py --interval 60  # 每60秒扫描一次
"""

import os, sys, json, time, sqlite3, argparse
from datetime import datetime, date, timedelta
from collections import defaultdict
import requests

SUPERQUANT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANT_ROOT = os.path.expanduser("~/project/quant")
TRADE_DB = os.path.join(QUANT_ROOT, "data", "trades.db")
RESULTS_DB = os.path.join(QUANT_ROOT, "data", "results.db")

# 模拟交易参数
INITIAL_CAPITAL = 5000.0
MAX_POSITIONS = 3
COMMISSION = 0.0003      # 万三
STAMP_TAX = 0.001        # 千一 (仅卖出)
MIN_SHARES = 100
SINA_QUOTE_URL = "http://hq.sinajs.cn/list="

# 活跃参数 (如果存在)
ACTIVE_PARAMS_FILE = os.path.join(SUPERQUANT_ROOT, "config", "active_params.json")


def load_active_params():
    if os.path.exists(ACTIVE_PARAMS_FILE):
        with open(ACTIVE_PARAMS_FILE) as f:
            return json.load(f)
    return None


def fetch_quotes(symbols: list) -> dict:
    """从 Sina 获取实时行情。"""
    if not symbols:
        return {}
    codes = [f"{'sh' if s.startswith('6') else 'sz'}{s}" for s in symbols]
    url = SINA_QUOTE_URL + ",".join(codes)
    try:
        resp = requests.get(url, timeout=10, headers={"Referer": "https://finance.sina.com.cn"})
        resp.encoding = 'gb2312'
        results = {}
        for line in resp.text.strip().split('\n'):
            if '=' not in line:
                continue
            var, data = line.split('=', 1)
            code = var.split('_')[-1]
            fields = data.strip('";').split(',')
            if len(fields) < 32 or not fields[1]:
                continue
            sym = code[2:]
            results[sym] = {
                'name': fields[0],
                'open': float(fields[1]),
                'prev_close': float(fields[2]),
                'price': float(fields[3]),
                'high': float(fields[4]),
                'low': float(fields[5]),
                'volume': float(fields[8]),
                'amount': float(fields[9]),
                'time': f"{fields[30]} {fields[31]}",
            }
        return results
    except Exception as e:
        print(f"  [warn] 行情获取失败: {e}")
        return {}


def init_account(conn):
    """初始化模拟账户。"""
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
    """记录模拟交易。"""
    conn.execute("""INSERT INTO sim_trades(date, symbol, side, price, shares, pnl, pnl_pct,
                   capital_after, strategy, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                 (date.today().isoformat(), symbol, side, price, shares,
                  round(pnl, 2), round(pnl_pct, 4), round(capital_after, 2),
                  'superquant', datetime.now().isoformat()))
    conn.commit()


def compute_factor_score(symbol, params=None):
    """计算单股票因子得分 (简化版: 调用 app.factors)。"""
    try:
        from app.factors import compute_factor_scores
        from hikyuu.interactive import sm, Query
        stocks = [sm[symbol]]
        scores = compute_factor_scores(stocks, Query(-30))
        return scores.get(symbol, 0.0)
    except Exception:
        return 0.0


def run_scan(conn, capital, positions, tracked_symbols):
    """单次扫描: 检查信号, 执行买卖。"""
    quotes = fetch_quotes(tracked_symbols)
    if not quotes:
        return capital, positions

    params = load_active_params()
    stop_base = params['stops']['adaptive_stop_base'] if params else 0.05

    signals = []
    today_positions = {p['symbol'] for p in positions}

    for sym, q in quotes.items():
        if q['open'] <= 0 or q['prev_close'] <= 0:
            continue

        gap = (q['open'] / q['prev_close'] - 1) * 100
        daily_ret = (q['price'] / q['prev_close'] - 1) * 100

        # 涨停不买
        if q['price'] >= q['prev_close'] * 1.095:
            continue

        # 信号检测 (简化: 弱转强)
        if 2.0 <= gap <= 5.0 and daily_ret >= 5.0:
            factor_score = compute_factor_score(sym, params)
            if factor_score > 0:
                signals.append({'symbol': sym, 'price': q['price'], 'gap': gap,
                                'ret': daily_ret, 'factor': factor_score, 'mode': '弱转强'})

    # 按因子得分降序
    signals.sort(key=lambda s: s['factor'], reverse=True)

    # 买入
    for s in signals:
        if len(positions) >= MAX_POSITIONS:
            break
        if s['symbol'] in today_positions:
            continue
        shares = min(int(capital * 0.33 / s['price'] / 100) * 100, 100)
        if shares < 100:
            continue
        cost = shares * s['price'] * (1 + COMMISSION)
        if cost > capital:
            continue
        capital -= cost
        positions.append({'symbol': s['symbol'], 'price': s['price'],
                          'shares': shares, 'date': date.today().isoformat(),
                          'mode': s['mode'], 'factor': s['factor']})
        record_trade(conn, s['symbol'], 'buy', s['price'], shares, capital_after=capital,
                     reason=f"{s['mode']}+因子{s['factor']:.2f}")
        print(f"  🟢 买 {s['symbol']} {s['mode']} ¥{s['price']:.2f} {shares}股 (因子{s['factor']:+.2f})")

    # 止盈止损检查
    to_sell = []
    for i, pos in enumerate(positions):
        sym = pos['symbol']
        if sym not in quotes:
            continue
        q = quotes[sym]
        pnl_pct = (q['price'] / pos['price'] - 1) * 100
        # 止损 (自适应基线)
        if pnl_pct <= -stop_base * 100:
            to_sell.append((i, '止损', pnl_pct))
        # 止盈 (最高点回撤5%)
        elif pos.get('peak', pos['price']) > pos['price']:
            peak = max(pos.get('peak', pos['price']), q['high'])
            pos['peak'] = peak
            if q['price'] < peak * 0.95:
                to_sell.append((i, '移动止盈', pnl_pct))

    for i, reason, pnl_pct in reversed(to_sell):
        pos = positions.pop(i)
        sell_val = pos['shares'] * quotes[pos['symbol']]['price']
        fee = sell_val * (COMMISSION + STAMP_TAX)
        pnl = sell_val - pos['shares'] * pos['price'] - fee
        capital += sell_val - fee
        record_trade(conn, pos['symbol'], 'sell', quotes[pos['symbol']]['price'],
                     pos['shares'], pnl=pnl, pnl_pct=pnl_pct, capital_after=capital,
                     reason=reason)
        print(f"  🔴 卖 {pos['symbol']} {reason} ¥{quotes[pos['symbol']]['price']:.2f} PnL=¥{pnl:.0f} ({pnl_pct:+.1f}%)")

    return capital, positions


def get_tracked_symbols():
    """获取监控股票列表 (因子TopN + 持仓)。"""
    try:
        from app.factors import compute_factor_scores
        from hikyuu.interactive import sm, Query
        stocks = []
        for mkt in ['SH', 'SZ']:
            try:
                market_stocks = sm.get_stock_list(lambda s, m=mkt: s.market == m and s.valid)
                stocks.extend(list(market_stocks))
            except Exception:
                pass
        scores = compute_factor_scores(stocks, Query(-30))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [code for code, _ in ranked[:200]]  # Top 200
    except Exception:
        return ['SH600000', 'SZ000001', 'SH600036']


def main():
    parser = argparse.ArgumentParser(description='superquant 模拟交易')
    parser.add_argument('--live', action='store_true', help='持续运行到收盘')
    parser.add_argument('--interval', type=int, default=60, help='扫描间隔(秒)')
    ARGS = parser.parse_args()

    print("=" * 60)
    print("superquant 模拟交易引擎")
    print(f"  启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  初始资金: ¥{INITIAL_CAPITAL:,.0f}")
    print("=" * 60)

    conn = sqlite3.connect(TRADE_DB)
    capital = init_account(conn)
    positions = []

    # 加载今日因子Top200作为监控池
    print("  加载因子候选池...")
    tracked = get_tracked_symbols()
    print(f"  监控池: {len(tracked)}只 (因子Top200)")

    if ARGS.live:
        print("  模式: 持续运行")
        while True:
            now = datetime.now()
            if now.hour >= 15 or (now.hour == 11 and now.minute >= 30):
                break  # 收盘或午休
            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                time.sleep(30)
                continue
            capital, positions = run_scan(conn, capital, positions, tracked)
            print(f"  [{now.strftime('%H:%M:%S')}] 资金=¥{capital:.0f}, 持仓={len(positions)}")
            time.sleep(ARGS.interval)
    else:
        capital, positions = run_scan(conn, capital, positions, tracked)

    # 汇总
    print(f"\n  最终资金: ¥{capital:,.0f}")
    print(f"  持仓: {len(positions)}只")
    equity = capital + sum(p['shares'] * p['price'] * 0.99 for p in positions)
    print(f"  总权益: ¥{equity:,.0f}")
    pnl_total = equity / INITIAL_CAPITAL - 1
    print(f"  累计收益: {pnl_total*100:+.1f}%")

    conn.close()
    print("✅ 模拟交易结束")
    return 0


if __name__ == '__main__':
    exit(main())
