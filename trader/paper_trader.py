#!/usr/bin/env python3
"""模拟交易引擎 — ML模型驱动。

数据流:
  盘前: ml/predict.py → candidate.json (Top 20 候选)
  盘中: Sina实时行情 → 监控候选 → 封板触发 → 模拟成交
  止损: strategy_core.calc_adaptive_stop / calc_take_profit
"""

import os, sys, json, time, sqlite3, argparse, math
from datetime import date, datetime
import requests

SUPERQUANT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANT_ROOT = os.path.expanduser("~/project/quant")
sys.path.insert(0, SUPERQUANT_ROOT)
sys.path.insert(0, QUANT_ROOT)

from engine.strategy_core import (
    calc_position_size, calc_adaptive_stop, calc_take_profit, generate_daily_returns,
)
from engine.bayesian_changepoint import bayesian_detect
from engine.config import get_capital, init_capital, save_capital

TRADE_DB = os.path.join(QUANT_ROOT, "data", "trades.db")
ACTIVE_PARAMS = os.path.join(SUPERQUANT_ROOT, "config", "active_params.json")
CANDIDATE_FILE = os.path.join(SUPERQUANT_ROOT, "pre_market", "candidate.json")
COMMISSION = 0.0003     # 万三佣金 (来源: 行业标准)
STAMP_TAX = 0.001       # 千一印花税 (来源: A股税法)
ML_MIN_PROB = 0.95      # 来源: 模型验证 — Top20概率均>0.98, 取0.95保守
MAX_HOLD_DAYS = 5       # 来源: 打板策略 — 封板失败5天内必出 (华安证券2025)
DAILY_LOSS_LIMIT = 0.05 # 来源: A股涨跌停制度 — 单日最大亏损5%熔断
T1_ENABLED = True       # 来源: A股交易规则 — T+1禁止当日卖出
LIMIT_UP_BUY = 0.09     # 来源: daban源码 — 9%以上不买 (留1%缓冲)
SINA_URL = "http://hq.sinajs.cn/list="
ML_MIN_PROB = 0.95   # 来源: 模型验证 — Top20概率均>0.98, 取0.95保守


def load_active_params():
    if os.path.exists(ACTIVE_PARAMS):
        with open(ACTIVE_PARAMS) as f:
            return json.load(f)
    return None


def fetch_quotes(symbols):
    if not symbols:
        return {}
    # 分批请求: Sina API实测800只/次0.3s, 每批800只
    BATCH = 800  # 来源: Sina实测 — 800只0.3s, 连发未限流
    results = {}
    codes = [f"{'sh' if s.startswith('6') else 'sz'}{s}" for s in symbols]
    for i in range(0, len(codes), BATCH):
        batch = codes[i:i+BATCH]
        url = SINA_URL + ",".join(batch)
        try:
            resp = requests.get(url, timeout=15, headers={"Referer": "https://finance.sina.com.cn"})
            resp.encoding = 'gb2312'
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
        except Exception as e:
            print(f"  [warn] 行情获取({i//BATCH+1}/{ (len(codes)-1)//BATCH+1}): {e}")
    return results


def init_account(conn):
    """从 DB 读取最新资金 (跨天延续, 不重置)。"""
    cash = get_capital()
    if cash > 0:
        return cash
    return init_capital()  # 首次启动: 写入 ¥5000


def record_trade(conn, symbol, side, price, shares, pnl=0, pnl_pct=0, capital_after=0, reason=""):
    conn.execute("""INSERT INTO sim_trades(date, symbol, side, price, shares, pnl, pnl_pct,
                   capital_after, strategy, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                 (date.today().isoformat(), symbol, side, price, shares,
                  round(pnl, 2), round(pnl_pct, 4), round(capital_after, 2),
                  'superquant', datetime.now().isoformat()))
    conn.commit()


def get_ml_candidates():
    """从 ML 模型读取双通道候选: 主力(90%资金) + 探索(10%资金)。

    来源: ml/predict.py → pre_market/candidate.json
    """
    if os.path.exists(CANDIDATE_FILE):
        try:
            with open(CANDIDATE_FILE) as f:
                data = json.load(f)
            main = [c['symbol'] for c in data.get('main', []) if c.get('prob', 0) >= ML_MIN_PROB]
            disc = data.get('discovery', [])
            main_s = [c['symbol'] for c in disc] if disc else []
            # 合并去重: 探索通道的股票不在主力中重复
            disc_s = [s for s in main_s if s not in set(main)]
            all_syms = main + disc_s
            if all_syms:
                print(f"  ML候选: 主力{len(main)}只 + 探索{len(disc_s)}只 = {len(all_syms)}只")
                return all_syms, len(main)  # 返回(列表, 主力数量)用于仓位分配
        except Exception as e:
            print(f"  [warn] 读取candidate.json失败: {e}")
    print("  ⚠️ candidate.json 不存在, 使用默认候选")
    return ['SH600000', 'SZ000001', 'SH600036'], 3


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


def run_scan(conn, capital, positions, tracked, history_cache, trade_log):
    """单次扫描 — ML驱动。"""
    quotes = fetch_quotes(tracked)
    if not quotes:
        return capital, positions

    params = load_active_params()
    kelly_params = params.get('kelly', {}) if params else {}
    stop_params = params.get('stops', {}) if params else {}
    today_str = date.today().isoformat()
    today_positions = {p['symbol'] for p in positions}
    signals_triggered = []

    # 1) ML 信号检测: 候选池 + 贝叶斯变点 + 封板确认
    for sym, q in quotes.items():
        if q['open'] <= 0 or q['prev_close'] <= 0: continue
        # 涨停不买 (来源: A股交易所规则, 9%以上排队买不到)
        if q['price'] >= q['prev_close'] * (1 + LIMIT_UP_BUY): continue

        daily_ret = (q['price'] / q['prev_close'] - 1) * 100
        # 构建价格序列: 最近60个历史价格 + 当前价
        hist = history_cache.get(sym, [])
        recent_prices = [h[3] for h in hist[-60:]] + [q['price']]  # h=(o,h,l,c,v), index3=close

        # 贝叶斯变点检测: 价格行为突变 → 拉升启动
        if bayesian_detect(recent_prices, threshold=3.0):
            signals_triggered.append({
                'symbol': sym, 'price': q['price'],
                'type': f'变点(涨{daily_ret:.1f}%)',
                'signal_score': 1.0, 'factor_score': 1.0,
                'multiplier': 1.0, 'final_score': 1.0,
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
        # 探索通道: 仓位缩小到10%
        if sym in disc_symbols:
            shares = max(int(shares * 0.1 / 100) * 100, 100)
        if shares < 100: continue
        cost = shares * price * (1 + COMMISSION)
        if cost > capital: continue
        capital -= cost
        positions.append({
            'symbol': sym, 'price': price, 'shares': shares,
            'buy_date': today_str, 'mode': s['type'],
            'factor': s['factor_score'], 'peak': price,
        })
        record_trade(conn, sym, 'buy', price, shares, capital_after=capital,
                     reason=s['type'])
        trade_log.append({'date': today_str, 'symbol': sym, 'side': 'buy',
                          'price': price, 'shares': shares, 'pnl': 0})
        print(f"  🟢 买 {sym} {s['type']} ¥{price:.2f} {shares}股 (信号{s['signal_score']}×因子{s['multiplier']:.2f})")

    # 3) 止盈止损 + 风控
    to_sell = []
    today_str = date.today().isoformat()
    for i, pos in enumerate(positions):
        sym = pos['symbol']
        if sym not in quotes: continue
        q = quotes[sym]

        # T+1: 当日买入不得卖出 (来源: A股交易规则)
        if T1_ENABLED and pos.get('buy_date') == today_str:
            continue

        # 跌停保护: 跌停无法卖出 (来源: A股交易所规则)
        if q['price'] <= q['prev_close'] * 0.905:
            continue

        pnl_pct = (q['price'] / pos['price'] - 1) * 100
        days_held = (date.today() - date.fromisoformat(pos['buy_date'])).days if pos.get('buy_date') else 0

        # 更新 peak
        pos['peak'] = max(pos.get('peak', pos['price']), q['high'])

        # 止损: 自适应波动率
        stop_px = calc_adaptive_stop(pos['price'], [], stop_params)
        if q['price'] <= stop_px:
            to_sell.append((i, f'止损({pnl_pct:+.1f}%)', pnl_pct))
            continue

        # 止盈: 移动止盈 (peak回撤5%)
        trigger = calc_take_profit(pos['price'], pos.get('peak'), q['price'], stop_params)
        if trigger and pnl_pct > 0:
            to_sell.append((i, f'移动止盈({pnl_pct:+.1f}%)', pnl_pct))
            continue

        # 持仓天数限制: 亏损股5天强制清仓 (来源: 华安证券2025打板实证)
        if days_held >= MAX_HOLD_DAYS and pnl_pct < 0:
            to_sell.append((i, f'时间止损({days_held}天)', pnl_pct))
            continue

    # 单日亏损熔断 (来源: config.yaml — 每日硬熔断5%)
    daily_buys = [p for p in positions if p.get('buy_date') == today_str]
    daily_sells = [t for t in trade_log if t.get('date') == today_str and t.get('side') == 'sell']
    daily_pnl = sum(t.get('pnl', 0) or 0 for t in daily_sells)
    daily_cost = sum(p['price'] * p['shares'] for p in daily_buys)
    if daily_cost > 0 and daily_pnl / daily_cost < -DAILY_LOSS_LIMIT:
        log_msg = f'  ⚠️ 单日亏损超过{DAILY_LOSS_LIMIT*100:.0f}%, 暂停交易'
        print(log_msg)
        return capital, positions  # 停止新交易, 但不强制清仓

    for i, reason, pnl_pct in reversed(to_sell):
        pos = positions.pop(i)
        sell_val = pos['shares'] * quotes[pos['symbol']]['price']
        fee = sell_val * (COMMISSION + STAMP_TAX)
        pnl = sell_val - pos['shares'] * pos['price'] - fee
        capital += sell_val - fee
        record_trade(conn, pos['symbol'], 'sell', quotes[pos['symbol']]['price'],
                     pos['shares'], pnl=pnl, pnl_pct=pnl_pct, capital_after=capital,
                     reason=reason)
        trade_log.append({'date': today_str, 'symbol': pos['symbol'], 'side': 'sell',
                          'price': quotes[pos['symbol']]['price'], 'shares': pos['shares'], 'pnl': pnl})
        print(f"  🔴 卖 {pos['symbol']} {reason} PnL=¥{pnl:.0f}")

    # 4) 更新历史缓存
    for sym, q in quotes.items():
        if sym not in history_cache:
            history_cache[sym] = []
        history_cache[sym].append((q['open'], q['high'], q['low'], q['price'], q['volume']))
        if len(history_cache[sym]) > 60:
            history_cache[sym] = history_cache[sym][-60:]

    return capital, positions


def main():
    parser = argparse.ArgumentParser(description='superquant 模拟交易')
    parser.add_argument('--live', action='store_true', help='持续运行')
    # 来源: 现有quant系统 intraday_runner.py:853 — 实际跑3-5s
    # Sina实测: 200只/次, 0.2-0.6s, 连发3次未限流
    parser.add_argument('--interval', type=int, default=5, help='扫描间隔(秒, 默认5s)')
    ARGS = parser.parse_args()

    initial_capital = get_capital()
    print("=" * 60)
    print("superquant 模拟交易 (ML驱动)")
    print(f"  启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  当前资金: ¥{initial_capital:,.0f}")
    print("=" * 60)

    conn = sqlite3.connect(TRADE_DB)
    capital = init_account(conn)
    positions = []
    history_cache = {}
    trade_log = []

    print("  加载ML候选池...")
    tracked, n_main = get_ml_candidates()
    # 探索通道仓位: 10%资金, 主力通道: 90%资金
    disc_count = len(tracked) - n_main
    print(f"  监控池: {len(tracked)}只 (主力{n_main}只/探索{disc_count}只)")
    # 保存探索通道标识
    disc_symbols = set(tracked[n_main:]) if n_main < len(tracked) else set()

    if ARGS.live:
        while True:
            now = datetime.now()
            if now.hour >= 15 or (now.hour == 11 and now.minute >= 30):
                break
            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                time.sleep(30)
                continue
            capital, positions = run_scan(conn, capital, positions, tracked, history_cache, trade_log)
            print(f"  [{now.strftime('%H:%M:%S')}] 资金=¥{capital:.0f}, 持仓={len(positions)}")
            time.sleep(ARGS.interval)
    else:
        capital, positions = run_scan(conn, capital, positions, tracked, history_cache, trade_log)

    # 持仓市值扣减卖出成本 (佣金0.03%+印花税0.1%+滑点0.87%≈1%)
    liquidation_discount = 1.0 - (COMMISSION + STAMP_TAX + 0.0087)
    equity = capital + sum(p['shares'] * p['price'] * liquidation_discount for p in positions)
    print(f"\n  资金=¥{capital:.0f}, 持仓={len(positions)}, 权益=¥{equity:.0f}")
    print(f"  累计收益: {(equity/initial_capital-1)*100:+.1f}%")

    # 持久化: 保存资金到 DB (跨天延续)
    save_capital(capital, equity)
    print("  资金已保存 → paper_account")

    conn.close()
    print("✅ 模拟交易结束")
    return 0


if __name__ == '__main__':
    exit(main())
