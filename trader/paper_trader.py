#!/usr/bin/env python3
"""模拟交易引擎 — ML模型驱动。

数据流:
  盘前: ml/predict.py → candidate.json (Top 200 候选, 板块中性化)
  盘中: Sina实时行情 → 监控候选 → 4信号检测(A/B/C/D) → 盘口确认 → 模拟成交
  止损: strategy_core.calc_adaptive_stop / calc_take_profit
  在线学习: 每笔交易结果 → signal_stats.json → 实证权重更新 (P2-11)

信号:
  A=贝叶斯变点 | B=量价背离 | C=买盘堆积 | D=竞价异常(VWAP, P2-13)
"""

import os, sys, json, time, sqlite3, argparse, math
from datetime import date, datetime
import requests

SUPERQUANT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANT_ROOT = os.path.expanduser("~/project/quant")
sys.path.insert(0, SUPERQUANT_ROOT)
sys.path.insert(0, QUANT_ROOT)

from engine.strategy_core import (
    calc_adaptive_stop, calc_take_profit,
)
from engine.bayesian_changepoint import bayesian_detect
from engine.config import get_capital, init_capital, save_capital
from engine.buy_rules import calculate_buys

TRADE_DB = os.path.join(QUANT_ROOT, "data", "trades.db")
ACTIVE_PARAMS = os.path.join(SUPERQUANT_ROOT, "config", "active_params.json")
# candidates now in trades.db
COMMISSION = 0.0003     # 万三佣金 (来源: 行业标准)
STAMP_TAX = 0.001       # 千一印花税 (来源: A股税法)
MAX_HOLD_DAYS = 5       # 来源: 打板策略 — 封板失败5天内必出 (华安证券2025)
DAILY_LOSS_LIMIT = 0.05 # 来源: config.yaml — 每日硬熔断5%
T1_ENABLED = True       # 来源: A股交易规则 — T+1禁止当日卖出
SINA_URL = "http://hq.sinajs.cn/list="
# JSON→DB 重构: signal_events, signal_stats, rejected_signals 全迁入 trades.db
# 以下常量仅保留文件路径引用 (用于兼容), 实际读写走 DB

# 信号权重 — 等权, 启动时从DB加载可覆盖
DEFAULT_SIGNAL_WEIGHTS = {'A': 0.33, 'B': 0.33, 'E': 0.33}

def load_params_from_db(mode):
    """从 strategy_params 表加载参数 (Item 3: 参数入DB)。
    优先级: data_driven > hand (默认值)
    """
    try:
        conn = sqlite3.connect(TRADE_DB)
        rows = conn.execute(
            "SELECT param_name, param_value, source FROM strategy_params WHERE mode=? ORDER BY source DESC",
            (mode,)
        ).fetchall()
        conn.close()
        params = {}
        for name, val, src in rows:
            if name not in params:  # 第一个=最高优先级source
                params[name] = val
        return params
    except Exception:
        return {}

def check_signal_circuit_breaker():
    """信号熔断: 连续亏损3次降权50%, 5次暂停。

    来源: 幻方量化 — "单策略连续亏损超阈值自动熔断"。
    Returns: {signal_code: multiplier} 1.0=正常, 0.5=降权, 0=暂停
    """
    try:
        conn = sqlite3.connect(TRADE_DB)
        breakers = {}
        for code in ['A', 'B', 'E']:
            # 查最近5笔该信号的交易
            rows = conn.execute("""
                SELECT s.pnl_pct FROM sim_trades s
                JOIN signal_events e ON s.symbol = e.symbol AND s.date = e.date
                WHERE s.side = 'sell' AND e.signal LIKE ? AND s.pnl_pct IS NOT NULL
                ORDER BY s.id DESC LIMIT 5
            """, (f'%{code}%',)).fetchall()
            if not rows:
                breakers[code] = 1.0
                continue
            # 统计连续亏损次数
            consecutive_losses = 0
            for r in rows:
                if r[0] < 0:
                    consecutive_losses += 1
                else:
                    break  # 遇到盈利就停
            if consecutive_losses >= 5:
                breakers[code] = 0.0   # 暂停
            elif consecutive_losses >= 3:
                breakers[code] = 0.5   # 降权
            else:
                breakers[code] = 1.0
        conn.close()
        return breakers
    except Exception:
        return {'A': 1.0, 'B': 1.0, 'E': 1.0}


def load_signal_weights():
    """从 signal_stats 表加载实证权重 (JSON->DB 重构)。"""
    try:
        conn = sqlite3.connect(TRADE_DB)
        rows = conn.execute("SELECT signal, win_rate FROM signal_stats WHERE total_count > 0").fetchall()
        conn.close()
        if rows:
            rates = {r[0]: r[1] for r in rows}
            max_rate = max(rates.values()) if rates else 0.5
            weights = {k: max(0.3, min(1.5, v / max(max_rate, 0.01))) for k, v in rates.items()}
            if weights: return weights
    except Exception:
        pass
    return dict(DEFAULT_SIGNAL_WEIGHTS)

def log_signal_event(sym, signal_type, action, pnl=0):
    """记录信号到 signal_events 表 (JSON->DB 重构)。"""
    try:
        conn = sqlite3.connect(TRADE_DB)
        conn.execute("INSERT INTO signal_events(time, date, symbol, signal, action, pnl) VALUES(?,?,?,?,?,?)",
                     (datetime.now().strftime('%H:%M:%S'), date.today().isoformat(), sym, signal_type, action, round(pnl, 2)))
        conn.commit()
        conn.execute("DELETE FROM signal_events WHERE id NOT IN (SELECT id FROM signal_events ORDER BY id DESC LIMIT 500)")
        conn.commit(); conn.close()
    except Exception:
        pass


def update_signal_stats(signal_type, pnl_pct):
    """在线学习: 更新 signal_stats 表 (JSON->DB 重构)。"""
    try:
        conn = sqlite3.connect(TRADE_DB)
        codes = signal_type.split('+')
        for code in codes:
            row = conn.execute("SELECT win_count, total_count, total_pnl FROM signal_stats WHERE signal=?", (code,)).fetchone()
            if row: wc, tc, tp = row
            else:
                wc, tc, tp = 0, 0, 0.0
                conn.execute("INSERT INTO signal_stats(signal) VALUES(?)", (code,))
            tc += 1
            if pnl_pct > 0: wc += 1
            tp += pnl_pct
            conn.execute("UPDATE signal_stats SET win_count=?, total_count=?, win_rate=?, total_pnl=?, avg_return=? WHERE signal=?",
                         (wc, tc, round(wc/tc, 4), round(tp, 4), round(tp/tc, 4), code))
        conn.commit(); conn.close()
    except Exception:
        pass


def load_active_params():
    if os.path.exists(ACTIVE_PARAMS):
        with open(ACTIVE_PARAMS) as f:
            return json.load(f)
    return None


def fetch_quotes(symbols):
    """获取实时行情 + 五档盘口。来源: Sina API (hq.sinajs.cn)。"""
    if not symbols:
        return {}
    # 分批请求: Sina API实测800只/次0.3s, 每批800只
    BATCH = 800  # 来源: Sina实测 — 800只0.3s, 连发未限流
    results = {}
    # 市场前缀: SH(6/5/9), BJ(920), SZ(0/3)
    codes = []
    for s in symbols:
        if s.startswith('920'): codes.append(f'bj{s}')
        elif s.startswith(('6','5','9')): codes.append(f'sh{s}')
        else: codes.append(f'sz{s}')
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

                def _f(i):
                    """安全解析字段为 float, 空值返回 0。"""
                    try: return float(fields[i]) if fields[i] else 0.0
                    except ValueError: return 0.0

                results[sym] = {
                    'open': _f(1), 'prev_close': _f(2),
                    'price': _f(3), 'high': _f(4), 'low': _f(5),
                    'volume': _f(8), 'amount': _f(9),
                    # 五档盘口 — 来源: Sina API 字段 10-29
                    # 字段索引: 买一量10,买一价11,卖一量20,卖一价21
                    'buy_vol_1': _f(10), 'buy_price_1': _f(11),
                    'sell_vol_1': _f(20), 'sell_price_1': _f(21),
                }
        except Exception as e:
            print(f"  [warn] 行情获取({i//BATCH+1}/{ (len(codes)-1)//BATCH+1}): {e}")
    return results


def get_market_regime(conn):
    """市场状态感知。来源: Grinold 条件IC — IC随市场状态变化。

    用上证指数 20 日趋势判断牛熊, 返回仓位乘数:
      Bull(+5%+): 1.2x | Neutral(-5%~+5%): 1.0x | Bear(-5%-): 0.5x
    """
    try:
        row = conn.execute(
            "SELECT close FROM daily WHERE symbol='000001' ORDER BY date DESC LIMIT 20"
        ).fetchall()
        if len(row) >= 20:
            ret20 = (row[0][0] / row[-1][0] - 1)
            if ret20 > 0.05:
                return 1.2, f'Bull({ret20:+.1%})'
            elif ret20 < -0.05:
                return 0.5, f'Bear({ret20:+.1%})'
            else:
                return 1.0, f'Neutral({ret20:+.1%})'
    except Exception:
        pass
    return 1.0, 'Unknown'


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
    """从 candidates 表读取 ML 候选 (JSON->DB 重构)。"""
    try:
        conn = sqlite3.connect(TRADE_DB)
        today = date.today().isoformat()
        rows = conn.execute("SELECT symbol, channel FROM candidates WHERE date=? ORDER BY prob DESC", (today,)).fetchall()
        if not rows:
            rows = conn.execute("SELECT symbol, channel FROM candidates ORDER BY date DESC, prob DESC LIMIT 200").fetchall()
        conn.close()
        main = [r[0] for r in rows if r[1] == 'main']
        disc = [r[0] for r in rows if r[1] == 'discovery']
        disc_s = [s for s in disc if s not in set(main)]
        all_syms = main + disc_s
        if all_syms:
            print(f"  ML候选: 主力{len(main)}只 + 探索{len(disc_s)}只 = {len(all_syms)}只 (from DB)")
            return all_syms, len(main)
    except Exception as e:
        print(f"  [warn] 读取candidates表失败: {e}")
    print("  ⚠️ candidates表为空, 使用默认候选")
    return ['SH600000', 'SZ000001', 'SH600036'], 3


def get_board_limit_ratio(symbol):
    """各板块涨跌停幅度。来源: A股交易所规则。"""
    if symbol.startswith('920'): return 0.30           # BSE ±30%
    if symbol.startswith(('300','301','688')): return 0.20  # 创业板/科创板 ±20%
    return 0.10  # 主板 ±10%


def check_buyable(symbol, quote):
    """检查涨停股是否可买入。

    来源: A股涨停板交易机制 — 涨停价上:
      - 卖一量 > 0 → 有人在卖, 可排队买入
      - 卖一量 = 0 → 封死涨停, 买不到

    Returns: (buyable: bool, reason: str)
    """
    limit_ratio = get_board_limit_ratio(symbol)
    prev_close = quote.get('prev_close', 0)
    if prev_close <= 0:
        return False, "昨收无效"

    limit_up_price = prev_close * (1 + limit_ratio)
    price = quote.get('price', 0)

    # 未到涨停价 → 正常买入
    if price < limit_up_price * 0.995:
        return True, ""

    # 在涨停价 → 检查封板状态
    sell_vol = quote.get('sell_vol_1', 0)
    if sell_vol <= 0:
        return False, f"涨停封死(卖一量=0, 涨停价¥{limit_up_price:.2f})"
    else:
        return True, ""


def save_rejected(rejected):
    """持久化拒绝信号到 rejected_signals 表 (JSON->DB 重构)。"""
    try:
        conn = sqlite3.connect(TRADE_DB)
        today = date.today().isoformat()
        for r in rejected:
            conn.execute("INSERT INTO rejected_signals(date, time, symbol, price, reason) VALUES(?,?,?,?,?)",
                         (today, r.get('time',''), r.get('symbol',''), r.get('price',0), r.get('reason','')))
        conn.execute("DELETE FROM rejected_signals WHERE date=? AND id NOT IN (SELECT id FROM rejected_signals WHERE date=? ORDER BY id DESC LIMIT 50)", (today, today))
        conn.commit(); conn.close()
    except Exception:
        pass


def run_scan(conn, capital, positions, tracked, history_cache, trade_log, rejected_signals):
    """单次扫描 — ML驱动。所有异常在内部处理, 保证不崩溃。"""
    try:
        return _run_scan_impl(conn, capital, positions, tracked, history_cache, trade_log, rejected_signals)
    except Exception as e:
        print(f"  ⚠️ run_scan 异常: {e}, 跳过本轮")
        import traceback; traceback.print_exc()
        return capital, positions


def _run_scan_impl(conn, capital, positions, tracked, history_cache, trade_log, rejected_signals):
    quotes = fetch_quotes(tracked)
    if not quotes:
        return capital, positions

    today_str = date.today().isoformat()
    today_positions = {p['symbol'] for p in positions}
    signals_triggered = []
    regime_mult, regime_label = get_market_regime(conn)

    # ═════════════════════════════════════════════════════════
    # 1) 多信号实时检测 (三级漏斗 第二级)
    #   四个并行信号, 任何一个触发→进入第三级盘口确认
    # ═════════════════════════════════════════════════════════
    sw = load_signal_weights()  # 一次扫描内权重不变, 提到循环外
    cb = check_signal_circuit_breaker()  # 信号熔断: 连续亏损→降权/暂停
    for sym, q in quotes.items():
        if q['open'] <= 0 or q['prev_close'] <= 0: continue

        daily_ret = (q['price'] / q['prev_close'] - 1) * 100
        hist = history_cache.get(sym, [])
        prices = [h[3] for h in hist[-60:]] + [q['price']]
        volumes = [h[4] for h in hist[-60:]] if hist else []

        signals_fired = []

        # 信号 A: BOCPD 趋势变点 (Adams & MacKay 2007)
        if bayesian_detect(prices):
            signals_fired.append(('A', sw.get('A', 0.33)))

        # 信号 B: 量价背离
        if len(hist) >= 10:
            recent_vols = volumes[-10:] if len(volumes) >= 10 else volumes
            avg_vol = sum(recent_vols[:-1]) / max(len(recent_vols)-1, 1)
            cur_vol = q.get('volume', 0)
            if cur_vol > avg_vol * 3 and abs(daily_ret) < 2.0:
                signals_fired.append(('B', sw.get('B', 0.33)))

        # 信号 E: 均值回复 (来源: Narang 趋势+均值回复互补)
        # 近5日累计跌幅>8% + 今日放量 → 超跌反弹
        if len(prices) >= 6:
            ret_5d = (prices[-1] / prices[-6] - 1) if prices[-6] > 0 else 0
            # 来源: 回测最优区间10-12%, 3d均值+0.46% 胜率54%
            if ret_5d < -0.10 and q.get('volume', 0) > 0:
                signals_fired.append(('E', sw.get('E', 0.33)))

        if signals_fired:
            # ── 等权线性组合 (来源: Narang方法一, Grinold "无信息时等权最优") ──
            signal_codes = [code for code, _ in signals_fired]
            # 条件模型 (来源: Narang方法三 — 熊市倾向均值回复, 牛市倾向趋势)
            if regime_mult <= 0.5:  # Bear
                n_effective = 3.0
                has_E = 1.5 if 'E' in signal_codes else 0  # E 1.5x
                has_A = 0.5 if 'A' in signal_codes else 0  # A 0.5x
                has_B = 1.0 if 'B' in signal_codes else 0
                signal_score = (has_A + has_B + has_E) / n_effective
            elif regime_mult >= 1.2:  # Bull
                n_effective = 3.0
                has_A = 1.5 if 'A' in signal_codes else 0  # A 1.5x
                has_E = 0.5 if 'E' in signal_codes else 0  # E 0.5x
                has_B = 1.0 if 'B' in signal_codes else 0
                signal_score = (has_A + has_B + has_E) / n_effective
            else:  # Neutral — 等权
                signal_score = len(signals_fired) / 3.0
            # 信号熔断乘数 (来源: 幻方 — 连续亏损→降权/暂停)
            cb_mult = min(cb.get(code, 1.0) for code in signal_codes)
            signal_score *= cb_mult
            signal_type = '+'.join(signal_codes)
            # ── 第三级: 盘口确认 ──
            buyable, reason = check_buyable(sym, q)
            if not buyable:
                rejected_signals.append({
                    'symbol': sym, 'price': q['price'],
                    'reason': f'{reason} [信号:{signal_type}]',
                    'time': datetime.now().strftime('%H:%M:%S'),
                })
                continue

            # ── 拉升确认: 卖压萎缩→加仓, 卖压正常→正常仓 (来源: 订单流阻力分析) ──
            multiplier = 1.0
            sell_vol = q.get('sell_vol_1', 0)
            buy_vol = q.get('buy_vol_1', 0)
            if buy_vol > 0 and sell_vol > 0:
                ratio = sell_vol / buy_vol
                if ratio < 0.5:
                    multiplier = 1.5   # 卖压显著萎缩→加仓
                elif ratio < 1.0:
                    multiplier = 1.2   # 卖压小于买压→偏积极
                elif ratio > 3.0:
                    multiplier = 0.5   # 卖压远大于买压→减仓

            final_score = signal_score * multiplier * regime_mult
            signals_triggered.append({
                'symbol': sym, 'price': q['price'],
                'type': f'{signal_type}×{multiplier:.1f}·{regime_label}',
                'signal_codes': signal_type,  # 干净码 "A+B+E" 供 update_signal_stats 用
                'signal_score': signal_score,
                'factor_score': multiplier,
                'multiplier': multiplier,
                'final_score': final_score,
            })

    # 2) 买入: 攻击期规则 (纯函数 calculate_buys)
    signals_triggered.sort(key=lambda s: s['final_score'], reverse=True)
    # 构建 symbol→信号类型 映射 (来源: P2-11 在线学习 — 记录每笔交易由哪个信号触发)
    signal_type_map = {s['symbol']: s.get('signal_codes', '') for s in signals_triggered}
    candidates = [(s['symbol'], s['price'], s['final_score']) for s in signals_triggered]
    orders = calculate_buys(capital, candidates)
    for sym, shares, price, phase in orders:
        if sym in today_positions: continue
        cost = shares * price * (1 + COMMISSION)
        capital -= cost
        # 记录信号触发 (来源: Web 实时信号展示)
        sig_type = signal_type_map.get(sym, '')
        if sig_type:
            log_signal_event(sym, sig_type, f'买入 ¥{price:.2f}×{shares}股')
        positions.append({
            'symbol': sym, 'price': price, 'shares': shares,
            'buy_date': today_str, 'mode': f'变点({phase})',
            'factor': 0, 'peak': price,
            'signal_type': signal_type_map.get(sym, ''),  # 来源: P2-11 在线学习
        })
        record_trade(conn, sym, 'buy', price, shares, capital_after=capital,
                     reason=f'变点+{phase}')
        trade_log.append({'date': today_str, 'symbol': sym, 'side': 'buy',
                          'price': price, 'shares': shares, 'pnl': 0})
        print(f"  🟢 买 {sym} ¥{price:.2f} {shares}股 ({phase})")

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

        # 跌停保护: 封死跌停无法卖出 (来源: A股交易所规则)
        # 与 check_buyable 对称: 卖看买一量, 买一=0 → 无人接盘, 卖不掉
        limit_ratio = get_board_limit_ratio(sym)
        limit_down = q['prev_close'] * (1 - limit_ratio)
        if q['price'] <= limit_down * 1.005 and q.get('buy_vol_1', 0) == 0:
            continue

        pnl_pct = (q['price'] / pos['price'] - 1) * 100
        days_held = (date.today() - date.fromisoformat(pos['buy_date'])).days if pos.get('buy_date') else 0

        # 更新 peak
        pos['peak'] = max(pos.get('peak', pos['price']), q['high'])

        # 止损: 自适应波动率 (用历史缓存中的价格计算实际波动率)
        hist_prices = [h[3] for h in history_cache.get(sym, [])[-20:]]
        if len(hist_prices) >= 5:
            from engine.strategy_core import generate_daily_returns
            returns = generate_daily_returns(hist_prices)
        else:
            returns = []
        stop_px = calc_adaptive_stop(pos['price'], returns)
        if q['price'] <= stop_px:
            to_sell.append((i, f'止损({pnl_pct:+.1f}%)', pnl_pct))
            continue

        # 止盈: 移动止盈 (peak回撤5%)
        trigger = calc_take_profit(pos['price'], pos.get('peak'), q['price'])
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
        # 在线学习: 交易结果反馈更新信号权重 (来源: P2-11, Chan 实证贝叶斯更新)
        sig_type = pos.get('signal_type', '')
        if sig_type:
            update_signal_stats(sig_type, pnl_pct)
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

    # 交易日检查 — 周末/节假日不启动
    if datetime.now().weekday() >= 5:
        print(f"  非交易日 (weekday={datetime.now().weekday()}), 退出")
        return 0

    initial_capital = get_capital()
    print("=" * 60)
    print("superquant 模拟交易 (ML驱动)")
    print(f"  启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  当前资金: ¥{initial_capital:,.0f}")
    print("=" * 60)

    conn = sqlite3.connect(TRADE_DB)
    capital = init_account(conn)
    history_cache = {}
    trade_log = []

    # 从 DB 恢复当日持仓 (来源: 午休重启安全 — 防重复买入)
    today_str = date.today().isoformat()
    positions = []
    db_buys = conn.execute(
        "SELECT symbol, price, shares, capital_after FROM sim_trades WHERE side='buy' AND date=? "
        "AND symbol NOT IN (SELECT symbol FROM sim_trades WHERE side='sell' AND date=?) "
        "ORDER BY id", (today_str, today_str)
    ).fetchall()
    for r in db_buys:
        positions.append({
            'symbol': r[0], 'price': r[1], 'shares': r[2],
            'buy_date': today_str, 'mode': '恢复',
            'factor': 0, 'peak': r[1],
        })
        print(f"  📦 恢复持仓: {r[0]} ¥{r[1]:.2f}×{r[2]}股")
    if db_buys:
        # 用最后一笔 capital_after 作为当前资金
        capital = db_buys[-1][3]
    print(f"  当前资金: ¥{capital:,.0f}, 持仓: {len(positions)}只")

    print("  加载ML候选池...")
    tracked, n_main = get_ml_candidates()
    disc_count = len(tracked) - n_main
    print(f"  监控池: {len(tracked)}只 (主力{n_main}只/探索{disc_count}只)")

    rejected_signals = []  # 今日被拒绝的信号 (供 Web 展示)

    if ARGS.live:
        while True:
            now = datetime.now()
            # 收盘 → 退出 (来源: A股交易时间 9:30-11:30, 13:00-15:00)
            if now.hour >= 15:
                break
            # 午休 → 等待 (来源: A股午休 11:30-13:00)
            if (now.hour == 11 and now.minute >= 30) or now.hour == 12:
                time.sleep(30)
                continue
            # 盘前 → 等待
            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                time.sleep(30)
                continue
            capital, positions = run_scan(conn, capital, positions, tracked, history_cache, trade_log, rejected_signals)
            save_rejected(rejected_signals)
            print(f"  [{now.strftime('%H:%M:%S')}] 资金=¥{capital:.0f}, 持仓={len(positions)}")
            time.sleep(ARGS.interval)
    else:
        capital, positions = run_scan(conn, capital, positions, tracked, history_cache, trade_log, rejected_signals)
        save_rejected(rejected_signals)

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
