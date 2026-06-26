"""superquant Web — 统一缓存层, 全 DB 读取 (JSON→DB 重构)。

API:
  /api/state /api/candidates /api/positions /api/trades
  /api/northstar /api/signals /api/signal-stats /api/rejected
  /api/l3-progress
"""

import json, os, sys, sqlite3, time, re, threading
from datetime import date, datetime
from flask import Flask, jsonify, render_template

SUPERQUANT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANT_ROOT = os.path.expanduser("~/project/quant")
sys.path.insert(0, SUPERQUANT_ROOT)
sys.path.insert(0, QUANT_ROOT)

from engine.config import get_capital

TRADE_DB = os.path.join(QUANT_ROOT, "data", "trades.db")
MKT_DB = os.path.join(QUANT_ROOT, "data", "market.db")

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# ═══════════════════════════════
# 缓存层: TTL 2s
# ═══════════════════════════════
_cache = {}
_cache_lock = threading.Lock()

def _cached(key, ttl, fn):
    now = time.time()
    with _cache_lock:
        e = _cache.get(key)
        if e and (now - e[0]) < ttl:
            return e[1]
    val = fn()
    with _cache_lock:
        _cache[key] = (now, val)
    return val

def _cached_quotes(symbols):
    if not symbols: return {}
    key = 'q_' + ','.join(sorted(symbols))
    return _cached(key, 2, lambda: __import__('execution.quote', fromlist=['fetch_quotes']).fetch_quotes(symbols))

def _cached_account():
    def _calc():
        try:
            conn = sqlite3.connect(TRADE_DB)
            today = date.today().isoformat()
            row = conn.execute("SELECT capital_after FROM sim_trades WHERE capital_after IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
            pos_rows = conn.execute(
                "SELECT symbol, SUM(shares) FROM sim_trades WHERE side='buy' AND date=? "
                "AND symbol NOT IN (SELECT symbol FROM sim_trades WHERE side='sell' AND date=?) "
                "GROUP BY symbol", (today, today)).fetchall()
            conn.close()
            cash = row[0] if row else get_capital()
            pos_value = 0
            if pos_rows:
                quotes = _cached_quotes([r[0] for r in pos_rows])
                for sym, shares in pos_rows:
                    pos_value += shares * quotes.get(sym, {}).get('price', 0)
            return {'cash': round(cash, 2), 'equity': round(cash + pos_value, 2),
                    'date': today, 'n_positions': len(pos_rows)}
        except Exception:
            return {'cash': get_capital(), 'equity': get_capital(), 'date': date.today().isoformat(), 'n_positions': 0}
    return _cached('acct', 2, _calc)

def _cached_performance():
    def _calc():
        try:
            conn = sqlite3.connect(TRADE_DB)
            today = date.today().isoformat()
            sells = conn.execute("SELECT COALESCE(SUM(pnl),0), COUNT(*) FROM sim_trades WHERE side='sell' AND date=?", (today,)).fetchone()
            realized, sell_count = sells[0] or 0, sells[1] or 0
            wins = conn.execute("SELECT COUNT(*) FROM sim_trades WHERE side='sell' AND pnl>0 AND date=?", (today,)).fetchone()[0]
            buyss = conn.execute("SELECT COUNT(*) FROM sim_trades WHERE side='buy' AND date=?", (today,)).fetchone()[0]
            pos_rows = conn.execute(
                "SELECT symbol, SUM(shares), SUM(price*shares) FROM sim_trades WHERE side='buy' AND date=? "
                "AND symbol NOT IN (SELECT symbol FROM sim_trades WHERE side='sell' AND date=?) "
                "GROUP BY symbol", (today, today)).fetchall()
            conn.close()
            unrealized = 0
            if pos_rows:
                quotes = _cached_quotes([r[0] for r in pos_rows])
                for sym, shares, cost in pos_rows:
                    unrealized += shares * quotes.get(sym, {}).get('price', 0) - cost
            return {'realized_pnl': round(realized, 2), 'unrealized_pnl': round(unrealized, 2),
                    'total_pnl': round(realized + unrealized, 2), 'buy_count': buyss, 'sell_count': sell_count,
                    'win_rate': round((wins / sell_count * 100) if sell_count > 0 else 0, 1)}
        except Exception:
            return {'realized_pnl': 0, 'unrealized_pnl': 0, 'total_pnl': 0, 'buy_count': 0, 'sell_count': 0, 'win_rate': 0}
    return _cached('perf', 2, _calc)

# ═══════════════════════════════
# DB 读取 (原 JSON 文件 → DB 表)
# ═══════════════════════════════

def _get_candidates():
    try:
        conn = sqlite3.connect(TRADE_DB)
        today = date.today().isoformat()
        rows = conn.execute("SELECT date, symbol, name, prob, channel FROM candidates WHERE date=? ORDER BY prob DESC", (today,)).fetchall()
        if not rows:
            rows = conn.execute("SELECT date, symbol, name, prob, channel FROM candidates ORDER BY date DESC LIMIT 200").fetchall()
        conn.close()
        main, disc = [], []
        for r in rows:
            entry = {'symbol': r[1], 'name': r[2], 'prob': r[3]}
            (disc if r[4] == 'discovery' else main).append(entry)
        return {'date': rows[0][0] if rows else date.today().isoformat(),
                'count': len(main) + len(disc), 'model': 'XGBRanker (Rank IC=0.37, 板块中性化)',
                'main': main, 'discovery': disc}
    except Exception:
        return {'date': date.today().isoformat(), 'count': 0, 'model': '待运行', 'main': [], 'discovery': []}

def _get_signal_events(limit=30):
    try:
        conn = sqlite3.connect(TRADE_DB)
        today = date.today().isoformat()
        rows = conn.execute("SELECT time, symbol, signal, action FROM signal_events WHERE date=? ORDER BY id DESC LIMIT ?", (today, limit)).fetchall()
        conn.close()
        return [{'time': r[0], 'symbol': r[1], 'signal': r[2], 'action': r[3]} for r in rows]
    except Exception:
        return []

def _get_signal_stats():
    try:
        conn = sqlite3.connect(TRADE_DB)
        rows = conn.execute("SELECT signal, win_count, total_count, win_rate, total_pnl, avg_return FROM signal_stats").fetchall()
        conn.close()
        return {r[0]: {'win_count': r[1], 'total_count': r[2], 'win_rate': r[3], 'total_pnl': r[4], 'avg_return': r[5]} for r in rows}
    except Exception:
        return {}

def _get_rejected():
    try:
        conn = sqlite3.connect(TRADE_DB)
        today = date.today().isoformat()
        rows = conn.execute("SELECT time, symbol, price, reason FROM rejected_signals WHERE date=? ORDER BY id DESC LIMIT 50", (today,)).fetchall()
        conn.close()
        return {'date': today, 'rejected': [{'time': r[0], 'symbol': r[1], 'price': r[2], 'reason': r[3]} for r in rows]}
    except Exception:
        return {'date': date.today().isoformat(), 'rejected': []}

def _get_l3_progress():
    log_file = os.path.join(SUPERQUANT_ROOT, 'ml', 'build_features.log')
    if not os.path.exists(log_file): return {'status': '未启动'}
    try:
        with open(log_file) as f:
            log_lines = [l.strip() for l in f.readlines() if 'L3-' in l]
        threads = {}
        for l in log_lines:
            m = re.search(r'\[(L3-\d)\]\s+(\d+)/(\d+)\(成功(\d+)/失败(\d+)\)\s+(\d+)s\s+预计(\d+)s', l)
            if m:
                tid, cur, total, ok, fail, elapsed, eta = m.groups()
                threads[tid] = {'cur': int(cur), 'total': int(total), 'ok': int(ok), 'fail': int(fail), 'elapsed': int(elapsed), 'eta': int(eta)}
            elif '完成' in l:
                m2 = re.search(r'\[(L3-\d)\].*成功(\d+)/失败(\d+),\s*(\d+)s', l)
                if m2:
                    tid, ok, fail, elapsed = m2.groups()
                    threads[tid] = {'ok': int(ok), 'fail': int(fail), 'elapsed': int(elapsed), 'done': True}
        conn = sqlite3.connect(MKT_DB)
        row = conn.execute("SELECT COUNT(DISTINCT date), COUNT(*), MIN(date), MAX(date) FROM daily_features WHERE main_net_in IS NOT NULL").fetchone()
        conn.close()
        running = any(not t.get('done') for t in threads.values())
        return {'status': '运行中' if running else ('完成' if threads else '待机'),
                'days': row[0] if row else 0, 'rows': row[1] if row else 0,
                'date_range': f"{row[2]}~{row[3]}" if row and row[2] else '-',
                'threads': {k: v for k, v in sorted(threads.items())}}
    except Exception:
        return {'status': '错误'}

def _get_market_regime():
    def _calc():
        try:
            conn = sqlite3.connect(MKT_DB)
            row = conn.execute("SELECT close FROM daily WHERE symbol='000001' ORDER BY date DESC LIMIT 20").fetchall()
            conn.close()
            if len(row) >= 20:
                ret20 = (row[0][0] / row[-1][0] - 1)
                if ret20 > 0.05: return {'label': '牛市 🐂', 'mult': 1.2, 'desc': f'指数20日 +{ret20*100:.1f}%'}
                elif ret20 < -0.05: return {'label': '熊市 🐻', 'mult': 0.5, 'desc': f'指数20日 {ret20*100:.1f}%'}
                else: return {'label': '震荡 ➡️', 'mult': 1.0, 'desc': f'指数20日 {ret20*100:+.1f}%'}
        except Exception: pass
        return {'label': 'Unknown', 'mult': 1.0, 'desc': ''}
    return _cached('regime', 60, _calc)

def _get_trades(limit=10):
    try:
        conn = sqlite3.connect(TRADE_DB)
        rows = conn.execute("SELECT date, symbol, side, price, shares, pnl, pnl_pct, strategy, created_at FROM sim_trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [{'date': r[0], 'symbol': r[1], 'side': r[2], 'price': r[3], 'shares': r[4], 'pnl': r[5], 'pnl_pct': r[6], 'reason': r[7], 'time': r[8]} for r in rows]
    except Exception:
        return []

# ═══════════════════════════════
# API 路由
# ═══════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def api_state():
    account = _cached_account()
    cands = _get_candidates()
    return jsonify({
        'cash': account['cash'], 'equity': account['equity'],
        'date': account['date'], 'n_positions': account['n_positions'],
        'initial_capital': get_capital(),
        'ml_candidates': len(cands.get('main', [])),
        'ml_date': cands.get('date', ''),
        'performance': _cached_performance(),
        'regime': _get_market_regime(),
    })

@app.route('/api/candidates')
def api_candidates():
    return jsonify(_get_candidates())

@app.route('/api/metrics')
def api_metrics():
    """绩效指标: 夏普/最大回撤/卡尔玛/胜率/盈亏比/期望值。"""
    def _calc():
        try:
            conn = sqlite3.connect(TRADE_DB)
            # 所有已平仓交易
            rows = conn.execute("SELECT pnl FROM sim_trades WHERE side='sell' AND pnl IS NOT NULL ORDER BY date, id").fetchall()
            # 计算每日收益序列 (从 paper_account 或 sim_trades capital_after)
            caps = conn.execute("SELECT date, capital_after FROM sim_trades WHERE capital_after IS NOT NULL ORDER BY id").fetchall()
            conn.close()

            pnls = [r[0] for r in rows if r[0] is not None]
            n = len(pnls)
            if n < 3:
                return {'sharpe': 0, 'max_drawdown': 0, 'calmar': 0, 'win_rate': 0, 'profit_factor': 0, 'expectancy': 0, 'total_trades': n, 'winning_trades': 0, 'total_pnl': sum(pnls)}

            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            win_rate = len(wins) / n
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = abs(sum(losses) / len(losses)) if losses else 1
            profit_factor = sum(wins) / abs(sum(losses)) if losses else 999
            expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

            # 从 capital_after 构建权益曲线
            if caps:
                eq_curve = [c[1] for c in caps]
                # 日收益率
                rets = []
                peak = eq_curve[0]
                max_dd = 0
                for v in eq_curve:
                    if v > peak: peak = v
                    dd = (peak - v) / peak if peak > 0 else 0
                    if dd > max_dd: max_dd = dd
                # 近似夏普 (假设等间隔)
                if len(eq_curve) >= 5:
                    import numpy as np
                    rets = np.diff(eq_curve) / np.array(eq_curve[:-1])
                    mu, sigma = np.mean(rets), np.std(rets)
                    sharpe = (mu / sigma * np.sqrt(252)) if sigma > 0 else 0
                    ann_ret = mu * 252
                else:
                    sharpe, ann_ret = 0, 0
                calmar = ann_ret / max_dd if max_dd > 0 else 0
            else:
                sharpe, max_dd, calmar, ann_ret = 0, 0, 0, 0

            total_pnl = sum(pnls)
            return {'sharpe': round(sharpe, 2), 'max_drawdown': round(max_dd * 100, 2),
                    'calmar': round(calmar, 2), 'win_rate': round(win_rate * 100, 1),
                    'profit_factor': round(profit_factor, 2), 'expectancy': round(expectancy, 2),
                    'total_trades': n, 'winning_trades': len(wins),
                    'total_pnl': round(total_pnl, 2), 'annual_return': round(ann_ret * 100, 2)}
        except Exception:
            return {'sharpe': 0, 'max_drawdown': 0, 'calmar': 0, 'win_rate': 0, 'profit_factor': 0, 'expectancy': 0, 'total_trades': 0}
    return jsonify(_cached('metrics', 60, _calc))

@app.route('/api/l3-progress')
def api_l3_progress():
    return jsonify(_get_l3_progress())

@app.route('/api/northstar')
def api_northstar():
    account = _cached_account()
    equity = account['equity']
    initial = get_capital(); target = 1_000_000
    earned = equity - initial; need = target - initial
    progress = (earned / need * 100) if need > 0 else 0
    return jsonify({'initial': initial, 'current': equity, 'target': target,
                    'progress_pct': round(progress, 4), 'daily_target_pct': 2.1,
                    'remaining': round(target - equity, 0),
                    'est_days': int(need / max(earned, 0.01)) if earned > 0 else 99999,
                    'date': account['date']})

@app.route('/api/positions')
def api_positions():
    positions = []
    try:
        conn = sqlite3.connect(TRADE_DB)
        today = date.today().isoformat()
        buys = conn.execute("SELECT symbol, price, shares, date FROM sim_trades WHERE side='buy' AND date=? AND symbol NOT IN (SELECT symbol FROM sim_trades WHERE side='sell' AND date=?) ORDER BY date", (today, today)).fetchall()
        conn.close()
        if buys:
            quotes = _cached_quotes([r[0] for r in buys])
            for r in buys:
                sym = r[0]; q = quotes.get(sym, {})
                positions.append({"symbol": sym, "name": q.get('name', ''), "shares": r[2], "price": r[1], "current": q.get('price', r[1]), "buy_date": r[3]})
    except Exception: pass
    return jsonify(positions)

@app.route('/api/trades')
def api_trades():
    return jsonify(_get_trades(30))

@app.route('/api/rejected')
def api_rejected():
    return jsonify(_get_rejected())

@app.route('/api/signals')
def api_signals():
    return jsonify(_get_signal_events(30))

@app.route('/api/signal-stats')
def api_signal_stats():
    return jsonify(_get_signal_stats())

if __name__ == '__main__':
    print("=" * 50)
    print("superquant Web — 攻击性涨停捕捉")
    print(f"  地址: http://localhost:8522")
    print(f"  初始资金: {get_capital():,.0f}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8522, debug=True)
