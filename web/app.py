"""superquant Web — 自主交易代理监控面板。

API:
  /api/state         — 实时状态 (资金/持仓/信号计数)
  /api/factors       — 因子截面 Top20 + IC
  /api/factor-score  — 单股票因子明细
  /api/paper-account — 模拟账户摘要
  /api/northstar     — 北极星进度
  /api/signals       — 信号列表
  /api/trades        — 交易记录
  /api/positions     — 当前持仓
  /                  — 主页面
"""

import json, os, sys, sqlite3
from datetime import date, datetime
from flask import Flask, jsonify, render_template, request

SUPERQUANT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANT_ROOT = os.path.expanduser("~/project/quant")
# Flask子进程需要此路径才能 import app.factors
sys.path.insert(0, SUPERQUANT_ROOT)
sys.path.insert(0, QUANT_ROOT)
TRADE_DB = os.path.join(QUANT_ROOT, "data", "trades.db")
ACTIVE_PARAMS = os.path.join(SUPERQUANT_ROOT, "config", "active_params.json")
AUTO_TUNING = os.path.join(SUPERQUANT_ROOT, "config", "auto_tuning.json")
INITIAL_CAPITAL = 5000.0

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default or {}


def _get_account():
    """获取模拟账户最新状态。"""
    try:
        conn = sqlite3.connect(TRADE_DB)
        row = conn.execute(
            "SELECT cash, equity, date FROM paper_account ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            return {'cash': row[0], 'equity': row[1], 'date': row[2]}
    except Exception:
        pass
    return {'cash': INITIAL_CAPITAL, 'equity': INITIAL_CAPITAL, 'date': date.today().isoformat()}


def _get_positions():
    """获取当前持仓。"""
    try:
        conn = sqlite3.connect(TRADE_DB)
        rows = conn.execute(
            "SELECT symbol, date, price, shares, pnl, pnl_pct, reason FROM sim_trades "
            "WHERE side='buy' ORDER BY date DESC, id DESC LIMIT 50"
        ).fetchall()
        conn.close()
        # 简化: 最近买入且未卖出的为持仓
        positions = []
        seen = set()
        for r in rows:
            sym = r[0]
            if sym not in seen:
                seen.add(sym)
                positions.append({
                    'symbol': sym, 'buy_date': r[1], 'price': r[2],
                    'shares': r[3], 'reason': r[6] or ''
                })
        return positions[:3]
    except Exception:
        return []


def _get_trades(limit=10):
    """获取最近交易记录。"""
    try:
        conn = sqlite3.connect(TRADE_DB)
        rows = conn.execute(
            "SELECT date, symbol, side, price, shares, pnl, pnl_pct, reason, created_at "
            "FROM sim_trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [{'date': r[0], 'symbol': r[1], 'side': r[2], 'price': r[3],
                 'shares': r[4], 'pnl': r[5], 'pnl_pct': r[6], 'reason': r[7],
                 'time': r[8]} for r in rows]
    except Exception:
        return []


def _get_factor_topn(n=20):
    """获取因子截面 Top N。"""
    try:
        from hikyuu.interactive import sm, Query
        from app.factors import compute_factor_scores
        stocks = []
        for mkt in ['SH', 'SZ']:
            try:
                market_stocks = sm.get_stock_list(
                    lambda s, m=mkt: s.market == m and s.valid)
                stocks.extend(list(market_stocks))
            except:
                pass
        scores = compute_factor_scores(stocks, Query(-30))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{'symbol': c, 'score': round(s, 4), 'name': sm[c].name if c in sm else '?'}
                for c, s in ranked[:n]]
    except Exception as e:
        return [{'error': str(e)}]


def _get_factor_ic():
    """获取因子IC数据。"""
    report = _load_json(AUTO_TUNING, {})
    ic_data = report.get('ic', {})
    result = {}
    for name, data in ic_data.items():
        result[name] = {
            'description': data.get('description', name),
            'n_stocks': data.get('n_stocks', 0),
        }
    return result


# ═══════════════════════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/state')
def api_state():
    account = _get_account()
    positions = _get_positions()
    trades = _get_trades(5)
    params = _load_json(ACTIVE_PARAMS, {})
    return jsonify({
        'cash': account['cash'],
        'equity': account['equity'],
        'date': account['date'],
        'n_positions': len(positions),
        'n_signals': len(trades),
        'initial_capital': INITIAL_CAPITAL,
        'weights': params.get('factor_weights', {}),
        'stop_base': params.get('stops', {}).get('adaptive_stop_base', 0.05),
        'kelly_win_rate': params.get('kelly', {}).get('win_rate', 0.55),
    })


@app.route('/api/factors')
def api_factors():
    topn = request.args.get('top', 20, type=int)
    return jsonify({
        'top': _get_factor_topn(topn),
        'ic': _get_factor_ic(),
        'source': 'auto_tuning.json' if os.path.exists(AUTO_TUNING) else 'default',
    })


@app.route('/api/factor-score')
def api_factor_score():
    symbol = request.args.get('symbol', '')
    if not symbol:
        return jsonify({'error': '?symbol=SH600000 required'}), 400
    try:
        from hikyuu.interactive import sm, Query
        from app.factors import compute_factor_scores
        scores = compute_factor_scores([sm[symbol]], Query(-30))
        detail = {
            'symbol': symbol,
            'name': sm[symbol].name if symbol in sm else '?',
            'score': scores.get(symbol, 0),
        }
        return jsonify(detail)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/paper-account')
def api_paper_account():
    return jsonify(_get_account())


@app.route('/api/northstar')
def api_northstar():
    account = _get_account()
    progress = (account['equity'] / INITIAL_CAPITAL - 1) * 100
    target = 100_0000
    remaining = target - account['equity']
    # 按当前速度估算天数
    days = 1
    daily_rate = progress / max(days, 1)
    est_days = int(remaining / max(account['equity'] * daily_rate / 100, 0.001))
    return jsonify({
        'initial': INITIAL_CAPITAL,
        'current': account['equity'],
        'target': target,
        'progress_pct': round(progress, 2),
        'daily_target_pct': 2.1,
        'remaining': round(remaining, 0),
        'est_days': est_days,
        'date': account['date'],
    })


@app.route('/api/signals')
def api_signals():
    return jsonify(_get_trades(20))


@app.route('/api/positions')
def api_positions():
    return jsonify(_get_positions())


@app.route('/api/trades')
def api_trades():
    return jsonify(_get_trades(30))


# ═══════════════════════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 50)
    print("superquant Web — 自主交易代理监控面板")
    print(f"  地址: http://localhost:8522")
    print(f"  初始资金: ¥{INITIAL_CAPITAL:,.0f}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8522, debug=True)
