"""superquant Web — 攻击性涨停捕捉监控面板。

API:
  /api/state         — 实时状态 (资金/持仓/候选)
  /api/candidates    — 今日候选池 (L1-L5筛选结果)
  /api/paper-account — 模拟账户摘要
  /api/northstar     — 北极星进度
  /api/positions     — 当前持仓
  /api/trades        — 交易记录
  /                  — 主页面
"""

import json, os, sys, sqlite3
from datetime import date, datetime
from flask import Flask, jsonify, render_template, request

SUPERQUANT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANT_ROOT = os.path.expanduser("~/project/quant")
sys.path.insert(0, SUPERQUANT_ROOT)
sys.path.insert(0, QUANT_ROOT)

TRADE_DB = os.path.join(QUANT_ROOT, "data", "trades.db")
ACTIVE_PARAMS = os.path.join(SUPERQUANT_ROOT, "config", "active_params.json")
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
    try:
        conn = sqlite3.connect(TRADE_DB)
        rows = conn.execute(
            "SELECT symbol, date, price, shares, pnl, pnl_pct, reason FROM sim_trades "
            "WHERE side='buy' ORDER BY date DESC, id DESC LIMIT 50"
        ).fetchall()
        conn.close()
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


def _get_candidates():
    """获取今日候选池状态。L1-L5 筛选进度。"""
    today = date.today().isoformat()
    try:
        # 从内存/缓存读取 (暂时回退到 Hikyuu 基础数据)
        from hikyuu.interactive import sm
        # 简化: 返回候选池统计框架，待 pre_market_scanner 实现后填充
        return {
            'date': today,
            'layers': [
                {'name': 'L1 竞价筛选', 'count': 0, 'threshold': 'gap≥2%, 竞价量>昨量×5%'},
                {'name': 'L2 技术形态', 'count': 0, 'threshold': 'KSFT+SLOPE+PTC'},
                {'name': 'L3 资金流', 'count': 0, 'threshold': 'AData 大单净买入>0'},
                {'name': 'L4 龙虎榜', 'count': 0, 'threshold': '昨日上榜+游资买入'},
                {'name': 'L5 板块共振', 'count': 0, 'threshold': '板块≥3涨停'},
            ],
            'final': [],  # 最终候选
        }
    except Exception:
        return {'date': today, 'layers': [], 'final': []}


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
        'stop_base': params.get('stops', {}).get('adaptive_stop_base', 0.05),
        'candidates': _get_candidates()['final'],
    })


@app.route('/api/candidates')
def api_candidates():
    return jsonify(_get_candidates())


@app.route('/api/paper-account')
def api_paper_account():
    return jsonify(_get_account())


@app.route('/api/northstar')
def api_northstar():
    account = _get_account()
    progress = (account['equity'] / INITIAL_CAPITAL - 1) * 100
    target = 1_000_000
    remaining = target - account['equity']
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


@app.route('/api/positions')
def api_positions():
    return jsonify(_get_positions())


@app.route('/api/trades')
def api_trades():
    return jsonify(_get_trades(30))


if __name__ == '__main__':
    print("=" * 50)
    print("superquant Web — 攻击性涨停捕捉")
    print(f"  地址: http://localhost:8522")
    print(f"  初始资金: ¥{INITIAL_CAPITAL:,.0f}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8522, debug=True)
