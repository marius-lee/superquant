"""superquant Web — ML 驱动的涨停预测监控面板。

API:
  /api/state         — 实时状态 (资金/持仓)
  /api/candidates    — ML 预测 Top20 候选 + 概率
  /api/l3-progress   — L3 资金流回补进度
  /api/paper-account — 模拟账户
  /api/northstar     — 北极星进度
  /api/positions     — 持仓
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
CANDIDATE_FILE = os.path.join(SUPERQUANT_ROOT, "pre_market", "candidate.json")
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
    """从 ML 预测结果读取今日候选。"""
    if os.path.exists(CANDIDATE_FILE):
        try:
            with open(CANDIDATE_FILE) as f:
                data = json.load(f)
                cands = data.get('candidates', [])
                return {
                    'date': data.get('date', date.today().isoformat()),
                    'count': len(cands),
                    'model': 'XGBoost (AUC=0.96)',
                    'candidates': [{'symbol': c['symbol'], 'prob': c['prob']} for c in cands[:10]],
                }
        except Exception:
            pass
    return {'date': date.today().isoformat(), 'count': 0, 'model': '待运行', 'candidates': []}


def _get_l3_progress():
    """读取 L3 资金流回补进度。"""
    log_file = os.path.join(SUPERQUANT_ROOT, 'ml', 'build_features.log')
    if not os.path.exists(log_file):
        return {'status': '未启动', 'last_line': ''}
    try:
        with open(log_file) as f:
            lines = f.readlines()
        last_lines = [l.strip() for l in lines[-5:] if 'L3' in l or '全部完成' in l or 'ERROR' in l]
        # 统计 DB 中的 L3 数据量
        conn = sqlite3.connect(os.path.join(QUANT_ROOT, 'data', 'market.db'))
        row = conn.execute("SELECT COUNT(DISTINCT date) as days, COUNT(*) as rows, MIN(date), MAX(date) FROM daily_features WHERE main_net_in IS NOT NULL").fetchone()
        conn.close()
        return {
            'status': '运行中' if any('完成' not in l for l in last_lines[:1]) else '待机',
            'days': row[0] if row else 0,
            'rows': row[1] if row else 0,
            'date_range': f"{row[2]}~{row[3]}" if row and row[2] else '无',
            'last_log': last_lines[-1] if last_lines else '',
        }
    except Exception:
        return {'status': '错误', 'last_line': ''}


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
    cands = _get_candidates()
    return jsonify({
        'cash': account['cash'],
        'equity': account['equity'],
        'date': account['date'],
        'n_positions': len(positions),
        'n_signals': len(trades),
        'initial_capital': INITIAL_CAPITAL,
        'ml_candidates': len(cands.get('candidates', [])),
        'ml_date': cands.get('date', ''),
    })


@app.route('/api/candidates')
def api_candidates():
    return jsonify(_get_candidates())


@app.route('/api/l3-progress')
def api_l3_progress():
    return jsonify(_get_l3_progress())


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
