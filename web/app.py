"""superquant Web — 统一缓存层, 全 DB 读取。"""

import json, os, sys, sqlite3, time, re, threading, math
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

# ═══════════════
# 缓存
# ═══════════════
_cache = {}
_cache_lock = threading.Lock()

def _cached(key, ttl, fn):
    now = time.time()
    with _cache_lock:
        e = _cache.get(key)
        if e and (now - e[0]) < ttl: return e[1]
    val = fn()
    with _cache_lock: _cache[key] = (now, val)
    return val

def _trade_date():
    """最后交易日 — 周末/节假日用最后数据日。"""
    try:
        c = sqlite3.connect(TRADE_DB)
        r = c.execute("SELECT MAX(date) FROM sim_trades").fetchone()[0]
        c.close()
        return r or date.today().isoformat()
    except: return date.today().isoformat()

def _quotes(symbols):
    if not symbols: return {}
    key = 'q_' + ','.join(sorted(symbols))
    return _cached(key, 2, lambda: __import__('execution.quote', fromlist=['fetch_quotes']).fetch_quotes(symbols))

def _account():
    def calc():
        try:
            c = sqlite3.connect(TRADE_DB); td = _trade_date()
            row = c.execute("SELECT capital_after FROM sim_trades WHERE capital_after IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
            pos = c.execute("SELECT symbol, SUM(shares) FROM sim_trades WHERE side='buy' AND date=? AND symbol NOT IN (SELECT symbol FROM sim_trades WHERE side='sell' AND date=?) GROUP BY symbol", (td,td)).fetchall()
            c.close()
            cash = row[0] if row else get_capital(); pv = 0
            if pos:
                qs = _quotes([r[0] for r in pos])
                for sym, sh in pos: pv += sh * qs.get(sym,{}).get('price',0)
            return {'cash': round(cash,2), 'equity': round(cash+pv,2), 'date': td, 'n_positions': len(pos)}
        except: return {'cash': get_capital(), 'equity': get_capital(), 'date': date.today().isoformat(), 'n_positions': 0}
    return _cached('acct', 2, calc)

def _perf():
    def calc():
        try:
            c = sqlite3.connect(TRADE_DB); td = _trade_date()
            s = c.execute("SELECT COALESCE(SUM(pnl),0), COUNT(*) FROM sim_trades WHERE side='sell' AND date=?",(td,)).fetchone()
            rl, sc = s[0] or 0, s[1] or 0
            w = c.execute("SELECT COUNT(*) FROM sim_trades WHERE side='sell' AND pnl>0 AND date=?",(td,)).fetchone()[0]
            b = c.execute("SELECT COUNT(*) FROM sim_trades WHERE side='buy' AND date=?",(td,)).fetchone()[0]
            pr = c.execute("SELECT symbol, SUM(shares), SUM(price*shares) FROM sim_trades WHERE side='buy' AND date=? AND symbol NOT IN (SELECT symbol FROM sim_trades WHERE side='sell' AND date=?) GROUP BY symbol",(td,td)).fetchall()
            c.close(); ur = 0
            if pr:
                qs = _quotes([r[0] for r in pr])
                for sym, sh, cost in pr: ur += sh * qs.get(sym,{}).get('price',0) - cost
            return {'realized_pnl':round(rl,2),'unrealized_pnl':round(ur,2),'total_pnl':round(rl+ur,2),'buy_count':b,'sell_count':sc,'win_rate':round((w/sc*100)if sc>0 else 0,1)}
        except: return {'realized_pnl':0,'unrealized_pnl':0,'total_pnl':0,'buy_count':0,'sell_count':0,'win_rate':0}
    return _cached('perf', 2, calc)

# ═══════════════
# DB 读取
# ═══════════════

def _candidates():
    try:
        c = sqlite3.connect(TRADE_DB); td = _trade_date()
        rows = c.execute("SELECT date, symbol, name, prob, channel FROM candidates WHERE date=? ORDER BY prob DESC",(td,)).fetchall()
        if not rows: rows = c.execute("SELECT date, symbol, name, prob, channel FROM candidates ORDER BY date DESC LIMIT 200").fetchall()
        c.close(); main, disc = [], []
        for r in rows:
            e = {'symbol':r[1],'name':r[2],'prob':r[3]}
            (disc if r[4]=='discovery' else main).append(e)
        return {'date': rows[0][0] if rows else date.today().isoformat(), 'count': len(main)+len(disc),
                'model':'XGBRanker (Rank IC=0.37)', 'main':main, 'discovery':disc}
    except: return {'date':date.today().isoformat(),'count':0,'model':'待运行','main':[],'discovery':[]}

def _signals(limit=30):
    try:
        c = sqlite3.connect(TRADE_DB)
        rows = c.execute("SELECT time, symbol, signal, action FROM signal_events WHERE date=? ORDER BY id DESC LIMIT ?",(_trade_date(),limit)).fetchall()
        c.close()
        return [{'time':r[0],'symbol':r[1],'signal':r[2],'action':r[3]} for r in rows]
    except: return []

def _signal_stats():
    try:
        c = sqlite3.connect(TRADE_DB)
        rows = c.execute("SELECT signal, win_count, total_count, win_rate, total_pnl, avg_return FROM signal_stats").fetchall()
        c.close()
        return {r[0]:{'win_count':r[1],'total_count':r[2],'win_rate':r[3],'total_pnl':r[4],'avg_return':r[5]} for r in rows}
    except: return {}

def _rejected():
    try:
        c = sqlite3.connect(TRADE_DB); td = _trade_date()
        rows = c.execute("SELECT time, symbol, price, reason FROM rejected_signals WHERE date=? ORDER BY id DESC LIMIT 50",(td,)).fetchall()
        c.close()
        return {'date':td, 'rejected':[{'time':r[0],'symbol':r[1],'price':r[2],'reason':r[3]} for r in rows]}
    except: return {'date':date.today().isoformat(),'rejected':[]}

def _l3():
    lf = os.path.join(SUPERQUANT_ROOT, 'ml', 'build_features.log')
    if not os.path.exists(lf): return {'status':'未启动'}
    try:
        with open(lf) as f: ll = [l.strip() for l in f.readlines() if 'L3-' in l]
        th = {}
        for l in ll:
            m = re.search(r'\[(L3-\d)\]\s+(\d+)/(\d+)\(成功(\d+)/失败(\d+)\)\s+(\d+)s\s+预计(\d+)s', l)
            if m:
                tid, cur, total, ok, fail, elapsed, eta = m.groups()
                th[tid] = {'cur':int(cur),'total':int(total),'ok':int(ok),'fail':int(fail),'elapsed':int(elapsed),'eta':int(eta)}
            elif '完成' in l:
                m2 = re.search(r'\[(L3-\d)\].*成功(\d+)/失败(\d+),\s*(\d+)s', l)
                if m2:
                    tid, ok, fail, elapsed = m2.groups()
                    th[tid] = {'ok':int(ok),'fail':int(fail),'elapsed':int(elapsed),'done':True}
        conn = sqlite3.connect(MKT_DB)
        row = conn.execute("SELECT COUNT(DISTINCT date), COUNT(*), MIN(date), MAX(date) FROM daily_features WHERE main_net_in IS NOT NULL").fetchone()
        conn.close()
        running = any(not t.get('done') for t in th.values())
        return {'status':'运行中' if running else ('完成' if th else '待机'), 'days':row[0] if row else 0, 'rows':row[1] if row else 0,
                'date_range':f"{row[2]}~{row[3]}" if row and row[2] else '-', 'threads':{k:v for k,v in sorted(th.items())}}
    except: return {'status':'错误'}

def _regime():
    def calc():
        try:
            c = sqlite3.connect(MKT_DB)
            row = c.execute("SELECT close FROM daily WHERE symbol='000001' ORDER BY date DESC LIMIT 20").fetchall()
            c.close()
            if len(row)>=20:
                r20 = row[0][0]/row[-1][0]-1
                if r20>0.05: return {'label':'牛市 🐂','mult':1.2,'desc':f'指数20日 +{r20*100:.1f}%'}
                elif r20<-0.05: return {'label':'熊市 🐻','mult':0.5,'desc':f'指数20日 {r20*100:.1f}%'}
                else: return {'label':'震荡 ➡️','mult':1.0,'desc':f'指数20日 {r20*100:+.1f}%'}
        except: pass
        return {'label':'Unknown','mult':1.0,'desc':''}
    return _cached('regime', 60, calc)

def _trades(limit=10):
    try:
        c = sqlite3.connect(TRADE_DB)
        rows = c.execute("SELECT date, symbol, side, price, shares, pnl, pnl_pct, strategy, created_at FROM sim_trades ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
        c.close()
        return [{'date':r[0],'symbol':r[1],'side':r[2],'price':r[3],'shares':r[4],'pnl':r[5],'pnl_pct':r[6],'reason':r[7],'time':r[8]} for r in rows]
    except: return []

# ═══════════════
# API
# ═══════════════

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/state')
def api_state():
    acct = _account(); cands = _candidates()
    return jsonify({'cash':acct['cash'],'equity':acct['equity'],'date':acct['date'],'n_positions':acct['n_positions'],
                    'initial_capital':get_capital(),'ml_candidates':len(cands.get('main',[])),'ml_date':cands.get('date',''),
                    'performance':_perf(),'regime':_regime()})

@app.route('/api/candidates')
def api_candidates(): return jsonify(_candidates())

@app.route('/api/l3-progress')
def api_l3_progress(): return jsonify(_l3())

@app.route('/api/northstar')
def api_northstar():
    acct = _account(); eq = acct['equity']; init = get_capital(); tgt = 1_000_000
    earned = eq - init; need = tgt - init; prog = (earned/need*100) if need>0 else 0
    return jsonify({'initial':init,'current':eq,'target':tgt,'progress_pct':round(prog,4),
                    'daily_target_pct':2.1,'remaining':round(tgt-eq,0),
                    'est_days':int(need/max(earned,0.01)) if earned>0 else 99999,'date':acct['date']})

@app.route('/api/positions')
def api_positions():
    pos = []
    try:
        c = sqlite3.connect(TRADE_DB); td = _trade_date()
        buys = c.execute("SELECT symbol, price, shares, date FROM sim_trades WHERE side='buy' AND date=? AND symbol NOT IN (SELECT symbol FROM sim_trades WHERE side='sell' AND date=?) ORDER BY date",(td,td)).fetchall()
        c.close()
        if buys:
            qs = _quotes([r[0] for r in buys])
            for r in buys:
                sym=r[0]; q=qs.get(sym,{})
                pos.append({'symbol':sym,'name':q.get('name',''),'shares':r[2],'price':r[1],'current':q.get('price',r[1]),'buy_date':r[3]})
    except: pass
    return jsonify(pos)

@app.route('/api/trades')
def api_trades(): return jsonify(_trades(30))

@app.route('/api/rejected')
def api_rejected(): return jsonify(_rejected())

@app.route('/api/signals')
def api_signals(): return jsonify(_signals(30))

@app.route('/api/signal-stats')
def api_signal_stats(): return jsonify(_signal_stats())

@app.route('/api/metrics')
def api_metrics():
    def calc():
        try:
            c = sqlite3.connect(TRADE_DB)
            rows = c.execute("SELECT pnl FROM sim_trades WHERE side='sell' AND pnl IS NOT NULL ORDER BY date, id").fetchall()
            caps = c.execute("SELECT date, capital_after FROM sim_trades WHERE capital_after IS NOT NULL ORDER BY id").fetchall()
            c.close()
            pnls = [r[0] for r in rows if r[0] is not None]; n = len(pnls)
            if n<3: return {'sharpe':0,'max_drawdown':0,'calmar':0,'win_rate':0,'profit_factor':0,'expectancy':0,'total_trades':n}
            wins = [p for p in pnls if p>0]; losses = [p for p in pnls if p<0]
            wr = len(wins)/n; aw = sum(wins)/len(wins) if wins else 0
            al = abs(sum(losses)/len(losses)) if losses else 1
            pf = sum(wins)/abs(sum(losses)) if losses else 999
            ex = wr*aw - (1-wr)*al
            import numpy as np
            if caps:
                eq = [c[1] for c in caps]; peak=eq[0]; mdd=0
                for v in eq:
                    if v>peak: peak=v
                    dd=(peak-v)/peak if peak>0 else 0
                    if dd>mdd: mdd=dd
                if len(eq)>=5:
                    rets=np.diff(eq)/np.array(eq[:-1])
                    mu,sigma=np.mean(rets),np.std(rets)
                    sh=(mu/sigma*np.sqrt(252)) if sigma>0 else 0; ann=mu*252
                else: sh=ann=0
                cal=ann/mdd if mdd>0 else 0
            else: sh=mdd=cal=ann=0
            return {'sharpe':round(sh,2),'max_drawdown':round(mdd*100,2),'calmar':round(cal,2),'win_rate':round(wr*100,1),
                    'profit_factor':round(pf,2),'expectancy':round(ex,2),'total_trades':n,'winning_trades':len(wins),
                    'total_pnl':round(sum(pnls),2),'annual_return':round(ann*100,2)}
        except: return {'sharpe':0,'max_drawdown':0,'calmar':0,'win_rate':0,'profit_factor':0,'expectancy':0,'total_trades':0}
    return jsonify(_cached('metrics',60,calc))

if __name__ == '__main__':
    print("="*50); print("superquant Web"); print(f"  http://localhost:8522"); print("="*50)
    app.run(host='0.0.0.0', port=8522, debug=True)
