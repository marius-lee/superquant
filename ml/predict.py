#!/usr/bin/env python3
"""XGBoost 涨停预测 — 盘前运行, 输出 Top N 候选。用法: python ml/predict.py"""
import os, sys, time, sqlite3, json
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.expanduser("~/project/quant"), "data", "market.db")
CANDIDATE_FILE = os.path.join(os.path.dirname(MODEL_DIR), "pre_market", "candidate.json")
TOP_N = 20  # 输出TopN

print("=" * 60)
print("XGBoost 涨停预测 — 盘前扫描")
print("=" * 60)

# ── 1. 加载模型 ──
model_path = os.path.join(MODEL_DIR, 'model.json')
if not os.path.exists(model_path):
    print(f"❌ 模型未找到: {model_path}"); sys.exit(1)
import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model(model_path)
print(f"✅ 模型加载: {model_path}")

# ── 2. 加载今日日线 ──
t0 = time.time()
conn = sqlite3.connect(DB_PATH)
has_ft = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_features'").fetchone() is not None
if has_ft:
    sql = """SELECT d.symbol, d.date, d.open, d.high, d.low, d.close, d.volume, d.amount, d.turnover,
        COALESCE(f.ksft,0), COALESCE(f.slope,0), COALESCE(f.ptc,0), COALESCE(f.vol_5min,0), COALESCE(f.max_ret,0), COALESCE(f.min_ret,0),
        COALESCE(f.main_net_in,0), COALESCE(f.main_net_ratio,0), COALESCE(f.super_large_in,0), COALESCE(f.large_in,0),
        COALESCE(f.lhb_net_buy,0), COALESCE(f.lhb_buy_ratio,0), COALESCE(f.lhb_count,0), COALESCE(f.lhb_exists,0),
        COALESCE(f.sector_limit_count,0), COALESCE(f.sector_rank,0)
    FROM daily d LEFT JOIN daily_features f ON d.symbol=f.symbol AND d.date=f.date
    WHERE d.date >= (SELECT DATE(MAX(date), '-20 days') FROM daily) ORDER BY d.symbol, d.date"""
else:
    sql = """SELECT symbol,date,open,high,low,close,volume,amount,turnover
    FROM daily WHERE date >= (SELECT DATE(MAX(date), '-20 days') FROM daily) ORDER BY symbol, date"""
rows = conn.execute(sql).fetchall()
conn.close()
today = max(r[1] for r in rows) if rows else '?'
print(f"  加载: {len(rows)}行, 日期={today}, {time.time()-t0:.0f}s")

# ── 3. 特征工程 ──
t0 = time.time()
daily_data = defaultdict(list)
for r in rows: daily_data[r[0]].append(r)
X_list, symbols = [], []
n_feat = len(rows[0]) - 2
for sym, rs in daily_data.items():
    rs_10 = rs[-10:] if len(rs) >= 10 else rs
    if len(rs_10) < 10: continue
    closes = np.array([float(r[4]) for r in rs_10])
    r = rs_10[-1]
    if closes[-2] <= 0: continue
    ret_1d = closes[-1]/closes[-2]-1
    ret_5d = closes[-1]/closes[-6]-1 if len(closes)>=2 and closes[-6]>0 else 0
    vols_5 = np.array([float(rs_10[j][5]) for j in range(-5,0)])
    vol_ratio = float(r[5])/max(np.mean(vols_5),1)
    rets_5d = [(closes[j]/closes[j-1]-1) for j in range(-4,0) if closes[j-1]>0]
    vol_5d = np.std(rets_5d) if len(rets_5d)>2 else 0
    gap = float(r[2])/closes[-2]-1
    turnover = float(r[8]) if r[8] else float(r[5])/10000.0
    amt_log = np.log(max(float(r[6]),1))
    hl_ratio = float(r[3])/max(float(r[4]),0.01)-1
    close_pos = (float(r[4])-float(r[4]))/(max(float(r[3])-float(r[4]),0.01)+0.001)
    ma20 = np.mean(closes[-21:-1]) if len(closes)>=21 else np.mean(closes)
    ma_dev = closes[-1]/ma20-1 if ma20>0 else 0
    base = [ret_1d,ret_5d,vol_ratio,vol_5d,gap,turnover,amt_log,hl_ratio,close_pos,ma_dev]
    extra = [float(r[i]) if r[i] is not None else 0.0 for i in range(9, len(r))]
    feat = base + extra
    # 补齐到模型期望的特征数
    while len(feat) < 26:
        feat.append(0.0)
    feat = feat[:26]
    if any(abs(v)>100 for v in feat): continue
    X_list.append(feat); symbols.append(sym)
X = np.array(X_list, dtype=np.float32)
print(f"  特征: {X.shape}, {time.time()-t0:.0f}s")

# ── 4. 预测 ──
t0 = time.time()
probs = model.predict_proba(X)[:, 1]
results = sorted(zip(symbols, probs), key=lambda x: -x[1])
print(f"  预测: {len(results)}只, {time.time()-t0:.0f}s")

# ── 5. 输出 ──
print(f"\n🎯 今日涨停候选 Top {TOP_N}:")
print(f"{'排名':<5} {'股票':<10} {'名称':<10} {'P(涨停)':>8} {'昨收':>8}")
conn = sqlite3.connect(DB_PATH)
for i, (sym, prob) in enumerate(results[:TOP_N], 1):
    row = conn.execute("SELECT name,close FROM stocks s JOIN daily d ON s.symbol=d.symbol WHERE d.symbol=? ORDER BY d.date DESC LIMIT 1",(sym,)).fetchone()
    name = row[0] if row else '?'
    px = row[1] if row else 0
    print(f"{i:<5} {sym:<10} {name:<10} {prob:>8.4f} {px:>8.2f}")
conn.close()

# 保存候选
os.makedirs(os.path.dirname(CANDIDATE_FILE), exist_ok=True)
candidates = [{'symbol': s, 'prob': round(float(p), 4), 'name': ''} for s, p in results[:TOP_N]]
with open(CANDIDATE_FILE, 'w') as f:
    json.dump({'date': today, 'candidates': candidates}, f, indent=2, ensure_ascii=False)
print(f"\n✅ 候选保存: {CANDIDATE_FILE} ({len(candidates)}只)")
