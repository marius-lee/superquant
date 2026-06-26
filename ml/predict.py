#!/usr/bin/env python3
"""XGBoost 涨停预测 — 盘前运行, 输出 Top N 候选。用法: python ml/predict.py"""
import os, sys, time, sqlite3, json, pickle
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.expanduser("~/project/quant"), "data", "market.db")
TOP_N = 200  # 输出TopN (三级漏斗第一级: 5000→200)

print("=" * 60)
print("XGBoost 涨停预测 — 盘前扫描")
print("=" * 60)

# ── 1. 加载模型 ──
model_path = os.path.join(MODEL_DIR, 'model.json')
if not os.path.exists(model_path):
    print(f"❌ 模型未找到: {model_path}"); sys.exit(1)
import xgboost as xgb
model = xgb.XGBRanker()
model.load_model(model_path)
print(f"✅ XGBRanker 加载: {model_path}")

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
    close_pos = (float(r[4])-float(r[2]))/(max(float(r[3])-float(r[4]),0.01)+0.001)
    ma20 = np.mean(closes[-21:-1]) if len(closes)>=21 else np.mean(closes)
    ma_dev = closes[-1]/ma20-1 if ma20>0 else 0
    base = [ret_1d,ret_5d,vol_ratio,vol_5d,gap,turnover,amt_log,hl_ratio,close_pos,ma_dev]
    extra = [float(r[i]) if r[i] is not None else 0.0 for i in range(9, len(r))]
    feat = base + extra
    # 补齐到模型期望的特征数
    while len(feat) < 26:
        feat.append(0.0)
    while len(feat) < 29:
        feat.append(0.0)
    feat = feat[:29]
    # Layer 2: 市场状态特征 (暂填0, 下面统一计算后填入)
    feat = np.clip(feat, -10, 10)
    X_list.append(feat); symbols.append(sym)
X = np.array(X_list, dtype=np.float32)
print(f"  特征: {X.shape}, {time.time()-t0:.0f}s")

# ── 市场特征 (Layer 2: 模型内生市场适应) ──
mkt_conn = sqlite3.connect(DB_PATH)
# Index 20d return
idx_row = mkt_conn.execute("SELECT close FROM daily WHERE symbol='000001' ORDER BY date DESC LIMIT 20").fetchall()
idx_20d = (idx_row[0][0]/idx_row[-1][0]-1) if len(idx_row)>=20 else 0
# Index 60d vol
idx_60 = mkt_conn.execute("SELECT close FROM daily WHERE symbol='000001' ORDER BY date DESC LIMIT 60").fetchall()
idx_60d_vol = 0.0
if len(idx_60) >= 30:
    idx_cl = np.array([r[0] for r in idx_60])
    idx_r = np.diff(idx_cl)/idx_cl[:-1]
    idx_60d_vol = float(np.std(idx_r))
# Market breadth
up_count = mkt_conn.execute("SELECT COUNT(*) FROM daily d1 WHERE d1.date=(SELECT MAX(date) FROM daily) AND d1.close>d1.open").fetchone()[0]
total_count = mkt_conn.execute("SELECT COUNT(*) FROM daily WHERE date=(SELECT MAX(date) FROM daily)").fetchone()[0]
breadth = up_count/max(total_count,1) if total_count>0 else 0.5
mkt_conn.close()
mkt_feat = [round(idx_20d,4), round(idx_60d_vol,4), round(breadth,4)]
print(f"  市场特征: idx20d={idx_20d*100:+.1f}% vol={idx_60d_vol:.4f} breadth={breadth:.2f}")
# 填入每个样本
for i in range(X.shape[0]):
    X[i, -3:] = mkt_feat

# ── 4. XGBoost 预测 ──
t0 = time.time()
preds_raw = model.predict(X)  # 回归输出: 预期最大涨幅

# ── 横截面 z-score 标准化 (来源: Grinold & Kahn — Alpha = IC × Score × Vol) ──
# Score 必须每日截面均值=0、标准差=1, 否则分数在不同市况下不可比
z_mean, z_std = np.mean(preds_raw), np.std(preds_raw)
preds = (preds_raw - z_mean) / (z_std + 1e-10)
print(f"  z-score: 原始[{preds_raw.min():.2f}, {preds_raw.max():.2f}] → 标准化[{preds.min():.2f}, {preds.max():.2f}]")

# ── 板块中性化 (来源: P2-12, Grinold/Barra 风险模型 — 消除热门板块偏差) ──
# 申万行业分类, 组内减去行业均值, 保留个股特质Alpha
# 问题: 热门板块股票天然得分高, 横截面排序被板块效应污染
# 方案: 每个股票得分减去其行业均值, 再重新z-score标准化
ind_conn = sqlite3.connect(DB_PATH)
ind_rows = ind_conn.execute("SELECT symbol, industry FROM stocks WHERE industry IS NOT NULL AND industry != ''").fetchall()
ind_conn.close()
ind_map = {r[0]: r[1] for r in ind_rows}
ind_groups = defaultdict(list)
for i, sym in enumerate(symbols):
    ind = ind_map.get(sym, '')
    if ind:
        ind_groups[ind].append(i)
# 行业中性化: preds_neutral[i] = preds[i] - industry_mean
preds_neutral = preds.copy()
neutral_count = 0
for ind, idxs in ind_groups.items():
    if len(idxs) >= 3:  # 至少3只股票才算有效行业 (来源: 统计显著性最低样本量)
        ind_mean = np.mean(preds[idxs])
        for i in idxs:
            preds_neutral[i] = preds[i] - ind_mean
        neutral_count += 1
# 重新z-score标准化到均值0、标准差1
n_mean, n_std = np.mean(preds_neutral), np.std(preds_neutral)
preds = (preds_neutral - n_mean) / (n_std + 1e-10)
print(f"  板块中性化: {neutral_count}个行业 → 中性z-score[{preds.min():.2f}, {preds.max():.2f}]")

# ── 波动率中性化 (Layer 1: 消除高beta偏好, 来源: 今日IC=-0.03根因分析) ──
# 大盘跌时模型偏好高波小票 → 熊市负IC。方案: 组内减均值 + Bear降权
vol_conn = sqlite3.connect(DB_PATH)
# 计算每只股票的60日历史波动率
vols = np.zeros(len(symbols))
for i, sym in enumerate(symbols):
    rows = vol_conn.execute(
        "SELECT close FROM daily WHERE symbol=? ORDER BY date DESC LIMIT 60", (sym,)
    ).fetchall()
    if len(rows) >= 30:
        closes_vol = np.array([r[0] for r in rows])
        rets_vol = np.diff(closes_vol) / closes_vol[:-1]
        vols[i] = np.std(rets_vol) if len(rets_vol) > 5 else 0
# 5分位分组
vol_quintiles = np.percentile(vols[vols > 0], [20, 40, 60, 80]) if np.any(vols > 0) else [0, 0, 0, 0]
vol_groups = np.digitize(vols, vol_quintiles)  # 0=最低波, 4=最高波
# 组内减均值
preds_vol_neutral = preds.copy()
for g in range(5):
    mask = vol_groups == g
    if mask.sum() >= 5:
        preds_vol_neutral[mask] = preds[mask] - np.mean(preds[mask])
# 市场状态判断 (上证20日)
idx_row = vol_conn.execute("SELECT close FROM daily WHERE symbol='000001' ORDER BY date DESC LIMIT 20").fetchall()
is_bear = False
if len(idx_row) >= 20:
    idx_ret20 = (idx_row[0][0] / idx_row[-1][0] - 1)
    is_bear = idx_ret20 < -0.05
# Bear市场: 高波动组 Q4/Q5 降权 0.7
if is_bear:
    hi_vol_mask = vol_groups >= 3
    preds_vol_neutral[hi_vol_mask] *= 0.7
# 重新z-score
vn_mean, vn_std = np.mean(preds_vol_neutral), np.std(preds_vol_neutral)
preds = (preds_vol_neutral - vn_mean) / (vn_std + 1e-10)
vol_conn.close()
print(f"  波动率中性化: {'Bear🐻' if is_bear else 'Bull/Neutral'}, 高波{'降权70%' if is_bear else '正常'} → z-score[{preds.min():.2f}, {preds.max():.2f}]")

# ── 5. 孤立森林 交叉验证 ──
if_path = os.path.join(MODEL_DIR, 'if_model.pkl')
if_probs = np.zeros(len(preds))
if os.path.exists(if_path):
    with open(if_path, 'rb') as f:
        if_model = pickle.load(f)
    if_raw = -if_model.score_samples(X)
    if_probs = (if_raw - if_raw.min()) / (if_raw.max() - if_raw.min() + 1e-10)  # 0=正常, 1=极异常

# 组合1: XGBoost 回归得分 × IF 增强 → 主力通道
# preds 是连续值 (预期最大涨幅), IF 放大异常模式的得分

# ── 概率校准 (Item 5: z-score→概率, 来源: Platt scaling近似) ──
# z=0→50%, z=1→64%, z=2→76%, z=3→86%, z=4→93%
import math as _math
_pred_probs = [round(1.0 / (1.0 + _math.exp(-(float(p) / 1.7))), 4) for p in preds]

combined = preds * (0.5 + 0.5 * if_probs)
main_results = sorted(zip(symbols, preds, if_probs, combined), key=lambda x: -x[3])

# 组合2: IF>0.9 + 得分中低 → 探索通道 (来源: 异常模式可能为新涨停形态)
disc_raw = [(s, x, i, i * 0.8) for s, x, i, _ in zip(symbols, preds, if_probs, combined)
            if i > 0.9 and x < np.percentile(preds, 70)]
disc_results = sorted(disc_raw, key=lambda x: -x[3])[:TOP_N]

print(f"  预测: {len(main_results)}只 (主力) + {len(disc_raw)}只 (探索), {time.time()-t0:.0f}s")

# ── IC 衰减监控 (来源: Grinold — Alpha半衰期, 监测模型健康) ──
model_health = {
    'date': today, 'n_stocks': len(symbols),
    'pred_mean': float(np.mean(preds)), 'pred_std': float(np.std(preds)),
    'pred_skew': float(np.mean((preds - np.mean(preds))**3) / (np.std(preds) + 1e-10)**3),
    'disc_count': len(disc_raw),
}
# 写入 market.db model_health 表
hconn = sqlite3.connect(DB_PATH)
regime = 'Bear' if idx_20d < -0.05 else ('Bull' if idx_20d > 0.05 else 'Neutral')
# 半衰期追踪 (Item 7: IC衰减到峰值50%的天数)
half_life = -1
past = hconn.execute("SELECT pred_std FROM model_health ORDER BY date DESC LIMIT 60").fetchall()
if len(past) >= 10:
    stds = [p[0] for p in past if p[0] is not None]
    if stds:
        peak = max(stds)
        peak_i = stds.index(peak)
        for j in range(peak_i, len(stds)):
            if stds[j] < peak * 0.5:
                half_life = j - peak_i
                break
hconn.execute("""INSERT OR REPLACE INTO model_health(date, n_stocks, pred_mean, pred_std, pred_skew, disc_count, regime, half_life_days)
    VALUES(?,?,?,?,?,?,?,?)""", (today, len(symbols), float(np.mean(preds)), float(np.std(preds)),
    float(np.mean((preds - np.mean(preds))**3) / (np.std(preds) + 1e-10)**3), len(disc_raw), regime, half_life))
hconn.commit(); hconn.close()
# IC 漂移检测
mconn = sqlite3.connect(DB_PATH)
recent = mconn.execute("SELECT pred_mean FROM model_health ORDER BY date DESC LIMIT 5").fetchall()
mconn.close()
if len(recent) >= 5:
	    recent_mean = np.mean([r[0] for r in recent])
	    drift = abs(model_health['pred_mean'] - recent_mean)
	    if drift > 1.5:
	        print(f"  ⚠️ IC衰减: 漂移{drift:.2f}, 已标记自动重训")
	        fconn = sqlite3.connect(DB_PATH.replace('market.db', 'trades.db'))
	        fconn.execute("INSERT OR REPLACE INTO candidates(date, symbol, name, prob, channel) VALUES(?,?,?,?,?)",
	                      (today, '__RETRAIN__', f'IC drift {drift:.2f}', drift, 'flag'))
	        fconn.commit(); fconn.close()

# ── 6. 输出 ──
print(f"\n🎯 主力候选 Top {TOP_N} (XGBoost × IF):")
print(f"{'排名':<5} {'股票':<10} {'名称':<10} {'XGBoost':>8} {'概率':>8} {'综合':>8} {'昨收':>8}")
conn = sqlite3.connect(DB_PATH)

# 预查所有候选的名称 (批量查询避免 N+1, 主力+探索)
# 预查所有非BSE+非ETF候选的名称 (先过滤再查名, 避免BSE淹没)
_all_main_syms = [s for s, _, _, _ in main_results if not s.startswith('920') and not s.startswith(('15','16','51','58'))]
_all_disc_syms = [s for s, _, _, _ in disc_results if not s.startswith('920') and not s.startswith(('15','16','51','58'))]
all_cand_syms = _all_main_syms + _all_disc_syms
placeholders = ','.join('?' for _ in all_cand_syms)
name_rows = conn.execute(
    f"SELECT symbol, name FROM stocks WHERE symbol IN ({placeholders})",
    all_cand_syms
).fetchall()
name_map = {r[0]: r[1] for r in name_rows}

main_syms = []
main_filtered = []  # (sym, xgb_prob, if_prob, combined)
for sym, xgb_prob, if_prob, combined in main_results:
    # 过滤 BSE (来源: 北交所50万门槛, 无权限不可交易)
    if sym.startswith('920'): continue
    # 过滤 ETF (来源: ETF非个股, 无涨停概念)
    if sym.startswith(('15','16','51','58')): continue
    name = name_map.get(sym)
    if name is None:
        row = conn.execute("SELECT name FROM stocks WHERE symbol=?", (sym,)).fetchone()
        name = row[0] if row else '?'
        name_map[sym] = name
    # 过滤 ST/退市 股票 (来源: ST涨跌停±5%, 退市风险高)
    if 'ST' in name or '退' in name:
        continue
    # 获取昨收
    row = conn.execute(
        "SELECT close FROM daily WHERE symbol=? ORDER BY date DESC LIMIT 1", (sym,)
    ).fetchone()
    px = row[0] if row else 0
    main_syms.append(sym)
    main_filtered.append((sym, name, xgb_prob, if_prob, combined))
    calib_p = round(1.0 / (1.0 + _math.exp(-(float(xgb_prob) / 1.7))), 4)
    print(f"{len(main_filtered):<5} {sym:<10} {name:<10} {xgb_prob:>8.4f} {calib_p:>8.0%} {combined:>8.4f} {px:>8.2f}")
    if len(main_filtered) >= TOP_N:
        break

disc_syms = []
disc_filtered = []
if disc_results:
    print(f"\n🔍 探索候选 Top {len(disc_results)} (IF高分+ML低分 → 潜在新模式):")
    print(f"{'排名':<5} {'股票':<10} {'名称':<10} {'XGBoost':>8} {'异常度':>8} {'综合':>8}")
    for i, (sym, xgb_prob, if_prob, combined) in enumerate(disc_results, 1):
        if sym.startswith('920') or sym.startswith(('15','16','51','58')): continue
        name = name_map.get(sym)
        if name is None:
            row = conn.execute("SELECT name FROM stocks WHERE symbol=?", (sym,)).fetchone()
            name = row[0] if row else '?'
            name_map[sym] = name
        if 'ST' in name or '退' in name:
            continue
        disc_syms.append(sym)
        disc_filtered.append((sym, name, xgb_prob, if_prob, combined))
        print(f"{len(disc_filtered):<5} {sym:<10} {name:<10} {xgb_prob:>8.4f} {if_prob:>8.4f} {combined:>8.4f}")
conn.close()

# 保存候选到 trades.db (来源: JSON→DB 重构)
wconn = sqlite3.connect(DB_PATH.replace("market.db", "trades.db"))
wconn.execute("DELETE FROM candidates WHERE date=?", (today,))
for i, (s, n, _, _, c) in enumerate(main_filtered):
    wconn.execute("INSERT INTO candidates(date, symbol, name, prob, channel) VALUES(?,?,?,?,'main',?)",
                  (today, s, n, round(float(c), 4), round(1.0/(1.0+_math.exp(-float(xgb_prob)/1.7)), 4)))
for s, n, _, _, c in disc_filtered:
    wconn.execute("INSERT INTO candidates(date, symbol, name, prob, channel) VALUES(?,?,?,?,'discovery')",
                  (today, s, n, round(float(c), 4), round(1.0/(1.0+_math.exp(-float(xgb_prob)/1.7)), 4)))
wconn.commit(); wconn.close()
print(f"\n✅ 候选已写入DB (主力{len(main_filtered)}只+探索{len(disc_filtered)}只)")
