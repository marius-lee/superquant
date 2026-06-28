"""XGBoost V3: Learning to Rank — 多日最大涨幅排序。

三级漏斗 第一级:
  5000只 → Top 200
  XGBRanker objective='rank:ndcg'
  target = max(1d_return, 3d_return, 5d_return)

来源:
  Grinold & Kahn: Alpha = Volatility × IC × Score. 回归输出=Score.
  Narang: α模型的终极目标是横截面收益排序能力.
  Chan: ML做条件收益预测, 不预测方向.

用法: python ml/train.py
"""
import os, sys, time, sqlite3, pickle
from collections import defaultdict
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.expanduser("~/project/quant"), "data", "market.db")

print("=" * 60)
print("XGBoost V3: Learning to Rank (排除BSE, 多日最大涨幅)")
print("=" * 60)

# ── 1. 加载数据 ──
print("\n[1/4] 加载...")
t0 = time.time()
conn = sqlite3.connect(DB_PATH)

has_features = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='daily_features'"
).fetchone() is not None

if has_features:
    sql = """SELECT d.symbol, d.date, d.open, d.high, d.low, d.close, d.volume, d.amount, d.turnover,
                f.ksft, f.slope, f.ptc, f.vol_5min, f.max_ret, f.min_ret,
                f.main_net_in, f.main_net_ratio, f.super_large_in, f.large_in,
                f.lhb_net_buy, f.lhb_buy_ratio, f.lhb_count, f.lhb_exists,
                f.sector_limit_count, f.sector_rank
             FROM daily d LEFT JOIN daily_features f ON d.symbol=f.symbol AND d.date=f.date
             WHERE d.date>='2025-06-01' ORDER BY d.symbol, d.date"""
    print("  ✅ daily_features 表存在, 使用完整L2-L5特征")
else:
    sql = """SELECT symbol,date,open,high,low,close,volume,amount,turnover
             FROM daily WHERE date>='2025-06-01' ORDER BY symbol,date"""
    print("  ⚠️ daily_features 不存在, 仅用日线特征 (先运行 ml/build_features.py --all)")

rows = conn.execute(sql).fetchall()
conn.close()
print(f"  加载: {len(rows)}行, {time.time()-t0:.0f}s")

# ── 2. 特征 + target ──
print("\n[2/4] 特征工程 (target = max(1d, 3d, 5d return))...")
t0 = time.time()
daily_data = defaultdict(list)
for r in rows:
    daily_data[r[0]].append(r)

X_list, y_list, qid_list = [], [], []
date_to_qid = {}
_next_qid = 0
n_cols = len(rows[0])

# ── 预计算市场特征 (Layer 2: 每交易日大盘指标) ──
mkt_features = {}
all_dates = sorted(set(r[1] for rs in daily_data.values() for r in rs))
idx_data = {r[1]: float(r[4]) for sym, rs in daily_data.items() if sym == '000001' for r in rs}
for d in all_dates:
    prev_dates = [dd for dd in all_dates if dd <= d]
    d_idx = prev_dates.index(d)
    i20 = prev_dates[max(0,d_idx-19):d_idx+1]
    idx_20d = (idx_data[i20[-1]] / idx_data[i20[0]] - 1) if len(i20)>=10 and i20[0] in idx_data and i20[-1] in idx_data else 0
    i60 = prev_dates[max(0,d_idx-59):d_idx+1]
    if len(i60) >= 30:
        i_cl = [idx_data[dd] for dd in i60 if dd in idx_data]
        if len(i_cl)>5:
            i_rets = np.diff(i_cl) / np.array(i_cl[:-1])
            idx_60d_vol = float(np.std(i_rets))
        else: idx_60d_vol = 0.0
    else: idx_60d_vol = 0.0
    up = total = 0
    for _, rs in daily_data.items():
        for r in rs:
            if r[1] == d:
                total += 1
                if float(r[5]) > float(r[2]): up += 1  # close>open=上涨
                break
    breadth = up/max(total,1)
    mkt_features[d] = [round(idx_20d,4), round(idx_60d_vol,4), round(breadth,4)]
mkt_computed = True

for sym, rs in daily_data.items():
    if sym.startswith('920'): continue  # 排除 BSE
    if len(rs) < 38: continue           # 需要未来5天数据
    closes = np.array([float(r[5]) for r in rs])  # r[5]=close (r[4]是low)

    for i in range(30, len(rs) - 6):    # 留6天做未来计算
        r = rs[i]
        if closes[i-1] <= 0: continue

        # ── 日线特征 (11个, +abs_ret_1d关注度) ──
        ret_1d = closes[i]/closes[i-1]-1
        abs_ret_1d = abs(ret_1d)  # 来源: A股U型效应 — 极端涨跌都有后续, 关注度代理
        ret_5d = closes[i]/closes[i-5]-1 if i>=5 and closes[i-5]>0 else 0
        vols_5 = np.array([float(rs[j][6]) for j in range(i-4,i+1)])
        vol_ratio = float(r[6])/max(np.mean(vols_5),1)
        rets_5d = [(closes[j]/closes[j-1]-1) for j in range(i-4,i+1) if closes[j-1]>0]
        vol_5d = np.std(rets_5d) if len(rets_5d)>2 else 0
        gap = float(r[2])/closes[i-1]-1
        turnover = float(r[8]) if r[8] else float(r[5])/10000.0
        amt_log = np.log(max(float(r[7]),1))
        hl_ratio = float(r[3])/max(float(r[4]),0.01)-1
        close_pos = (float(r[5])-float(r[2]))/(max(float(r[3])-float(r[4]),0.01)+0.001)
        ma20 = np.mean(closes[max(0,i-20):i+1])
        ma_dev = closes[i]/ma20-1 if ma20>0 else 0
        base = [ret_1d,abs_ret_1d,ret_5d,vol_ratio,vol_5d,gap,turnover,amt_log,hl_ratio,close_pos,ma_dev]

        # ── L2-L5特征 ──
        extra = []
        if has_features and len(r) > 9:
            for v in r[9:]:
                extra.append(float(v) if v is not None else 0.0)
        else:
            extra = [0.0] * 16

        feat = base + extra
        # Layer 2: 市场状态特征 (idx_20d_ret, idx_60d_vol, mkt_breadth)
        feat += [0.0, 0.0, 0.0]
        if len(feat) < 30:
            feat += [0.0] * (30 - len(feat))
        feat = feat[:30]
        date_str = r[1]  # date 字段 — 提前定义, Layer 2 需要
        if date_str in mkt_features:
            feat[-3:] = mkt_features[date_str]
        feat = np.clip(feat, -10, 10).tolist()

        # ── target: 多日最大涨幅 (来源: 三级漏斗 — 找连涨/大涨潜力) ──
        r1 = closes[i+1]/closes[i] - 1
        r3 = closes[min(i+3, len(closes)-2)]/closes[i] - 1
        r5 = closes[min(i+5, len(closes)-2)]/closes[i] - 1
        max_ret = max(r1, r3, r5)
        # Sortino 调整: 过程中最低点惩罚 (来源: 下跌波动率惩罚, 过滤会被止损斩仓的票)
        min_path = r1
        for offset in range(2, 6):
            if i + offset < len(closes):
                min_path = min(min_path, closes[i+offset]/closes[i] - 1)
        downside = abs(min(0, min_path))
        target = max_ret / (1 + downside)

        # ── qid: 同一日期=同一排名组 (来源: XGBRanker 要求) ──
        if date_str not in date_to_qid:
            date_to_qid[date_str] = _next_qid
            _next_qid += 1
        qid = date_to_qid[date_str]

        X_list.append(feat)
        y_list.append(target)
        qid_list.append(qid)

X = np.array(X_list, dtype=np.float32)
y_raw = np.array(y_list, dtype=np.float32)  # 连续 target
groups = np.array(qid_list, dtype=np.int32)

# ── 离散化: 每日组内按 target 排名 → 5级相关性 (XGBoost ranking 要求整数标签) ──
y_discrete = np.zeros_like(y_raw, dtype=np.int32)
for g in np.unique(groups):
    mask = groups == g
    n = mask.sum()
    if n < 5:
        continue
    ranks = np.argsort(np.argsort(y_raw[mask]))  # 组内排名 0~(n-1)
    # 分5级: 0=底20%,1=次20%,2=中20%,3=次高20%,4=顶20%
    y_discrete[mask] = np.clip((ranks * 5 // n).astype(np.int32), 0, 4)

y = y_discrete
print(f"  {X.shape}, target范围: [{y_raw.min():.4f}, {y_raw.max():.4f}], "
      f"均值={y_raw.mean():.4f}, 中位数={np.median(y_raw):.4f}, "
      f"排名组: {len(np.unique(groups))}天, "
      f"标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"  耗时: {time.time()-t0:.0f}s")

# ── 3. Walk-Forward 训练 + 验证 ──
print(f"\n[3/4] Walk-Forward 验证 XGBRanker ({X.shape[1]}特征)...")
t0 = time.time()

# 按 group 排序 (XGBRanker 要求)
sort_idx = np.argsort(groups)
X, y, y_raw, groups = X[sort_idx], y[sort_idx], y_raw[sort_idx], groups[sort_idx]

# Walk-forward: 3个扩展窗口 (来源: Chan — 时间序列必须用walk-forward避免look-ahead)
all_ic = []
windows = [0.4, 0.6, 0.8]  # 训练占比
last_split = 0
final_model = None

for wi, train_pct in enumerate(windows):
    split = int(len(X) * train_pct)
    while split < len(X) and groups[split] == groups[split-1]:
        split += 1

    X_tr, X_te = X[:split], X[split:]  # 扩展窗: 所有历史数据训练
    y_tr, y_te = y[:split], y[split:]
    y_raw_te = y_raw[split:]
    grp_tr, grp_te = groups[:split], groups[split:]

    if len(np.unique(grp_te)) < 3:
        continue

    n_days = len(np.unique(grp_te))
    m = xgb.XGBRanker(
        objective='rank:pairwise',
        n_estimators=min(100 + wi*50, 200), max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=4,
    )
    m.fit(X_tr, y_tr, qid=grp_tr, verbose=False)
    preds = m.predict(X_te)

    # 每日 Rank IC
    win_ic = []
    for g in np.unique(grp_te):
        mask = grp_te == g
        if mask.sum() < 10:
            continue
        ic = np.corrcoef(preds[mask], y_raw_te[mask])[0, 1]
        if not np.isnan(ic):
            all_ic.append(ic)
            win_ic.append(ic)

    if wi == len(windows) - 1:
        final_model = m

    print(f"  窗{wi+1}({train_pct:.0%}训练→{n_days}天验证): "
          f"IC={np.mean(win_ic):.4f}±{np.std(win_ic):.4f}, IC>0={np.mean(np.array(win_ic)>0)*100:.0f}%")

mean_ic = np.mean(all_ic)
print(f"\n[4/4] 评估 (Walk-Forward, {len(all_ic)}天验证)...")
print(f"  Rank IC 均值: {mean_ic:.4f}  ({'✅' if mean_ic>0.03 else '⚠️' if mean_ic>0 else '❌'})")
print(f"  Rank IC 标准差: {np.std(all_ic):.4f}")
print(f"  IC > 0 占比: {np.mean(np.array(all_ic)>0)*100:.1f}%")
print(f"  IC 衰减: 窗1→窗3 = 检查输出趋势")
print(f"  耗时: {time.time()-t0:.0f}s")

# 最终模型用全数据训练
model = xgb.XGBRanker(
    objective='rank:pairwise',
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=4,
)
model.fit(X, y, qid=groups, verbose=False)

# 特征重要性
names = ['ret_1d','abs_ret_1d','ret_5d','vol_ratio','vol_5d','gap','turnover','amt_log','hl_ratio','close_pos','ma_dev',
         'ksft','slope','ptc','vol_5min','max_ret','min_ret',
         'main_net_in','main_net_ratio','super_large_in','large_in',
         'lhb_net_buy','lhb_buy_ratio','lhb_count','lhb_exists',
         'sector_limit_count','sector_rank',
         'idx_20d_ret','idx_60d_vol','mkt_breadth']
imp = sorted(zip(names[:X.shape[1]], model.feature_importances_), key=lambda x: -x[1])
print("特征重要性 (Top 10):")
for n, i in imp[:10]:
    print(f"  {n:<20} {i:.4f} {'█'*int(i*100)}")

path = os.path.join(MODEL_DIR, 'model.json')
model.save_model(path)
print(f"\n✅ XGBRanker: {path} | Rank IC={mean_ic:.4f}")

# ── 孤立森林 ──
t_if = time.time()
if_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=4)
if_model.fit(X[:500000])  # 内存限制, 取前50万样本
if_path = os.path.join(MODEL_DIR, 'if_model.pkl')
with open(if_path, 'wb') as f:
    pickle.dump(if_model, f)
print(f"✅ IF: {if_path} | anomaly_ratio={float(np.mean(if_model.predict(X[:500000])==-1)):.2%} | {time.time()-t_if:.0f}s")

print(f"\n📊 对比: V2(分类 AUC=0.95)→V3(排序 Rank IC={mean_ic:.4f})")
