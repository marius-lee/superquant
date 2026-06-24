"""XGBoost V2: 日线 + DB特征 (L2-L5预计算) → 终极模型。用法: python ml/train_v2.py"""
import os, sys, time, sqlite3, json
from collections import defaultdict
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.expanduser("~/project/quant"), "data", "market.db")

print("=" * 60)
print("XGBoost V2: 日线 + L2-L5 DB特征")
print("=" * 60)

# ── 1. 加载数据 ──
print("\n[1/4] 加载...")
t0 = time.time()
conn = sqlite3.connect(DB_PATH)

# 检查 daily_features 表是否存在
has_features = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='daily_features'"
).fetchone() is not None

# 主查询: 日线 JOIN 预计算特征
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

# ── 2. 特征 + 标签 ──
print("\n[2/4] 特征工程...")
t0 = time.time()
daily_data = defaultdict(list)
for r in rows:
    daily_data[r[0]].append(r)

X_list, y_list = [], []
n_cols = len(rows[0])
n_feat = min(n_cols - 2, 26)  # 减去symbol+date

for sym, rs in daily_data.items():
    if len(rs) < 32: continue
    closes = np.array([float(r[4]) for r in rs])
    for i in range(25, len(rs)-1):
        r = rs[i]
        if closes[i-1] <= 0: continue

        # 日线特征 (10个)
        ret_1d = closes[i]/closes[i-1]-1
        ret_5d = closes[i]/closes[i-5]-1 if i>=5 and closes[i-5]>0 else 0
        vols_5 = np.array([float(rs[j][5]) for j in range(i-4,i+1)])
        vol_ratio = float(r[5])/max(np.mean(vols_5),1)
        rets_5d = [(closes[j]/closes[j-1]-1) for j in range(i-4,i+1) if closes[j-1]>0]
        vol_5d = np.std(rets_5d) if len(rets_5d)>2 else 0
        gap = float(r[2])/closes[i-1]-1
        turnover = float(r[8]) if r[8] else float(r[5])/10000.0
        amt_log = np.log(max(float(r[6]),1))
        hl_ratio = float(r[3])/max(float(r[4]),0.01)-1
        close_pos = (float(r[4])-float(r[4]))/(max(float(r[3])-float(r[4]),0.01)+0.001)
        ma20 = np.mean(closes[max(0,i-20):i+1])
        ma_dev = closes[i]/ma20-1 if ma20>0 else 0
        base = [ret_1d,ret_5d,vol_ratio,vol_5d,gap,turnover,amt_log,hl_ratio,close_pos,ma_dev]

        # L2-L5特征 (从DB, 如果存在)
        extra = []
        if has_features and len(r) > 9:
            for v in r[9:]:
                extra.append(float(v) if v is not None else 0.0)
        else:
            extra = [0.0] * 16

        feat = base + extra
        if any(abs(v)>100 for v in feat): continue
        next_ret = closes[i+1]/closes[i]-1 if closes[i]>0 else 0
        label = 1 if next_ret>=0.095 else 0
        X_list.append(feat)
        y_list.append(label)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
print(f"  {X.shape}, 正样本: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"  耗时: {time.time()-t0:.0f}s")

# ── 3. 训练 ──
print(f"\n[3/4] 训练 XGBoost ({X.shape[1]}特征)...")
t0 = time.time()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
sw = (len(y_tr)-y_tr.sum())/max(y_tr.sum(),1)
model = xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,
    subsample=0.8,colsample_bytree=0.8,scale_pos_weight=sw,
    objective='binary:logistic',eval_metric='auc',random_state=42,n_jobs=4)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
print(f"  AUC-ROC: {auc:.4f} | Baseline=0.50 | 提升: {(auc-0.5)/0.5*100:+.0f}%")
print(f"  耗时: {time.time()-t0:.0f}s")

# ── 4. 评估 ──
print(f"\n[4/4] 评估...")
yp = model.predict(X_te)
print(classification_report(y_te, yp, target_names=['未涨停','涨停'], zero_division=0))
names = ['ret_1d','ret_5d','vol_ratio','vol_5d','gap','turnover','amt_log','hl_ratio','close_pos','ma_dev',
         'ksft','slope','ptc','vol_5min','max_ret','min_ret',
         'main_net_in','main_net_ratio','super_large_in','large_in',
         'lhb_net_buy','lhb_buy_ratio','lhb_count','lhb_exists',
         'sector_limit_count','sector_rank']
imp = sorted(zip(names[:X.shape[1]], model.feature_importances_), key=lambda x:-x[1])
print("特征重要性 (Top 10):")
for n,i in imp[:10]: print(f"  {n:<20} {i:.4f} {'█'*int(i*100)}")

path = os.path.join(MODEL_DIR,'model.json')
model.save_model(path)
print(f"\n✅ 模型: {path} | AUC={auc:.4f} {'✅' if auc>0.55 else '⚠️'}")
print(f"\n📊 对比: POC(0.75)→V1(0.78)→V2({auc:.4f})")
