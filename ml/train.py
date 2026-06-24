"""XGBoost 完整版: L1-L5 全部特征 + 全量训练数据。用法: python ml/train.py"""
import os, sys, time, sqlite3, json
from collections import defaultdict
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.features import build_all_features, FEATURE_NAMES

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.expanduser("~/project/quant"), "data", "market.db")

print("=" * 60)
print("XGBoost 完整版: L1-L5 特征")
print("=" * 60)

# ── 1. 加载数据 ──
print("\n[1/4] 加载数据...")
t0 = time.time()
conn = sqlite3.connect(DB_PATH)
rows = conn.execute(
    "SELECT symbol,date,open,high,low,close,volume,amount,turnover "
    "FROM daily WHERE date>='2025-06-01' ORDER BY symbol,date"
).fetchall()
conn.close()

daily_data = defaultdict(list)
for r in rows:
    daily_data[r[0]].append((r[1], float(r[2]), float(r[3]), float(r[4]),
                              float(r[5]), float(r[6]), float(r[7]), r[8]))
print(f"  加载: {len(rows)}行, {len(daily_data)}只, {time.time()-t0:.0f}s")

# 板块涨停统计
sector_counts = defaultdict(int)
for r in rows:
    if float(r[2]) > 0 and float(r[5]) / float(r[2]) - 1 >= 0.095:
        conn2 = sqlite3.connect(DB_PATH)
        row = conn2.execute("SELECT market FROM stocks WHERE symbol=?", (r[0],)).fetchone()
        conn2.close()
        if row:
            sector_counts[row[0]] += 1

# ── 2. 特征工程 ──
print("\n[2/4] 特征工程 (1年数据)...")
t0 = time.time()
X, y, meta = build_all_features(daily_data, sector_counts)
print(f"  {X.shape}, 正样本: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"  耗时: {time.time()-t0:.0f}s")

# ── 3. 训练 ──
print(f"\n[3/4] 训练 XGBoost ({X.shape[1]}特征)...")
t0 = time.time()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
sw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=sw, objective='binary:logistic',
    eval_metric='auc', random_state=42, n_jobs=4,
)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
print(f"  AUC-ROC: {auc:.4f} | Baseline=0.50 | 提升: {(auc-0.5)/0.5*100:+.0f}%")
print(f"  耗时: {time.time()-t0:.0f}s")

# ── 4. 评估 ──
print(f"\n[4/4] 评估...")
yp = model.predict(X_te)
print(classification_report(y_te, yp, target_names=['未涨停','涨停'], zero_division=0))
imp = sorted(zip(FEATURE_NAMES[:X.shape[1]], model.feature_importances_), key=lambda x: -x[1])
print("特征重要性 (Top 10):")
for n, i in imp[:10]:
    bar = '█' * int(i * 100)
    print(f"  {n:<20} {i:.4f} {bar}")

path = os.path.join(MODEL_DIR, 'model.json')
model.save_model(path)
print(f"\n✅ 模型: {path} | AUC={auc:.4f} {'✅' if auc>0.55 else '⚠️'}")

# 对比 POC
print(f"\n📊 对比:")
print(f"  POC (仅日线, 3月): AUC=0.7513")
print(f"  完整 (日线, 1年): AUC={auc:.4f} ({(auc-0.7513)/0.7513*100:+.1f}%)")
