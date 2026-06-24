"""XGBoost POC: 预测A股涨停概率。数据: market.db 日线, 标签: 次日涨停(9.5%+)。用法: python ml/train.py"""
import os, sys, time, sqlite3, json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.expanduser("~/project/quant"), "data", "market.db")

print("=" * 60)
print("XGBoost 涨停预测 POC")
print("=" * 60)

# ── 1. 数据加载 ──
print("\n[1/4] 加载数据...")
t0 = time.time()
conn = sqlite3.connect(DB_PATH)
rows = conn.execute("SELECT symbol,date,open,high,low,close,volume,amount,turnover FROM daily WHERE date>='2026-03-01' ORDER BY symbol,date").fetchall()
conn.close()
print(f"  加载: {len(rows)}行, {time.time()-t0:.0f}s")

# ── 2. 特征工程 ── (SQL: symbol=0,date=1,open=2,high=3,low=4,close=5,volume=6,amount=7,turnover=8)
print("\n[2/4] 特征工程...")
t0 = time.time()
data = defaultdict(list)
for r in rows: data[r[0]].append(r)

X_list, y_list = [], []
for sym, rs in data.items():
    if len(rs) < 32: continue
    closes = np.array([float(r[5]) for r in rs])
    volumes = np.array([float(r[6]) for r in rs])
    opens = np.array([float(r[2]) for r in rs])
    highs = np.array([float(r[3]) for r in rs])
    lows = np.array([float(r[4]) for r in rs])
    amounts = np.array([float(r[7]) for r in rs])
    for i in range(25, len(rs)-1):
        if closes[i-1] <= 0: continue
        ret_1d = closes[i]/closes[i-1]-1
        ret_5d = closes[i]/closes[i-5]-1 if i>=5 and closes[i-5]>0 else 0
        vol_5d_avg = np.mean(volumes[i-5:i]) if i>=5 else volumes[i]
        vol_ratio = volumes[i]/max(vol_5d_avg,1)
        rets_5d = [(closes[j]/closes[j-1]-1) for j in range(i-4,i+1) if closes[j-1]>0]
        vol_5d = np.std(rets_5d) if len(rets_5d)>2 else 0
        gap = opens[i]/closes[i-1]-1
        turnover = float(rs[i][8]) if rs[i][8] else volumes[i]/10000.0
        amt_log = np.log(max(amounts[i],1))
        hl_ratio = highs[i]/max(lows[i],0.01)-1
        close_pos = (closes[i]-lows[i])/max(highs[i]-lows[i],0.01)
        ma20 = np.mean(closes[max(0,i-20):i+1])
        ma_dev = closes[i]/ma20-1 if ma20>0 else 0
        feat = [ret_1d,ret_5d,vol_ratio,vol_5d,gap,turnover,amt_log,hl_ratio,close_pos,ma_dev]
        if any(abs(v)>100 for v in feat): continue
        X_list.append(feat)
        next_ret = closes[i+1]/closes[i]-1 if closes[i]>0 else 0
        y_list.append(1 if next_ret>=0.095 else 0)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
print(f"  特征: {X.shape}, 正样本(涨停): {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"  耗时: {time.time()-t0:.0f}s")

# ── 3. 训练 ──
print("\n[3/4] 训练 XGBoost...")
t0 = time.time()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
sw = (len(y_tr)-y_tr.sum())/max(y_tr.sum(),1)
model = xgb.XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,subsample=0.8,
    colsample_bytree=0.8,scale_pos_weight=sw,objective='binary:logistic',eval_metric='auc',random_state=42)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
print(f"  AUC-ROC: {auc:.4f} | Baseline(random)=0.5000 | 提升: {(auc-0.5)/0.5*100:+.0f}%")
print(f"  耗时: {time.time()-t0:.0f}s")

# ── 4. 评估 ──
print(f"\n[4/4] 评估...")
yp = model.predict(X_te)
print(classification_report(y_te, yp, target_names=['未涨停','涨停'], zero_division=0))
names = ['ret_1d','ret_5d','vol_ratio','vol_5d','gap','turnover','amt_log','hl_ratio','close_pos','ma_dev']
imp = sorted(zip(names, model.feature_importances_), key=lambda x:-x[1])
print("特征重要性 (Top5):")
for n,i in imp[:5]: print(f"  {n:<15} {i:.4f} {'█'*int(i*100)}")

path = os.path.join(MODEL_DIR,'model.json')
model.save_model(path)
print(f"\n✅ 模型: {path} | AUC={auc:.4f} {'✅' if auc>0.55 else '⚠️'}")
