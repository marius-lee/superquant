#!/usr/bin/env python3
"""批量预计算 L1-L5 特征 — 全部带日志。用法: python ml/build_features.py --l3 --start 1 --count 10"""
import os, sys, time, sqlite3, argparse, threading, logging
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_features.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(threadName)s] %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUANT_ROOT = os.path.expanduser("~/project/quant"); DB_PATH = os.path.join(QUANT_ROOT, "data", "market.db")
HDF5_DIR = os.path.expanduser("~/stock"); t_start = time.time()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS daily_features (
        symbol TEXT, date TEXT, ksft REAL, slope REAL, ptc REAL, vol_5min REAL, max_ret REAL, min_ret REAL,
        main_net_in REAL, main_net_ratio REAL, super_large_in REAL, large_in REAL,
        lhb_net_buy REAL, lhb_buy_ratio REAL, lhb_count INTEGER, lhb_exists INTEGER,
        sector_limit_count INTEGER, sector_rank INTEGER, PRIMARY KEY (symbol, date))""")
    conn.commit(); conn.close()

def build_l2_batch(conn):
    import h5py
    log.info("L2 技术特征")
    dates = {r[0] for r in conn.execute("SELECT DISTINCT date FROM daily WHERE date>='2025-06-01'").fetchall()}
    syms = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM daily ORDER BY symbol").fetchall()]
    total, inserted = len(syms), 0
    for i, sym in enumerate(syms):
        mkt = 'sh' if sym.startswith(('6','5','9')) else 'sz'
        path = os.path.join(HDF5_DIR, f'{mkt}_5min.h5')
        if not os.path.exists(path): continue
        try:
            with h5py.File(path, 'r') as h5:
                ds_name = f"{'SH' if mkt=='sh' else 'SZ'}{sym}"
                if ds_name not in h5['data']: continue
                ds = h5['data'][ds_name]
                if len(ds) < 10: continue
                dt_arr = np.array([r['datetime'] for r in ds], dtype=np.uint64)
                o_arr = np.array([r['openPrice'] for r in ds], dtype=np.float64)/1000.0
                h_arr = np.array([r['highPrice'] for r in ds], dtype=np.float64)/1000.0
                l_arr = np.array([r['lowPrice'] for r in ds], dtype=np.float64)/1000.0
                c_arr = np.array([r['closePrice'] for r in ds], dtype=np.float64)/1000.0
                v_arr = np.array([r['transCount'] for r in ds], dtype=np.float64)
                batch = []
                for d in dates:
                    mask = np.array([str(dt)[:8]==d.replace('-','') for dt in dt_arr])
                    if mask.sum()<10: continue
                    idx = np.where(mask)[0]; cd,hd,ld,od,vd = c_arr[idx],h_arr[idx],l_arr[idx],o_arr[idx],v_arr[idx]
                    ksft = float(np.mean((hd-ld)/np.maximum(np.abs(od-cd[0]),0.01)))
                    x=np.arange(min(10,len(cd))); y=cd[-len(x):]
                    slope = float(np.polyfit(x,y,1)[0]) if len(x)>=5 else 0.0
                    ad=cd*vd; c=np.corrcoef(cd[-20:],ad[-20:])[0,1] if len(cd)>=20 else 0
                    ptc = float(0 if np.isnan(c) else c)
                    rt=np.diff(cd)/cd[:-1]; vol=float(np.std(rt)) if len(rt)>3 else 0.0
                    batch.append((sym,d,ksft,slope,ptc,vol,float(np.max(rt)) if len(rt)>0 else 0.0,float(np.min(rt)) if len(rt)>0 else 0.0))
                if batch: conn.executemany("INSERT OR REPLACE INTO daily_features(symbol,date,ksft,slope,ptc,vol_5min,max_ret,min_ret) VALUES(?,?,?,?,?,?,?,?)",[(s,d,k,sl,p,v,mx,mn) for s,d,k,sl,p,v,mx,mn in batch]); inserted+=len(batch)
        except Exception as e:
            if i<3: log.warning(f"  {sym}: {e}")
        if (i+1)%500==0: conn.commit(); log.info(f"  L2 {i+1}/{total} ({inserted}) {time.time()-t_start:.0f}s")
    conn.commit(); log.info(f"L2 完成: {inserted}条, {time.time()-t_start:.0f}s")

def _l3_worker(syms_chunk, chunk_id, start_day, count_days):
    import easy_tdx as et
    tid = f"L3-{chunk_id}"
    client = et.TdxClient()
    try: client.connect()
    except Exception as e: log.error(f"  [{tid}] 连接失败: {e}"); return []
    local, success, fail = [], 0, 0
    t0 = time.time()
    for i, sym in enumerate(syms_chunk):
        try:
            mkt = et.Market.SZ if sym.startswith(('0','3')) else et.Market.SH
            df = client.get_history_fund_flow(mkt, sym, start=start_day, count=count_days)
            if df is not None and len(df) > 0:
                for _, row in df.iterrows():
                    d = str(row['date'])[:10]
                    si=float(row.get('super_in',0)or 0); li=float(row.get('large_in',0)or 0)
                    so=float(row.get('super_out',0)or 0); lo=float(row.get('large_out',0)or 0)
                    mn=(si+li)-(so+lo); mr=mn/(abs(si)+abs(li)+abs(so)+abs(lo)+1)
                    local.append((sym,d,mn,mr,si,li))
                success += 1
            else: fail += 1
        except: fail += 1
        if (i+1)%200==0:
            elapsed = time.time()-t0
            est = elapsed/(i+1)*len(syms_chunk) if i>0 else 0
            log.info(f"  [{tid}] {i+1}/{len(syms_chunk)}(成功{success}/失败{fail}) {elapsed:.0f}s 预计{est:.0f}s")
    elapsed = time.time()-t0
    log.info(f"  [{tid}] 完成: 成功{success}/失败{fail}, {elapsed:.0f}s")
    client.close(); return local

def build_l3_batch(conn, start_day=0, count_days=10):
    """L3: easy-tdx 资金流 4线程并行。
    用法: --start 1 --count 10 → start=1=昨天,count=10=往回10天"""
    import easy_tdx as et
    log.info(f"开始 L3 资金流 (start={start_day}, count={count_days}, 4线程)")
    syms = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM daily ORDER BY symbol").fetchall()]
    log.info(f"  总{len(syms)}只, 全量拉取 (INSERT OR REPLACE 自动去重)")
    n=len(syms); cs=(n+3)//4; chunks=[syms[i:i+cs] for i in range(0,n,cs)]
    log.info(f"  4组: {[len(c) for c in chunks]}")
    results={}; lock=threading.Lock()
    with ThreadPoolExecutor(max_workers=4) as ex:
        fs={ex.submit(_l3_worker,ch,i,start_day,count_days):i for i,ch in enumerate(chunks)}
        for f in as_completed(fs): results[fs[f]]=f.result()
    total=0
    for cid in range(4):
        data=results.get(cid,[])
        if data: conn.executemany("INSERT OR REPLACE INTO daily_features(symbol,date,main_net_in,main_net_ratio,super_large_in,large_in) VALUES(?,?,?,?,?,?)",data); total+=len(data)
    conn.commit()
    log.info(f"L3 完成: {total}条, {time.time()-t_start:.0f}s ({ (time.time()-t_start)/60:.1f}min)")

def build_l4_batch(conn):
    import akshare as ak
    log.info("L4 龙虎榜")
    d, end, inserted = date(2025,6,1), date(2026,6,24), 0
    while d<=end:
        try:
            df=ak.stock_lhb_detail_daily_sina(date=d.strftime('%Y%m%d'))
            if df is not None and len(df)>0:
                batch=[(r.get('股票代码',''),d.isoformat(),float(r.get('成交额',0)or 0),0.5,1,1) for _,r in df.iterrows() if r.get('股票代码','')]
                if batch: conn.executemany("INSERT OR REPLACE INTO daily_features(symbol,date,lhb_net_buy,lhb_buy_ratio,lhb_count,lhb_exists) VALUES(?,?,?,?,?,?)",batch); inserted+=len(batch)
        except: pass
        d+=timedelta(days=1)
        if d.day==1: log.info(f"  L4 {d.strftime('%Y-%m')} ({inserted}) {time.time()-t_start:.0f}s")
    conn.commit(); log.info(f"L4 完成: {inserted}条, {time.time()-t_start:.0f}s")

def build_l5_batch(conn):
    log.info("L5 板块")
    rows=conn.execute("SELECT d.date,s.market,COUNT(*) FROM daily d JOIN stocks s ON d.symbol=s.symbol WHERE d.close>=d.open*1.095 AND d.date>='2025-06-01' GROUP BY d.date,s.market").fetchall()
    batch=[(sym,ds,cnt,0) for ds,mkt,cnt in rows for (sym,) in conn.execute("SELECT symbol FROM stocks WHERE market=?",(mkt,)).fetchall()]
    if batch: conn.executemany("INSERT OR REPLACE INTO daily_features(symbol,date,sector_limit_count,sector_rank) VALUES(?,?,?,?)",batch)
    conn.commit(); log.info(f"L5 完成: {len(batch)}条, {time.time()-t_start:.0f}s")

if __name__=='__main__':
    p=argparse.ArgumentParser(description='特征预计算 (日志: ml/build_features.log)')
    p.add_argument('--l2',action='store_true'); p.add_argument('--l3',action='store_true')
    p.add_argument('--l4',action='store_true'); p.add_argument('--l5',action='store_true')
    p.add_argument('--all',action='store_true')
    p.add_argument('--start',type=int,default=0,help='L3 start 交易日偏移 (0=今天)')
    p.add_argument('--count',type=int,default=10,help='L3 count 拉取天数')
    ARGS=p.parse_args()
    log.info(f"=== 开始特征预计算 (PID={os.getpid()}) ===")
    init_db()
    conn=sqlite3.connect(DB_PATH)
    if ARGS.all or ARGS.l2: build_l2_batch(conn)
    if ARGS.all or ARGS.l3: build_l3_batch(conn, start_day=ARGS.start, count_days=ARGS.count)
    if ARGS.all or ARGS.l4: build_l4_batch(conn)
    if ARGS.all or ARGS.l5: build_l5_batch(conn)
    conn.close()
    log.info(f"=== 全部完成 (总耗时{time.time()-t_start:.0f}s) ===")
