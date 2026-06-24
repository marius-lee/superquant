#!/usr/bin/env python3
"""全模式数据驱动阈值优化 — 全量扫描 + NumPy向量化。

数据来源: market.db daily 表 (16M行)
优化方法: 逐只查询 + NumPy向量化处理 (预计 ~60s)

用法:
  python engine/threshold_optimizer.py              # 全量扫描 (默认)
  python engine/threshold_optimizer.py --dry-run    # 快速测试
"""

import os, sys, time, sqlite3, argparse, json, random
import numpy as np

SUPERQUANT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUANT_ROOT = os.path.expanduser("~/project/quant")
DB_PATH = os.path.join(QUANT_ROOT, "data", "market.db")
# 来源: 因子日历 page 36 — 短线策略典型目标3%, 3日窗口匹配A股T+1持有周期
TARGET_RETURN = 0.03; FORWARD_DAYS = 3; MIN_SAMPLE = 20  # 来源: 统计经验 — 最小样本≥20


def _process_stock_vectorized(rows):
    """NumPy向量化: 单只股票日线 → 事件提取。"""
    if len(rows) < FORWARD_DAYS + 3:
        return [], [], []
    n = len(rows)
    o = np.array([r[0] for r in rows], dtype=np.float64)
    h = np.array([r[1] for r in rows], dtype=np.float64)
    l_arr = np.array([r[2] for r in rows], dtype=np.float64)
    c = np.array([r[3] for r in rows], dtype=np.float64)
    v = np.array([r[4] for r in rows], dtype=np.float64)

    valid = (c > 0) & (o > 0)
    if valid.sum() < 8:
        return [], [], []

    pc = np.roll(c, 1); pc[0] = np.nan
    ret = (c - pc) / pc
    vm = valid & (pc > 0)

    # 涨停判断 + 板计数
    limit_up = (ret >= 0.095) & vm
    not_board = (~limit_up).astype(int)
    island_id = np.cumsum(np.where(vm, not_board, 1))
    board_count = np.zeros(n, dtype=int)
    for gid in np.unique(island_id[vm]):
        m = (island_id == gid) & vm
        board_count[m] = np.arange(1, m.sum() + 1)

    # 炸板判断
    limit_price = np.round(pc * 1.10, 2)
    touched = (h >= limit_price * 0.995) & vm
    held = (c >= h * 0.995) & vm
    broken = touched & (~held) & vm

    # 换手率
    turnover = v / 10000.0

    # 前向最高价 → 标签 (FORWARD_DAYS=3: 次日/第3日/第4日, 不含当天)
    fwd_max = np.array([np.max(h[j+1:j+FORWARD_DAYS+1]) if j < n-FORWARD_DAYS else np.nan for j in range(n)])
    fwd_ret = (fwd_max - c) / c
    label = ((fwd_ret >= TARGET_RETURN) & vm).astype(int)

    broken_events = []; board_events = []; single_events = []
    for i in range(2, n - FORWARD_DAYS):
        if not vm[i] or not vm[i-1]:
            continue
        g = (o[i] / c[i-1] - 1) * 100
        vr = v[i] / max(v[i-1], 1)
        dr = ret[i] * 100
        to = turnover[i]
        lb = label[i]
        if broken[i]:
            broken_events.append((g, vr, dr, to, lb))
        nb = int(board_count[i])
        if nb >= 2:
            board_events.append((nb, vr, to, g, lb))
        elif nb == 1 and to >= 0.10:
            single_events.append((vr, to, g, lb))
    return broken_events, board_events, single_events


def fetch_events(conn, symbols):
    """全量扫描 — 逐只查询 + NumPy向量化。"""
    print(f"  扫描 {len(symbols)} 只股票 (NumPy 向量化)...")
    t0 = time.time()
    all_b = []; all_bo = []; all_si = []
    for i, sym in enumerate(symbols):
        rows = conn.execute(
            "SELECT open, high, low, close, volume FROM daily WHERE symbol=? ORDER BY date",
            (sym,)
        ).fetchall()
        if len(rows) < FORWARD_DAYS + 3:
            continue
        try:
            b, bo, si = _process_stock_vectorized(rows)
            all_b.extend(b); all_bo.extend(bo); all_si.extend(si)
        except Exception:
            continue
        if (i + 1) % 500 == 0:
            e = time.time() - t0
            print(f"    {i+1}/{len(symbols)} ({len(all_b)}炸/{len(all_bo)}连/{len(all_si)}首) {e:.0f}s")
    e = time.time() - t0
    print(f"  完成: 炸板{len(all_b)}件, 连板{len(all_bo)}件, 首板{len(all_si)}件, {e:.0f}s")
    return {'broken': all_b, 'boards': all_bo, 'single': all_si}


def optimize_s1(broken):
    if len(broken) < MIN_SAMPLE: return None
    a = np.array(broken); a = a[(a[:,0]>=-5)&(a[:,0]<=10)]
    gs = [0.2, 0.4, 0.5, 1.0, 1.5, 2.0]; vs = [0.3, 0.5, 0.6, 1.0, 2.0, 3.0]; rs = [0.5, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0]
    best, best_sc = None, 0
    for gm in gs:
        for vm in vs:
            for rm in rs:
                sel = a[(a[:,0]>=gm)&(a[:,1]>=vm)&(a[:,2]>=rm)]
                if len(sel) < MIN_SAMPLE: continue
                pos = sel[sel[:,4]==1]
                if len(pos) < MIN_SAMPLE/2: continue
                neg = sel[sel[:,4]==0]
                if len(neg) < 5: continue
                wr = len(pos)/len(sel); wl = np.mean(pos[:,2])/max(abs(np.mean(neg[:,2])),0.001)
                sc = wr*wr*wl*np.log1p(len(sel))
                if sc > best_sc: best_sc = sc; best = {'gap_min':round(gm,2),'vol_ratio':round(vm,2),'daily_ret':round(rm,2),'wr':round(wr,3),'wl':round(wl,2),'n':int(len(sel))}
    return best


def optimize_s2(broken):
    if len(broken) < MIN_SAMPLE: return None
    a = np.array(broken)
    best, best_sc = None, 0
    for gm in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for tl in [0.05, 0.10, 0.15]:
            for th in [0.25, 0.30, 0.40, 0.50]:
                sel = a[(a[:,0]>=gm)&(a[:,3]>=tl)&(a[:,3]<=th)]
                if len(sel) < MIN_SAMPLE: continue
                pos = sel[sel[:,4]==1]
                if len(pos) < MIN_SAMPLE/2: continue
                wr = len(pos)/len(sel); sc = wr*wr*np.log1p(len(sel))
                if sc > best_sc: best_sc = sc; best = {'gap_min':gm,'turnover_min':tl,'turnover_max':th,'wr':round(wr,3),'n':int(len(sel))}
    return best


def optimize_s3(boards):
    if len(boards) < MIN_SAMPLE: return None
    a = np.array(boards); pos = a[a[:,4]==1]
    return {'n':len(a),'wr':round(len(pos)/len(a),3),'opt_vr':round(np.percentile(pos[:,1],25) if len(pos)>10 else 0.67,2)}


def optimize_s4(single):
    if len(single) < MIN_SAMPLE: return None
    a = np.array(single)
    pos = a[a[:,3]==1] if a.shape[1] > 3 else a
    return {'n':len(a),'wr':round(len(pos)/len(a),3),'min_to':round(np.percentile(np.abs(a[:,1]),25),2) if len(a)>10 else 0.10,'min_gap':round(np.percentile(a[a[:,2]>0,2],25),2) if len(a[a[:,2]>0])>5 else 2.0}


def save_to_db(mode, params):
    if not params: return
    from engine.db_schema import save_params
    th = {k:v for k,v in params.items() if not k.startswith('n_') and k not in ('wr','wl','opt_vr','min_to','min_gap')}
    if th: save_params(mode, th, 'data_driven')
    print(f"  {mode}: {params}")


def main():
    p = argparse.ArgumentParser(description='全模式阈值优化')
    p.add_argument('--dry-run', action='store_true')
    ARGS = p.parse_args()

    conn = sqlite3.connect(DB_PATH)
    all_syms = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM daily ORDER BY symbol").fetchall()]
    random.seed(42)

    if ARGS.dry_run:
        syms = random.sample(all_syms, 100)
        print("=" * 60)
        print(f"全模式阈值优化 (快速测试: {len(syms)}只)")
    else:
        syms = all_syms
        print("=" * 60)
        print(f"全模式阈值优化 (全量: {len(syms)}只, NumPy向量化)")
    print("=" * 60)

    events = fetch_events(conn, syms)
    conn.close()

    s1 = optimize_s1(events['broken'])
    if s1: save_to_db('S1_弱转强', s1)
    s2 = optimize_s2(events['broken'])
    if s2: save_to_db('S2_首阴反包', s2)
    s3 = optimize_s3(events['boards'])
    if s3: save_to_db('S3_连板接力', s3)
    s4 = optimize_s4(events['single'])
    if s4: save_to_db('S4_首板试探', s4)

    print(f"\n✅ 优化完成 → {os.path.join(QUANT_ROOT, 'data', 'trades.db')} strategy_params")


if __name__ == '__main__':
    main()
