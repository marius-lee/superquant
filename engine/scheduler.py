#!/usr/bin/env python3
"""定时调度引擎 — cron 编排。

时间表:
  08:45 → 日线同步 + 日线导出 + ML 预测
  09:30-15:00 → 模拟交易 (trader/paper_trader)
  15:05 → 分钟数据存储 + ML 重训 + L3 回补

用法:
  python engine/scheduler.py --mode daily     # 每日运行模式
  python engine/scheduler.py --mode live      # 盘中实时模式
  python engine/scheduler.py --mode all       # 全流程 (默认)

cron 配置 (macOS launchd):
  盘前: 每天 8:45 执行 scheduler.py --mode pre-market
  盘中: 每天 9:29 启动 live 模式 (常驻)
  盘后: 每天 15:05 执行 scheduler.py --mode post-market
"""

import os, sys, time, subprocess, argparse
from datetime import datetime, date

SUPERQUANT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.expanduser("~/project/quant/.venv/bin/python")
ENV = {"PYTHONPATH": f"{os.path.expanduser('~/project/quant')}:{SUPERQUANT_ROOT}"}


def run_step(name, module, args=None):
    """运行一个步骤, 打印耗时。"""
    t0 = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {name}...")
    # 直接脚本执行 (避免 python -m 的 __main__ 问题)
    path = os.path.join(SUPERQUANT_ROOT, module.replace('.', os.sep))
    if os.path.isdir(path):
        script = os.path.join(path, '__init__.py')
    else:
        script = path + '.py'
    cmd = [PYTHON, script]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, cwd=SUPERQUANT_ROOT, env={**os.environ, **ENV},
                           capture_output=True, text=True)
    elapsed = time.time() - t0
    status = "✅" if result.returncode == 0 else f"❌({result.returncode})"
    print(f"  {status} {elapsed:.0f}s")
    if result.returncode != 0:
        # 关键步骤失败时输出完整错误 (来源: 2026-06-26 train.py崩了被吞)
        print(f"  [stderr] {result.stderr.strip()[-1000:]}" if result.stderr else "  [stderr] (empty)")
        if result.stdout.strip():
            print(f"  [stdout] {result.stdout.strip()[-500:]}")
    return result.returncode


def pre_market():
    """盘前流程: 8:45"""
    errors = 0
    errors += run_step("日线同步", "data.daily_sync")
    errors += run_step("ML 预测", "ml.predict")
    # 08:46 预检: candidates非空 + model_health写入
    if errors == 0:
        errors += _pre_check()
    else:
        print(f"⚠️ 盘前流程 {errors} 步失败")
    return errors


def _pre_check():
    """预检: 确认 candidates 和 model_health 已生成。"""
    import sqlite3
    today = date.today().isoformat()
    try:
        conn = sqlite3.connect(os.path.join(os.path.expanduser("~/project/quant"), "data", "trades.db"))
        n_cand = conn.execute("SELECT COUNT(*) FROM candidates WHERE date=?", (today,)).fetchone()[0]
        conn.close()
        mkt = sqlite3.connect(os.path.join(os.path.expanduser("~/project/quant"), "data", "market.db"))
        n_health = mkt.execute("SELECT COUNT(*) FROM model_health WHERE date=?", (today,)).fetchone()[0]
        mkt.close()
        if n_cand == 0:
            print(f"  ❌ 预检失败: candidates={n_cand}只, model_health={n_health}条")
            return 1
        print(f"  ✅ 预检通过: candidates={n_cand}只, model_health={n_health}条")
        # 连续无交易告警
        return _alert_check()
    except Exception as e:
        print(f"  ❌ 预检异常: {e}")
        return 1


def _alert_check():
    """告警: 连续2天无交易则打印警告。"""
    import sqlite3
    today = date.today().isoformat()
    try:
        conn = sqlite3.connect(os.path.join(os.path.expanduser("~/project/quant"), "data", "trades.db"))
        days_with_trades = conn.execute("""
            SELECT DISTINCT date FROM sim_trades WHERE side='buy'
            ORDER BY date DESC LIMIT 2
        """).fetchall()
        conn.close()
        if len(days_with_trades) == 0 or days_with_trades[0][0] < today:
            # 检查最近是否有交易
            last_trade = days_with_trades[0][0] if days_with_trades else '无'
            if last_trade != today and last_trade != '无':
                # 一天无交易正常, 连续两天告警
                prev_days = [d[0] for d in days_with_trades]
                if len(prev_days) >= 2:
                    print(f"  ⚠️ 连续两天无交易 (最后交易: {prev_days[0]})")
    except Exception:
        pass
    return 0


def live_trading():
    """盘中流程: 9:29 启动"""
    print("=" * 60)
    print("superquant 模拟交易启动")
    print(f"  时间: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    return run_step("模拟交易", "trader.paper_trader")


def post_market():
    """盘后流程: 15:05"""
    errors = 0
    errors += run_step("分钟存储", "data.minute_store", ["--market", "all", "--today"])
    errors += run_step("ML 训练", "ml.train")
    # 15:06 复盘
    _post_review()
    if errors > 0:
        print(f"⚠️ 盘后流程 {errors} 步失败")
    return errors


def _post_review():
    """复盘: 今日收益/信号/异常 + 蒙特卡洛信号验证。"""
    import sqlite3
    today = date.today().isoformat()
    print(f"\n{'='*40}")
    print(f"📊 今日复盘 — {today}")
    print(f"{'='*40}")
    try:
        conn = sqlite3.connect(os.path.join(os.path.expanduser("~/project/quant"), "data", "trades.db"))
        buys = conn.execute("SELECT COUNT(*), COALESCE(SUM(price*shares),0) FROM sim_trades WHERE side='buy' AND date=?", (today,)).fetchone()
        sells = conn.execute("SELECT COUNT(*), COALESCE(SUM(pnl),0) FROM sim_trades WHERE side='sell' AND date=?", (today,)).fetchone()
        sigs = conn.execute("SELECT signal, COUNT(*) FROM signal_events WHERE date=? GROUP BY signal", (today,)).fetchall()
        conn.close()
        print(f"  买入: {buys[0]}笔, ¥{buys[1]:,.0f}")
        print(f"  卖出: {sells[0]}笔, 盈亏¥{sells[1]:+,.0f}")
        print(f"  信号触发: {sum(s[1] for s in sigs)}次")
        for s in sigs:
            print(f"    {s[0]}: {s[1]}次")
        if buys[0] == 0:
            print(f"  ⚠️ 今日无交易 — 检查候选质量或市场状态")

        # 蒙特卡洛信号检验 (来源: Liang Wenfeng — 每个信号必须过统计验证)
        print(f"\n  🎲 蒙特卡洛检验:")
        try:
            from ops.monte_carlo import test_signal, test_strategy
            for code in ['A', 'B', 'E']:
                r = test_signal(code, n_permutations=500)  # 每日500次, 每周3000+次够用
                if r['n_trades'] >= 5:
                    print(f"    {code}: p={r['p_value']:.3f} {r['verdict']}")
                    if not r['is_significant'] and r['percentile'] < 50:
                        # 自动标记无效信号
                        import sqlite3 as sq
                        fc = sq.connect(os.path.join(os.path.expanduser("~/project/quant"), "data", "trades.db"))
                        fc.execute("UPDATE signal_stats SET win_rate=0 WHERE signal=? AND total_count>0", (code,))
                        fc.commit(); fc.close()
                        print(f"      ⚠️ {code} 不如随机, 已自动降权")
                else:
                    print(f"    {code}: 数据不足 (n={r['n_trades']})")
            # 策略整体检验
            s = test_strategy(n_permutations=500)
            if s.get('n_trades', 0) >= 10:
                print(f"  策略整体: p={s.get('p_total',1):.3f} {s.get('verdict','')}")
        except Exception as e:
            print(f"    ❌ 蒙特卡洛异常: {e}")
    except Exception as e:
        print(f"  ❌ 复盘异常: {e}")


def full_cycle():
    """全流程 (非交易时间可用)"""
    errors = 0
    errors += run_step("日线同步", "data.daily_sync")
    errors += run_step("ML 预测", "ml.predict")
    errors += run_step("ML 训练", "ml.train")
    if errors > 0:
        print(f"⚠️ 全流程 {errors} 步失败")
    return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='superquant 调度引擎')
    parser.add_argument('--mode', default='all',
                       choices=['pre-market', 'live', 'post-market', 'daily', 'all'])
    ARGS = parser.parse_args()

    now = datetime.now()
    print(f"superquant 调度引擎 — {now.strftime('%Y-%m-%d %H:%M:%S')}")

    if ARGS.mode == 'pre-market':
        exit(pre_market())
    elif ARGS.mode == 'live':
        exit(live_trading())
    elif ARGS.mode == 'post-market':
        exit(post_market())
    elif ARGS.mode == 'daily':
        # 完整日循环
        errors = pre_market()
        if errors:
            print("盘前失败, 跳过盘中")
        else:
            live_trading()
        post_market()
    else:
        exit(full_cycle())
