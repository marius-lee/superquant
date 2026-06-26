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
    if errors > 0:
        print(f"⚠️ 盘前流程 {errors} 步失败")
    return errors


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
    if errors > 0:
        print(f"⚠️ 盘后流程 {errors} 步失败")
    return errors


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
