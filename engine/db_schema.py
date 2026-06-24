"""DB 表结构 — 策略参数持久化。

表: strategy_params
  - 存储四模式的优化阈值
  - 记录来源和更新时间
  - 支持回滚到默认值

用法:
  python engine/db_schema.py  # 创建表 + 写入手写基线
"""

import os, sqlite3, json
from datetime import date

QUANT_ROOT = os.path.expanduser("~/project/quant")
TRADE_DB = os.path.join(QUANT_ROOT, "data", "trades.db")

# 手写基线 (来源: chen-xiaoqun-final-signal-design.md, 17次搜索交叉验证)
BASELINE = {
    'S1_弱转强': {
        'gap_min': 2.0, 'gap_max': 5.0, 'vol_ratio': 3.0, 'daily_ret': 5.0,
        'score': 0.90, 'source': '手写基线 (17次搜索, 2026-06-15)'
    },
    'S2_首阴反包': {
        'gap_min': 3.0, 'gap_max': 10.0, 'vol_ratio': 0.0, 'daily_ret': 0.0,
        'turnover_min': 0.10, 'turnover_max': 0.30,
        'score': 0.85, 'source': '手写基线 (17次搜索, 2026-06-15)'
    },
    'S3_连板接力': {
        'gap_min': 0.0, 'gap_max': 10.0, 'vol_ratio': 0.67, 'daily_ret': 0.0,
        'turnover_min': 0.10, 'turnover_max': 1.0, 'min_boards': 2,
        'score': 0.70, 'source': '手写基线 (17次搜索, 2026-06-15)'
    },
    'S4_首板试探': {
        'gap_min': 2.0, 'gap_max': 10.0, 'vol_ratio': 0.0, 'daily_ret': 0.0,
        'turnover_min': 0.10, 'turnover_max': 1.0, 'min_boards': 1,
        'score': 0.30, 'source': '手写基线 (17次搜索, 2026-06-15)'
    },
}

DEFAULTS = {
    'hand': '手写基线 (17次搜索交叉验证)',
    'data_driven': '数据驱动优化 (全量历史回测)',
    'auto_tuned': '自动调整 (近期绩效反馈)',
}


def init_db():
    """创建 strategy_params 表。"""
    conn = sqlite3.connect(TRADE_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS strategy_params (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mode TEXT NOT NULL,           -- S1_弱转强 / S2_首阴反包 / S3_连板接力 / S4_首板试探
        param_name TEXT NOT NULL,      -- gap_min / vol_ratio / ...
        param_value REAL NOT NULL,     -- 参数值
        source TEXT NOT NULL,          -- hand / data_driven / auto_tuned
        description TEXT,              -- 来源说明
        updated_at TEXT NOT NULL,      -- 更新时间
        UNIQUE(mode, param_name, source)
    )""")
    conn.commit()
    conn.close()
    print(f"✅ strategy_params 表 → {TRADE_DB}")


def save_params(mode, params, source):
    """保存参数到 DB。params = {key: value, ...}"""
    conn = sqlite3.connect(TRADE_DB)
    today = date.today().isoformat()
    for k, v in params.items():
        if k in ('score', 'source'):
            continue  # 元数据字段不存为参数
        conn.execute(
            """INSERT OR REPLACE INTO strategy_params(mode, param_name, param_value, source, updated_at)
               VALUES(?,?,?,?,?)""",
            (mode, k, float(v), source, today)
        )
    conn.commit()
    conn.close()
    print(f"  [{mode}] {len(params)} 参数 → DB ({source})")


def save_baseline():
    """写入手写基线。"""
    init_db()
    for mode, params in BASELINE.items():
        save_params(mode, params, 'hand')
    print("✅ 手写基线已写入")


def load_params(mode, source=None):
    """从 DB 加载参数。

    Args:
        mode: S1_弱转强 / S2_首阴反包 / S3_连板接力 / S4_首板试探
        source: None=取最新, 'hand'=手写, 'data_driven'=数据驱动
    Returns: {param_name: value, ...}
    """
    conn = sqlite3.connect(TRADE_DB)
    if source:
        rows = conn.execute(
            "SELECT param_name, param_value FROM strategy_params WHERE mode=? AND source=?",
            (mode, source)
        ).fetchall()
    else:
        # 取最新的 source
        rows = conn.execute(
            """SELECT param_name, param_value FROM strategy_params WHERE mode=?
               AND updated_at=(SELECT MAX(updated_at) FROM strategy_params WHERE mode=?)""",
            (mode, mode)
        ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows} if rows else BASELINE.get(mode, {})


if __name__ == '__main__':
    save_baseline()
    # 验证
    for mode in ['S1_弱转强', 'S2_首阴反包', 'S3_连板接力', 'S4_首板试探']:
        p = load_params(mode)
        print(f"  {mode}: {p}")
