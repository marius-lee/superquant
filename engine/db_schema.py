"""superquant DB Schema — 统一数据层。

表位置:
  ~/project/quant/data/trades.db:
    sim_trades, paper_account (已存在)
    strategy_params (策略参数)
    candidates, signal_events, signal_stats, rejected_signals (JSON→DB)

  ~/project/quant/data/market.db:
    daily, daily_features, stocks (已存在)
    model_health (JSON→DB)
"""

import os, sqlite3, json
from datetime import date

QUANT_ROOT = os.path.expanduser("~/project/quant")
TRADE_DB = os.path.join(QUANT_ROOT, "data", "trades.db")
MKT_DB = os.path.join(QUANT_ROOT, "data", "market.db")

# ═══════════════════════════════════════════════════════════
# 策略参数 (已存在, 保留)
# ═══════════════════════════════════════════════════════════

BASELINE = {
    'S1_弱转强': {
        'gap_min': 2.0, 'gap_max': 5.0, 'vol_ratio': 3.0, 'daily_ret': 5.0,
        'score': 0.90, 'source': '17+8轮搜索'
    },
    'S2_首阴反包': {
        'gap_min': 1.0, 'gap_max': 10.0, 'vol_ratio': 1.5, 'daily_ret': 0.0,
        'turnover_min': 0.10, 'turnover_max': 0.30,
        'score': 0.85, 'source': '8轮搜索修正'
    },
    'S3_连板接力': {
        'gap_min': 0.0, 'gap_max': 10.0, 'vol_ratio': 0.67, 'daily_ret': 0.0,
        'turnover_min': 0.10, 'turnover_max': 1.0, 'min_boards': 2,
        'market_cap_min': 30, 'market_cap_max': 80, 'sector_limit_min': 3,
        'score': 0.70, 'source': '8轮搜索修正+市值+板块'
    },
    'S4_首板试探': {
        'gap_min': 0.5, 'gap_max': 10.0, 'vol_ratio': 0.0, 'daily_ret': 0.0,
        'turnover_min': 0.10, 'turnover_max': 1.0, 'min_boards': 1,
        'score': 0.30, 'source': '数据驱动修正'
    },
}

# ═══════════════════════════════════════════════════════════
# 完整 Schema 定义
# ═══════════════════════════════════════════════════════════

ALL_SCHEMAS = {
    TRADE_DB: [
        # 策略参数 (已有)
        """CREATE TABLE IF NOT EXISTS strategy_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT NOT NULL,
            param_name TEXT NOT NULL,
            param_value REAL NOT NULL,
            source TEXT NOT NULL,
            description TEXT,
            updated_at TEXT NOT NULL,
            UNIQUE(mode, param_name, source)
        )""",
        # ── JSON→DB 新表 ──
        """CREATE TABLE IF NOT EXISTS candidates (
            date TEXT NOT NULL, symbol TEXT NOT NULL,
            name TEXT DEFAULT '', prob REAL NOT NULL,
            channel TEXT NOT NULL DEFAULT 'main',
            PRIMARY KEY (date, symbol, channel)
        )""",
        """CREATE TABLE IF NOT EXISTS signal_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT NOT NULL, date TEXT NOT NULL,
            symbol TEXT NOT NULL, signal TEXT NOT NULL,
            action TEXT NOT NULL, pnl REAL DEFAULT 0
        )""",
        """CREATE INDEX IF NOT EXISTS idx_se_date ON signal_events(date)""",
        """CREATE TABLE IF NOT EXISTS signal_stats (
            signal TEXT PRIMARY KEY,
            win_count INTEGER DEFAULT 0, total_count INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0.5, total_pnl REAL DEFAULT 0.0,
            avg_return REAL DEFAULT 0.0
        )""",
        """CREATE TABLE IF NOT EXISTS rejected_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL, time TEXT NOT NULL,
            symbol TEXT NOT NULL, price REAL NOT NULL,
            reason TEXT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_rej_date ON rejected_signals(date)""",
        """CREATE TABLE IF NOT EXISTS performance_metrics (
            date TEXT PRIMARY KEY,
            sharpe REAL, max_drawdown REAL, calmar REAL,
            win_rate REAL, profit_factor REAL, expectancy REAL,
            total_trades INTEGER, winning_trades INTEGER,
            total_pnl REAL, annual_return REAL
        )""",
    ],
    MKT_DB: [
        """CREATE TABLE IF NOT EXISTS model_health (
            date TEXT PRIMARY KEY, n_stocks INTEGER,
            pred_mean REAL, pred_std REAL,
            pred_skew REAL, disc_count INTEGER
        )""",
    ],
}


def init_all():
    """幂等创建所有表。"""
    for db_path, stmts in ALL_SCHEMAS.items():
        conn = sqlite3.connect(db_path)
        for sql in stmts:
            conn.execute(sql)
        conn.commit()
        conn.close()

def init_params():
    """策略参数表的创建+基线写入。"""
    conn = sqlite3.connect(TRADE_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS strategy_params (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mode TEXT NOT NULL, param_name TEXT NOT NULL,
        param_value REAL NOT NULL, source TEXT NOT NULL,
        description TEXT, updated_at TEXT NOT NULL,
        UNIQUE(mode, param_name, source)
    )""")
    conn.commit()
    conn.close()

def save_baseline():
    init_params()
    for mode, params in BASELINE.items():
        save_params(mode, params, 'hand')

def save_params(mode, params, source):
    conn = sqlite3.connect(TRADE_DB)
    today = date.today().isoformat()
    for k, v in params.items():
        if k in ('score', 'source'): continue
        conn.execute("""INSERT OR REPLACE INTO strategy_params
            (mode, param_name, param_value, source, updated_at)
            VALUES(?,?,?,?,?)""", (mode, k, float(v), source, today))
    conn.commit(); conn.close()

def load_params(mode, source=None):
    conn = sqlite3.connect(TRADE_DB)
    if source:
        rows = conn.execute("SELECT param_name, param_value FROM strategy_params WHERE mode=? AND source=?", (mode, source)).fetchall()
    else:
        rows = conn.execute("""SELECT param_name, param_value FROM strategy_params WHERE mode=?
            AND updated_at=(SELECT MAX(updated_at) FROM strategy_params WHERE mode=?)""", (mode, mode)).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows} if rows else BASELINE.get(mode, {})

if __name__ == '__main__':
    init_all()
    save_baseline()
    print("✅ 所有表已创建 + 基线已写入")
