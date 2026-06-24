"""配置管理 — 从 DB 读取, 不硬编码。

来源: trades.db paper_account 表
北极星起点: ¥5000 (仅首次初始化时写入)
"""
import os, sqlite3
from datetime import date

QUANT_ROOT = os.path.expanduser("~/project/quant")
TRADE_DB = os.path.join(QUANT_ROOT, "data", "trades.db")
DEFAULT_CAPITAL = 5000.0  # 北极星起点 (来源: 北极星目标 ¥5000→¥100万)


def get_capital():
    """获取当前可用资金。读取 paper_account 最新记录, 无则创建首行。"""
    conn = sqlite3.connect(TRADE_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS paper_account (
        id INTEGER PRIMARY KEY, date TEXT, cash REAL, equity REAL, updated_at TEXT)""")
    row = conn.execute(
        "SELECT cash FROM paper_account ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row:
        return row[0]
    return DEFAULT_CAPITAL


def init_capital():
    """首次启动: 写入初始资金记录。已有记录则跳过。"""
    conn = sqlite3.connect(TRADE_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS paper_account (
        id INTEGER PRIMARY KEY, date TEXT, cash REAL, equity REAL, updated_at TEXT)""")
    existing = conn.execute("SELECT COUNT(*) FROM paper_account").fetchone()[0]
    if existing == 0:
        today = date.today().isoformat()
        conn.execute("INSERT INTO paper_account(date, cash, equity, updated_at) VALUES(?,?,?,?)",
                     (today, DEFAULT_CAPITAL, DEFAULT_CAPITAL, today))
        conn.commit()
    conn.close()
    return get_capital()


def save_capital(cash, equity):
    """更新当日资金记录。"""
    conn = sqlite3.connect(TRADE_DB)
    today = date.today().isoformat()
    row = conn.execute("SELECT id FROM paper_account WHERE date=? ORDER BY id DESC LIMIT 1", (today,)).fetchone()
    if row:
        conn.execute("UPDATE paper_account SET cash=?, equity=?, updated_at=? WHERE id=?",
                     (cash, equity, today, row[0]))
    else:
        conn.execute("INSERT INTO paper_account(date, cash, equity, updated_at) VALUES(?,?,?,?)",
                     (today, cash, equity, today))
    conn.commit()
    conn.close()
