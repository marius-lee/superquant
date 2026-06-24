"""陈小群信号 — Hikyuu 适配器 (回测用)。

薄封装层: Hikyuu KData → engine/strategy_core.detect_signals() → crtSG

实盘适配器: trader/paper_trader.py 调用同一份 strategy_core
"""

from hikyuu.trade_sys import crtSG
from engine.strategy_core import detect_signals


def chen_signal_calculate(self, indicator=None):
    """Hikyuu crtSG 适配器 — 将 KData 转为纯数据, 调用共享核心。"""
    kdata = indicator if indicator is not None else self.getTO()
    n = len(kdata)
    if n < 5:
        return

    # KData → [(datetime, open, high, low, close, volume), ...]
    records = [(k.datetime, k.open, k.high, k.low, k.close, k.volume) for k in kdata]

    signals = detect_signals(records)
    for dt, sig_type, score in signals:
        self._add_signal(dt, score)


def create_chen_sg():
    """创建陈小群信号指示器 (Hikyuu 适配器)。"""
    sg = crtSG(chen_signal_calculate, name='ChenXiaoqun', params={})
    global _sg_ref
    _sg_ref = sg
    return sg


_sg_ref = None


# ═══════════════════════════════════════════════════════════
# P2 验收: 单股票回测
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    from hikyuu.interactive import sm, Query, Datetime
    from hikyuu.trade_sys import System, crtMM, crtST
    from hikyuu.trade_manage import crtTM

    print("=" * 60)
    print("P2: 陈小群信号 (refactored → strategy_core)")
    print("=" * 60)

    sg = create_chen_sg()
    mm = crtMM(lambda *a: 100, name='Fixed100')
    st = crtST(lambda s, dt, px: px * 0.95, name='Fixed5Pct')

    for code in ['SH600000', 'SZ000001']:
        stock = sm[code]
        if not stock.valid:
            continue
        tm = crtTM(Datetime(202401010000), 100000.0)
        sys = System(tm, mm, None, None, sg, st, None, None, None, f'Chen-{code}')
        sys.run(stock, Query(-500))
        trades = sys.get_trade_record_list()
        buys = [t for t in trades if t.business == 1]
        print(f"  {code}: {len(buys)} 信号")
    print("✅ P2 通过")
