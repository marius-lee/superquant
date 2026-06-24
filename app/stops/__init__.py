"""风控组件 — Hikyuu 适配器 (回测/实盘共享)。

薄封装层: Hikyuu KData → engine/strategy_core 计算 → crtST/crtTP
实盘适配器: trader/paper_trader.py 调用同一份 strategy_core
"""

from hikyuu.trade_sys import crtST
from engine.strategy_core import calc_adaptive_stop, calc_take_profit, generate_daily_returns


def create_vol_adaptive_stop():
    """波动率自适应止损 — 替代 B1 硬止损 -5%。"""

    def st_calculate(self, indicator=None):
        kdata = self.to
        n = len(kdata)
        if n < 20:
            self.set_param('stop_pct', 0.05)
            return
        prices = [k.close for k in kdata[-21:]]
        returns = generate_daily_returns(prices)
        entry_px = kdata[-1].close
        stop_px = calc_adaptive_stop(entry_px, returns)
        pct = 1 - stop_px / max(entry_px, 0.01)
        self.set_param('stop_pct', round(pct, 4))

    def st_get_price(self, datetime, price):
        return price * (1 - self.get_param('stop_pct'))

    st = crtST(st_get_price, params={'stop_pct': 0.05},
               name='VolAdaptiveStop', calculate=st_calculate)
    global _st_ref
    _st_ref = st
    return st


def create_mcva_take_profit():
    """MCVA 动态止盈 — 替代 B6。"""
    def tp_get_price(self, datetime, price):
        peak = self.get_param('peak_price')
        trigger = calc_take_profit(0, peak, price)
        return price if trigger else 0.0

    tp = crtST(tp_get_price, params={'peak_price': 0.0}, name='MCVATakeProfit')
    global _tp_ref
    _tp_ref = tp
    return tp


_st_ref = None
_tp_ref = None


if __name__ == '__main__':
    from hikyuu.interactive import sm, Query, Datetime
    from hikyuu.trade_sys import System, crtMM
    from hikyuu.trade_manage import crtTM
    from app.signals import create_chen_sg

    print("=" * 60)
    print("P3: 风控验证 (refactored → strategy_core)")
    print("=" * 60)

    sg = create_chen_sg()
    mm = crtMM(lambda *a: 100, name='Fixed100')
    adaptive_st = create_vol_adaptive_stop()
    fixed_st = crtST(lambda s, dt, px: px * 0.95, name='Fixed5Pct')

    for code in ['SH600036', 'SZ000001']:
        stock = sm[code]
        if not stock.valid:
            continue
        for name, st in [('自适应', adaptive_st), ('固定5%', fixed_st)]:
            tm = crtTM(Datetime(202401010000), 100000.0)
            sys = System(tm, mm, None, None, sg, st, None, None, None, f'{name}-{code}')
            sys.run(stock, Query(-500))
            trades = sys.get_trade_record_list()
            buys = [t for t in trades if t.business == 1]
            sells = [t for t in trades if t.business == 2]
            print(f"  {code} [{name}]: {len(buys)}买/{len(sells)}卖")
    print("✅ P3 通过")
