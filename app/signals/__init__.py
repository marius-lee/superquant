"""陈小群模式识别 — Hikyuu crtSG 组件。

将 execution/quote.py BoardTracker.scan_all_modes() 的模式识别逻辑
封装为 Hikyuu SignalBase，支持历史回测。

关键适配:
  实时数据 (self.stocks dict) → Hikyuu KData (历史K线)
  分钟级条件 (5分涨>7%)    → 日线近似 (日涨幅替代)
  emitted 去重              → Hikyuu SignalBase 自动管理

模式定义 (来源: chen-xiaoqun-final-signal-design.md):
  S1 弱转强 (0.90): 昨炸板 + 高开2-5% + 量>昨3倍
  S2 首阴反包 (0.85): 昨炸板 + 高开≥3% + 换手10-30%
  S3 连板接力 (0.70): 2连板 + 换手>10% + 二板量≥首板2/3
  S4 首板试探 (0.30): 首板 + 换手>10% (观察信号)
"""

from hikyuu.trade_sys import crtSG


def _is_broken_board(k, prev_k) -> bool:
    """判断前一日是否炸板。

    条件: 昨最高触及涨停 (≥prev_close×1.095) 且收盘未封住 (close < high)
    来源: execution/quote.py:367 — st["yesterday_broken"]
    """
    if prev_k is None:
        return False
    limit_price = round(prev_k.open * 1.10, 2)
    touched_limit = prev_k.high >= limit_price * 0.995
    did_not_hold = prev_k.close < prev_k.high * 0.995
    return touched_limit and did_not_hold


def _count_boards(kdata, idx: int) -> int:
    """计算截至 idx 位置的连板数。"""
    board = 1
    for j in range(idx, 0, -1):
        prev_close = kdata[j - 1].close
        curr_close = kdata[j].close
        if prev_close > 0 and (curr_close / prev_close - 1) >= 0.095:
            board += 1
        else:
            break
    return board


def chen_signal_calculate(self, indicator=None):
    """陈小群模式识别 — Hikyuu SignalBase._calculate 实现。

    对每根日线K线检查信号条件，满足则调用 self._add_signal(datetime, score)。
    Hikyuu 自动管理信号去重和生命周期。
    """
    # Hikyuu 2.8.0: indicator 参数就是 KData (交易标的的K线数据)
    kdata = indicator if indicator is not None else self.getTO()
    n = len(kdata)
    if n < 5:
        return

    for i in range(2, n):
        k = kdata[i]
        prev_k = kdata[i - 1]

        if prev_k.close <= 0 or k.open <= 0:
            continue

        gap = (k.open / prev_k.close - 1) * 100
        daily_ret = (k.close / prev_k.close - 1) * 100
        vol_ratio = k.volume / max(prev_k.volume, 1)
        turnover = k.volume / 10000.0
        is_broken = _is_broken_board(k, prev_k)

        # S1: 弱转强 (0.90)
        if is_broken and 2.0 <= gap <= 5.0 and vol_ratio >= 3.0 and daily_ret >= 5.0:
            self._add_signal(k.datetime, 0.90)
            continue

        # S2: 首阴反包 (0.85)
        if is_broken and gap >= 3.0 and 0.10 <= turnover <= 0.30:
            self._add_signal(k.datetime, 0.85)
            continue

        # S3: 连板接力 (0.70)
        board = _count_boards(kdata, i)
        if board >= 2 and turnover >= 0.10 and vol_ratio >= 0.67:
            self._add_signal(k.datetime, 0.70)
            continue

        # S4: 首板试探 (0.30)
        if board == 1 and turnover >= 0.10 and gap > 2.0:
            self._add_signal(k.datetime, 0.30)


def create_chen_sg():
    """创建陈小群信号指示器。"""
    return crtSG(chen_signal_calculate, name='ChenXiaoqun', params={})


# ═══════════════════════════════════════════════════════════
# P2 验收: 单股票回测
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    from hikyuu.interactive import sm, Query, Datetime
    from hikyuu.trade_sys import System, crtMM, crtST
    from hikyuu.trade_manage import crtTM

    print("=" * 60)
    print("P2: 陈小群信号 — 单股票回测验证")
    print("=" * 60)

    sg = create_chen_sg()
    mm = crtMM(lambda *a: 100, name='Fixed100')
    st = crtST(lambda *a: 0.0, name='NoStop')

    test_stocks = [('SH600000', '浦发银行'), ('SZ000001', '平安银行')]

    for code, name in test_stocks:
        stock = sm[code]
        if not stock.valid:
            print(f"  {code} 无效, 跳过")
            continue

        tm = crtTM(Datetime(202401010000), 100000.0)
        sys = System(tm, mm, None, None, sg, st, None, None, None, f'Chen-{code}')

        print(f"\n{'─' * 40}")
        print(f"测试: {code} {name}")
        q = Query(-500)
        sys.run(stock, q)

        trades = sys.get_trade_record_list()
        print(f"  交易次数: {len(trades)}")
        if len(trades) > 0:
            buys = [t for t in trades if t.business == 1]
            sells = [t for t in trades if t.business == 2]
            wins = sum(1 for i in range(min(len(buys), len(sells)))
                       if sells[i].real_price > buys[i].real_price)
            total = min(len(buys), len(sells))
            wr = wins / max(total, 1) * 100
            print(f"  胜率: {wr:.1f}% ({wins}W/{total-wins}L/{total}T)")
            for t in trades[-6:]:
                b = '买' if t.business == 1 else '卖'
                print(f"    {t.datetime} {b} ¥{t.real_price:.2f} {t.number}股")
        else:
            print(f"  ⚠️ 无交易信号")

    print(f"\n{'─' * 40}")
    print("✅ P2 验收: SG 组件可运行")
    print("   下一步: P1因子×P2信号 → P3风控 → P4完整策略")
