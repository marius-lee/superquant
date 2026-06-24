"""strategy_core 纯函数测试 — 零依赖。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.strategy_core import *

def test_is_broken_board():
    assert not is_broken_board(11.0, 11.0, 10.0)
    assert is_broken_board(11.0, 10.5, 10.0)
    print("  ✅ is_broken_board")

def test_count_boards():
    assert count_boards([10.0, 11.0, 12.1, 13.31], 3) == 4
    assert count_boards([10.0, 10.5, 11.55], 2) == 2
    print("  ✅ count_boards")

def test_detect_signals():
    b = [(0, 10.0, 10.0, 10.0, 10.0, 1000)]*3
    r1 = b + [(1, 10.0, 11.0, 10.0, 10.5, 300), (2, 10.9, 11.5, 10.8, 11.5, 1200)]
    assert any(s[1]=='弱转强' for s in detect_signals(r1))
    r2 = b + [(10, 10.0, 11.0, 10.0, 10.95, 1200), (20, 11.0, 12.0, 11.0, 12.0, 1200)]
    assert any(s[1]=='连板接力' for s in detect_signals(r2))
    print("  ✅ detect_signals")

def test_factor_multiplier():
    assert compute_factor_multiplier(2.0) > 1.2
    assert compute_factor_multiplier(-2.0) < 0.8
    print("  ✅ factor_multiplier")

def test_position_size():
    s = calc_position_size(5000, 10.0, 0.5)
    assert 0 <= s <= 2000
    print("  ✅ position_size")

def test_adaptive_stop():
    low = [0.001, -0.001]*10
    high = [0.08, -0.09]*10
    sl = calc_adaptive_stop(10.0, low)
    sh = calc_adaptive_stop(10.0, high)
    assert 9.2 <= sl <= 9.98
    assert 9.2 <= sh <= 9.98
    print("  ✅ adaptive_stop")

def test_take_profit():
    assert calc_take_profit(10.0, 11.0, 10.4) is not None
    assert calc_take_profit(10.0, 11.0, 10.6) is None
    print("  ✅ take_profit")

def test_generate_returns():
    r = generate_daily_returns([10.0, 10.5, 9.8, 10.2])
    assert len(r) == 3 and abs(r[0]-0.05)<0.001
    print("  ✅ generate_returns")

if __name__ == '__main__':
    print("=" * 50)
    for t in [test_is_broken_board,test_count_boards,test_detect_signals,
              test_factor_multiplier,test_position_size,
              test_adaptive_stop,test_take_profit,test_generate_returns]:
        t()
    print("✅ 8/8 全部通过")
