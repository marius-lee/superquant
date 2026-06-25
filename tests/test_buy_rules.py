import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.buy_rules import calculate_buys

def test_all_in():
    o = calculate_buys(5000, [('A', 13, 0.9)])
    assert len(o)==1 and o[0][1]==300; print("  ✅ all_in")

def test_two():
    o = calculate_buys(5000, [('A', 5, 0.9), ('B', 4, 0.8)])
    assert len(o)==2; print("  ✅ two")

def test_three():
    o = calculate_buys(30000, [('A', 5, 0.9), ('B', 4, 0.8), ('C', 3, 0.7)])
    assert len(o)==2; print("  ✅ two_of_three")

def test_price_high():
    o = calculate_buys(5000, [('A', 60, 0.9)])
    assert len(o)==0; print("  ✅ price_high")

def test_kelly():
    o = calculate_buys(60000, [('A', 10, 0.9)])
    assert len(o)>=1 and o[0][3]=='kelly'; print("  ✅ kelly")

def test_empty():
    assert calculate_buys(5000, [])==[]
    assert calculate_buys(0, [('A',10,0.9)])==[]
    print("  ✅ empty")

if __name__=='__main__':
    print("buy_rules 测试")
    for t in [test_all_in, test_two, test_three, test_price_high, test_kelly, test_empty]:
        t()
    print("✅ 6/6 通过")
