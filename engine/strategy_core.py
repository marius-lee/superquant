"""策略核心 — 纯函数, 零依赖, 回测和实盘共享唯一真相源。

设计原则:
  1. 不依赖 Hikyuu (框架无关)
  2. 不依赖任何数据库/网络
  3. 纯函数: 输入→输出, 无副作用
  4. 单元测试可覆盖

回测适配器: app/signals/__init__.py → 调用 detect_signals (包装为 crtSG)
实盘适配器:  trader/paper_trader.py  → 调用 detect_signals (实时行情驱动)

参数来源:
  手写基线: chen-xiaoqun-final-signal-design.md (17次搜索交叉验证)
  数据驱动: engine/threshold_optimizer.py 网格搜索 → config/optimal_thresholds.json
  优先级:   params参数 > optimal_thresholds.json > 手写基线

来源:
  P1 app/factors   — 因子定义
  P2 app/signals   — 信号检测
  P3 app/stops     — 止损止盈
  P4 app/strategy  — 策略组装
  execution/quote.py — 原陈小群模式识别

北极星: ¥5000 → ¥100万, 所有参数必须可配置、可回测、可自调整
"""

import json
import math
import os


# ═══════════════════════════════════════════════════════════
# 信号检测 (从 app/signals 提取)
# ═══════════════════════════════════════════════════════════

def is_broken_board(prev_high, prev_close, prev_open):
    """判断前一日是否炸板。

    来源: execution/quote.py:367 — st["yesterday_broken"]
    条件: 昨最高触及涨停 (≥prev_close×1.095) 且收盘未封住
    """
    limit_price = round(prev_open * 1.10, 2)
    touched_limit = prev_high >= limit_price * 0.995
    did_not_hold = prev_close < prev_high * 0.995
    return touched_limit and did_not_hold


def count_boards(closes, idx):
    """计算截至 idx 位置的连板数。closes[-1] 是最新。"""
    board = 1
    for j in range(idx, 0, -1):
        if closes[j-1] > 0 and (closes[j] / closes[j-1] - 1) >= 0.095:
            board += 1
        else:
            break
    return board


def load_params_for_mode(mode):
    """从 DB 加载单模式参数。来源: DB strategy_params 表 → 手写基线回退。"""
    try:
        from engine.db_schema import load_params
        p = load_params(mode)
        if p: return p
    except Exception:
        pass
    # 手写基线回退 (来源: chen-xiaoqun-final-signal-design.md, 17次搜索交叉验证)
    defaults = {
        'S1_弱转强': {'gap_min':2.0,'gap_max':5.0,'vol_ratio':3.0,'daily_ret':5.0},
        'S2_首阴反包': {'gap_min':3.0,'gap_max':10.0,'vol_ratio':0.0,'daily_ret':0.0,'turnover_min':0.10,'turnover_max':0.30},
        'S3_连板接力': {'gap_min':0.0,'gap_max':10.0,'vol_ratio':0.67,'daily_ret':0.0,'turnover_min':0.10,'turnover_max':1.0,'min_boards':2},
        'S4_首板试探': {'gap_min':2.0,'gap_max':10.0,'vol_ratio':0.0,'daily_ret':0.0,'turnover_min':0.10,'turnover_max':1.0,'min_boards':1},
    }
    return defaults.get(mode, {})


def detect_signals(kdata_records, params=None):
    """陈小群模式识别 — 纯函数，参数全部来自 DB。

    Args:
        kdata_records: K线记录列表, 每项(datetime,open,high,low,close,volume)
        params: None=自动从DB加载各模式参数, 也可传入dict覆盖

    Returns: [(datetime, signal_type, score), ...]

    参数来源链: DB strategy_params → 手写基线 → 代码默认值
    优化机制: engine/threshold_optimizer.py 定期更新 DB
    """
    if params is None:
        p1 = load_params_for_mode('S1_弱转强')
        p2 = load_params_for_mode('S2_首阴反包')
        p3 = load_params_for_mode('S3_连板接力')
        p4 = load_params_for_mode('S4_首板试探')
    else:
        p1 = p2 = p3 = p4 = params

    signals = []
    n = len(kdata_records)
    if n < 5:
        return signals

    closes = [r[4] for r in kdata_records]

    for i in range(2, n):
        dt, op, hi, lo, cl, vol = kdata_records[i]
        _, pop, phi, plo, pcl, pvol = kdata_records[i-1]

        if pcl <= 0 or op <= 0:
            continue

        gap = (op / pcl - 1) * 100
        daily_ret = (cl / pcl - 1) * 100
        vol_ratio = vol / max(pvol, 1)
        turnover = vol / 10000.0
        broken = is_broken_board(phi, pcl, pop)

        # S1: 弱转强 (p1 parameters)
        if broken and gap >= p1.get('gap_min',2.0) and gap <= p1.get('gap_max',5.0) \
           and vol_ratio >= p1.get('vol_ratio',3.0) and daily_ret >= p1.get('daily_ret',5.0):
            signals.append((dt, '弱转强', 0.90))
            continue

        # S2: 首阴反包 — 修正: 换手率检查炸板日(prev_k), 非信号日
        # 来源: 8轮搜索 — 换手率10-30%是分歧日的条件, 量比1.5x(非S1的3x)
        prev_turnover = pvol / 10000.0  # 炸板日换手率 (来源: 8轮搜索修正)
        if broken and gap >= p2.get('gap_min',1.0) \
           and prev_turnover >= p2.get('turnover_min',0.10) \
           and prev_turnover <= p2.get('turnover_max',0.30) \
           and vol_ratio >= p2.get('vol_ratio',1.5):
            signals.append((dt, '首阴反包', 0.85))
            continue

        # S3: 连板接力 — 修正: 添加市值过滤和板块联动条件
        # 来源: 8轮搜索 — 龙头需带动≥3家涨停, 流通市值30-80亿
        # 市值和板块条件在paper_trader层检查 (需实时数据)
        board = count_boards(closes, i)
        if board >= p3.get('min_boards',2) \
           and turnover >= p3.get('turnover_min',0.10) \
           and vol_ratio >= p3.get('vol_ratio',0.67):
            signals.append((dt, '连板接力', 0.70))
            continue

        # S4: 首板试探 — gap放宽到0.5 (数据驱动)
        if board == p4.get('min_boards',1) \
           and turnover >= p4.get('turnover_min',0.10) \
           and gap > p4.get('gap_min',0.5):
            signals.append((dt, '首板试探', 0.30))

    return signals


# ═══════════════════════════════════════════════════════════
# 因子评分 (从 app/factors 提取)
# ═══════════════════════════════════════════════════════════

def compute_factor_multiplier(stock_factor_score, params=None):
    """因子得分 → 信号乘数。

    公式: multiplier = 1 + stock_factor_score / 3
    范围: 0.5 (低分) ~ 1.5 (高分)

    Args:
        stock_factor_score: 单股票 Z-Score 标准化后的因子得分 (-3 ~ +3)
    Returns:
        multiplier: 0.5 ~ 1.5
    """
    return max(0.5, min(1.5, 1.0 + stock_factor_score / 3.0))


# ═══════════════════════════════════════════════════════════
# 仓位计算 — 半Kelly (从 ops/performance.py + app/strategy.py 提取)
# ═══════════════════════════════════════════════════════════

def calc_position_size(capital, price, risk_per_share, params=None):
    """半Kelly 仓位计算。

    来源: Kelly 1956, Thorp 2006, Chan 量化交易 §3

    f* = (b × p - q) / b
    half_kelly = f* × 0.5
    adjusted = half_kelly / (1 + ρ × (n-1))

    Args:
        capital: 可用资金
        price: 买入价
        risk_per_share: 每股风险 (止损价与买入价之差, ≥0.01)
        params: dict with win_rate, avg_win_loss, rho, n_positions
    Returns:
        shares: 买入股数 (100 的整数倍)
    """
    if params is None:
        params = {}
    win_rate = params.get('win_rate', 0.55)
    avg_win_loss = params.get('avg_win_loss', 2.0)
    rho = params.get('rho', 0.3)
    n_positions = params.get('n_positions', 3)
    max_position_pct = params.get('max_position_pct', 0.33)

    if risk_per_share <= 0:
        risk_per_share = 0.01

    # Kelly: f* = (b×p - q) / b
    kelly_f = (avg_win_loss * win_rate - (1 - win_rate)) / max(avg_win_loss, 1.0)
    half_kelly = kelly_f * 0.5
    kelly_adj = half_kelly / (1 + rho * max(n_positions - 1, 0))

    # 单笔最大仓位限制
    max_bet = capital * max_position_pct
    risk_amount = min(capital * kelly_adj, max_bet)
    shares = int(risk_amount / price / 100) * 100

    return max(shares, 0)


# ═══════════════════════════════════════════════════════════
# 止损计算 — 波动率自适应 (从 app/stops 提取)
# ═══════════════════════════════════════════════════════════

def calc_adaptive_stop(entry_price, daily_returns, params=None):
    """波动率自适应止损价。

    公式: stop = entry × (1 - adaptive_pct)
          adaptive_pct = base × (stock_down_vol / 0.30)
          floor: 2%, ceiling: 8%

    来源: ops/performance.py — A股残差波动率年化30%中位 (Grinold 表3-1)
          陈小群固定止损3-5%是另一套机制, 不适用于自适应公式
    """
    if params is None:
        params = {}
    base = params.get('adaptive_stop_base', 0.05)     # 来源: 默认5%, 与auto_tuner一致
    floor = params.get('adaptive_stop_floor', 0.02)   # 来源: 防止低波股票止损过紧
    ceiling = params.get('adaptive_stop_ceiling', 0.08)

    down_rets = [r for r in daily_returns if r < 0]
    if len(down_rets) < 5:
        return entry_price * (1 - base)

    daily_down_vol = _std(down_rets)
    annual_down_vol = daily_down_vol * math.sqrt(252)
    adaptive_pct = base * (annual_down_vol / 0.30)
    adaptive_pct = max(floor, min(ceiling, adaptive_pct))

    return entry_price * (1 - adaptive_pct)


def calc_take_profit(entry_price, peak_price, current_price, params=None):
    """移动止盈: 从最高点回撤 5% 触发。

    Returns: trigger_price 或 None (不触发)
    """
    if params is None:
        params = {}
    trail_pct = params.get('trail_stop_pct', 0.05)
    peak = max(entry_price, peak_price or entry_price)
    trigger = peak * (1 - trail_pct)
    return trigger if current_price <= trigger else None


# ═══════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════

def _std(values):
    """无依赖标准差。"""
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def generate_daily_returns(prices):
    """从价格序列生成日收益率。prices[-1] 是最新。"""
    return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
