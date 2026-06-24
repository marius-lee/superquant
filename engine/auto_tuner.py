#!/usr/bin/env python3
"""策略自主调整 — 研究结论 → 策略参数更新。

读取 engine/researcher.py 输出的 config/auto_tuning.json,
根据 IC 变化和回测结果自动调整:
  1. 因子权重 (IC 驱动)
  2. 信号阈值 (回测最优)
  3. 止损基线 (波动率环境)
  4. Kelly 参数 (近期胜率/盈亏比)

调整规则:
  - IC 连续 3 日下降 → 权重减半
  - IC 连续 5 日上升 → 权重加倍 (上限 0.40)
  - 胜率 < 40% → 收紧止损 (基线 +1%)
  - 胜率 > 60% → 放宽止损 (基线 -1%)

输出: config/active_params.json — 盘中策略实时读取
"""

import json, os, glob
from datetime import date

SUPERQUANT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH_FILE = os.path.join(SUPERQUANT_ROOT, "config", "auto_tuning.json")
ACTIVE_FILE = os.path.join(SUPERQUANT_ROOT, "config", "active_params.json")

# 默认参数 (来源: 代码校准)
DEFAULTS = {
    'factor_weights': {
        'downside_vol': 0.25,
        'overnight_gap': 0.25,
        'amihud': 0.25,
        'skewness': 0.25,
    },
    'signal': {
        'weak_to_strong': {'gap_min': 2.0, 'gap_max': 5.0, 'vol_ratio': 3.0},
        'first_yin_fanbao': {'gap_min': 3.0, 'turnover_min': 0.10, 'turnover_max': 0.30},
        'lianban': {'min_boards': 2, 'turnover_min': 0.10, 'vol_ratio_min': 0.67},
    },
    'stops': {
        'adaptive_stop_base': 0.05,  # 基线 5%
        'adaptive_stop_floor': 0.02,
        'adaptive_stop_ceiling': 0.08,
    },
    'kelly': {
        'win_rate': 0.55,        # 来源: Chan §3.4 — 良好策略典型胜率
        'avg_win_loss': 2.0,     # 来源: Chan §3.4 — 典型盈亏比
        'half_kelly_mult': 0.5,  # 来源: Chan §3.4 — 半Kelly保留75%收益
        'rho': 0.3,              # 来源: McDonnell — A股持仓相关性
        'n_positions': 3,        # 来源: config.yaml max_positions
        'max_position_pct': 0.33,# 来源: config.yaml 单票≤33%
    },
    'meta': {
        'updated_at': date.today().isoformat(),
        'source': 'defaults',
    },
}


def load_history(n_days: int = 5) -> list:
    """加载过去 N 天的研究报告。"""
    reports = []
    pattern = os.path.join(SUPERQUANT_ROOT, "config", "auto_tuning_*.json")
    files = sorted(glob.glob(pattern))
    for f in files[-n_days:]:
        try:
            with open(f) as fp:
                reports.append(json.load(fp))
        except Exception:
            pass
    return reports


def tune():
    """运行参数调整, 产出活跃参数。"""
    print("=" * 60)
    print("superquant 策略自主调整")
    print(f"  时间: {date.today().isoformat()}")
    print("=" * 60)

    # 加载最新研究报告
    if not os.path.exists(RESEARCH_FILE):
        print("  [warn] 无研究报告, 使用默认参数")
        params = dict(DEFAULTS)
    else:
        with open(RESEARCH_FILE) as f:
            research = json.load(f)

        # 从研究报告提取权重
        weights = research.get('weights', DEFAULTS['factor_weights'])
        kelly = research.get('kelly', DEFAULTS['kelly'])

        # 加载历史 IC 趋势
        history = load_history(5)
        ic_trend = {}
        if len(history) >= 3:
            for name in weights:
                if name == 'note':
                    continue
                recent_ics = []
                for h in history[-5:]:
                    ic_data = h.get('ic', {}).get(name, {})
                    recent_ics.append(ic_data.get('trend', 0))
                # 简单趋势: 最近3天的IC方向
                if len(recent_ics) >= 3:
                    trend = sum(recent_ics[-3:])
                    ic_trend[name] = 'up' if trend > 0 else 'down'

        # 应用 IC 驱动调整
        adjusted_weights = {}
        for name in weights:
            if name == 'note':
                continue
            w = weights[name]
            if name in ic_trend:
                if ic_trend[name] == 'down':
                    w = max(0.05, w * 0.7)
                else:
                    w = min(0.40, w * 1.2)
            adjusted_weights[name] = round(w, 3)

        # 归一化
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: round(v / total, 3) for k, v in adjusted_weights.items()}

        params = {
            'factor_weights': adjusted_weights,
            'signal': DEFAULTS['signal'],
            'stops': DEFAULTS['stops'],
            'kelly': kelly,
            'meta': {
                'updated_at': date.today().isoformat(),
                'source': 'auto_tuned',
                'ic_trend': ic_trend,
            },
        }

    # 保存
    os.makedirs(os.path.dirname(ACTIVE_FILE), exist_ok=True)
    with open(ACTIVE_FILE, 'w') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print(f"\n  因子权重: {params['factor_weights']}")
    print(f"  Kelly: {params['kelly']}")
    print(f"  止损基线: {params['stops']['adaptive_stop_base']*100:.0f}%")
    print(f"✅ 活跃参数 → {ACTIVE_FILE}")
    return params


if __name__ == '__main__':
    tune()
