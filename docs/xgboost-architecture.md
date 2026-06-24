# XGBoost 在 superquant 中的位置

## 架构注入点

```
superquant/
├── engine/
│   ├── strategy_core.py        # 止损/止盈/Kelly (保留)
│   ├── scheduler.py            # 调度框架 (保留)
│   ├── auto_tuner.py           # 参数调整 (保留)
│   ├── db_schema.py            # DB 管理 (保留)
│   └── ml_trainer.py           # ← 新增: 模型训练
│
├── ml/                         # ← 新增目录
│   ├── features.py             # 特征工程 (L1-L5 → 特征向量)
│   ├── train.py                # 训练脚本
│   ├── predict.py              # 预测脚本
│   └── model.json              # 训练好的模型文件
│
├── pre_market/                 # ← 新增目录 (盘前五层筛选)
│   ├── scanner.py              # L1-L5 筛选
│   └── candidate.json          # 今日候选
│
├── trader/
│   └── paper_trader.py         # 执行 (接收到ML选股)
│
└── web/
    └── app.py                  # 展示ML预测结果
```

## 数据流

```
每天 15:10 (盘后):
  ml/train.py
    ├── 读取 market.db 最近 1 年日线
    ├── 读取 HDF5 5分钟线
    ├── 读取 easy-tdx 资金流
    ├── 标签: 当日是否涨停 (binary)
    ├── 训练 XGBoost
    └── 保存 ml/model.json

每天 8:55 (盘前):
  pre_market/scanner.py
    ├── L1: Sina竞价 → 300只
    ├── L2: HDF5 技术形态 → 90只
    ├── L3: easy-tdx 资金流 → 18只
    ├── L4: AKShare 龙虎榜 → 5只
    └── L5: Sina 板块共振 → 2-3只

每天 9:00 (盘前):
  ml/predict.py
    ├── 读取 L1候选
    ├── 对每只计算特征 (features.py)
    ├── XGBoost 预测 P(涨停)
    ├── 按概率排序 → Top 5-10
    └── 写入 candidate.json

每天 9:30 (盘中):
  trader/paper_trader.py
    ├── 读取 candidate.json
    ├── 监控这 5-10 只
    ├── 封板触发 → 模拟买入
    └── 止损止盈 (strategy_core)
```

## 特征向量 (features.py)

```python
# 对每只股票计算的特征 (全部免费):
features = {
    # L1 集合竞价
    'gap_pct': 高开幅度,
    'auction_vol_ratio': 竞价量/昨量,
    'auction_price_change': 竞价期间涨速,
    
    # L2 技术形态  
    'ksft_5min': K线位移,
    'slope_5min': 动量斜率,
    'ptc_5min': 量价相关性,
    'volatility_5min': 波动率,
    'relative_position': 相对位置,
    
    # L3 资金流
    'super_large_net': 超大单净流入,
    'large_net': 大单净流入,
    'main_net_ratio': 主力净占比,
    'small_net': 小单净流入,
    
    # L4 龙虎榜
    'lhb_net_buy': 龙虎榜净买入,
    'lhb_buy_ratio': 买入占比,
    'lhb_institution': 机构参与度,
    
    # L5 板块
    'sector_limit_count': 同板块涨停数,
    'sector_rank': 板块热度排名,
    
    # 基础
    'market_cap': 流通市值,
    'turnover_rate': 换手率,
    'prev_day_ret': 昨日涨幅,
}
```

## 与现有系统关系

| 现有 | 改为 |
|------|------|
| 手写 L1-L5 if/else | XGBoost 预测概率 |
| 固定候选 2-3 只 | 概率 Top 5-10 |
| 无模型更新 | 每天盘后自动重训 |
| 无特征记录 | ml/features.py 统一管理 |

## 北极星贡献

```
现在: 手写规则 → 30% 封板率 → 日期望 2.1%
ML后: XGBoost → 预期 50%+ 封板率 → 日期望 3.5%+
      ¥5000 × 1.035^250 = ¥590,000
```
