# superquant 完整业务逻辑 — 2026-06-25

## 一、盘前 (8:45, launchd 自动执行)

```
engine/scheduler.py --mode pre-market
  环境变量: PYTHONPATH=~/project/quant:~/project/superquant
  日志: logs/scheduler.log, logs/scheduler.err

Step 1: 日线更新
  执行: scripts/export_daily.py --market all
  来源: Sina 日线 API (http://money.finance.sina.com.cn/.../getKLineData?symbol=...&scale=240)
  目标: market.db daily 表 + HDF5 sh_day.h5/sz_day.h5
  数据: 全市场 5532 只股票, 增量更新最新一天
  耗时: ~120s

Step 2: ML 预测
  执行: ml/predict.py
  流程:
    2a. 加载 XGBoost 模型 (ml/model.json, AUC=0.9532)
    2b. 加载 孤立森林 (ml/if_model.pkl, anomaly_ratio=5%)
    2c. SQL 查询: 最近 20 天日线 + LEFT JOIN daily_features (L2-L5特征)
        返回字段: symbol,date,open,high,low,close,volume,amount,turnover
                 + ksft,slope,ptc,vol_5min,max_ret,min_ret (L2技术)
                 + main_net_in,main_net_ratio,super_large_in,large_in (L3资金)
                 + lhb_net_buy,lhb_buy_ratio,lhb_count,lhb_exists (L4龙虎)
                 + sector_limit_count,sector_rank (L5板块)
        共 26 列, 其中 L2-L4 缺失时用 COALESCE 填充 0
    2d. 逐只股票计算 10 个日线特征:
        ret_1d, ret_5d, vol_ratio, vol_5d, gap, turnover, amt_log, hl_ratio, close_pos, ma_dev
    2e. 合并 16 个 L2-L5 特征 → 26 维向量
    2f. XGBoost.predict_proba() → P(涨停) 概率 [0,1]
    2g. IF.score_samples() → 异常度 → 归一化到 [0,1]
    2h. 主力通道: combined = XGBoost × (0.5 + 0.5 × IF) → Top 20
    2i. 探索通道: IF > 0.85 AND XGBoost < 0.5 → IF × 0.7 → Top N
    2j. 输出: pre_market/candidate.json
        {"date":"2026-06-25",
         "main":[{"symbol":"920045","prob":0.9644},...],
         "discovery":[{"symbol":"920402","prob":0.6688}]}
  耗时: ~13s

Step 3: 参数调整
  执行: engine/auto_tuner.py
  读取: config/auto_tuning.json (上次研究结论)
  更新: config/active_params.json (活跃参数: 止损基线, Kelly, 因子权重)
  耗时: ~1s
```

## 二、盘中 (9:30-15:00, 手动启动)

```
trader/paper_trader.py --live

══════ 启动阶段 ══════

1. 获取资金
   engine/config.py: get_capital()
   └── SELECT cash FROM paper_account ORDER BY id DESC LIMIT 1
       首次启动 → 写入 ¥5000 → 返回 ¥5000
       后续启动 → 返回上次余额 (跨天延续)

2. 加载候选池
   get_ml_candidates()
   └── 读取 pre_market/candidate.json
       主力: data['main'] 中 prob >= 0.95 的股票
       探索: data['discovery'] 中的股票 (去重后)
       合并: 主力 + 探索 → 监控池
       回退: 文件不存在 → 默认候选 ['SH600000','SZ000001','SH600036']

3. 初始化状态
   positions = []          # 当前持仓 [{symbol,price,shares,buy_date,mode,peak}]
   history_cache = {}      # 价格历史 {symbol: [(open,high,low,close,volume),...]}
   trade_log = []          # 今日交易记录 [{date,symbol,side,price,shares,pnl}]

══════ 每 5 秒循环 ══════

while 9:30 ≤ now ≤ 15:00 且 now 不在午休:

  try: _run_scan_impl()
  except: 打印异常, 继续下一轮 (不崩溃)

  ┌─ _run_scan_impl() ────────────────────────────────────┐
  │                                                        │
  │  A. 拉取行情                                            │
  │    fetch_quotes(tracked)                                │
  │    └── Sina API: http://hq.sinajs.cn/list=              │
  │        市场前缀: 920→bj, 6/5/9→sh, 0/3→sz              │
  │        每批 800 只, 0.3s/批                             │
  │        返回: {symbol: {open,prev_close,price,high,      │
  │                        low,volume,amount}}              │
  │                                                        │
  │  B. 信号检测 (对每只候选)                                │
  │    for sym, q in quotes.items():                        │
  │      ├─ 跳过: open≤0 或 prev_close≤0 (无效报价)         │
  │      ├─ 跳过: price ≥ prev_close × 1.09 (涨停不买)      │
  │      ├─ 取历史缓存最近 60 根 K 线收盘价                 │
  │      │   history_cache[sym][-60:] → [close,...]         │
  │      ├─ 追加当前价: [历史..., 当前价]                   │
  │      ├─ bayesian_detect(prices, threshold=3.0):         │
  │      │   ├─ 扫描变点位置 τ (5~20个点前)                 │
  │      │   ├─ H0: 全序列同一分布 N(μ₀,σ₀²)               │
  │      │   ├─ H1: 变点后 μ₁ > μ₀ (拉升)                  │
  │      │   ├─ 贝叶斯因子 = P(data|H1)/P(data|H0)          │
  │      │   ├─ 波动率放大加权: var₂>var₁×1.5 → BF×1.5     │
  │      │   └─ BF > 3 → True (变点检测到)                  │
  │      └─ True → signals_triggered.append({symbol,price})  │
  │                                                        │
  │  C. 买入决策                                            │
  │    signals_triggered.sort(key=final_score, reverse=True) │
  │    candidates = [(symbol, price, score), ...]            │
  │    orders = calculate_buys(capital, candidates)          │
  │    │                                                     │
  │    │ calculate_buys() 内部:                              │
  │    │   if capital < 50000:                               │
  │    │     攻击期: candidates 按评分降序                    │
  │    │       for sym, price, score in candidates:          │
  │    │         max_shares = int(capital/price/100)*100     │
  │    │         while max_shares≥100:                       │
  │    │           cost = max_shares × price × 1.0003        │
  │    │           if cost ≤ capital: break                   │
  │    │           max_shares -= 100                          │
  │    │         if max_shares < 100: continue               │
  │    │         capital -= cost                             │
  │    │         → order (sym, shares, price, 'attack')      │
  │    │         持仓满 3 或资金不足 → 停止                  │
  │    │   else:                                             │
  │    │     Kelly: calc_position_size() 分散                │
  │    │                                                     │
  │    for sym, shares, price, phase in orders:              │
  │      ├─ 跳过: sym 已在今日持仓中                         │
  │      ├─ cost = shares × price × 1.0003                   │
  │      ├─ capital -= cost                                  │
  │      ├─ positions.append({symbol,price,shares,            │
  │      │     buy_date=today, mode=变点+phase,              │
  │      │     factor=0, peak=price})                        │
  │      ├─ record_trade(conn, sym, 'buy', ...)              │
  │      └─ trade_log.append({date,sym,'buy',price,shares})  │
  │                                                        │
  │  D. 止盈止损 + 风控                                      │
  │    for pos in positions:                                 │
  │      ├─ 跳过: sym 不在 quotes 中 (行情缺失)              │
  │      ├─ T+1 检查: pos.buy_date == today → continue      │
  │      │   来源: A股交易规则, 当日买入不得卖出             │
  │      ├─ 跌停保护: price ≤ prev_close × 0.905 → continue │
  │      │   来源: 交易所规则, 跌停无法卖出                  │
  │      ├─ pnl_pct = (price / pos.price - 1) × 100         │
  │      ├─ days_held = today - pos.buy_date                 │
  │      ├─ pos.peak = max(pos.peak, q.high)                 │
  │      │                                                    │
  │      ├─ 止损: calc_adaptive_stop(pos.price, returns)     │
  │      │   ├─ 从 history_cache 取最近 20 根K线价格          │
  │      │   ├─ generate_daily_returns(prices) → 收益率      │
  │      │   ├─ 只取负收益 → down_returns                    │
  │      │   ├─ daily_down_vol = std(down_rets)              │
  │      │   ├─ annual_vol = daily_down_vol × √252           │
  │      │   ├─ adaptive_pct = 0.05 × (annual_vol / 0.30)    │
  │      │   ├─ clamped: [0.02, 0.08]                       │
  │      │   └─ stop_px = entry × (1 - adaptive_pct)         │
  │      │   if price ≤ stop_px → to_sell (止损)             │
  │      │                                                    │
  │      ├─ 止盈: calc_take_profit(entry, peak, price)       │
  │      │   ├─ trigger = peak × 0.95                        │
  │      │   └─ if price ≤ trigger AND pnl_pct > 0           │
  │      │       → to_sell (移动止盈)                         │
  │      │                                                    │
  │      └─ 时间止损: if days_held ≥ 5 AND pnl_pct < 0      │
  │           → to_sell (时间止损)                            │
  │                                                            │
  │  E. 单日熔断                                              │
  │    daily_pnl = 今日所有卖出盈亏之和                       │
  │    daily_cost = 今日所有买入成本之和                      │
  │    if daily_cost > 0 AND daily_pnl/daily_cost < -0.05:   │
  │      → 暂停新交易 (不强制清仓)                            │
  │                                                            │
  │  F. 执行卖出                                              │
  │    for i, reason, pnl_pct in reversed(to_sell):           │
  │      pos = positions.pop(i)                               │
  │      sell_val = pos.shares × quotes[pos.symbol].price     │
  │      fee = sell_val × (0.0003 + 0.001)                    │
  │      pnl = sell_val - pos.shares × pos.price - fee        │
  │      capital += sell_val - fee                            │
  │      record_trade(conn, sym, 'sell', ...)                 │
  │      trade_log.append({sold})                             │
  │                                                            │
  │  G. 更新历史缓存                                          │
  │    for sym, q in quotes.items():                          │
  │      history_cache[sym].append((open,high,low,price,vol)) │
  │      保留最近 60 条                                       │
  └──────────────────────────────────────────────────────────┘

  打印: [HH:MM:SS] 资金=¥xxxx, 持仓=x

午休: 11:30-13:00 → 停止扫描, time.sleep()

══════ 收盘阶段 ══════

收盘条件: now.hour >= 15

1. 计算总权益
   liquidation_discount = 1.0 - (0.0003 + 0.001 + 0.0087) = 0.99
   equity = capital + Σ(pos.shares × pos.price × 0.99)

2. 持久化资金
   save_capital(capital, equity)
   └── INSERT/UPDATE paper_account(date, cash, equity, updated_at)
       后续启动: get_capital() 读取最新记录 → 跨天延续

3. 打印汇总
   资金=¥xxxx, 持仓=x, 权益=¥xxxx, 累计收益=+x.x%
```

## 三、盘后 (手动或定时)

```
1. 分钟数据存储
   data/minute_store.py --market all --today
   └── Sina 5分钟K线 → HDF5 sh_5min.h5/sz_5min.h5

2. L3 资金流回补
   ml/build_features.py --l3 --start N --count 10
   └── easy-tdx 4线程并行 → daily_features 表
       每批 10 个交易日, 25 批覆盖全年

3. ML 模型重训
   ml/train.py
   └── 日线(1年) + L2-L5特征 → XGBoost(AUC~0.95) + 孤立森林
       模型: ml/model.json + ml/if_model.pkl

4. 盘后研究 (待实现)
   - 今日交易分析: 胜率, 盈亏比, 最大回撤
   - ML 特征重要性变化
   - 参数自动优化 (auto_tuner)
```

## 四、关键参数来源

| 参数 | 值 | 来源 |
|------|-----|------|
| 扫描间隔 | 5s | quant/intraday_runner.py:853 |
| 涨停阈值 | 9% | daban 源码 raTh=8 + 1%缓冲 |
| 跌停阈值 | 90.5% | A股交易所规则 |
| 贝叶斯阈值 | 3.0 | Adams & MacKay 2007 |
| 攻击期上限 | ¥50,000 | 北极星10x后Kelly启用 |
| 止损基线 | 5% | strategy_core 默认 |
| 止盈回撤 | 5% | 行业惯例 |
| 持仓上限 | 5天 | 华安证券2025打板实证 |
| 单日熔断 | 5% | config.yaml |
| T+1 | true | A股交易规则 |
| 佣金 | 0.03% | 券商行业标准 |
| 印花税 | 0.1% | 国家税务总局 |
| Sina批量 | 800只/次 | 实测:800只0.3s |

## 五、当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| 盘前自动 | ✅ | launchd 每天 8:45 |
| ML预测 | ✅ | XGBoost+IF 双通道 |
| 盘中交易 | ✅ | PID 7269 --live |
| L3回补 | 🔄 | Batch 2 进行中 (PID 1758) |
| 资金 | ✅ | DB持久化, 跨天延续 |
| 测试 | ✅ | buy_rules(6/6) + strategy_core(8/8) |
