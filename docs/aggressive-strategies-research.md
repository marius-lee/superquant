# 激进型量化打板策略 — 海量搜索 2026-06-24

## 核心发现：因子是慢的，技术+高频是快的

| 策略类型 | 年化收益 | 回撤 | 来源 |
|------|------|------|------|
| 首板回调 | +20.72% | -8.99% | 华安证券2025实证 |
| 开盘大单净买入 | +8.23% YTD | — | 海通证券2024 |
| AI多颗粒度DL | +22.41% YTD | — | 海通证券2024 |
| KSFT技术因子(小盘) | +39% | — | 瑞银量化 |
| 聚类增强反转 | +28-30% | — | 学术论文2025 |
| 强化学习日内交易 | +15.7%超额 | — | 民生证券2024 |

## 关键模式
1. 小盘股是主战场 (中证1000/2000, 技术因子年化40-50%)
2. 涨停板制度是A股独有Alpha来源
3. 高频+量价碾压基本面
4. DL/ML显著优于线性因子
5. 成本控制决定成败 (滑点0.05→0.30, 收益151→18%)

## 补充搜索 (2026-06-24 Round 2)

### 一、集合竞价+Level2 大单因子 (广发证券 2024)
- 94个大/小单因子: 多头年化 **36.61%**, 最大回撤17.52%, 夏普2.03
- 241个复合订单因子: Top30年化 **31.33%**, 夏普1.86
- 集合竞价9:15-9:25: 买单方向因子RankIC=-10.1%

### 二、隔夜+日内高频因子 (东吴/西部/中金 2024-2025)
- 东吴"日与夜"新动量因子: 多空年化 **18.37%**, IR=2.09, 月度胜率78%
- 西部TOI隔夜-日内拉锯因子: 月均IC=0.035, IC胜率 **83%**
- 中金高频动量复合: 周度多空 **64.2%**, 拥挤度复合 **69.9%**
- OVP因子(隔夜vs下午): 年化 **18.04%**, ICIR=4.17, 月度胜率86%

### 三、深度学习打板预测
- 华泰: 基于逐笔成交的深度学习选股模型 (2025)
- GitHub开源LSTM: 分钟级涨跌预测准确率~81% (2024)
- A股LOB数据集: 2020-2024秒级订单簿, 5364只股票

### 四、核心结论
- 开盘集合竞价是alpha最密集的时间段
- Level2逐笔数据比日线数据信息量高100倍+
- 隔夜收益率预测力 > 日内收益率
- 小盘股(中证1000/2000)技术因子回报是大盘股的3-5倍
- 深度学习模型(GPT因子工厂)已能自动化挖掘高频因子

## Round 3: 免费攻击性系统与数据 (2026-06-24)

### 免费开源打板系统
| 系统 | 特点 | 地址 |
|------|------|------|
| **daban** | 自动盯盘打板, 跨平台, LGPL | github.com/freevolunteer/daban |
| **xman** | C++ 高频抢板/半路板, 全内存 | github.com/showmsg/xman |
| **Ashare** | 免费分钟线 API, 双数据源 | github.com/wangpeng711/AutoTrade |
| **AData** | 免费量化数据库, 多源融合 | github.com/dodge-quant/A-shares-data |

### 免费数据获取路径
- Sina/腾讯: 分钟K线 (已有, 正在用)
- Ashare: 1m/5m/15m/30m/60m K线, 免费无限制
- AData: 资金流向/概念板块/龙虎榜, pip install
- jvQuant: WebSocket Level2 实时行情, 有免费额度
- PTrade/QMT: 券商提供, 免费Level2逐笔交易数据

### 关键结论
- 分钟K线: 免费且充足 (Sina/腾讯/Ashare多个源)
- Level2逐笔: PTrade券商免费, jvQuant有免费额度
- 打板执行: daban/xman提供完整的盯盘+自动下单
- 不存在完全免费+完整的"涨停预测系统"
- 但组件齐全: 数据+策略+执行 三块各自有免费方案

## Round 4: 免费攻击性系统深度搜索 (2026-06-24)

### 免费自动打板系统
| 系统 | 语言 | 特点 | 来源 |
|------|------|------|------|
| **daban** | Python | 自动盯盘打板, 跨平台, LGPL | github.com/freevolunteer/daban |
| **xman** | C++ | 抢板/半路板/集合竞价, 全内存, 微秒级 | github.com/showmsg/xman |
| **king-pin** | Python | 封板王, 多因子+舆情+风控, 2025年 | github.com/JunFuXu/king-pin |
| **ZGNB** | Python | Z哥战法(BBI+KDJ/Peak/放量), 短线 | github.com/zhouyu102030/ZGNB |
| **StockTradebyZ** | Python | Z哥战法升级版(暴力K/上穿60), 2025 | github.com/SebastienZh/StockTradebyZ |

### 免费实时行情系统
| 系统 | 数据 | 特点 |
|------|------|------|
| **Ashare** | 分钟/日线 | 新浪+腾讯双源, pip install |
| **AData** | 全品类 | 多源融合, pip install adata |
| **XTick** | Tick/分钟 | WebSocket, 2024年起 |
| **jvQuant** | Level2 | WebSocket实时, 有免费额度 |
| **tdx2db** | 日线/分钟 | DuckDB, 30s建库 |

### 免费情绪/选股/分析系统
| 系统 | 特点 |
|------|------|
| **InStock** (myhhub/stock) | 200+指标, 11策略, Docker |
| **longhubang-stock** | AI龙虎榜分析, GLM-4 |
| **aiagents-stock** | 6AI Agent协作, miniQMT |
| **vnpy-CTA** | CPV因子, 情绪周期 |

### 最终结论
1. 免费攻击性系统已覆盖完整链路: 数据→分析→信号→执行
2. 最佳组合: Ashare(数据) + daban(打板执行) 或 xman(高频抢板)
3. king-pin 架构最完整 (数据→处理→决策→执行→风控)
4. 不存在免费+完整的"一键打板赚钱"系统, 需要自行组合

## Round 5: 更激进的系统 + 完整打板系统 (2026-06-24)

### 新发现 (之前未收录)
| 系统 | 语言 | 特点 | 链接 |
|------|------|------|------|
| **TradingAgents-astock** | Python | 7 AI分析师, 含游资追踪师, 龙虎榜/解禁/政策专项, 免费数据源 | github.com/simonlin1212/TradingAgents-astock |
| **easytrader** | Python | 同花顺/国金/华泰自动交易, 均线/打板/打新 | github.com/huadi/easytrader |
| **STIP** | Python/Streamlit | 交互式技术指标回测, K线形态识别, 多条件组合 | github.com/cn-vhql/STIP |
| **QTYX** | Python | 涨停板选股+题材+形态, miniQMT实盘 | CSDN |
| **JoinQuant打板** | Python | 一进二+首板低开+弱转强, 5年950x | joinquant.com |

### 新增克隆 (手动下载)
```
git clone --depth 1 https://github.com/simonlin1212/TradingAgents-astock.git
git clone --depth 1 https://github.com/huadi/easytrader.git
git clone --depth 1 https://github.com/cn-vhql/STIP.git
```

## Round 6: 极深搜索 (2026-06-24)

### 新发现
| 系统 | 类型 | 亮点 |
|------|------|------|
| **QMT-QuantLimit** | Python | miniQMT实盘打板 + Kimi/阶跃星辰大模型AI选股 + 韭研公社舆情 |
| **QTYX V3.1.9** | Python | 涨停数据库 + 热门题材跟踪 + DeepSeek题材分析 + 形态选股 |
| **龙系选股系统** | Python/Flask/Vue3 | 龙头战法三层门控评分 + 10项指标 + 情景计划 |
| **gitee: yeapllg** | Python | AI量化打板软件, 专为A股设计 |
| **百果量化** | 策略 | 突破上升打板 + 集合竞价打三板 (策略思路+源码) |

### 系统总览 (6轮搜索, 60+来源)
现已覆盖:
  打板执行: daban, xman, QMT-QuantLimit, easytrader
  架构参考: king-pin, QTYX, 龙系选股
  AI智能: TradingAgents-astock, aiagents-stock, longhubang-stock
  短线选股: ZGNB, StockTradebyZ, InStock
  数据: Ashare, easy-tdx, AData, AKShare

## Round 7: 极限搜索 (2026-06-24)

### 新发现
| 系统 | 类型 | 关键 |
|------|------|------|
| **QMT-QuantLimit** | Python | 完整打板系统, QMT实盘, Kimi/阶跃AI选股, 源代码完整 |
| **KuaiT** | C++ | 同花顺自动下单HTTP API, OCR验证码识别 |
| **THSAutoTrader** | Python/Flask | 129⭐同花顺闪电下单, Web API |
| **BitSoulStockSkill** | Python | 100+因子, MoE选股, 龙虎榜, 完整回测 |
| **小鸭量化** | Python/PyQt5 | 图形化量化工具, 20+策略内置 |
| **百果量化** | 策略社区 | PTrade打板: 连板/首板/二板/三板策略代码 |
| **QMT打板** | 社区 | 打板源码框架: subscribe_quote→涨停检测→封板买入 |

### 累计覆盖 (7轮, 80+来源)
执行: daban, xman, QMT-QuantLimit, easytrader, THSAutoTrader, KuaiT
架构: king-pin, QTYX, 龙系选股, 小鸭量化
AI: TradingAgents-astock, aiagents-stock, longhubang, BitSoulStock
选股: ZGNB, StockTradebyZ, InStock
数据: Ashare, easy-tdx, AKShare, AData, Sina, tdx2db
策略: 百果量化(PTrade), JoinQuant打板, 通达信公式
