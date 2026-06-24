# CLAUDE.md — superquant

Hikyuu 驱动的量化短线交易系统。陈小群模式识别 + 因子日历因子验证 + 波动率自适应风控。

## 项目结构

```
superquant/
├── app/                   # 核心应用
│   ├── factors/           # 自定义因子 Indicator (因子日历日频 + 分钟)
│   ├── signals/           # 陈小群信号 (crtSG)
│   ├── stops/             # 止损止盈 (crtST/crtTP)
│   ├── moneymgmt/         # 资金管理 (crtMM: 半Kelly)
│   ├── environment/       # 市场环境 (crtEV: 情绪周期)
│   └── strategy.py        # 主策略组装
├── data/                  # 数据层
│   └── minute_store.py    # Sina 5分钟线 → HDF5
├── docs/                  # 设计文档
├── tests/                 # 测试
├── scripts/               # 辅助脚本
└── web/                   # 前端 (未来)
```

## 架构

```
Hikyuu Strategy
├── EV: 市场情绪周期 (crtEV)
├── MF: 因子日历因子合成 (MF_ICIRWeight)
├── SE: 因子选股过滤 (SE_MultiFactor2)
├── System (per-stock)
│   ├── SG: 陈小群模式识别 (crtSG)
│   ├── ST: 波动率自适应止损 (crtST)
│   ├── TP: MCVA 动态止盈 (crtST)
│   └── MM: 半Kelly 资金管理 (crtMM)
└── PF: 多标的组合 (PF_Simple)
```

## 依赖

- Hikyuu 2.8.0+ (pip install hikyuu)
- 共享 quant 项目的 data/store.py (Sina 行情抓取)
- 共享 quant 项目的 config/ (策略配置)
- 共享 quant 项目的 utils/ (日志/日期工具)

## 与 quant 项目的关系

- `quant/` 保持不变，继续运行现有陈小群系统
- `superquant/` 独立开发，独立回测
- 共享代码通过 `sys.path` 导入，不复制
- 数据独立：superquant 使用 Hikyuu HDF5 (~/stock/)
- 双轨验证阶段 (P5): 两系统信号对比

## 数据目录

```
~/stock/
├── stock.db          # Hikyuu 元数据
├── sh_day.h5         # 上海日线
├── sz_day.h5         # 深圳日线
├── sh_5min.h5        # 上海5分钟线
└── sz_5min.h5        # 深圳5分钟线
```
