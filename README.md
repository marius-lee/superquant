# superquant

Hikyuu 驱动的量化短线交易系统。
陈小群模式识别 + 因子日历因子验证 + 波动率自适应风控。

## 安装

```bash
cd ~/project/superquant
pip install hikyuu numpy pandas matplotlib seaborn sqlalchemy click tqdm pyecharts
```

## 数据准备

```bash
# 从 quant 项目导出日线到 Hikyuu HDF5
PYTHONPATH=. python scripts/export_daily.py
```

## 运行

```bash
# 回测
PYTHONPATH=. python app/strategy.py
```
