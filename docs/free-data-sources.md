# superquant 免费数据源手册 — 2026-06-24

## 实测可用 ✅

### 实时行情
| 源 | 安装/接入 | 频率 | 延迟 | 单次量 | 限流 |
|------|------|------|------|------|------|
| **Sina** | `http://hq.sinajs.cn/list=` | 实时 | <1s | 800只/0.3s | 未见限流 |
| 用法: `requests.get(url+','.join(codes[:800]), headers={'Referer':'https://finance.sina.com.cn'})` |

### 分钟K线
| 源 | 安装/接入 | 频率 | 存储 |
|------|------|------|------|
| **Sina 5分钟线** | `money.finance.sina.com.cn/...getKLineData?symbol=sh600000&scale=5` | 5min | HDF5 |

### 资金流向 (大单/中单/小单)
| 源 | 安装 | API | 性能 |
|------|------|------|------|
| **easy-tdx** | `pip install easy-tdx` | `TdxClient().get_fund_flow(market, code)` | 700ms/只 |
| 返回字段: `super_in, large_in, medium_in, small_in, super_out, large_out, medium_out, small_out` |
| 注意: 需通达信服务器连接 (免费, 无需安装通达信客户端) |

### 龙虎榜
| 源 | 安装 | API | 性能 |
|------|------|------|------|
| **AKShare** | `pip install akshare` | `stock_lhb_detail_daily_sina(date='YYYYMMDD')` | 0.8s |
| **AData** | `pip install adata` | `sentiment.hot.list_a_list_daily()` | <1s |

### 日线
| 源 | 存储 | 说明 |
|------|------|------|
| **market.db** | SQLite (16M行) | quant项目已有, P0a每日更新 |
| **HDF5** | sh_day.h5 + sz_day.h5 | Hikyuu兼容格式 |

### 股票基础信息
| 源 | 安装 | API |
|------|------|------|
| **easy-tdx** | 已安装 | `TdxClient().get_security_list(market)` |
| **AData** | 已安装 | `stock.info.all_code()` |

## 测试过但不可用 ❌

| 源 | 原因 |
|------|------|
| 东方财富 push2 | 限流 (2025-2026升级反爬) |
| 东方财富 push2his | 限流 |
| AKShare 资金流 | 底层连东方财富, 随东方财富一起毙 |
| AData 资金流 | 底层连东方财富 |
| zzshare 资金流 | API不存在 |
| Sina MoneyFlow | 返回空模板 (已废弃) |
| 腾讯资金流 | 返回空 |
| 网易/和讯/百度 | HTTP 502/404/403 |
| Baostock | 无资金流接口 |
| Tushare 免费版 | 资金流需付费Pro版 |

## 五层筛选数据映射

| 层 | 数据需求 | 数据源 | 时机 |
|------|---------|------|------|
| L1 竞价 | real-time OHLCV | Sina | 9:25 |
| L2 技术形态 | 5min K线 | HDF5 (P0b) | 9:25-9:30 |
| L3 资金流 | 大单/中单/小单 | easy-tdx | 9:25-9:30 |
| L4 龙虎榜 | 昨日上榜 | AKShare | 盘前 |
| L5 板块 | 板块涨停数 | Sina | 9:30 |

全部免费, 全部无需注册。
