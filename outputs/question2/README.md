# 问题二 结果目录（已重组）

> 本目录集中存放问题二的全部结果、分析与历史对照。**当前正式答案 = LDR（负载周持久化＋光伏七天均值）**（`current/ldr/result2.xlsx`，2—12 月实际总购电费 13,978,077.23 元）。2025-01-01 为冻结初始条件日（无计划、电池不动作、SOC 恒 6000 kWh、不入统计），首个计划日为 1/2，全年账本为 1/2—12/31 共 364 天、52416 区间。该方案在当前数据口径、参数边界、搜索预算和已完成对照范围内费用最低；相似日高斯核暂不接入本轮费用回测，因此不在该排序范围内。

## 目录结构

```
outputs/question2/
├── current/      当前主方案（正式答案）
│   ├── ldr/      LDR 正式结果：result2.xlsx + 全年明细 + 参数诊断 + QA 图
│   └── paper/    论文正文《问题二完整解答_LDR主方案.md》+ figures/ + tables/
├── benchmark/    基准与下界
│   └── perfect_foresight/   完美预见 LP 下界（matched_ldr：按现行LDR期初库存匹配）
├── analysis/     分析检验（不进入 result2.xlsx）
│   ├── forecast/        光伏预测模块质量指标
│   ├── load_forecast/   负载预测四方案对比、b1 接入重算对比
│   ├── decomposition/   主方案 vs 情景方案费用差拆分（历史口径记录）
│   ├── g_search/        g 下界搜索 / α 扫描（参数探索记录）
│   ├── gap_audit/       固定 g 下界差额复核（历史口径记录）
│   └── beta_audit/      β 保留机制配对日分析（历史口径记录）
└── archive/      历史方案（已被现行方案取代，仅留对照）
    ├── baseline/         旧基线（α=0.8, β=1, 七天均值），2—12 月 16,022,551 元
    ├── scenarios/        情景价值控制（负载周持久化，现行对照方案）
    ├── scenarios_mean7d/ 原情景法（七天均值），2—12 月 15,868,380 元
    └── ldr_mean7d/       旧 LDR（七天均值负载预测），2—12 月 15,819,069 元
```

## 快速导航

| 想找什么 | 去哪里 |
|---|---|
| 正式答案 Excel | `current/ldr/result2.xlsx` |
| 论文正文 | `current/paper/问题二完整解答_LDR主方案.md` |
| 模型演进与审查记录 | `../../docs/问题二/问题二完整方案与审查修订.md` |
| 负载预测对比与 b1 切换说明 | `analysis/load_forecast/负载预测方案对比说明.md`、`analysis/load_forecast/b1接入重算对比.md` |
| 完美预见下界说明 | `benchmark/perfect_foresight/完美预见下限测算说明.md` |
| 旧基线说明 | `archive/baseline/问题二实施与结果说明.md` |
| 原情景法说明（七天均值时期） | `archive/scenarios_mean7d/原情景法检验与对比.md` |

## 结果版本对照（2—12 月正式期）

| 版本 | 路径 | 实际总购电费 |
|---|---|---:|
| **LDR 周持久化负载（当前主方案）** | `current/ldr/result2.xlsx` | 13,978,077.23 |
| 情景价值控制（周持久化，对照） | `archive/scenarios/` | 13,983,219.29 |
| 旧 LDR 七天均值负载 | `archive/ldr_mean7d/result2.xlsx` | 15,819,068.62 |
| 原情景法（七天均值） | `archive/scenarios_mean7d/result2.xlsx` | 15,868,380 |
| 旧基线（α=0.8, β=1） | `archive/baseline/result2.xlsx` | 16,022,551 |

> 说明：`archive/` 三版完整保留（含明细 CSV），供对照读取；仅清除了 debug 中间文件（`*.inspect.ndjson`、`checkpoint.pkl`、`pilot/`）。
