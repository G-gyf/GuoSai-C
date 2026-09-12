# 问题二 结果目录（已重组）

> 本目录集中存放问题二的全部结果、分析与历史对照。**当前正式答案 = LDR**（`current/ldr/result2.xlsx`，2—12 月实际总购电费 15,819,068.62 元）。

## 目录结构

```
outputs/question2/
├── current/      当前主方案（正式答案）
│   ├── ldr/      LDR 正式结果：result2.xlsx + 全年明细 + 参数诊断 + QA 图
│   └── paper/    论文正文《问题二完整解答_LDR主方案.md》+ figures/ + tables/
├── benchmark/    基准与下界
│   └── perfect_foresight/   完美预见 LP 下界（衡量信息缺口，非可达策略）
├── analysis/     分析检验（不进入 result2.xlsx）
│   ├── forecast/        光伏预测模块质量指标
│   ├── decomposition/   主方案 vs 情景方案费用差拆分
│   ├── g_search/        g 下界搜索 / α 扫描（参数探索记录）
│   ├── gap_audit/       固定 g 下界差额复核
│   └── beta_audit/      β 保留机制配对日分析
└── archive/      历史方案（已被 LDR 取代，仅留对照）
    ├── baseline/         旧基线（α=0.8, β=1），2—12 月 16,022,551 元
    └── scenarios/        原情景法（价值函数控制），2—12 月 15,868,380 元
```

## 快速导航

| 想找什么 | 去哪里 |
|---|---|
| 正式答案 Excel | `current/ldr/result2.xlsx` |
| 论文正文 | `current/paper/问题二完整解答_LDR主方案.md` |
| 模型演进与审查记录 | `../../docs/问题二/问题二完整方案与审查修订.md` |
| 完美预见下界说明 | `benchmark/perfect_foresight/完美预见下限测算说明.md` |
| 旧基线说明 | `archive/baseline/问题二实施与结果说明.md` |
| 原情景法说明 | `archive/scenarios/原情景法检验与对比.md` |

## 结果版本对照（2—12 月正式期）

| 版本 | 路径 | 实际总购电费 |
|---|---|---:|
| **LDR（当前主方案）** | `current/ldr/result2.xlsx` | 15,819,068.62 |
| 原情景法 | `archive/scenarios/result2.xlsx` | 15,868,380 |
| 旧基线（α=0.8, β=1） | `archive/baseline/result2.xlsx` | 16,022,551 |

> 说明：`archive/` 两版完整保留（含明细 CSV），供 LDR 脚本重跑时读取对照；仅清除了 debug 中间文件（`*.inspect.ndjson`、`checkpoint.pkl`、`pilot/`）。
