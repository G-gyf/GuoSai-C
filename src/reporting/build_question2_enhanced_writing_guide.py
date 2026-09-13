"""Create the evidence-enhanced complete Q2 writing guide from the reviewed HTML."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "问题二" / "问题二写作指导_思路与结果呈现及图表建议.html"
OUTPUT = ROOT / "docs" / "问题二" / "问题二写作指导_增强论证完整稿.html"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one marker, found {count}: {old[:80]!r}")
    return text.replace(old, new, 1)


def figure(path: str, title: str, caption: str) -> str:
    return f'''<figure class="figure-card">
  <a href="figures_增强论证/{path}"><img src="figures_增强论证/{path}" alt="{title}"></a>
  <figcaption><strong>{title}</strong><br>{caption}</figcaption>
</figure>'''


def main() -> None:
    html = SOURCE.read_text(encoding="utf-8")
    html = replace_once(
        html,
        "<title>问题二论文写作指导：建模思路 · 结果呈现 · 图表生成</title>",
        "<title>问题二论文写作指导（增强论证完整稿）：模型逻辑 · 结果解释 · 稳健性 · 图表</title>",
    )
    html = replace_once(
        html,
        "<h1>问题二论文写作指导：建模思路 · 结果呈现 · 图表生成</h1>",
        "<h1>问题二论文写作指导（增强论证完整稿）</h1>",
    )
    html = replace_once(
        html,
        '<p class="subtitle">以本队问题二实际实施方案（周持久化负载预测＋光伏七天均值＋80% 分位数风险修正＋四阶段截断仿射保留规则 LDR，正式期实际总费 13,978,077.23 元）为依据，结合可查证文献，给出论文该部分的写作思路、结果呈现与图表生成建议。</p>',
        '<p class="subtitle">在原有“周持久化＋光伏七天均值＋80%风险修正＋截断仿射保留规则”方案上，补入逐区间覆盖校准、规则工作状态、配对证据边界、有限敏感性与结构消融。主结果仍按正式期实际总费 13,978,077.23 元口径呈现，同时明确哪些结论已经被数据支持、哪些仍属于合理解释或待验证假设。</p>',
    )

    css = r'''

  .figure-card{margin:24px 0 30px; padding:14px; border:1px solid var(--line);
    border-radius:12px; background:#fbfcfe; box-shadow:0 5px 18px rgba(31,95,168,.07);}
  .figure-card img{display:block; width:100%; height:auto; border-radius:7px; background:#fff;}
  .figure-card figcaption{margin:10px 6px 2px; color:var(--muted); font-size:13.5px; line-height:1.65;}
  .evidence-summary{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0;}
  .evidence-summary .metric{border:1px solid var(--line); border-radius:10px; padding:14px 12px;
    background:#fbfcfe; min-height:105px;}
  .metric .value{font-size:23px; font-weight:700; color:var(--blue-dark); font-variant-numeric:tabular-nums;}
  .metric .label{font-size:13px; color:var(--muted); margin-top:3px;}
  .metric .note{font-size:12px; color:#7b8792; margin-top:5px; line-height:1.45;}
  .verdict{border:1px solid #b8d3df; border-left:6px solid #6f9db7; border-radius:10px;
    padding:16px 20px; background:#f2f8fb; margin:18px 0;}
  .verdict strong{font-size:17px;}
  .priority-high{color:#9d433d; font-weight:700;} .priority-mid{color:#8c6a25; font-weight:700;}
  .source-note{font-size:12.5px; color:var(--muted); border-top:1px dashed var(--line); padding-top:8px; margin-top:10px;}
  @media(max-width:820px){.evidence-summary{grid-template-columns:1fr 1fr;}}
  @media(max-width:520px){.evidence-summary{grid-template-columns:1fr;}}
'''
    html = replace_once(html, "</style>", css + "\n</style>")

    html = replace_once(
        html,
        '<ol>\n    <li><a href="#s1">一、评委视角：问题二到底考什么</a></li>',
        '<ol>\n    <li><a href="#s0">零、增强稿结论与使用边界</a></li>\n    <li><a href="#s1">一、评委视角：问题二到底考什么</a></li>',
    )
    html = replace_once(
        html,
        '    <li><a href="#s4">四、结果呈现建议（表格清单与叙事线）</a></li>',
        '    <li><a href="#s3x">三-A、核心论证补强：新增实证结果</a></li>\n    <li><a href="#s4">四、结果呈现建议（表格清单与叙事线）</a></li>',
    )

    summary = f'''

<h2 id="s0">零、增强稿结论与使用边界</h2>

<div class="verdict"><strong>总判断：主线合理、结果可复算，但论证强度必须分层。</strong><br>
预测—情景—风险计划—规则校准—日内执行—跨日库存的闭环是成立的；物理、账单与信息边界校验充分。当前证据可以支持“主方案可执行、在已完成正式口径对照中费用较低”，但不能支持“21日窗口最优”“反馈斜率必然增益”“最坏情形不劣于基线”或“相对情景价值控制具有稳定显著优势”。论文应把<strong>已验证事实、机制解释、启发式设计、未完成验证</strong>四类内容明确分开。</div>

<div class="evidence-summary">
  <div class="metric"><div class="value">1,397.81 万元</div><div class="label">LDR 正式期实际总费</div><div class="note">计划费 1,333.28 万元＋紧急费 64.52 万元</div></div>
  <div class="metric"><div class="value">76.94%</div><div class="label">80%风险曲线实际区间覆盖率</div><div class="note">低于名义80%，说明经验尾部仍偏乐观</div></div>
  <div class="metric"><div class="value">42.11%</div><div class="label">保留阈值位于SOC下限的区间占比</div><div class="note">规则相当一部分时间退化为即时补缺</div></div>
  <div class="metric"><div class="value">−6.96 万元</div><div class="label">14日残差窗相对21日主方案</div><div class="note">同年度事后敏感性，不是独立外部验证</div></div>
</div>

<table>
<tr><th style="width:17%">论证层级</th><th>可以写什么</th><th>不能写什么</th></tr>
<tr><td><strong>已验证事实</strong></td><td>真实账单、逐时物理可行、因果信息边界、正式期策略排序、配对回放结果</td><td>把数值排序扩展成普遍最优性</td></tr>
<tr><td><strong>有数据支持的解释</strong></td><td>紧急购电加权单价较低，说明节费与补购时序相关；风险曲线明显提高点预测覆盖</td><td>仅凭年度汇总识别每个参数的独立因果贡献</td></tr>
<tr><td><strong>启发式设计</strong></td><td>max{点预测,Q<sub>0.8</sub>}、21日窗、四阶段、端值代理和参数边界</td><td>称其由理论唯一推出或已达到最优</td></tr>
<tr><td><strong>未完成验证</strong></td><td>外部年度泛化、完整同初始SOC对照、更多窗口/种子/边界、预测模型的端到端费用比较</td><td>隐藏缺口或用“已足够”代替敏感性分析</td></tr>
</table>

{figure('fig01_workflow.png', '图1　问题二的因果决策链与证据闭环', '图中虚线表示0:00时的信息边界；实际路径随后揭示，日内动作仅依赖已完成区间信息，期末SOC作为跨日状态传递。')}
'''
    marker = "\n<!-- ===================== 一 ===================== -->"
    html = replace_once(html, marker, summary + marker)

    # Correct claims that were too strong or internally inconsistent.
    html = html.replace(
        "该接受规则把'最坏情形不劣于基线'作为硬性保证",
        "该接受规则仅保证候选在当前21条历史情景的等权平均经验目标上不劣于零参数基线；它不构成逐情景、最坏情形或样本外保证",
    )
    html = html.replace(
        '四策略对照即"计划×控制器"的消融；预测方案对比；下界差额讨论；局限逐条列出',
        '四策略提供整套方案的初步对照；同计划、同日初库存的配对回放用于补充隔离控制器影响，但日末库存价值仍需单列；另加入残差窗口、随机种子、搜索预算、参数边界与反馈斜率的有限敏感性',
    )
    html = html.replace(
        '证明费用优势来自紧急购电时序（4.48/5.73/5.20/4.97 元/kWh）',
        '支持“费用优势与紧急购电时序有关”的解释（4.48/5.73/5.20/4.97 元/kWh），但不能仅凭加权均价完成因果识别',
    )
    html = html.replace(
        '配对对照表证明改善来自控制器本身（计划不变、库存相同）',
        '配对对照表支持改善与控制器有关（计划不变、日初库存相同）；由于日末库存未强制一致，应同时披露库存差异，不能写成严格因果证明',
    )
    html = html.replace(
        '四策略对照＝计划×控制器消融；预测四方案对比；下界定位；已足够，在讨论节如实说明未做多种子/窗口敏感性',
        '四策略与预测对比只能算初步证据，不能替代敏感性分析；本增强稿已补跑随机种子、14/21/28日窗口、4/8/12代预算、参数边界与λ=0结构消融，并明确仍缺外部年度验证',
    )
    html = html.replace(
        '均已核实可查证；<strong>第 9 条作者名单请以期刊官网为准再行核对</strong>',
        '第1—8条已按现有信息整理；<strong>第9条作者名单尚未完成最终核对，正式提交前必须以期刊官网为准补全</strong>',
    )
    html = html.replace(
        '说明"零紧急购电≠经济最优"',
        '提示“零紧急购电并不自动等于经济最优”；若要对该日作最优性判断，仍需相同边界条件下的反事实费用比较',
    )
    html = html.replace(
        '（取 max 保证规划净需求不低于点预测）。',
        '（取 max 保证规划净需求不低于点预测）。该外层 max 是单边保守保护，不由报童分位数公式唯一推出，应明确列为启发式设计；如篇幅允许，增加去掉 max 的对照。',
    )

    evidence = f'''

<!-- ===================== 三-A：增强论证 ===================== -->
<h2 id="s3x">三-A、核心论证补强：新增实证结果</h2>

<h3>3A.1 预测选型应呈现“指标权衡”，不能只报MAPE</h3>
<p>周持久化的正式期 MAPE 为4.13%，相似日高斯核为4.02%；但周持久化的 RMSE 为244 kW、日电量 MAE 为1,635 kWh，均优于高斯核的277 kW与2,347 kWh。因此，现有证据支持的是“周持久化在极端误差和日电量偏差方面更稳、实现简单”，而不是“它在所有预测指标上最好”。更严格的选型标准应与最终费用一致，例如0.8分位损失、欠预测尾部损失或端到端购电费用。</p>
{figure('fig03_forecast_tradeoffs.png', '图2　负载预测方案的多指标权衡', '高斯核在MAPE上略优，周持久化在RMSE和日电量误差上更稳。正文应同时展示二者，而不是只挑有利指标。')}

<h3>3A.2 名义80%风险曲线实际覆盖76.94%，风险解释必须降一级</h3>
<p>按正式期48,096个10分钟区间检验，实际净负荷不超过风险曲线的比例为<strong>76.94%</strong>，点预测对应比例为50.85%。风险曲线显著改善了欠预测问题，但没有达到名义80%；分月覆盖率约75.1%—79.6%，四阶段约76.7%—77.4%。这说明最近21日经验残差对尾部仍略偏乐观，且“80%”只能称为<strong>设计分位水平</strong>，不能称为实证保证。</p>
<div class="box warn"><span class="bt">推荐正文表述</span>
“由五倍补购价格得到0.8的单时段经济分位，并将其作为日前风险设计。正式期逐区间回测覆盖率为76.94%，低于名义80%，表明有限滚动样本对上尾仍有低估；因此该参数用于解释风险偏好，而不被解释为供电可靠率或严格概率约束。”</div>
{figure('fig04_risk_calibration.png', '图3　80%风险曲线的样本外区间覆盖表现', '覆盖率按正式期区间计算。虚线是名义80%参考线；图表同时给出点预测覆盖，避免把风险修正效果与概率保证混为一谈。')}

<h3>3A.3 LDR经常触及截断边界，反馈项存在但独立价值尚未被证明</h3>
<p>正式期约42.11%的区间满足 R<sub>t</sub>=E<sub>min</sub>，1.19%的区间位于上限；按阶段看，6—12时和18—24时的下限截断率均接近68%。同时，|λa|&gt;1 kWh 的区间占63.77%，说明反馈项并非完全失活，但其量级通常远小于δ造成的阈值平移。结构消融进一步显示，λ=0的“仅截距”版本费用反而比主方案低3,610.97元（0.0258%）。因此，当前证据支持“截断仿射规则提供了可解释的低维反馈结构”，但<strong>不支持把节费单独归因于λ</strong>。</p>
{figure('fig05_controller_interpretability.png', '图4　截断仿射保留规则的实际工作状态', '左图展示阈值处于下限、内部和上限的比例；右图展示日—阶段粒度的反馈项分布。大量下限截断说明控制器常退化为即时补缺。')}

<h3>3A.4 配对回放是支持证据，不是严格控制器因果证明</h3>
<p>同计划、同日初库存配对下，LDR相对β=1累计节费193,363.14元，相对β=0累计节费64,236.94元；但配对按日重置，且不同控制器的日末SOC未被强制一致。故它能隔离一部分计划差异，却没有完整计入库存的跨日机会价值。论文中应写“配对结果支持改善与控制器有关”，不能写“证明改善完全来自控制器”。更严格的版本应对日末库存作价值调整，或从同一2月1日SOC连续回放全部控制器。</p>
{figure('fig06_paired_savings.png', '图5　同计划、同日初库存配对试验的节费路径', '蓝线显示相对β=1的节费较稳定；相对β=0的改善更集中于少数日期。正值表示LDR当日费用更低。')}

<h3>3A.5 有限敏感性显示主结论方向稳定，但21日与8代并非最优参数</h3>
<table>
<tr><th>变体</th><th class="num">正式期总费 / 元</th><th class="num">相对主方案 / 元</th><th>解释</th></tr>
<tr><td>主方案：21日、seed 20250912、8代</td><td class="num">13,978,077.23</td><td class="num">0.00</td><td>预先固定主口径</td></tr>
<tr><td>随机种子 20250911</td><td class="num">13,988,638.78</td><td class="num">+10,561.55</td><td>搜索随机性可影响约0.076%</td></tr>
<tr><td>随机种子 20250913</td><td class="num">13,979,099.29</td><td class="num">+1,022.06</td><td>与主结果接近</td></tr>
<tr><td>14日残差窗</td><td class="num">13,908,465.96</td><td class="num">−69,611.28</td><td>本年度更低；不应再声称21日已充分优化</td></tr>
<tr><td>28日残差窗</td><td class="num">13,974,689.99</td><td class="num">−3,387.24</td><td>与主结果接近</td></tr>
<tr><td>4代 / 12代预算</td><td class="num">13,970,901.94 / 13,967,840.49</td><td class="num">−7,175.29 / −10,236.75</td><td>更强经验优化不必然单调改善真实账单</td></tr>
<tr><td>δ边界±4800 / λ边界±1</td><td class="num">13,979,959.88 / 13,977,041.68</td><td class="num">+1,882.64 / −1,035.55</td><td>边界变化影响较小</td></tr>
<tr><td>仅截距 λ=0</td><td class="num">13,974,466.27</td><td class="num">−3,610.97</td><td>尚不能证明反馈斜率有独立费用增益</td></tr>
</table>
<p>上述试验都使用同一年度，因此属于<strong>事后敏感性</strong>，不是外部年度验证。14日窗更低并不自动意味着应在定稿中改用14日：若在看到全年结果后改主参数，会引入新的模型选择偏差。稳妥写法是保留21日为预设主口径，把14日作为敏感性发现；若确需切换主方案，应重新生成全部结果表，并将选择过程说明为嵌套历史验证或单独训练—验证划分。</p>
{figure('fig07_sensitivity.png', '图6　LDR主方案的有限敏感性与结构消融', '横轴为替代配置相对主方案的费用变化。负值代表该配置在本年度事后回测中更低，不代表独立样本上的必然优势。')}

<div class="box tip"><span class="bt">补强后的核心结论</span>
主方案的物理可行性、信息因果性和真实账单均可验证；相对β=1和β=0的费用改善具有明确数据支持。相对情景价值控制的0.0368%优势应视为近似持平；21日窗口、8代预算与反馈斜率均未获得唯一最优或独立增益证明。这样写不会削弱论文，反而能显示对近似模型、有限样本和启发式求解边界的清醒认识。</div>
'''
    html = replace_once(html, "\n<!-- ===================== 四 ===================== -->", evidence + "\n<!-- ===================== 四 ===================== -->")

    # Place the two remaining result visuals beside the result narrative.
    cost_fig = figure(
        "fig02_strategy_costs.png",
        "图7　四种策略正式期费用与相对差额",
        "左图保留零起轴以诚实展示绝对规模；右图使用聚焦尺度显示小差额。情景价值控制与LDR的差距仅0.514万元，应解释为数值近似持平。",
    )
    html = replace_once(html, '<h3>4.3 结果段的"叙事线"（按此顺序写，评委会跟得很顺）</h3>', cost_fig + '\n\n<h3>4.3 结果段的"叙事线"（按此顺序写，评委会跟得很顺）</h3>')
    key_fig = figure(
        "fig08_key_dates.png",
        "图8　四个指定日期的供需响应与库存保留",
        "供需与库存分列绘制，避免原双轴方案把区间电量与SOC混在同一面板。9月23日两段紧急购电在左图以浅橙色标示。",
    )
    html = replace_once(html, '<!-- ===================== 五 ===================== -->', key_fig + '\n\n<!-- ===================== 五 ===================== -->')

    generated_table = '''<h3>5.1 已生成图表与正文落点</h3>
<table>
<tr><th style="width:10%">图号</th><th style="width:28%">文件</th><th>回答的问题</th><th style="width:18%">建议位置</th></tr>
<tr><td>图1</td><td><code>fig01_workflow.png</code></td><td>日前锁定、日内信息和跨日状态如何形成闭环</td><td>问题分析末尾</td></tr>
<tr><td>图2</td><td><code>fig03_forecast_tradeoffs.png</code></td><td>为什么周持久化与高斯核是不同指标下的权衡</td><td>预测方法选择</td></tr>
<tr><td>图3</td><td><code>fig04_risk_calibration.png</code></td><td>名义80%风险水平在真实区间上的覆盖情况</td><td>风险分位数之后</td></tr>
<tr><td>图4</td><td><code>fig05_controller_interpretability.png</code></td><td>阈值多大比例被截断，反馈项实际有多强</td><td>LDR规则之后</td></tr>
<tr><td>图5</td><td><code>fig06_paired_savings.png</code></td><td>控制器配对节费是否稳定、集中在哪些时期</td><td>配对对照之后</td></tr>
<tr><td>图6</td><td><code>fig07_sensitivity.png</code></td><td>窗口、种子、预算、边界和λ消融如何影响费用</td><td>稳健性与局限</td></tr>
<tr><td>图7</td><td><code>fig02_strategy_costs.png</code></td><td>费用绝对规模与细小差额如何同时诚实展示</td><td>总体结果</td></tr>
<tr><td>图8</td><td><code>fig08_key_dates.png</code></td><td>指定日期的供需、紧急购电、SOC和阈值如何演化</td><td>指定日期结果</td></tr>
</table>
<p class="caption">全部为320 dpi PNG，采用低饱和度蓝、杏、金、鼠尾草绿与淡紫色；背景近白、网格弱化、深灰文字，不依赖颜色作为唯一编码。按用户要求不另行生成PDF版图片。</p>

<h3>5.2 通用制图规范</h3>'''
    pattern = re.compile(r'<h3>5\.1 图清单（编号、用途、规格）</h3>.*?<h3>5\.2 通用制图规范</h3>', re.S)
    html, count = pattern.subn(generated_table, html, count=1)
    if count != 1:
        raise RuntimeError("could not replace the original figure inventory")

    replacement_code = '''<h3>5.3 图表复现与数据表</h3>
<p>增强图由统一脚本 <code>src/reporting/question2_enhanced_evidence_figures.py</code> 生成；原始数据均来自当前正式结果目录，不手工抄数。重新运行：</p>
<pre><code>python -m src.reporting.question2_enhanced_evidence_figures</code></pre>
<p>图像位于 <code>docs/问题二/figures_增强论证/</code>；图表背后的聚合数据位于 <code>docs/问题二/增强论证数据/</code>，包括分月/分阶段覆盖率、控制器阶段摘要、配对月度节费与敏感性汇总。论文中若只保留4—6张图，优先使用图1、图3、图4、图6、图7和图8；预测权衡与配对时间路径可移至附录。</p>

'''
    pattern = re.compile(r'<h3>5\.3 matplotlib 代码骨架（三张核心图，可直接基于现有 CSV 改造）</h3>.*?(?=<div class="box tip"><span class="bt">图的取舍建议</span>)', re.S)
    html, count = pattern.subn(replacement_code, html, count=1)
    if count != 1:
        raise RuntimeError("could not replace the outdated plotting skeleton")
    html = html.replace(
        '正文图控制在 <strong>4～6 张</strong>：F1～F4 必放，F5/F6 视篇幅选一即可。',
        '正文图控制在 <strong>4～6 张</strong>：因果流程、风险覆盖、控制器工作状态、敏感性、费用比较与指定日期优先；预测权衡和配对时间路径可移至附录。',
    )

    appendix_row = '''<tr><td>增强论证图与聚合证据</td><td><code>docs/问题二/figures_增强论证/*.png</code>、<code>docs/问题二/增强论证数据/*</code></td></tr>
<tr><td>增强图生成与敏感性脚本</td><td><code>src/reporting/question2_enhanced_evidence_figures.py</code>、<code>question2_sensitivity_evidence.py</code></td></tr>
'''
    html = replace_once(html, '<tr><td>主程序（复现）</td>', appendix_row + '<tr><td>主程序（复现）</td>')
    html = html.replace(
        '本指导文档基于本队问题二实际实施数据撰写（2026 高教社杯 C 题）',
        '本增强稿基于本队问题二实际实施数据及新增诊断结果撰写（2026 高教社杯 C 题）',
    )
    html = html.replace(
        '文中所有费用、电量与核验数字均引自重跑后的现有结果文件；',
        '文中主费用、电量与核验数字引自现有正式结果；新增覆盖率、截断比例、配对路径与敏感性数字由可复现脚本重新计算；',
    )
    html = html.replace(
        '文献条目经公开来源核实，正式投稿前请按期刊/组委会格式规范再次核对，尤其第 9 条作者名单。',
        '正式投稿前仍需按期刊/组委会格式复核全部文献，尤其补全第9条作者名单。',
    )

    OUTPUT.write_text(html, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
