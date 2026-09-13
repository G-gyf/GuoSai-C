from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper"
OUT_DOCX = OUT_DIR / "微网与外部电网电力调控策略_数学建模论文.docx"
IMAGE_MD = OUT_DIR / "图片需求.md"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="D9D9D9", size="6") -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        el = borders.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), size)
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_font(run, east="宋体", west="Times New Roman", size=11, bold=None, italic=None) -> None:
    run.font.name = west
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), west)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), west)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_field(paragraph, instruction: str, placeholder: str = "") -> None:
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = placeholder
    fld_char3 = OxmlElement("w:fldChar")
    fld_char3.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr, fld_char2, text, fld_char3])


def add_text(doc, text: str, *, indent=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=3.0, before=0.0):
    p = doc.add_paragraph(style="正文")
    p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.38
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    r = p.add_run(text)
    set_font(r)
    return p


def add_bullet(doc, text: str) -> None:
    p = doc.add_paragraph(style="列表")
    p.paragraph_format.left_indent = Cm(0.74)
    p.paragraph_format.first_line_indent = Cm(-0.37)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.line_spacing = 1.18
    r = p.add_run("• ")
    set_font(r, east="宋体", size=10.5)
    r = p.add_run(text)
    set_font(r)


def add_equation(doc, formula: str, number: int | None = None) -> None:
    eq_dir = OUT_DIR / ".eq"
    eq_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eq_{number:02d}" if number is not None else f"eq_{len(list(eq_dir.glob('eq_*.png'))) + 1:02d}"
    img_path = eq_dir / f"{stem}.png"
    fig = plt.figure(figsize=(0.4, 0.25), dpi=300)
    fig.patch.set_alpha(0)
    text = fig.text(0.02, 0.5, f"${formula}$", fontsize=11.5, color="black", va="center")
    fig.canvas.draw()
    bbox = text.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.04, 1.20)
    width_in = max(bbox.width / fig.dpi, 0.35)
    height_in = max(bbox.height / fig.dpi, 0.20)
    fig.set_size_inches(width_in, height_in)
    text.set_position((0.02, 0.5))
    fig.savefig(img_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    p = doc.add_paragraph(style="公式")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run()
    with Image.open(img_path) as image:
        aspect = image.width / image.height
    target_height = Cm(0.62)
    if aspect * 0.62 <= 13.2:
        r.add_picture(str(img_path), height=target_height)
    else:
        r.add_picture(str(img_path), width=Cm(13.2))
    if number is not None:
        n = p.add_run(f"  （{number}）")
        set_font(n, east="宋体", size=9.5)


def add_placeholder(doc, fig_no: int, caption: str) -> None:
    p = doc.add_paragraph(style="图占位")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(60)
    p.paragraph_format.space_after = Pt(60)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"[图 {fig_no} 占位符  具体要求见 图片需求.md]")
    set_font(r, east="宋体", size=10, italic=True)
    c = doc.add_paragraph(style="题注")
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.keep_with_next = True
    r = c.add_run(f"图 {fig_no}  {caption}")
    set_font(r, east="宋体", size=9.5)


def add_table(doc, headers, rows, widths=None, caption=None):
    if caption:
        p = doc.add_paragraph(style="题注")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(caption)
        set_font(r, east="宋体", size=9.5)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    set_repeat_table_header(table.rows[0])
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        set_cell_shading(cell, "365F91")
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        if widths:
            cell.width = Cm(widths[j])
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(h))
        set_font(r, east="宋体", size=9, bold=True)
        r.font.color.rgb = RGBColor(255, 255, 255)
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        for j, value in enumerate(row):
            cell = cells[j]
            if widths:
                cell.width = Cm(widths[j])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if i % 2 == 1:
                set_cell_shading(cell, "F3F6FA")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(value))
            set_font(r, east="宋体", size=8.8)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def h1(doc, text: str, page_break=True):
    if page_break and len(doc.paragraphs) > 0:
        doc.add_page_break()
    return doc.add_heading(text, level=1)


def h2(doc, text: str):
    return doc.add_heading(text, level=2)


def h3(doc, text: str):
    return doc.add_heading(text, level=3)


def setup_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(11)

    if "正文" not in styles:
        styles.add_style("正文", WD_STYLE_TYPE.PARAGRAPH)
    body = styles["正文"]
    body.font.name = "Times New Roman"
    body._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    body.font.size = Pt(11)
    body.paragraph_format.line_spacing = 1.38
    body.paragraph_format.first_line_indent = Cm(0.74)
    body.paragraph_format.space_after = Pt(3)

    for name in ("列表", "公式", "图占位", "题注"):
        if name not in styles:
            styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)

    title = styles["Title"]
    title.font.name = "黑体"
    title._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "黑体")
    title.font.size = Pt(22)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)
    title_ppr = title._element.get_or_add_pPr()
    title_border = title_ppr.find(qn("w:pBdr"))
    if title_border is not None:
        title_ppr.remove(title_border)

    for style_name, east, size, bold, before, after in (
        ("Heading 1", "黑体", 15, True, 10, 6),
        ("Heading 2", "黑体", 12.5, True, 7, 4),
        ("Heading 3", "楷体", 11, True, 5, 3),
    ):
        st = styles[style_name]
        st.font.name = "Times New Roman"
        st._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east)
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True


def add_cover(doc: Document) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(70)
    r = p.add_run("2026 年高教社杯全国大学生数学建模竞赛")
    set_font(r, east="黑体", size=15, bold=True)
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(46)
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("微网与外部电网电力调控策略")
    set_font(r, east="黑体", size=22, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("基于风险分位数 滚动优化与截断仿射决策规则")
    set_font(r, east="宋体", size=13)
    for _ in range(5):
        doc.add_paragraph()
    for label in ("参赛队号", "学校名称", "队员姓名"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(14)
        r = p.add_run(f"{label}：________________________")
        set_font(r, east="宋体", size=12)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(40)
    r = p.add_run("2026 年 9 月")
    set_font(r, east="宋体", size=11)
    doc.add_page_break()


def build_document() -> Document:
    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.1)
    sec.left_margin = Cm(2.6)
    sec.right_margin = Cm(2.3)
    sec.header_distance = Cm(1.0)
    sec.footer_distance = Cm(1.0)
    sec.different_first_page_header_footer = True

    footer = sec.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(p, "PAGE", "1")
    sec.first_page_footer.paragraphs[0].text = ""

    add_cover(doc)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("摘  要")
    set_font(r, east="黑体", size=15, bold=True)
    abstract = (
        "针对含光伏和储能的微网购电调控问题，本文在统一十分钟物理时间轴、严格信息边界和跨日储能状态的基础上，建立由确定性线性规划、风险修正日前计划、滚动调整与因果执行组成的分层模型。问题一采用两阶段词典序线性规划，在保持购电费用全局最优的同时最小化储能吞吐量；问题二用负载周持久化、光伏七日均值和二十一日配对残差构造百分之八十分位风险曲线，并以四阶段截断仿射保留阈值控制日内储能；问题三把零时计划、六时和十二时滚动调整纳入分段净结算，通过共同起点消融识别各次预报的经济价值；问题四进一步引入因果电价预测、价格加权分位数、负载光伏电价联合情景及发布执行双账本。"
        "结果表明，问题一购电费由 47921.77 元降至 35245.31 元，节省 26.45%。问题二正式期总费为 13978077.23 元，低于固定保留和贪心控制基线，且残差窗长与随机种子改变时排序保持稳定。问题三 M612 策略总费为 13848036.20 元，十二时更新贡献 262702.49 元，是主要信息价值来源；十八时新预报的纯增量仅 388.21 元。固定电价下十时自建光伏预报通过统计和经济门槛，十四时暂缓。波动电价下，问题四第二问与第三问重算费用分别为 14636473.85 元和 14638588.62 元，十二时更新仍有效而十八时调整转为负收益；十时与十四时新增光伏预报均通过稳健性门槛，单独新增电价更新预报则没有经济必要。"
        "全模型通过供需平衡、储能递推、跨日连续、结算复算、因果篡改、工作簿回读和完美信息下界等检验。各问的主要结论均给出基于已有数据的敏感性或鲁棒性证据，并明确不宣称非凸参数搜索的全局最优。"
    )
    add_text(doc, abstract, indent=True, after=5)
    p = doc.add_paragraph(style="正文")
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("关键词：")
    set_font(r, east="黑体", size=10.5, bold=True)
    r = p.add_run("微网调度；储能优化；风险分位数；滚动优化；线性决策规则；波动电价")
    set_font(r)

    doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("目  录")
    set_font(r, east="黑体", size=15, bold=True)
    p = doc.add_paragraph()
    add_field(p, 'TOC \\o "1-3" \\h \\z \\u', "目录将在打开文档时更新")

    h1(doc, "一 问题重述与总体分析")
    h2(doc, "1.1 问题背景与研究目标")
    add_text(doc, "微网内部的光伏出力、居民负荷和储能状态具有显著的时序耦合。计划购电过少会在实际执行时触发五倍电价的紧急购电，计划购电过多又会形成已付费但无法利用的供能；储能能够把低价或光伏富余电量转移到高价缺口时段，但其容量、功率和效率限制使这种转移并非无限。题目四问依次放松确定性假设，从单日已知信息扩展到全年预测误差、日内预报更新和波动电价，因此模型必须同时处理经济性、可执行性与信息非预见性。")
    add_text(doc, "本文的总体思路是把每次可获得的信息转化为风险修正后的净负荷曲线，再用线性规划生成计划或调整购电量，最后以不使用未来信息的储能规则在真实数据上连续执行。所有方案均按实际结算公式回放，不以规划目标值代替真实账单。")
    h2(doc, "1.2 四问递进关系")
    add_text(doc, "问题一是确定性单日基准。负荷、价格和光伏预测已知，核心是证明储在日初日末库存一致时，储能能否通过跨时段套利降低成本。问题二引入预测误差，决策从一次性优化变为日前计划与日内因果执行的组合。问题三增加多次光伏预报和调整费用，需要识别新信息、当前状态和重优化机会各自的价值。问题四把固定电价换成实际波动价格，风险曲线和控制参数都必须反映高价缺口的非对称后果，并解决计划发布区间与执行结算区间错位。")
    h2(doc, "1.3 关键难点")
    add_bullet(doc, "时刻语义：负荷和光伏记录对应区间终点，电价对应区间起点，官方模板又存在一列错位，必须先建立物理时间轴再回填。")
    add_bullet(doc, "非预见性：日初计划不能使用当天未来实测值，滚动调整也只能改变尚未执行的区间。")
    add_bullet(doc, "跨日状态：问题二至四的日末储电量必须传递到下一日，不能每日重置。")
    add_bullet(doc, "价值分解：预测精度改善不等于经济收益，需把状态重优化价值与预报纯增量分开，并控制共同初始库存和期末库存价值。")
    add_bullet(doc, "波动电价：紧急购电惩罚与当期价格同比变化，普通分位数会低估高价缺口的重要性。")
    h2(doc, "1.4 比较口径与评价指标")
    add_text(doc, "单日问题以购电费、总购电量、储能吞吐和初末库存为主指标；全年问题以正式期现金账单为最终评价，同时报告计划费、调整费、紧急费、紧急电量和未利用供能。对需要识别信息价值的方案，采用同一历史路径和同一时点 SOC 分叉，比较基准、状态重优化、新预报与完美信息四类分支。这样可以避免把库存起点、重优化机会和新预报混为同一收益。")
    add_text(doc, "敏感性分析遵循先固定主方案、后报告扰动结果的顺序。参数扰动若只改变绝对费用而不改变策略排序，则认为结论层面稳健；若跨过预设 0.5% 年费阈值或出现排序翻转，则明确披露。新增预报时点还要求同时通过预测精度、块自助置信区间、分月稳定性、随机种子、经济实质门槛、完美信息捕获率和安全性门槛。")
    add_placeholder(doc, 1, "四问统一的计划 调整 执行与检验技术路线")

    h1(doc, "二 模型假设")
    assumptions = [
        "调度步长为十分钟，每日一百四十四个区间；区间内功率视为常数，功率乘以六分之一小时得到电量。",
        "储能额定容量为 12000 kWh，安全运行区间为 1200 至 10800 kWh；最大充放电功率均为 5000 kW，故每区间上限为 833.3333 kWh。",
        "充电效率和放电效率分别取 0.9，往返效率为 0.81；同一时段不允许同时充电与放电。",
        "微网不向外网售电；光伏超过负荷和可充电能力的部分记为未利用供能，不获得收益。",
        "问题一日初和日末储电量均为 6000 kWh；问题二至四储能按真实时间跨日连续。",
        "紧急购电能够补足任意剩余缺口，价格为当期正常电价的五倍，因此执行层始终可行。",
        "问题三和问题四的调整购电采用分段净结算：保留部分按正常价，下调部分支付半价违约费，上调部分按一点五倍价格购买。",
        "实际负荷、光伏和电价只在其对应区间已经发生后用于结算、状态更新或后续训练，不进入之前的决策。",
        "异常尖峰可能是真实运行状态，预处理只标记而不自动删除、平滑或缩尾；夜间光伏零值视为有效观测。",
        "差分进化只用于低维控制参数搜索，模型结论限定为给定参数边界、搜索预算和比较集合下的最优或较优方案。",
    ]
    for i, item in enumerate(assumptions, 1):
        add_text(doc, f"假设 {i}  {item}", indent=False)

    h1(doc, "三 符号说明")
    add_text(doc, "各问共用下列符号。带上标零的量表示日初计划，带上标 A 的量表示滚动调整后最终生效值；上标 ω 表示历史残景。所有能量流变量均以 kWh 每十分钟计。")
    add_table(
        doc,
        ["符号", "含义", "单位或范围"],
        [
            ("t d k", "区间 日期与六小时阶段索引", "t=0…143 k=1…4"),
            ("L P N", "负荷 光伏与净负荷电量", "kWh 每十分钟"),
            ("c", "区间起点电价", "元每 kWh"),
            ("G C D U B", "计划购电 充电 放电 未利用供能与紧急购电", "kWh 每十分钟"),
            ("E", "区间起点储电量", "1200…10800 kWh"),
            ("ηc ηd", "充电与放电效率", "均为 0.9"),
            ("R Qα", "风险修正净负荷与经验分位数", "kWh 每十分钟"),
            ("δ λ a", "保留阈值平移系数 反馈系数与已实现误差信号", "模型参数"),
            ("q0 qA", "原计划与最终调整后购电量", "kWh 每十分钟"),
            ("ν", "期末库存价值系数", "元每 kWh"),
            ("V UB", "信息价值与完美信息上界", "元"),
        ],
        widths=[2.6, 8.5, 4.2],
        caption="表 1 主要符号及其含义",
    )

    h1(doc, "四 数据预处理与前置分析")
    h2(doc, "4.1 数据源与统一时间轴")
    add_text(doc, "附件一给出问题一的代表日价格、负荷和光伏预测；附件二给出 2025 年实际负荷与光伏；附件三给出每日零时、六时、十二时和十八时发布的未来二十四小时整点光伏预报；附件四给出十分钟波动电价。原始宽表被转换为以时间戳为主键的长表，并同时保留 interval start、interval end、observation timestamp、issue timestamp 和模板行列映射。")
    add_equation(doc, r"\Delta t=\frac{1}{6}\,\mathrm{h},\qquad X_{d,t}^{\mathrm{kWh}}=X_{d,t}^{\mathrm{kW}}\Delta t", 1)
    add_text(doc, "物理区间 t 对应当天第 t 个十分钟区间。负荷和光伏取区间终点记录，价格取区间起点记录。该映射避免把下一时段价格错配到当前供需平衡，并使问题一最后一个区间与次日零时状态连续。")
    h2(doc, "4.2 缺失 异常与单位处理")
    add_text(doc, "数据质量报告对，一百四十四个日内区间在每个日期均完整，主调度表共有 52560 行，正式评价期共有 48096 个区间。附件三日期列仅在每天首行出现，读取后按日向下填充，再与发布时间联合构成主键。数值字段无非法空值或无穷值。对极端负荷、光伏和价格只设置异常标记，不删除也不缩尾，以免改变真实缺口与结算费用。")
    h2(doc, "4.3 光伏预报十分钟化")
    add_text(doc, "附件三的预报 h 小时定义为发布时间后 h 小时。每次发布时，以当刻已经观测到的光伏和未来整点预报为锚点，采用 PCHIP 保形插值生成十分钟序列。PCHIP 保留锚点并避免普通高阶样条的过冲，插值后执行非负和夜间归零约束。每次新预报只覆盖发布时间之后尚未执行的区间。")
    add_equation(doc, r"\widehat P(t)=\mathrm{PCHIP}\!\left\{(\tau,P^{\mathrm{actual}}(\tau)),(\tau+h,\widehat P_h),\ h=1,\ldots,24\right\}", 2)
    h2(doc, "4.4 因果预测与冷热启动")
    add_text(doc, "2025 年 1 月 1 日在问题二和问题三中作为冻结初始条件日，不安排购电且储能维持 6000 kWh；首个计划日为 1 月 2 日。历史不足七日时，负荷退化为昨日持久化，光伏使用已有历史日均值；2 月 1 日以后进入正式评价。问题四因发布执行双账本需要，将 1 月 1 日定义为零计划真实运行日，其费用不进入正式期。该差异只影响预热状态，问题二三与问题四的费用不作跨口径直接比较。")
    h2(doc, "4.5 前置质量检验")
    add_text(doc, "共执行三十项自动检查，覆盖源表尺寸、主键唯一、十分钟连续性、日电量完整性、价格起点语义、负荷光伏终点语义、模板映射、功率电量换算、预报目标时刻和训练截止时间，结果全部通过。进一步以篡改未来数据的方式验证因果性：未来实测值被改写后，已发布预报和对应计划保持不变。")
    h2(doc, "4.6 前置规律与建模启示")
    add_text(doc, "光伏预报误差随预测步长扩大：一小时步长的 MAE 为 18.69 kW，十二小时为 159.71 kW，二十四小时达到 374.79 kW；长时域同时出现负偏差，说明仅使用点预测会系统性低估部分缺口。六时和十二时发布后，针对同一目标区间的误差显著下降，而十八时覆盖的大部分有效光伏已经接近零，这为问题三的滚动调整时点提供了数据依据。")
    add_text(doc, "波动电价存在明显的日内和周内结构：早晚高价与居民负荷峰值部分重合，周持久化能够保持日内形状与周周期。价格与净负荷并非独立，高净负荷叠加高价格时，缺口单位损失更大。因此问题四的情景必须按历史日期联合抽取，并在风险曲线中引入价格权重；仅预测平均净负荷或独立重采样会低估尾部费用。")
    add_placeholder(doc, 2, "统一时间轴 数据映射与信息可得性示意")

    h1(doc, "五 问题一模型建立与求解")
    h2(doc, "5.1 确定性调度模型")
    add_text(doc, "问题一的决策变量为外网购电量 Gt、充电量 Ct、放电量 Dt、未利用供能 Ut 与时段起点储电量 Et。供需平衡把外网、光伏和储能放电作为供给，把负荷、充电和未利用供能作为去向。")
    add_equation(doc, r"G_t+P_t+D_t=L_t+C_t+U_t,\qquad t=0,\ldots,143", 3)
    add_equation(doc, r"E_{t+1}=E_t+\eta_c C_t-\frac{D_t}{\eta_d}", 4)
    add_equation(doc, r"1200\leq E_t\leq10800,\qquad 0\leq C_t,D_t\leq833.3333", 5)
    add_equation(doc, r"E_0=E_{144}=6000,\qquad G_t\geq0,\qquad 0\leq U_t\leq P_t", 6)
    add_text(doc, "题面没有给出并网点或储能变流器的爬坡参数，因此主模型不设置硬爬坡约束。储能效率小于一时，同时充放电只会增加能量损耗，但在线性规划的退化最优解中仍可能出现数值意义上的无效循环，故采用词典序目标消除。")
    h2(doc, "5.2 两阶段词典序目标")
    add_equation(doc, r"C_1^*=\min\sum_{t=0}^{143}c_tG_t", 7)
    add_equation(doc, r"\min\sum_t(C_t+D_t),\qquad \mathrm{s.t.}\ \sum_tc_tG_t\leq C_1^*+\varepsilon_{\mathrm{num}}", 8)
    add_text(doc, "第一阶段以购电费为唯一目标，第二阶段将成本固定在第一阶段最优值的 3.52×10 的负六次方元容差内，再最小化充放电总吞吐量。两阶段均为连续线性规划，采用 HiGHS 内点与对偶单纯形混合实现获得全局最优解。")
    h2(doc, "5.3 求解结果与解读")
    add_table(
        doc,
        ["指标", "无储能基准", "两阶段优化", "变化"],
        [
            ("购电量 kWh", "61789.94", "59352.24", "减少 2437.69"),
            ("购电费 元", "47921.77", "35245.31", "节省 12676.47"),
            ("相对费用", "100%", "73.55%", "下降 26.45%"),
            ("充电 放电量 kWh", "0 0", "20054.04 16243.78", "跨时段转移"),
            ("SOC 范围 kWh", "6000", "1200 至 10800", "满足边界"),
        ],
        widths=[4.1, 3.6, 3.6, 4.0],
        caption="表 2 问题一主要结果",
    )
    add_text(doc, "优化后电费下降幅度大于购电量下降幅度，说明主要收益来自把购电从高价时段转移到低价时段，而不是单纯减少总能量。总充电量高于总放电量，差额与 0.81 往返效率造成的损耗一致；初末库存相同，因此不存在消耗期末库存换取表面节费。储能在低价和光伏富余时段充电，在晚高峰或高价缺口时放电，零弃光表明容量在该代表日足以吸收全部富余光伏。")
    h2(doc, "5.4 对偶机制与经济含义")
    add_text(doc, "线性规划的功率平衡对偶变量可解释为在某区间增加一单位净需求对最优成本的边际影响，SOC 递推对偶变量则表示储存在电池中的一单位能量对未来的边际价值。若当前电价低于考虑充电效率后的水价值，模型倾向于充电；若当前电价高于考虑放电效率后的水价值，模型倾向放电。SOC 触及上下限时，相应边界对偶变量非零，表示继续扩大容量具有边际收益。")
    add_text(doc, "代表日中，四至八时和十六至二十时的放电量分别为 5697.59 和 5064.26 kWh，均对应较高净负荷或较高价格；零至四时以及十二至十六时合计充电 10453.20 kWh，为后续高价缺口准备库存。该分布与对偶条件一致，也解释了为什么只比较总购电量无法说明储能价值。")
    add_placeholder(doc, 3, "问题一价格 净负荷 购电 储能动作与 SOC 联动结果")
    h2(doc, "5.5 敏感性分析与鲁棒性检验")
    add_text(doc, "将储能净功率相邻十分钟变化上限作为工程扩展，从 500 kW 至 10000 kW 扫描。500 kW 限制下第一阶段成本为 35887.44 元，比主模型增加 1.82%；1000、2000 和 4000 kW 时增幅依次为 0.83%、0.30% 和 0.04%；达到 5000 kW 后与无爬坡主模型费用相同。该结果说明节费结论对宽松爬坡限制稳健，严格爬坡会以不超过约 1.9% 的成本换取更平滑动作。由于题面未给真实爬坡能力，扩展结果不回写主模型。")
    h2(doc, "5.6 模型检验与评价")
    add_text(doc, "逐区间供需平衡最大残差为 1.14×10 的负十三次方 kWh，SOC 递推最大残差为 9.09×10 的负十三次方 kWh；初末 SOC 均为 6000 kWh，储能边界无越界，同时充放电时段为零，费用独立复算误差为零。模型结构清晰且可获得全局最优解，适合作为后三问的物理核心；局限在于不包含电池寿命、需求响应和并网功率硬约束，实际部署时需用设备参数补充。")

    h1(doc, "六 问题二模型建立与求解")
    h2(doc, "6.1 日前预测与配对残差情景")
    add_text(doc, "每日零时以周持久化预测负荷，以最近七日同区间均值预测光伏。负荷选择来自同一数据上的方案对比；周持久化既保持日型，又不使用未来信息。预测误差按同一历史日配对，使负荷误差与光伏误差的相关关系不被独立抽样破坏。")
    add_equation(doc, r"\widehat L_{d,t}=L_{d-7,t},\qquad \widehat P_{d,t}=\frac{1}{7}\sum_{j=1}^{7}P_{d-j,t}", 9)
    add_equation(doc, r"N_{d,t}^{\omega}=\widehat L_{d,t}-\widehat P_{d,t}+e_{L,t}^{\omega}-e_{P,t}^{\omega}", 10)
    add_text(doc, "使用最近二十一个完整历史日构造等权场景。单时段忽略储能时，计划量 g 的边际过量成本为 c，缺量后紧急购电的边际成本为 5c，报童临界分位数为 1-c/(5c)=0.8。因此风险水平由结算规则先验确定，而不是用全年结果反向选参。")
    add_equation(doc, r"R_{d,t}=\max\!\left\{\widehat N_{d,t},Q_{0.8}\!\left(N_{d,t}^{\omega}\right)\right\}", 11)
    h2(doc, "6.2 风险修正日前计划 LP")
    add_text(doc, "把风险修正净负荷 R 代入与问题一同构的储能线性规划，求得计划购电 G0 与参考库存轨迹 Ep。日末设置负的库存价值项，近似表示留存电量对次日最低价购电的替代价值，防止单日截断模型在午夜前耗尽电池。库存价值只用于决策边界，不进入实际账单。")
    add_equation(doc, r"\min\ \sum_tc_tG_{d,t}^{0}-\nu_dE_{d,144}+\varepsilon\sum_t(C_t+D_t)", 12)
    add_equation(doc, r"\nu_d=\frac{\min_t c_{d+1,t}}{\eta_d}", 13)
    h2(doc, "6.3 四阶段截断仿射保留规则")
    add_text(doc, "日内执行将全天划分为零至六时、六至十二时、十二至十八时和十八至二十四时四阶段。阶段开始时只能观测已经完成区间的净负荷预测误差，取其均值 ak 作为信号。储能保留阈值由参考轨迹、阶段平移量 δk 和误差反馈 λk ak 组成，并截断到安全范围。")
    add_equation(doc, r"a_k=\frac{1}{n_k}\sum_{t<t_k}\left(N_t^{\mathrm{actual}}-\widehat N_t\right)", 14)
    add_equation(doc, r"\theta_{k,t}=\operatorname{clip}\!\left(E_{t}^{p}+\delta_k+\lambda_ka_k,1200,10800\right)", 15)
    add_text(doc, "实际净负荷为负时优先充电，净负荷为正时只允许把库存从当前水平放到保留阈值，剩余缺口由紧急购电补足。该规则把未来信息隔离在日前计划之外，且紧急购电保证任何误差下可行。δ 与 λ 每日用最近二十一日场景和固定边界搜索，差分进化候选若不优于零参数则回退。")
    h3(doc, "6.3.1 非预见性实现")
    add_text(doc, "为避免场景模型在每个历史路径上自由选择不同的储能动作，控制参数在同一决策时点对所有场景共享。每个场景只能通过已经实现的误差信号改变阈值，不能直接读取未来场景编号。原规则含正部和截断运算，若写入一体化模型可用二元变量精确线性化为 MILP；考虑全年日滚动规模，主方案采用规则外层搜索加线性规划内层求解，完整 MILP 仅作结构验证。")
    h3(doc, "6.3.2 日内执行顺序")
    add_text(doc, "每个区间先叠加已承诺计划购电与实际光伏，再与实际负荷比较。若形成富余，则在充电功率与 SOC 上限内充电，其余记为未利用供能；若形成缺口，则在放电功率、效率和保留阈值共同限制下放电，仍不足部分才紧急购电。该顺序保证同一区间不会一边紧急购电一边充电，也避免已付费计划量被误记为可退款弃光。")
    h2(doc, "6.4 求解结果与解读")
    add_table(
        doc,
        ["策略", "计划费 元", "紧急费 元", "总费 元", "紧急电量 kWh"],
        [
            ("截断仿射 LDR", "13332849.50", "645227.73", "13978077.23", "129705.32"),
            ("固定保留 β=1", "13326707.15", "838479.81", "14165186.96", "187104.90"),
            ("贪心放电 β=0", "13333371.39", "709464.67", "14042836.06", "123759.74"),
            ("情景价值控制", "12983839.70", "999128.15", "13982967.84", "192142.96"),
        ],
        widths=[3.6, 3.2, 3.0, 3.2, 3.2],
        caption="表 3 问题二正式期策略比较",
    )
    add_text(doc, "LDR 相对 β=1 节省 187109.73 元，相对 β=0 节省 64758.83 元；共同期初 SOC 回放后排序不变。LDR 的紧急电量略高于 β=0，但紧急费用更低，说明控制器不是追求最少补购量，而是把电池保留给更昂贵的缺口时段。正式期 334 天中 136 天发生紧急购电，198 天没有紧急购电，共 1556 个区间触发补购。")
    add_text(doc, "四个指定日期揭示了不同运行机制。三月二十日、六月二十一日和九月二十三日中午均有负净负荷，储能吸收部分光伏；六月二十一日未利用供能约 17540.72 kWh，表明零紧急购电并不等价于资源完全利用。十二月二十一日净负荷全天为正，储能主要执行低价充电和高价放电。")
    add_text(doc, "配对回放进一步说明费用与电量指标可能方向不同。LDR 相对 β=0 多补购 5945.58 kWh 紧急电量，却少支出约 64236.94 元紧急费用，加权紧急单价明显下降。模型把储能留给晚间或高价缺口，同时容忍低价小缺口，符合以费用最小而非以补购量最小为目标的设定。")
    add_placeholder(doc, 4, "问题二风险修正 日内执行与月度费用分解")
    h2(doc, "6.5 敏感性分析与稳健性检验")
    add_text(doc, "残差窗口取十四、二十一和二十八日，每个窗口用三个随机种子重复。九个格子均保持 LDR 小于 β=0 小于 β=1，未发生排序翻转；同一窗口下种子极差不超过 6923.26 元，即年费的 0.05%。窗口改变使 LDR 总费最大相差 79488.34 元，约 0.57%，略高于预设 0.5% 稳健阈值。主方案仍保留预先登记的二十一日窗口，不根据事后最低费用回调。")
    add_text(doc, "消融试验把反馈系数 λ 固定为零，只搜索四阶段平移量 δ。独立全年账单为 13974466.27 元，比完整规则低 3610.97 元，差异小于随机种子噪声带；λ 仅增加约 1372 元的样本内经验改善，没有迁移为可测的真实账单收益。因此主方案保留完整规则族，但将阶段平移解释为主要收益来源。80% 风险曲线的实际覆盖率为 76.94%，低于名义水平约三个百分点；95.24% 的紧急购电区间发生在实际净负荷越过风险曲线时，验证了风险曲线与补购机制的一致性，也暴露了轻微乐观偏差。")
    h2(doc, "6.6 模型检验与评价")
    add_text(doc, "正式期最大供需平衡误差为 4.55×10 的负十三次方 kWh，SOC 递推和跨日连续误差均为零；SOC 始终位于 1200 至 10800 kWh，最大充放电量均不超过 833.3333 kWh，无同时充放电和边充电边紧急购电。337 个具备完整历史窗的日期均接受不劣于保底候选的搜索结果。模型兼顾因果性与计算可行性，但七维非凸校准不保证全局最优，且分位覆盖偏差提示后续可引入分月或日型校准。")

    h1(doc, "七 问题三模型建立与求解")
    h2(doc, "7.1 多阶段滚动调整与结算")
    add_text(doc, "零时先用附件三零时发布的 PCHIP 光伏预报生成原计划 q0，六时和十二时加入新预报并重解尚未执行的区间。若最终生效计划为 qA，则保留、下调和上调三部分分别按 c、0.5c 和 1.5c 结算，剩余缺口按 5c 紧急购电。")
    add_equation(doc, r"C_t^A=c_t\min(q_t^0,q_t^A)+0.5c_t(q_t^0-q_t^A)^++1.5c_t(q_t^A-q_t^0)^++5c_tB_t", 16)
    add_text(doc, "双面报童边际解释给出下调区九十分位和上调区七十分位：取消一单位计划的边际代价为 0.5c，对应 1-0.5/5=0.9；新增一单位计划的边际代价为 1.5c，对应 1-1.5/5=0.7。调整风险曲线取 Q70、Q90 与原计划 q0 的中位数，并以点预测为下限。")
    add_equation(doc, r"R_t^A=\max\!\left\{\widehat N_t^A,\operatorname{median}(Q_{0.7},Q_{0.9},q_t^0)\right\}", 17)
    h2(doc, "7.2 滚动求解和共同起点")
    add_text(doc, "每次调整固定已执行区间和当前真实 SOC，只优化未来区间。六时锁定六至十二时生效量，十二时锁定十二至二十四时生效量；十八时主策略不再调整购电，只将最新观测误差代入已锁定的 LDR。为公平比较，1 月由 M0 策略共同预热一次，五个策略在 2 月 1 日以 2148.5983 kWh 的相同库存分叉。")
    h2(doc, "7.3 官方预报时点的经济价值")
    add_table(
        doc,
        ["策略", "更新时间", "总费 元", "紧急费 元", "紧急电量 kWh"],
        [
            ("M0", "0 时", "14115380.10", "567113.77", "117280.98"),
            ("M6", "0 6 时", "14110738.69", "481410.16", "94299.74"),
            ("M612 主策略", "0 6 12 时", "13848036.20", "429263.29", "85542.48"),
            ("M61218 S", "加 18 时状态重优化", "13810955.96", "375233.50", "77403.03"),
            ("M61218 F", "加 18 时新预报", "13810567.76", "374666.87", "77320.01"),
        ],
        widths=[3.4, 4.8, 3.2, 3.0, 3.2],
        caption="表 4 问题三五种策略的正式期结果",
    )
    add_text(doc, "六时更新相对 M0 节省 4641.41 元，十二时更新进一步节省 262702.49 元，是主要信息价值来源。十八时仅利用当前 SOC 重优化可节省 37080.24 元，而把十二时预报替换为十八时新预报只再节省 388.21 元，约为总费的 0.003%。因此主策略选择 M612：使用六时和十二时预报，不把十八时新预报纳入购电调整。")
    add_text(doc, "M612 的总费比问题二 LDR 低约 130041 元，但这一差额同时包含预测输入、调整结算和共同预热状态的变化，只能说明多次预报在相应模型内有效，不能作单一因果归因。与相同期初库存的完美预见下界 12252452.93 元相比仍高 1595583.27 元，说明预测误差和近视滚动均留下改进空间。")
    h3(doc, "7.3.1 指定日期的调整特征")
    add_text(doc, "三月二十日和九月二十三日的十二时预报使计划量明显下调，分别降低白天光伏高估或低估造成的敞口；六月二十一日光伏充足，调整后仍无紧急购电，主要收益来自减少已付费未利用供能；十二月二十一日负荷高且光伏弱，上调与储能保留共同抑制五倍价格补购。四日均满足已执行区间不改写，说明滚动模型的经济改善来自未来区间而非追溯修改。")
    add_placeholder(doc, 5, "问题三五策略费用与预报时点价值分解")
    h2(doc, "7.4 分位数敏感性与求解稳健性")
    add_text(doc, "将调整上调侧分位数从 0.5、0.7、0.8 扫描到 0.9。M612 总费依次为 13884950.38、13848036.20、13873621.11 和 14073248.67 元，0.7 最低但并非事后选取，而是由结算边际推导。四档下 M612 相对 M6 的十二时更新价值均为正，核心结论不随分位改变。差分进化搜索以 β=1 和 β=0 双保底，13 次正常收敛，其余达到预设迭代上限但均返回可行且不劣于保底的解；因此以全年账单和物理校验评价，而不以终止文字单独判断失败。")
    h2(doc, "7.5 是否增加其他整数时点预报")
    add_text(doc, "在七至十一时和十三至十七时之间，构造最新官方预报加当日动态残差修正的自建光伏预报。候选时点 τ 的当前误差已经可观测，以最近二十一个完整日的岭收缩回归预测未来误差；样本不足、系数不稳定或滚动验证无改善时回退到官方预报。")
    add_equation(doc, r"\widehat e_{d,h}=\mu_{\tau,h}+\widetilde\rho_{\tau,h}\left(e_{d,\tau}-\mu_{\tau,\tau}\right)", 18)
    add_equation(doc, r"\widetilde P_{d,h}=\max\!\left\{0,\widehat P_{d,h}^{\,r}-\widehat e_{d,h}\right\}", 19)
    add_text(doc, "预测层筛选显示，十时相关系数 0.686，RMSE 和 MAE 分别改善 20.9% 和 21.5%；十四时分别改善 23.2% 和 22.5%。从同一 M612 路径和同一 SOC 分叉 B、S、F、O 四分支后，十时的库存调整后纯预报价值为 10067 元，五个种子均为正，块自助置信区间下界大于零，九个月为正，故建议进入主策略。十四时纯预报价值为 10498 元，但逐日块自助区间下界为负 5.07 元且仅七个月为正，按预先门槛暂缓。")
    add_text(doc, "相邻时点检验中，九时、十一时、十三时和十五时的纯预报价值分别约为 6507、4835、10841 和 5823 元。部分时点精度改善比例较高，却因剩余可操作时长短而经济价值有限。十时与十四时的选择不是单看 RMSE 最优，而是综合相关性、精度、剩余光伏能量和可调窗口后的结果。")
    h2(doc, "7.6 模型检验与评价")
    add_text(doc, "M612 的最大供需平衡误差为 2.27×10 的负十三次方 kWh，状态、跨日连续和结算复算误差均为零；调整前已执行区间未被改写，计划量非负，无同时充放电。十时新增预报对期末价值系数零、半倍、一倍和一点五倍的敏感性均保持正价值。模型能把预报信息价值与状态价值分开，缺点是零时计划没有显式内生未来调整期权，采用滚动 MPC 近似，且新增时点回归只利用本题数据，遇到结构突变时需回退。")

    h1(doc, "八 问题四模型建立与求解")
    h2(doc, "8.1 因果电价预测与联合情景")
    add_text(doc, "问题四在问题二和问题三结构上加入波动电价。日前电价主模型采用周持久化，即用上周同日同区间价格预测；四类日型相似日高斯核作为敏感性对照。2025 年价格具有明显日内双峰和周周期，日均价七日自相关达到 0.963，周持久化在透明性和稳定性之间取得较好平衡。模型在 1 月 8 日至 31 日预评价窗冻结，WP 与 γ=0 的费用只比最优 GK 组合高 0.35%，小于 0.5% 门槛，故按简单性优先规则选择 WP。")
    add_text(doc, "电价预测对比中，周持久化 MAPE 为 7.65%，MAE 为 0.0472 元每 kWh；相似日高斯核的 MAPE 和 MAE 分别为 6.59% 与 0.0409 元每 kWh。虽然 GK 精度更高，预评价窗内的费用优势不足以跨过预设切换门槛，保留 WP 可避免在正式期前增加模型复杂度。该选择体现预测评价最终服从调度目标，同时保留 GK 作为敏感性检验。")
    add_equation(doc, r"\widehat c_{d,t}=c_{d-7,t}", 20)
    add_text(doc, "历史情景从同一日期抽取负荷、光伏和价格误差，并使用四十二日指数衰减权重，避免打散高净负荷与高电价的相关性。二十一日和六十三日窗口仅用于敏感性。")
    add_equation(doc, r"w_j=\frac{\exp(-\ell_j/14)}{\sum_r\exp(-\ell_r/14)}", 21)
    h2(doc, "8.2 价格加权分位数与正则化 LDR")
    add_text(doc, "紧急缺口在高价时段的损失更大，因此将经验分位数中的每个场景权重再乘以对应电价，得到价格加权分位数。它优先覆盖高价缺口，而不是只追求无权覆盖率。")
    add_equation(doc, r"Q_{\alpha}^{\mathrm{pw}}=\inf\left\{x:\sum_jw_jc_j\mathbf{1}(N_j\leq x)\geq\alpha\sum_jw_jc_j\right\}", 22)
    add_text(doc, "问题四第二问使用价格加权 Q80 生成计划；问题四第三问在零时使用 Q80，在调整侧沿用 Q70 与 Q90 双面风险曲线。LDR 校准增加滚动验证日验收，正则强度 γ 在预评价窗冻结。少于十四个历史日不校准，十四至四十一日只调整截距，历史充分后再开放反馈项，以降低早期过拟合。")
    h2(doc, "8.3 发布执行双账本")
    add_text(doc, "附件四与官方结果模板的第一列语义不同。每天零时发布的一百四十四列覆盖从当日零时十分至次日零时十分，而物理执行账本以当日零时至二十四时为界。设 g issue 为发布量、g exec 为执行量，则当天首区间承接前一日末列，其余区间取当天发布行前一列。")
    add_equation(doc, r"g_{d,0}^{\mathrm{exec}}=g_{d-1,143}^{\mathrm{issue}},\qquad g_{d,t}^{\mathrm{exec}}=g_{d,t-1}^{\mathrm{issue}}", 23)
    add_text(doc, "正式期计划费以执行账本为准，并通过 carry in 和 carry out 与发布账本桥接。十二月三十一日末列由当日零时的因果计划生成，而不是机械填零；该列在物理上属于下一年首区间，通过 carry out 移出本年费用。")
    h2(doc, "8.4 波动电价下问题二重算结果")
    add_table(
        doc,
        ["方案", "计划费 元", "紧急费 元", "总费 元", "紧急电量 kWh"],
        [
            ("价格加权 Q80 正则化 LDR", "14178616.41", "457857.44", "14636473.85", "82947.51"),
            ("普通 Q80 β=1", "14084364.48", "912718.45", "14997082.93", "196519.12"),
            ("普通 Q80 β=0", "14092460.02", "755480.85", "14847940.87", "123852.30"),
            ("普通 Q80 原 LDR", "14091829.26", "704026.84", "14795856.10", "130354.42"),
            ("完美预见下界", "—", "—", "12793144.36", "—"),
        ],
        widths=[4.5, 3.0, 3.0, 3.2, 3.2],
        caption="表 5 问题四第二问正式期策略比较",
    )
    add_text(doc, "主方案相对 β=1、β=0 和原 LDR 分别节省 360609.08、211467.02 和 159382.25 元。其计划费略高于普通分位数基线，但紧急费至少减少 246169 元，说明价格加权分位数主动用适量低风险计划成本换取高价缺口保护。与完美预见下界相差 1843329.48 元，比例为 14.41%，该差额是预测和策略联合改进空间，不解释为任何单一误差来源。")
    h2(doc, "8.5 波动电价下问题三重算结果")
    add_table(
        doc,
        ["策略", "常规及调整费 元", "紧急费 元", "总费 元", "信息价值 元"],
        [
            ("M0", "14398170.44", "440414.31", "14838584.75", "基准"),
            ("M6", "14386156.74", "444476.17", "14830632.91", "V6=7951.84"),
            ("M612", "14292663.67", "345924.95", "14638588.62", "V12=192044.29"),
            ("M61218", "14267849.28", "602682.22", "14870531.50", "V18=-231942.89"),
        ],
        widths=[3.0, 4.0, 3.2, 3.5, 3.8],
        caption="表 6 问题四第三问滚动策略比较",
    )
    add_text(doc, "波动电价下 M612 仍最优，零时至十二时滚动调整合计节省 199996.13 元，其中十二时贡献约 96%。十八时调整使总费增加 231942.89 元：常规结算虽下降，但晚间紧急费由 345924.95 元升至 602682.22 元。固定电价下十八时状态重优化为正，而波动电价下转负，说明晚间电价水平和上调成本放大了错误重买的后果，必须按当前价格口径重新判定信息价值。")
    add_placeholder(doc, 6, "波动电价预测 价格加权风险曲线与费用结构")
    h2(doc, "8.6 敏感性分析与稳健性检验")
    add_text(doc, "问题四第二问分别改变电价模型、情景窗、分位数和期末价值。GK 电价模型总费 14573911.43 元，比主方案低 62562.42 元；二十一日窗和六十三日窗总费为 14665408.26 和 14634594.59 元；普通分位数为 14676421.23 元，且紧急费升至 630940.15 元；期末价值取零或同日最低价时总费为 14653372.21 和 14659404.07 元。单维变化均在约 0.5% 内，且普通分位数持续增加紧急费，支持价格加权设计。")
    add_text(doc, "问题四第三问的 GK、二十一日窗和普通分位数变体总费分别为 14589593.76、14662007.66 和 14678764.16 元，M612 优于 M0 与 M6 的结论未改变。发布与执行账本桥接误差为零，正式期执行计划费等于发布行费用 14178759.32 元加 carry in 144.59 元再减 carry out 287.49 元。")
    h2(doc, "8.7 是否新增光伏或电价预报")
    add_text(doc, "对十时和十四时分别构造六个分支：基准 B、仅状态重优化 S、新增光伏预报 Fpv、新增电价更新 Fp、两者同时 Fboth 与完美信息 O。新增电价更新使用候选时点以前已揭示的实际价，对日前 WP 预测进行四十二日滚动 OLS 收缩的日级乘性修正，乘子截断在 0.5 至 1.5。")
    add_text(doc, "十时和十四时新增光伏预报的库存调整后纯价值分别约为 10878 元和 11319 元，五个种子均为正，95% 块自助区间下界大于零，九个月为正，并超过年费 0.05% 即 7319 元的实质门槛；捕获完美信息上界的比例为 19.1% 和 33.9%。两时点组合方案的库存调整成本为 14572679.03 元，相对 M612 节省约 6.6 万元，其中包含状态重优化价值。")
    add_text(doc, "电价更新预报在预测层能改善剩余时域 MAE 2.8% 至 4.0%、RMSE 8.4% 至 9.2%，但单独经济价值为十时负 17 元、十四时负 591 元；叠加光伏预报后的边际价值也接近零。因此建议升级为 M612 加十时和十四时光伏预报，不为调整购电单独新增电价更新预报。")
    h2(doc, "8.8 模型检验与评价")
    add_text(doc, "候选调整 LP 存在储能不动、未满足量全部由辅助变量和紧急购电吸收的平凡可行解，因此按构造恒可行。54 个全年回测共求解 17368 次候选 LP，全部返回最优，失败和保底次数均为零；最大能量平衡误差 4.5×10 的负十三次方 kWh，SOC 递推和跨日误差为零。未来价格或光伏被篡改时，当日候选决策不变；扩展代码的 B 分支与锁定 M612 逐日费用和 SOC 完全一致。模型的主要局限是极端近零价格事件样本少、四十二日窗仍属设计选择，以及问题四的预热口径与问题二三不同。")

    h1(doc, "九 模型综合评价与推广")
    h2(doc, "9.1 模型优势")
    add_text(doc, "第一，四问共享同一能量平衡和储能状态定义，变化只发生在信息集、风险曲线和结算层，因而逻辑递进清楚。第二，模型把规划决策和真实执行账本分离，避免用含未来信息的规划轨迹替代可执行策略。第三，风险分位数直接由紧急购电与调整价格的边际关系推导，减少事后调参。第四，共同起点、配对分支和完美信息下界使不同策略的经济价值具有明确比较基准。第五，价格加权分位数和发布执行双账本解决了波动电价与模板错位两个工程关键问题。")
    add_text(doc, "在计算层面，问题一和各阶段计划均为线性规划，可获得确定的全局最优解；非凸性只集中在低维 LDR 参数搜索，并设置显式保底。数据层、规划层、执行层和结算层分别保存独立中间结果，使论文中的费用、库存与模板数字能够逐层追溯。")
    h2(doc, "9.2 模型局限")
    add_text(doc, "预测模型以透明和因果为优先，没有引入天气、辐照度和外部日历，极端天气下的精度受限。LDR 参数通过差分进化直接搜索，只保证在固定边界与预算内不劣于保底候选。库存价值采用线性近似，不能完全代表跨年机会成本。模型忽略电池衰减、循环寿命、需量电费、售电收益和并网容量，适合竞赛数据下的购电策略比较，不能直接替代现场 EMS。")
    add_text(doc, "此外，问题二的名义 80% 风险曲线实际覆盖为 76.94%，提示经验分位数受样本窗和日型分布影响；问题三零时计划没有把未来调整期权内生到同一多阶段随机规划；问题四对近零电价和结构突变的样本仍有限。这些限制不会推翻现有回测结论，但决定了模型外推时需要滚动再校准和在线监控。")
    h2(doc, "9.3 检验体系")
    add_text(doc, "本文使用九类证据形成闭环：逐区间物理残差检验、费用独立复算、模板回读、残差窗和随机种子网格、结构消融、共同起点回放、分月与块自助统计、完美信息下界以及因果篡改测试。敏感性结果只用于检验，不根据正式期最低费用回调已登记主参数，降低数据窥探风险。")
    add_placeholder(doc, 7, "四问模型检验体系与证据链")
    h2(doc, "9.4 推广价值")
    add_text(doc, "该框架可推广到虚拟电厂、用户侧储能、电动汽车聚合和含不确定需求的能源采购。报童分位数适合缺货成本显著高于过量成本的申报问题；价格加权分位数适合损失随实时价格变化的市场；发布执行双账本适合交易区间与结算日界线错位的系统；B S F O 或六分支价值分解可用于判断购买新气象数据或新增市场信息是否值得。")

    h1(doc, "十 结论")
    add_text(doc, "问题一的两阶段词典序 LP 在代表日将购电费降低 26.45%，并在数值容差内保持全局经济最优。问题二的风险分位数加截断仿射储能规则把正式期总费控制在 13978077.23 元，策略排序对残差窗和随机种子稳定，但费用水平对窗长约有 0.57% 敏感性，且实际覆盖率为 76.94%。")
    add_text(doc, "问题三中，六时预报价值较小，十二时预报贡献 262702.49 元，十八时新预报纯增量仅 388.21 元，因此固定电价主策略为 M612。额外时点分析表明十时自建光伏预报应纳入策略，十四时在固定电价下因置信区间和月份门槛未全部通过而暂缓。")
    add_text(doc, "问题四中，价格加权 Q80 与正则化 LDR 的总费为 14636473.85 元；滚动调整的 M612 总费为 14638588.62 元，十二时更新仍有显著价值，十八时调整因紧急费上升而产生 231942.89 元负价值。波动电价下十时和十四时新增光伏预报均通过经济与统计门槛，而日内电价更新虽提高预测精度，却没有可测经济收益。综合建议是在波动电价场景采用 M612 加十时和十四时光伏更新，并持续监控风险覆盖、极端价格和电池寿命成本。")
    add_placeholder(doc, 8, "四问主策略 成本与信息价值结论汇总")

    h1(doc, "参考文献")
    refs = [
        "[1] Dantzig G B. Linear Programming and Extensions. Princeton University Press, 1963.",
        "[2] Birge J R, Louveaux F. Introduction to Stochastic Programming. Springer, 2011.",
        "[3] Shapiro A, Dentcheva D, Ruszczynski A. Lectures on Stochastic Programming. SIAM, 2014.",
        "[4] Storn R, Price K. Differential Evolution A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. Journal of Global Optimization, 1997, 11: 341-359.",
        "[5] Fritsch F N, Carlson R E. Monotone Piecewise Cubic Interpolation. SIAM Journal on Numerical Analysis, 1980, 17(2): 238-246.",
        "[6] Powell W B. Approximate Dynamic Programming Solving the Curses of Dimensionality. Wiley, 2011.",
        "[7] Boyd S, Vandenberghe L. Convex Optimization. Cambridge University Press, 2004.",
        "[8] Efron B, Tibshirani R J. An Introduction to the Bootstrap. Chapman and Hall, 1993.",
    ]
    for ref in refs:
        add_text(doc, ref, indent=False, align=WD_ALIGN_PARAGRAPH.LEFT, after=2)

    h1(doc, "附录 A 求解流程与复现要点")
    h2(doc, "A.1 统一求解流程")
    steps = [
        "读取附件并依据数据契约建立物理时间轴、发布时刻和模板映射；",
        "使用决策时刻以前的数据生成负荷、光伏和价格预测，建立配对残差情景；",
        "求解日前计划 LP 或滚动调整 LP，并用词典序目标消除无效吞吐；",
        "固定已承诺购电量，以截断仿射保留阈值逐区间执行真实负荷和光伏；",
        "按自然账本或共同起点分支连续传递 SOC，逐区间计算正常、调整和紧急费用；",
        "复算约束残差、结算分项、上下界和模板合计，并执行敏感性与因果测试。",
    ]
    for i, step in enumerate(steps, 1):
        add_text(doc, f"步骤 {i}  {step}", indent=False)
    h2(doc, "A.2 结果口径注意事项")
    add_text(doc, "问题二和问题三的正式期为 2025 年 2 月 1 日至 12 月 31 日，1 月 1 日冻结；问题四为适应双账本，把 1 月 1 日作为零计划真实运行日。固定电价与波动电价、自然账本与共同起点回放、现金账单与库存调整成本分别服务于不同问题，表中已在相应章节明确，不能脱离口径直接相减。")
    add_text(doc, "正文图片均保留占位符，绘制要求与数据来源见同目录 图片需求.md。公式在 Word 中保存为可编辑数学对象；目录和页码字段在打开文档时更新。")

    settings = doc.settings._element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")
    return doc


IMAGE_REQUIREMENTS = """# 图片需求

本文正文已预留 8 个图片占位符。图片应采用中文学术论文风格，白底、黑色坐标轴、低饱和蓝橙配色，字体优先使用思源黑体或微软雅黑，正文可辨字号不低于 8 pt。位图建议 300 至 400 dpi，单栏宽约 8 cm，通栏宽约 15.5 cm；同时保留 PDF 或 SVG 矢量版本。所有数值必须从下列已有结果文件读取，不得手工估算。

## 图 1 四问统一的计划 调整 执行与检验技术路线

- 放置位置：正文 1.3 节后。
- 形式：横向流程图，通栏。
- 内容：数据清洗与时间对齐 → 因果预测与配对残差 → 日前计划 LP → 6 时和 12 时滚动调整 → 截断仿射储能执行 → 紧急购电兜底 → 费用结算 → 敏感性与物理校验；用四条支路标出问题一至四新增的机制。
- 强调：每次决策只使用已获得信息；问题四增加电价预测、价格加权分位数和发布执行双账本。
- 数据依据：README.md、数据预处理与前置分析口径说明.md。

## 图 2 统一时间轴 数据映射与信息可得性示意

- 放置位置：正文 4.5 节后。
- 形式：上下两层时间轴，通栏。
- 内容：上层展示负荷和光伏的区间终点记录、电价的区间起点记录；下层展示附件三 0 时 6 时 12 时 18 时发布、PCHIP 锚点和允许调整的未来区间。
- 必须标注：00:00 至 00:10 使用 00:10 负荷光伏与 00:00 起点价格；已执行区间锁定。
- 数据依据：config/data_contract.json、outputs/preanalysis/quality_checks.csv。

## 图 3 问题一价格 净负荷 购电 储能动作与 SOC 联动结果

- 放置位置：正文 5.3 节后。
- 形式：四联图，共享横轴 0 至 24 时，通栏。
- 面板：价格曲线；负荷 光伏 净负荷；最优购电与无储能购电差；充放电柱与 SOC 折线。
- 标注：SOC 安全带 1200 至 10800 kWh，初末 6000 kWh；总费 35245.31 元、节费 26.45%。
- 数据依据：outputs/question1/question1_schedule.csv、outputs/question1/question1_summary.json。

## 图 4 问题二风险修正 日内执行与月度费用分解

- 放置位置：正文 6.4 节后。
- 形式：三联图，通栏。
- 面板：点预测与 Q80 风险曲线；四阶段保留阈值和 SOC；LDR β等于0 β等于1 的月度计划费与紧急费堆叠柱。
- 强调：LDR 紧急电量可能高于 β等于0，但紧急费用更低，体现对高价缺口的选择性保护。
- 数据依据：outputs/question2/current/ldr/question2_daily.csv、ldr_daily_parameters.csv、outputs/question2/current/paper/tables/monthly_summary.csv。

## 图 5 问题三五策略费用与预报时点价值分解

- 放置位置：正文 7.3 节后。
- 形式：左侧总费瀑布或分组柱，右侧 V6 V12 V18state V18forecast 条形图。
- 必须使用数值：M0 14115380.10、M6 14110738.69、M612 13848036.20、M61218 S 13810955.96、M61218 F 13810567.76 元；V6 4641.41、V12 262702.49、V18state 37080.24、V18forecast 388.21 元。
- 数据依据：outputs/question3/current/question3_comparison.csv、outputs/question3/current/问题三实施与结果说明.md。

## 图 6 波动电价预测 价格加权风险曲线与费用结构

- 放置位置：正文 8.5 节后。
- 形式：三联图，通栏。
- 面板：实际价格与 WP 预测；普通 Q80 与价格加权 Q80 风险曲线对比；Q4 2 与 Q4 3 的计划 调整 紧急费用分解。
- 强调：高价缺口在加权分位数中权重更大；M61218 的紧急费显著上升。
- 数据依据：outputs/question4/analysis/price_forecast/、outputs/question4/v2/result4-2/run_summary.json、outputs/question4/v2/result4-3/run_summary.json。

## 图 7 四问模型检验体系与证据链

- 放置位置：正文 9.3 节后。
- 形式：证据链或矩阵图，通栏。
- 行：物理残差、费用复算、模板回读、窗口种子、结构消融、共同起点、块自助、完美信息下界、因果篡改。
- 列：问题一、问题二、问题三、问题四；用实心点表示已执行，用空心点表示不适用。
- 数据依据：模型优缺点与检验汇总.md 及各问 validation JSON。

## 图 8 四问主策略 成本与信息价值结论汇总

- 放置位置：正文 10 节末。
- 形式：不直接比较绝对费用的结论图。建议用四个并列小面板，分别显示各问自身基线与主方案的相对变化。
- 问题一：节费 26.45%。问题二：LDR 相对 β等于1 节费 187109.73 元。问题三：M612 相对 M0 节费 267343.90 元，突出 V12。问题四：Q4 2 主方案相对 β等于1 节费 360609.08 元；Q4 3 M612 相对 M0 节费 199996.13 元。
- 注意：问题二三与问题四预热及价格口径不同，不得把四问绝对费用画在同一纵轴上作横向排名。

## 统一制图规范

- 不使用三维图、渐变背景、阴影、装饰性图标或过多颜色。
- 数值统一保留两位小数，电量单位 kWh，费用单位元；金额超过一百万元时可在坐标轴标注为百万元，但图内注释给出精确值。
- 图题放在图下，子图标记使用小写英文字母 a b c。
- 每张图需提供源数据脚本或数据文件路径，保证可复算。
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = build_document()
    doc.core_properties.title = "微网与外部电网电力调控策略"
    doc.core_properties.subject = "2026 年高教社杯全国大学生数学建模竞赛 C 题论文"
    doc.core_properties.author = "参赛队"
    doc.core_properties.keywords = "微网 储能 风险分位数 滚动优化 波动电价"
    doc.save(OUT_DOCX)
    IMAGE_MD.write_text(IMAGE_REQUIREMENTS, encoding="utf-8")
    print(OUT_DOCX)
    print(IMAGE_MD)


if __name__ == "__main__":
    main()
