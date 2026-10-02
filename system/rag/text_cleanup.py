r"""
文本规整：修复 PDF 字形错映射 + 把 LaTeX 标记降级为可读纯文本。

背景
----
本项目的人教版 / 人教中图版电子教材 PDF 使用方正系列字体，**数学斜体字母**
以及部分数学符号在 PDF 里没有正确的 ToUnicode 映射，抽取出来会变成生僻汉字
（如 `x` → `狓`、`y` → `狔`）或私用区（PUA）字符。实测：

    狓/狔/狕 等斜体字母      约 3 万处，集中在 5 本数学教材
    私用区符号（∀ ⊂ ⇒ 等）   8264 处
    其它异体文字噪声          4134 处（0.07%）

如果不在入库前修掉，这些字符会一路传到检索、重排、大模型上下文，
最后原样出现在学生的答案里 —— 模型只会照抄，不会「猜」出正确字母。

映射表来源
----------
**不是猜的**：每一条都有教材原文作为判据。最有力的一条来自人教 A 版
数学选择性必修第三册第 7 章原文：

    「通常用大写英文字母表示随机变量，例如犡，犢，犣；
       用小写英文字母表示随机变量的取值，例如狓，狔，狕。」

据此直接锁定 犡=X、犢=Y、犣=Z、狓=x、狔=y、狕=z，再按字母序与公式语义
逐个外推验证（如「设犻，犼，犽是空间中三个两两垂直的向量」→ i/j/k；
「λ，μ∈犚」→ R；「点犖在线段犗犕上」→ N/M）。其余条目同理，
见下方 `PDF_GLYPH_MAP` 每行的行内注释。

用法
----
    from system.rag.text_cleanup import clean_text, fix_pdf_glyphs

    fix_pdf_glyphs("若函数狔＝犳（狓）")     # -> "若函数y＝f（x）"
    clean_text(r"\[ \lim_{\Delta狓 \to 0} \]")  # -> "lim(Δx→0)"

`fix_pdf_glyphs` 在**入库时**调用（保证索引本身干净）；
`strip_latex` 在**答案输出时**调用（GUI 的答案框只支持纯文本，
渲染不了 `\\frac` 这类标记）。
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 一、PDF 字形错映射：字符 → 真实字符
# ═══════════════════════════════════════════════════════════════════
# 判据：每条都取自教材原文（注释里给的是原文片段）。
PDF_GLYPH_MAP: Dict[str, str] = {
    # ---- 数学斜体大写字母 ----
    "犃": "A",   # 「向量犪的起点是犃，终点是犅」
    "犅": "B",
    "犆": "C",   # 「犗犃→，犗犅→，犗犆→」
    "犇": "D",   # 「平行六面体犃犅犆犇-犃′犅′犆′犇′」
    "犈": "E",   # 「在四条射线上分别取点犈，犉，犌，犎」
    "犉": "F",
    "犌": "G",
    "犎": "H",
    "犐": "I",   # 「犐，犑分别是棱犆′犇′，犇′犃′的中点」
    "犑": "J",
    "犓": "K",   # 「犈，犉，犌，犎，犓，犔分别是…各棱的中点」
    "犔": "L",
    "犕": "M",   # 「点犖在线段犗犕上」
    "犖": "N",
    "犗": "O",   # 「过同一个顶点犗」
    "犘": "P",   # 「对于直线犾上任意一点犘」
    "犙": "Q",   # 「使得犙犘→＝狕犽」
    "犚": "R",   # 「λ，μ∈犚」（实数集）
    "犛": "S",   # 「四棱锥犛-犃犅犆犇」
    "犜": "T",   # 「犚，犛，犜是线段犗犉的四等分点」
    "犝": "U",   # 「将集合犝分成一些两两不交的子集」
    "犞": "V",   # 「气球的半径狉与体积犞」「犞犪狉（犡）」= Var(X)
    "犠": "W",   # 「仓库的长犔大于宽犠的４倍」← 决定性判据
    "犡": "X",   # 「例如犡，犢，犣」← 决定性判据
    "犢": "Y",
    "犣": "Z",

    # ---- 数学斜体小写字母 ----
    "犪": "a",   # 「空间向量用字母犪，犫，犮，…表示」
    "犫": "b",
    "犮": "c",
    "犱": "d",   # 「犱＝（犱１，犱２，…）」
    "犲": "e",   # 「犲＝（犲１，犲２，…）」
    "犳": "f",   # 「若函数狔＝犳（狓）」
    "犵": "g",   # 「垂直于α内的任意一条直线犵」
    "犺": "h",   # 「设边犃犅上的高为犺」
    "犻": "i",   # 「设犻，犼，犽是空间中三个两两垂直的向量」
    "犼": "j",
    "犽": "k",
    "犾": "l",   # 「犗是直线犾上一点」
    "犿": "m",   # 「犿，狀是平面α内的两条相交直线」
    "狀": "n",   # ↑ 与上一行同时出现，互为佐证
    # 'o' 未在任何教材中出现（数学里小写 o 极少作为变量），故不设映射，
    # 避免把生僻汉字误判成字母。
    "狆": "p",   # 「任意一个向量狆可以写成狆＝狓犪＋狔犫」
    "狇": "q",   # 「狇＝犪－犫」
    "狉": "r",   # 「⊙犃的圆心犃的坐标为（犪，犫），半径为狉」
    "狊": "s",   # 注音「数（狊ǔ）出」
    "狋": "t",   # 「存在实数狋，使得犃犘→＝狋犪」
    "狌": "u",   # 「设狌１，狌２分别是直线犾１，犾２的方向向量」
    "狏": "v",   # 「若狏是直线犾的方向向量」
    "狑": "w",   # 「工资狑（单位：元）是他工作天数犱的函数」
    "狓": "x",
    "狔": "y",
    "狕": "z",

    # ---- 数学符号 ----
    "槡": "√",   # 「２槡２」= 2√2；「狉（犞）＝３３犞４π槡」= ∛(3V/4π)
    "瓚": "∥",   # 「在四边形犗犆犃犅中，犗犅瓚犆犃，所以犗犆犃犅是平行四边形」
    "狘": "|",   # 「狘犘犉１狘＝狘犘犉２狘＝犪」= |PF₁|=|PF₂|=a

    # ---- 私用区（PUA）里承载的数学符号 ----
    "\ue010": ".",   # 章节号分隔点：「１[.]１　空间向量及其运算」= 1.1
    "\ue011": "-",   # 图号/换行连字符：「图１．１[-]１」= 图1.1-1；「ｕｎｉｔｖｅｃ[-]ｔｏｒ」
    "\ue012": "*",   # 上标星号：「犿，狀∈犖[*]」= N*
    "\ue01b": "⊂",   # 「犪[⊂]β，犫[⊂]β，犪∩犫＝犘」
    "\ue02f": "∀",   # 「并用符号“[∀]”表示」（全称量词）
    "\ue03c": "⇒",   # 「犘（犅）＝犘（犃）[⇒]犘（犃犅）＝犘（犃）犘（犅）」
}

# 上表之外，需要**整字删除**的字符。
PDF_GLYPH_DROP = {
    "烄", "烅", "烆",        # 方程组 / 分段函数的大括号碎片（上、中、下）
    "熿", "燀", "燄", "燅",  # 特大号括号的碎片
}
# 注意：这几个字形**必须逐个列出**，不能写成 U+70C4–70C6 / U+71BF–U+71C5 这样的范围 ——
# 范围内混着常用汉字（如 U+71C3=燃、U+70C8=烈），用范围会把正常正文一起删掉。

# 剩余私用区字符无对应关系（长尾共 24 个码点、约 450 处），
# 留着会在界面上显示成方框，属纯噪声 → 删除。
# 异体文字块同理：中文教材里不会出现西里尔/泰文/卡纳达文，
# 实测这些位置的原文是流程图里字体损坏的标注文字（如
# 「ب᭛ႆP′᭛ႆPᡦ๊ልࡗڍ」，真值应为图注中的中文/英文），整段删除后
# 剩余可读内容（「图3-14 化学沉淀法废水处理工艺流程示意图」）反而更干净。
_NOISE_RE = re.compile(
    "["
    "\ue000-\uf8ff"      # 私用区（未在上表映射的部分）
    "\u0400-\u04ff"      # 西里尔
    "\u0530-\u058f"      # 亚美尼亚
    "\u0590-\u05ff"      # 希伯来
    "\u0600-\u06ff"      # 阿拉伯
    "\u0b80-\u0dff"      # 卡纳达 / 泰卢固 / 僧伽罗等
    "\u0e00-\u0e7f"      # 泰文
    "\u10a0-\u10ff"      # 格鲁吉亚
    "]"
)

# 一次性构造翻译表（str.translate 是 C 实现，逐块调用开销可忽略）
_TRANS = {ord(k): v for k, v in PDF_GLYPH_MAP.items()}
_DROP_TRANS = {ord(k): None for k in PDF_GLYPH_DROP}

# 反斜杠命令 + 待映射字符 的边界（用于插入分隔空格，见 fix_pdf_glyphs）
_CMD_BOUNDARY_RE = re.compile(
    r"(\\(?:[A-Za-z]+)?)([" + re.escape("".join(PDF_GLYPH_MAP)) + r"])"
)

# 换行符在本项目里承担段落语义，规整时保留
_WS_ONLY_LINE_RE = re.compile(r"\n{3,}")


def fix_pdf_glyphs(text: str) -> str:
    """
    修复 PDF 字形错映射。

    Args:
        text: 从 PDF 抽取出的原始文本

    Returns:
        修复后的文本。原始文本中本来就没有问题字符时，返回原文（不做任何改动）。

    说明：先做字符映射，再删除噪声 —— 顺序不能反。
    PUA 里既有需要保留其含义的符号（∀ ⊂ ⇒），也有纯噪声，
    必须先映射掉前者，后者才会被噪声正则捕获。

    边界保护：若目标字符紧跟在 `\\Delta` 这类反斜杠命令之后（模型输出里常见
    `\\Delta狓`），会在命令后补一个空格，得到 `\\Delta x`。否则字符映射会把
    两者粘成 `\\Deltax`，反斜杠命令被拼成一个不存在的命令名。
    正常抽取文本不含反斜杠，该保护不会生效。
    """
    if not text:
        return text
    if not (PDF_GLYPH_MAP.keys() | PDF_GLYPH_DROP) & set(text) and not _NOISE_RE.search(text):
        return text          # 快路径：绝大多数块不含问题字符
    guarded = _CMD_BOUNDARY_RE.sub(r"\1 \2", text)
    out = guarded.translate(_TRANS).translate(_DROP_TRANS)
    out = _NOISE_RE.sub("", out)
    return out


def glyph_stats(text: str) -> Tuple[Dict[str, int], int]:
    """
    统计一段文本里的字形问题，供入库日志与自检使用。

    Returns:
        (映射明细 {原字符: 次数}, 噪声字符总数)
    """
    if not text:
        return {}, 0
    mapped: Dict[str, int] = {}
    for ch in PDF_GLYPH_MAP:
        n = text.count(ch)
        if n:
            mapped[ch] = n
    for ch in PDF_GLYPH_DROP:
        n = text.count(ch)
        if n:
            mapped[ch] = n
    noise = len(_NOISE_RE.findall(text))
    return mapped, noise


def has_glyph_issues(text: str) -> bool:
    """文本是否含待修复的字形问题（用于决定是否需要重建索引）。"""
    if not text:
        return False
    if (PDF_GLYPH_MAP.keys() | PDF_GLYPH_DROP) & set(text):
        return True
    return bool(_NOISE_RE.search(text))


# ═══════════════════════════════════════════════════════════════════
# 二、LaTeX → 纯文本
# ═══════════════════════════════════════════════════════════════════
# GUI 的答案框是 CTkTextbox，**不支持渲染 LaTeX**。大模型习惯性输出
# `\\[ \\lim_{\\Delta x \\to 0} \\frac{...}` 这类标记，学生会直接看到反斜杠。
# 提示词里已要求模型改用纯文本数学，这里作为兜底再降级一次。

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "varpi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
}

_SYMBOLS = {
    "times": "×", "div": "÷", "cdot": "·", "cdots": "…", "ldots": "…",
    "dots": "…", "pm": "±", "mp": "∓", "le": "≤", "leq": "≤", "ge": "≥",
    "geq": "≥", "ne": "≠", "neq": "≠", "approx": "≈", "equiv": "≡",
    "sim": "∼", "propto": "∝", "infty": "∞", "to": "→", "rightarrow": "→",
    "leftarrow": "←", "Rightarrow": "⇒", "Leftarrow": "⇐",
    "leftrightarrow": "↔", "mapsto": "↦", "in": "∈", "notin": "∉",
    "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "cup": "∪", "cap": "∩",
    "emptyset": "∅", "varnothing": "∅", "forall": "∀", "exists": "∃",
    "sum": "∑", "prod": "∏", "int": "∫", "oint": "∮", "partial": "∂",
    "nabla": "∇", "angle": "∠", "perp": "⊥", "parallel": "∥",
    "circ": "∘", "degree": "°", "prime": "′", "ast": "*", "star": "⋆",
    "surd": "√", "log": "log", "ln": "ln", "sin": "sin", "cos": "cos",
    "tan": "tan", "cot": "cot", "sec": "sec", "csc": "csc", "lim": "lim",
    "max": "max", "min": "min", "lg": "lg", "exp": "exp",
    "lt": "<", "gt": ">", "mid": "|", "vert": "|", "Vert": "‖",
    "because": "∵", "therefore": "∴", "not": "¬", "land": "∧", "lor": "∨",
    "neg": "¬", "leqq": "≤", "geqq": "≥", "colon": ":", "implies": "⇒",
    "iff": "⇔", "Leftrightarrow": "⇔",
}

_SUPER = dict(zip("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ"))
_SUB = dict(zip("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))

# 需要「保留花括号内内容」的命令
_WRAP_CMDS = ("text", "mathrm", "mathbf", "mathit", "operatorname", "mbox",
              "textrm", "textit", "textbf", "mathsf", "mathtt", "bm",
              # 重音/装饰命令：内容有意义，装饰丢了可接受
              "bar", "hat", "tilde", "dot", "ddot", "overline", "underline",
              "widehat", "widetilde", "boldsymbol", "mit")
# 需要丢弃的命令（排版微调，无语义）
_DROP_CMDS = ("left", "right", "big", "Big", "bigg", "Bigg", "displaystyle",
              "textstyle", "scriptstyle", "limits", "nolimits", "quad",
              "qquad", "thinspace", "negthinspace")

# 变量前缀类符号：后面紧跟字母/数字时不留空格（Δ x -> Δx）。
# **不含**关系与箭头符号 —— 「x → 0」「a ∈ A」里的空格是必要的。
_PREFIX_SYMBOLS = ("Α-Ωα-ωΔ∇∂∑∏∫√±∓∘∀∃∠")

_BRACED_CMD_RE = re.compile(r"\\([A-Za-z]+)\s*\{")
_FRAC_RE = re.compile(r"\\(?:d|t)?frac\s*\{")
_SQRT_RE = re.compile(r"\\sqrt\s*(?:\[([^\]]*)\])?\s*\{")
_CMD_RE = re.compile(r"\\([A-Za-z]+)")
_SCRIPT_RE = re.compile(r"([_^])\s*(\{[^{}]*\}|\\[A-Za-z]+|[^\s{}])")


def _take_group(text: str, start: int) -> Tuple[str, int]:
    """
    从 text[start]（应为 '{'）起取出一对花括号内的内容，支持嵌套。

    Returns:
        (组内内容, 组结束后的下标)；start 位置不是 '{' 时返回 ("", start)。
    """
    if start >= len(text) or text[start] != "{":
        return "", start
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return text[start + 1:], len(text)   # 括号不闭合：取到结尾


def _to_script(body: str, table: Dict[str, str]) -> str:
    """把一段内容转成上标/下标；无法整体转换时退化为 ^(...) / _(...) 。"""
    body = body.strip()
    if body and all(c in table for c in body):
        return "".join(table[c] for c in body)
    return None


def strip_latex(text: str) -> str:
    """
    把 LaTeX 标记降级为可读的纯文本 / Unicode 数学。

    只处理「能无损或近无损转换」的常见构造；不认识的命令去掉反斜杠保留名字，
    保证**不会因为转不出来而丢内容**。
    """
    if not text or "\\" not in text and "$" not in text:
        return text

    s = text

    # 1) 分数：\frac{a}{b} -> (a)/(b)，从内向外反复处理以支持嵌套
    #    正则末尾正好落在 '{' 上，故括号起点是 m.end() - 1。
    for _ in range(24):
        m = _FRAC_RE.search(s)
        if not m:
            break
        brace = m.end() - 1
        num, i = _take_group(s, brace)
        if i < len(s) and s[i] == "{":
            den, j = _take_group(s, i)
        else:
            den, j = "", i
        s = f"{s[:m.start()]}({num})/({den}){s[j:]}"

    # 2) 根号：\sqrt[n]{a} -> ⁿ√(a) 或 √(a)
    for _ in range(24):
        m = _SQRT_RE.search(s)
        if not m:
            break
        deg = (m.group(1) or "").strip()
        body, j = _take_group(s, m.end() - 1)
        prefix = f"{deg}√(" if deg else "√("
        s = f"{s[:m.start()]}{prefix}{body}){s[j:]}"

    # 3) 保留内容的外壳命令：\text{...} -> ...（只处理白名单内的命令）
    for _ in range(24):
        target = None
        for mm in _BRACED_CMD_RE.finditer(s):
            if mm.group(1) in _WRAP_CMDS:
                target = mm
                break
        if target is None:
            break
        body, j = _take_group(s, target.end() - 1)
        s = f"{s[:target.start()]}{body}{s[j:]}"

    # 3b) 向量箭头：\vec{AB} -> AB→（教材自身就写作 犃犅→，保持一致）
    for _ in range(24):
        m = re.search(r"\\vec\s*\{", s)
        if not m:
            break
        body, j = _take_group(s, m.end() - 1)
        s = f"{s[:m.start()]}{body}→{s[j:]}"

    # 4) 定界符、环境标记与换行
    #    环境标记必须在这里处理：下面的第 6 步会把 `\begin` 剥成 `begin`，
    #    届时再也没有反斜杠可以识别了。
    s = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", "", s)
    s = re.sub(r"\\\[|\\\]|\\\(|\\\)", "", s)
    s = s.replace("$$", "").replace("$", "")
    s = re.sub(r"\\\\", "\n", s)

    # 5) 无意义的排版微调命令
    for cmd in _DROP_CMDS:
        s = re.sub(r"\\" + cmd + r"\b", "", s)

    # 6) 普通命令：希腊字母、运算符
    def _sub_cmd(m: "re.Match[str]") -> str:
        name = m.group(1)
        if name in _GREEK:
            return _GREEK[name]
        if name in _SYMBOLS:
            return _SYMBOLS[name]
        return name          # 不认识的命令：保留名字，仅去掉反斜杠
    s = _CMD_RE.sub(_sub_cmd, s)

    # 7) 转义符号
    s = s.replace("\\{", "{").replace("\\}", "}").replace("\\%", "%")
    s = s.replace("\\&", "&").replace("\\_", "_")

    # 8) 上下标
    #    (a) 算子名的下标写成括号形式：`\lim_{x \to 0}` -> lim(x→0)，
    #        而不是 lim_{x→0} / lim(x→0) 之类带残留花括号的写法。
    _OPS = (r"lim|max|min|log|ln|lg|exp|sin|cos|tan|cot|sec|csc|"
            r"sum|prod|int|oint|arg|det|gcd|deg")
    s = re.sub(r"\b(" + _OPS + r")\s*[_^]\s*\{([^{}]*)\}", r"\1(\2)", s)
    s = re.sub(r"\b(" + _OPS + r")\s*[_^]\s*([A-Za-z0-9])", r"\1(\2)", s)

    def _sub_script(m: "re.Match[str]") -> str:
        kind, raw = m.group(1), m.group(2)
        body = raw[1:-1] if raw.startswith("{") and raw.endswith("}") else raw
        out = _to_script(body, _SUPER if kind == "^" else _SUB)
        return out if out is not None else f"{kind}({body.strip()})"
    s = _SCRIPT_RE.sub(_sub_script, s)

    # 9) 变量与符号之间不留空格：`Δ x` → `Δx`、`∀ x` → `∀x`
    #    只针对「前缀型」希腊字母与数学符号，因此
    #    「x → 0」「a ∈ A」「求 α 的值」这些都不受影响。
    s = re.sub(r"([" + _PREFIX_SYMBOLS + r"])\s+(?=[A-Za-z0-9])", r"\1", s)

    # 10) 对齐符与残留外壳
    s = re.sub(r"\\?(?:begin|end)\s*\{[^{}]*\}", "", s)
    s = re.sub(r"(?<=[^\n])&(?=[^\n])", " ", s)      # 表格对齐符 -> 空格
    s = re.sub(r"\\(?![A-Za-z0-9])", "", s)          # 残留的孤立反斜杠（保留 \word 形态，避免误删路径）

    # 11) 收尾：去掉定界符残留的边距空格、压掉多余空行
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = _WS_ONLY_LINE_RE.sub("\n\n", s)
    return s


def clean_text(text: str) -> str:
    """
    完整规整：降级 LaTeX → 修字形。可重复调用（幂等）。

    顺序不能颠倒。大模型写的 `\\Delta狓` 里，`\\Delta` 是命令、`狓` 是被
    PDF 错映射的字母；必须**先**把 LaTeX 降级成 `Δ` + `狓`，字形修复再把它
    变成 `Δx`。若反过来先做字形修复，就会得到 `\\Deltax` —— 反斜杠命令被
    粘成一个不存在的命令名，反而更糟。
    """
    if not text:
        return text
    return fix_pdf_glyphs(strip_latex(text))
