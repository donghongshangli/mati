#!/usr/bin/env python3
"""
文本规整回归测试：PDF 字形错映射修复 + LaTeX 降级。

判据来源
--------
映射表不是猜的。最有力的一条来自人教 A 版数学选择性必修第三册原文：

    「通常用大写英文字母表示随机变量，例如犡，犢，犣；
       用小写英文字母表示随机变量的取值，例如狓，狔，狕。」

据此锁定 X/Y/Z 与 x/y/z，其余按字母序 + 公式语义逐条验证
（详见 system/rag/text_cleanup.py 映射表的行内注释）。

这些用例全部是纯逻辑断言，不依赖向量库，任何 Python 都能跑。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.rag.text_cleanup import (  # noqa: E402
    PDF_GLYPH_MAP,
    clean_text,
    fix_pdf_glyphs,
    glyph_stats,
    has_glyph_issues,
    strip_latex,
)

_PASS = []
_FAIL = []
_SKIP = []


def check(name, fn):
    try:
        fn()
        _PASS.append(name)
    except AssertionError as e:
        _FAIL.append((name, str(e) or "断言失败"))
    except Exception as e:  # noqa: BLE001
        _FAIL.append((name, f"{type(e).__name__}: {e}"))


# ══════════════════════════════════════════════════ 字形修复：字母
def test_user_reported_formula_is_fixed():
    """主人实际遇到的那段乱码：狓→x、狔→y。"""
    got = fix_pdf_glyphs("如果Δ狓→0时，Δ狔Δ狓无限趋近于某个定值，则该函数在点狓处的导数")
    assert "Δx→0" in got, got
    assert "ΔyΔx" in got, got
    assert "点x处" in got, got
    assert "狓" not in got and "狔" not in got, got


def test_random_variable_sentence_is_decoded():
    """教材原文：大写表示随机变量（犡犢犣），小写表示取值（狓狔狕）。"""
    src = "通常用大写英文字母表示随机变量，例如犡，犢，犣；用小写英文字母表示随机变量的取值，例如狓，狔，狕．"
    got = fix_pdf_glyphs(src)
    assert "例如X，Y，Z" in got, got
    assert "例如x，y，z" in got, got


def test_uppercase_letters_all_mapped():
    src = "犃犅犆犇犈犉犌犎犐犑犓犔犕犖犗犘犙犚犛犜犝犞犠犡犢犣"
    assert fix_pdf_glyphs(src) == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def test_lowercase_letters_all_mapped():
    # 注意：小写 o 未被任何教材使用，映射表里刻意不设，故此处也不含 o
    src = "犪犫犮犱犲犳犵犺犻犼犽犾犿狀狆狇狉狊狋狌狏狑狓狔狕"
    assert fix_pdf_glyphs(src) == "abcdefghijklmnopqrstuvwxyz".replace("o", "")


def test_real_chinese_characters_are_not_touched():
    """
    反向保护：这些是**常用汉字**，恰好和错映射字母同码段，绝不能被改。
    用范围替换的实现会在这里翻车（U+70C4–70C6 里有「烈」，U+71BF–71C5 里有「燃」）。
    """
    keep = "牛犊犍为武阳犒劳犬犯犰狳状犹狂狄狼狈狐狗狠犀利犁地现状燃烧烈火"
    assert fix_pdf_glyphs(keep) == keep, fix_pdf_glyphs(keep)


def test_normal_chinese_text_unchanged():
    """不含问题字符的正文必须原样返回（快路径）。"""
    src = "函数的单调性与导数密切相关。一般地，设函数 f(x) 在区间 D 上可导。"
    assert fix_pdf_glyphs(src) == src


# ══════════════════════════════════════════════════ 字形修复：符号
def test_math_symbols_are_restored():
    # 注意：源文用的是**全角**数字（２），字形修复不负责全角/半角转换，
    # 因此期望值里保留全角 —— 那是教材原文的排版，不是乱码。
    cases = {
        "｜犃犅｜＝２槡２": "｜AB｜＝２√２",           # 根号
        "犗犅瓚犆犃": "OB∥CA",                       # 平行
        "狘犘犉１狘＝犪": "|PF１|＝a",                 # 绝对值竖线
    }
    for src, want in cases.items():
        got = fix_pdf_glyphs(src)
        assert got == want, f"{src!r} -> {got!r}，期望 {want!r}"


def test_pua_math_symbols_are_restored():
    """私用区里承载的是真符号，不能当噪声删掉。"""
    cases = {
        "并用符号“\ue02f”表示": "并用符号“∀”表示",
        "犪\ue01bβ，犫\ue01bβ": "a⊂β，b⊂β",
        "记作狆\ue03c狇": "记作p⇒q",
        "犿，狀∈犖\ue012": "m，n∈N*",
        "１\ue010１\ue010２": "１.１.２",
        "（ｕｎｉｔｖｅｃ\ue011ｔｏｒ）": "（ｕｎｉｔｖｅｃ-ｔｏｒ）",
    }
    for src, want in cases.items():
        got = fix_pdf_glyphs(src)
        assert got == want, f"{src!r} -> {got!r}，期望 {want!r}"


def test_unknown_pua_noise_is_dropped():
    """未映射的私用区字符与异体文字块是纯噪声，删除而非显示成方框。"""
    got = fix_pdf_glyphs("废水处理工艺\ue999᠀ᮕဳ流程图")
    assert "\ue999" not in got, got
    assert "废水处理工艺" in got and "流程图" in got, got


def test_glyph_stats_reports_counts():
    mapped, noise = glyph_stats("狓狓狔\ue999")
    assert mapped.get("狓") == 2 and mapped.get("狔") == 1, mapped
    assert noise == 1, noise


def test_has_glyph_issues_detects_both_kinds():
    assert has_glyph_issues("向量犪") is True
    assert has_glyph_issues("噪声\ue999") is True
    assert has_glyph_issues("完全正常的句子") is False


def test_fix_is_idempotent():
    src = "若函数狔＝犳（狓），则犪\ue01bβ，２槡２"
    once = fix_pdf_glyphs(src)
    assert fix_pdf_glyphs(once) == once


# ══════════════════════════════════════════════════ LaTeX 降级
def test_latex_delimiters_removed():
    got = strip_latex(r"导数是 $f'(x)$，即 \[ f'(x)=2x \]")
    assert "$" not in got and r"\[" not in got and r"\]" not in got, got
    assert "f'(x)=2x" in got, got


def test_frac_and_sqrt_converted():
    assert strip_latex(r"\frac{\Delta y}{\Delta x}") == "(Δy)/(Δx)"
    assert strip_latex(r"\sqrt{3}") == "√(3)"
    assert strip_latex(r"\sqrt[3]{V}") == "3√(V)"


def test_operator_and_greek_commands_converted():
    got = strip_latex(r"\lim_{\Delta x \to 0} \frac{\Delta y}{\Delta x}")
    assert "lim(" in got and "Δx" in got, got
    assert "\\" not in got, got
    got2 = strip_latex(r"\alpha \in (0, \pi/2)")
    assert got2 == "α ∈ (0, π/2)", got2


def test_script_converted_to_unicode():
    got = strip_latex(r"$x_1 + x_2$")
    assert got == "x₁ + x₂", got
    got2 = strip_latex(r"$a^{2}$")
    assert got2 == "a²", got2


def test_delta_adjacent_to_glyph_is_not_glued():
    """
    关键边界：`\\Delta狓` 必须先降级 LaTeX 再修字形。
    顺序反了会得到 `\\Deltax` —— 反斜杠命令被粘成一个不存在的命令名。
    """
    got = clean_text(r"\Delta狓 \to 0")
    assert got == "Δx → 0", got
    assert "\\" not in got, got


def test_unknown_command_does_not_lose_content():
    """不认识的命令只去掉反斜杠，绝不吞内容。"""
    got = strip_latex(r"\foo{bar}")
    assert "bar" in got, got
    assert "\\" not in got, got


def test_real_chinese_text_is_not_mangled_by_latex_pass():
    src = "本章小结：导数刻画的是瞬时变化率，与函数的单调性有关。下划线_和星号*应保持原样。"
    got = strip_latex(src)
    assert got == src, got


def test_clean_text_end_to_end_on_reported_answer():
    """端到端：主人贴的那段答案，规整后应完全可读。"""
    raw = (
        "导数的定义是：一个函数的导数是在某点处的瞬时变化率。具体来说，如果Δ狓→0时，"
        "Δ狔Δ狓无限趋近于某个定值，则这个定值就是该函数在点狓处的导数。用数学符号表示即为：\n"
        "\\[\n"
        "\\text{狔}' = \\lim_{\\Delta狓 \\to 0} \\frac{\\Delta \\text{狔}}{\\Delta \\text{狓}}\n"
        "\\]"
    )
    got = clean_text(raw)
    assert "Δx→0" in got, got
    assert "y' = lim(" in got, got
    assert "狓" not in got and "狔" not in got, got
    assert "\\" not in got and "{" not in got and "}" not in got, got
    assert clean_text(got) == got, "不幂等"


def test_clean_text_is_idempotent():
    for src in ["向量犪\ue01bβ", r"$\frac{1}{2}$", "正常文本", r"\sqrt{x^2+1}"]:
        once = clean_text(src)
        assert clean_text(once) == once, src


# ══════════════════════════════════════════════════ 映射表自检
def test_map_has_no_duplicate_targets_conflicts():
    """一个源字符只能映射到一个目标；且不能把常用汉字映射成字母。"""
    assert len(PDF_GLYPH_MAP) == len(set(PDF_GLYPH_MAP)), "映射表有重复键"
    assert all(len(v) >= 1 for v in PDF_GLYPH_MAP.values())
    # 目标不能还是汉字（否则修复等于没做）
    for k, v in PDF_GLYPH_MAP.items():
        assert not ("\u4e00" <= v <= "\u9fff"), f"{k} 仍映射为汉字 {v}"


def main():
    checks = [
        ("主人报告的公式", test_user_reported_formula_is_fixed),
        ("随机变量句", test_random_variable_sentence_is_decoded),
        ("大写字母 26 个", test_uppercase_letters_all_mapped),
        ("小写字母 25 个", test_lowercase_letters_all_mapped),
        ("常用汉字不被误改", test_real_chinese_characters_are_not_touched),
        ("正常中文原样返回", test_normal_chinese_text_unchanged),
        ("数学符号 √ ∥ |", test_math_symbols_are_restored),
        ("PUA 真符号 ∀ ⊂ ⇒ * - .", test_pua_math_symbols_are_restored),
        ("PUA 噪声被删除", test_unknown_pua_noise_is_dropped),
        ("统计口径", test_glyph_stats_reports_counts),
        ("问题检测", test_has_glyph_issues_detects_both_kinds),
        ("字形修复幂等", test_fix_is_idempotent),
        ("LaTeX 定界符", test_latex_delimiters_removed),
        ("\\frac / \\sqrt", test_frac_and_sqrt_converted),
        ("\\lim / 希腊字母", test_operator_and_greek_commands_converted),
        ("上下标转 Unicode", test_script_converted_to_unicode),
        ("\\Delta 粘连边界", test_delta_adjacent_to_glyph_is_not_glued),
        ("未知命令不丢内容", test_unknown_command_does_not_lose_content),
        ("中文不被 LaTeX 步骤破坏", test_real_chinese_text_is_not_mangled_by_latex_pass),
        ("端到端：报告原文", test_clean_text_end_to_end_on_reported_answer),
        ("clean_text 幂等", test_clean_text_is_idempotent),
        ("映射表自检", test_map_has_no_duplicate_targets_conflicts),
    ]
    for name, fn in checks:
        check(name, fn)

    print(f"\n{'='*64}\n文本规整回归测试\n{'='*64}")
    for name in _PASS:
        print(f"  PASS  {name}")
    for name, reason in _SKIP:
        print(f"  SKIP  {name}  （{reason}）")
    for name, reason in _FAIL:
        print(f"  FAIL  {name}\n        {reason}")
    print("-" * 64)
    print(f"  共 {len(_PASS)} 通过 / {len(_FAIL)} 失败 / {len(_SKIP)} 跳过")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
