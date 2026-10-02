"""
教材目录解析器回归测试。

用 textbooks/ 下的**真实文件名**逐一断言 book_id 与 grade，
防止「必修上册」被「选择性必修上册」子串误吞 这类匹配回归。

可直接运行（不依赖 pytest）：
    python tests/test_curriculum_catalog.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.rag.curriculum_catalog import (  # noqa: E402
    CurriculumCatalog,
    collection_name_for,
    grade_label,
    normalize_name,
)

# 文件名主干（不含扩展名） -> (期望 book_id, 期望 grade, 期望 subject)
EXPECTED = {
    "2025人教版物理必修一": ("physics_required_1", "10", "science"),
    "2025人教版物理必修二": ("physics_required_2", "10", "science"),
    "2025人教版物理必修三": ("physics_required_3", "10", "science"),
    "2025人教版物理选必一": ("physics_elective_1", "11", "science"),
    "2025人教版物理选必二": ("physics_elective_2", "11", "science"),
    "2025人教版物理选必三": ("physics_elective_3", "11", "science"),
    "信息技术 数据与计算 人教中图版 必修 第一册 2019版":
        ("it_required_1", "10", "computer_science"),
    "信息技术 信息系统与社会 人教中图版 必修 第二册 2019版":
        ("it_required_2", "10", "computer_science"),
    "信息技术 数据与数据结构 人教中图版 选择性必修 第一册 2019版":
        ("it_elective_1", "11", "computer_science"),
    "信息技术 网络基础 人教中图版 选择性必修 第二册 2019版":
        ("it_elective_2", "11", "computer_science"),
    "信息技术 数据管理与分析 人教中图版 选择性必修 第三册 2019版":
        ("it_elective_3", "11", "computer_science"),
    "信息技术 人工智能初步 人教中图版 选择性必修 第四册 2019版":
        ("it_elective_4", "11", "computer_science"),
    "化学 人教版 必修 第一册 2019版": ("chemistry_required_1", "10", "science"),
    "化学 人教版 必修 第二册 2019版": ("chemistry_required_2", "10", "science"),
    "化学·化学反应原理 人教版 选择性必修 第一册 2019版":
        ("chemistry_elective_1", "11", "science"),
    "化学·物质结构与性质 人教版 选择性必修 第二册 2019版":
        ("chemistry_elective_2", "11", "science"),
    "化学·有机化学基础 人教版 选择性必修 第三册 2019版":
        ("chemistry_elective_3", "11", "science"),
    "数学 人教版 A 必修 第一册 2019版": ("math_required_1", "10", "math"),
    "数学 人教版 A 必修 第二册 2019版": ("math_required_2", "10", "math"),
    "数学 人教版 A 选择性必修 第一册 2019版": ("math_elective_1", "11", "math"),
    "数学 人教版 A 选择性必修 第二册 2019版": ("math_elective_2", "11", "math"),
    "数学 人教版 A 选择性必修 第三册 2019版": ("math_elective_3", "11", "math"),
    "语文 统编版 必修 上册 2019版": ("chinese_required_1", "10", "chinese"),
    "语文 统编版 必修 下册 2019版": ("chinese_required_2", "10", "chinese"),
    "语文 统编版 选择性必修 上册 2022版": ("chinese_elective_1", "11", "chinese"),
    "语文 统编版 选择性必修 中册 2022版": ("chinese_elective_2", "11", "chinese"),
    "语文 统编版 选择性必修 下册 2022版": ("chinese_elective_3", "11", "chinese"),
}


def _catalog():
    return CurriculumCatalog.load()


# --------------------------------------------------------------------- 单测
def test_all_expected_files_resolve_to_expected_book_and_grade():
    cat = _catalog()
    problems = []
    for stem, (book_id, grade, subject) in EXPECTED.items():
        r = cat.resolve(stem)
        if not r["matched"]:
            problems.append(f"{stem}: 未命中目录")
        elif r["book_id"] != book_id:
            problems.append(f"{stem}: book_id={r['book_id']} 期望 {book_id}")
        elif r["grade"] != grade:
            problems.append(f"{stem}: grade={r['grade']} 期望 {grade}")
        elif r["subject"] != subject:
            problems.append(f"{stem}: subject={r['subject']} 期望 {subject}")
    assert not problems, "解析不符：\n  " + "\n  ".join(problems)


def test_elective_not_swallowed_by_required_rule():
    """回归：'选择性必修上册' 含子串 '必修上册'，不得被判成必修。"""
    cat = _catalog()
    for stem in EXPECTED:
        if "选择性必修" in stem or "选必" in stem:
            r = cat.resolve(stem)
            assert r["grade"] == "11", f"{stem} 被误判为 grade={r['grade']}"
            assert not str(r["book_id"]).endswith(("required_1", "required_2")), (
                f"{stem} 被误匹配到必修条目 {r['book_id']}"
            )


def test_year_digits_not_treated_as_grade():
    """回归：文件名里的年份 2019/2022/2025 不得被当成年级。"""
    cat = _catalog()
    r = cat.resolve("未知教材 2019版")
    assert r["grade"] == "unknown", f"年份被误解析为 grade={r['grade']}"


def test_fallback_subject_by_keyword():
    cat = _catalog()
    r = cat.resolve("某校自编物理讲义.pdf")
    assert r["matched"] is False
    assert r["subject"] == "science", r


def test_fallback_grade_from_directory_convention():
    """
    回归：文档承诺「年级从文件夹名（grade_10/）提取」。
    P0 改造后解析器一度只看文件名而丢掉路径，此处锁定该行为。
    """
    cat = _catalog()
    cases = [
        ("textbooks/grade_10/cs_notes.pdf", "10"),
        ("textbooks/grade_11/my_math.pdf", "11"),
        ("textbooks/class_12/chem.pdf", "12"),
        ("textbooks/10/physics_extra.pdf", "10"),
        (r"textbooks\grade_11\english.pdf", "11"),   # Windows 反斜杠
    ]
    for path, expect in cases:
        r = cat.resolve(os.path.basename(path), path=path)
        assert r["grade"] == expect, f"{path}: grade={r['grade']} 期望 {expect}"

    # 目录未命中但学科命中
    r = cat.resolve("自编物理讲义.pdf", path="textbooks/grade_10/自编物理讲义.pdf")
    assert r["subject"] == "science"
    assert r["grade"] == "10"
    assert r["matched"] is False


def test_fallback_still_ignores_year_digits_in_path():
    cat = _catalog()
    r = cat.resolve("某教材 2025版.pdf", path="textbooks/某教材 2025版.pdf")
    assert r["grade"] == "unknown", r["grade"]


def test_path_directories_with_incidental_digits_are_not_grades():
    """
    回归：目录名里出现的任意数字不得被当成年级。
    目录必须是**恰好全数字**才算（如 textbooks/10/），
    否则 `_p1_probe`、`proj2_tmp` 这类目录会让整批教材被打成「一年级」。
    """
    cat = _catalog()
    for path in [
        "textbooks/_p1_probe/物理讲义.pdf",
        "textbooks/proj2_tmp/物理讲义.pdf",
        "build_v2/物理讲义.pdf",
        "tmp2025/物理讲义.pdf",
    ]:
        r = cat.resolve(os.path.basename(path), path=path)
        assert r["grade"] == "unknown", f"{path}: 误判为 grade={r['grade']}"

    # 恰好全数字的目录仍然有效
    assert cat.resolve("x.pdf", path="textbooks/10/x.pdf")["grade"] == "10"
    assert cat.resolve("x.pdf", path="textbooks/12/x.pdf")["grade"] == "12"


def test_catalog_match_takes_priority_over_path():
    """目录表命中时，不应被路径里的数字改写年级。"""
    cat = _catalog()
    r = cat.resolve("数学 人教版 A 选择性必修 第一册 2019版.pdf",
                    path="textbooks/grade_10/数学 人教版 A 选择性必修 第一册 2019版.pdf")
    assert r["matched"] is True
    assert r["grade"] == "11", r["grade"]


def test_collection_name_is_chroma_safe():
    for subject, grade in [("computer_science", "10"), ("math", "11"),
                           ("science", "unknown"), ("", ""), ("a", "12")]:
        name = collection_name_for(subject, grade)
        assert len(name) >= 3, name
        assert all(c.isalnum() or c in "._-" for c in name), name


def test_collection_name_never_collapses_to_bare_prefix():
    """回归：整段中文的文件名曾把学科清洗成空串，生成名为 'neb' 的垃圾集合。"""
    for subject in ["调研笔记运行时安全日志用户数据", "中文", "", "-", "   "]:
        name = collection_name_for(subject, "unknown")
        assert name != "neb", f"subject={subject!r} 生成了退化集合名"
        assert name.startswith("neb_"), name
        n = collection_name_for(subject, "10")
        assert n != "neb", f"subject={subject!r} grade=10 生成了退化集合名"
        assert n.startswith("neb_"), n
        assert "__grade" not in n, f"出现连续下划线：{n}"


def test_fallback_gives_ascii_safe_subject():
    cat = CurriculumCatalog.load()
    r = cat.resolve("调研笔记-运行时-安全-日志-用户数据.md")
    assert r["matched"] is False
    # 学科部分必须能被安全用作集合名
    assert r["subject"].isascii(), r["subject"]
    assert collection_name_for(r["subject"], r["grade"]).startswith("neb_")


def test_grade_label():
    assert grade_label("10") == "高一"
    assert grade_label("11") == "高二"
    assert grade_label("12") == "高三"
    assert grade_label("unknown") == ""


def test_normalize_strips_decoration():
    assert normalize_name("化学·化学反应原理 (人教版)") == "化学化学反应原理人教版"
    assert normalize_name("数学 人教版 A") == "数学人教版a"


# ------------------------------------------------------------ 独立运行入口
def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}\n        {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
