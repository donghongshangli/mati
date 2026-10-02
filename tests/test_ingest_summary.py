"""
摄取摘要与强制学科/年级的回归测试（P0-1 / P0-2 / P2-5 服务端行为）。

只依赖 chromadb —— 用例都不触发嵌入模型加载
（空文件在抽取阶段就被拦下，README 在发现阶段就被排除）。

    python tests/test_ingest_summary.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import chromadb  # noqa: F401
    from scripts.ingest_content import (
        SKIP_FILENAMES,
        UniversalContentIngester,
        format_ingest_summary,
    )
    HAVE_CHROMA = True
except ImportError as e:  # pragma: no cover
    HAVE_CHROMA = False
    _IMPORT_ERR = str(e)


def _tmp_db():
    return tempfile.mkdtemp(prefix="mati_ingest_")


def test_readme_is_not_ingested():
    """目录说明文件不应进入知识库（过去 textbooks/README.md 会生成 neb_readme）。"""
    d = tempfile.mkdtemp(prefix="mati_src_")
    db = _tmp_db()
    try:
        Path(d, "README.md").write_text("这是目录说明，不是教学内容。", encoding="utf-8")
        Path(d, "readme.txt").write_text("说明", encoding="utf-8")
        ing = UniversalContentIngester(db, "never")
        s = ing.ingest_directory(d, progress=False)
        assert s["found"] == 0, f"README 未被排除：{s}"
        assert s["files"] == [], s["files"]
        assert all(n in SKIP_FILENAMES for n in ("readme.md", "readme.txt"))
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(db, ignore_errors=True)


def test_empty_file_is_reported_as_skipped_with_reason():
    """空/无文本文件必须报出原因，而不是静默 0 块。"""
    d = tempfile.mkdtemp(prefix="mati_src_")
    db = _tmp_db()
    try:
        Path(d, "空白讲义.txt").write_text("   \n  ", encoding="utf-8")
        ing = UniversalContentIngester(db, "never")
        s = ing.ingest_directory(d, progress=False)
        assert s["found"] == 1
        assert s["ok"] == 0 and s["skipped"] == 1, s
        rec = s["files"][0]
        assert rec["status"] == "skipped"
        assert rec["reason"], "跳过却没有给出原因"
        report = format_ingest_summary(s)
        assert "跳过" in report
        assert "没有任何文件成功入库" in report, report
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(db, ignore_errors=True)


def test_missing_directory_is_error_not_silent():
    db = _tmp_db()
    try:
        ing = UniversalContentIngester(db, "never")
        s = ing.ingest_directory("no/such/dir", progress=False)
        assert s["failed"] == 1 and s["ok"] == 0, s
        assert s["files"][0]["reason"] == "目录不存在", s["files"]
    finally:
        shutil.rmtree(db, ignore_errors=True)


def test_force_subject_and_grade_override_inference():
    """P0-2：界面指定学科/年级时优先于按书名推断。"""
    db = _tmp_db()
    try:
        ing = UniversalContentIngester(db, "never",
                                       force_subject="生物", force_grade="11")
        m = ing.extract_metadata(Path("王老师自编物理讲义.pdf"))
        assert m["subject"] == "science", m
        assert m["discipline"] == "biology", m          # 细分到生物
        assert m["grade"] == "11" and m["grade_label"] == "高二", m
        assert m["collection_name"] == "neb_biology_grade_11", m["collection_name"]

        # 只指定到家族时退化为家族集合
        ing2 = UniversalContentIngester(db, "never", force_subject="Science")
        m2 = ing2.extract_metadata(Path("王老师自编物理讲义.pdf"))
        assert m2["collection_name"].startswith("neb_science"), m2

        # 非法年级被忽略（不写入库）
        ing3 = UniversalContentIngester(db, "never", force_grade="99")
        assert ing3.force_grade is None
    finally:
        shutil.rmtree(db, ignore_errors=True)


def test_catalog_match_still_wins_over_path_for_grade():
    db = _tmp_db()
    try:
        ing = UniversalContentIngester(db, "never")
        m = ing.extract_metadata(Path("textbooks/grade_10/数学 人教版 A 选择性必修 第一册 2019版.pdf"))
        assert m["grade"] == "11", m        # 以教材目录表为准，不被目录名改写
        assert m["collection_name"] == "neb_math_grade_11", m["collection_name"]
    finally:
        shutil.rmtree(db, ignore_errors=True)


def test_summary_shape_is_stable():
    """摘要字段是 GUI 展示契约，不能随意改名。"""
    db = _tmp_db()
    try:
        ing = UniversalContentIngester(db, "never")
        s = ing.ingest_directory(tempfile.mkdtemp(prefix="mati_src_"), progress=False)
        for key in ("input", "found", "ok", "skipped", "failed", "chunks", "files"):
            assert key in s, f"摘要缺少字段 {key}"
        assert isinstance(s["files"], list)
    finally:
        shutil.rmtree(db, ignore_errors=True)


def _main():
    if not HAVE_CHROMA:
        print(f"  SKIP  需要 chromadb：{_IMPORT_ERR}")
        return 0
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, t in tests:
        try:
            t()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n        {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}\n        {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
