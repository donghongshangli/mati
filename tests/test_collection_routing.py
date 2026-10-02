"""
集合路由回归测试（需要 chromadb）。

关键回归点：**无年级标注的集合（人工导入教材常落在这里）不得因学生选定年级而被排除**。
否则人工导入的书会在检索中静默隐身。

测试用临时 ChromaDB，不触碰正式索引。

    python tests/test_collection_routing.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import chromadb
    from system.rag.rag_retrieval_engine import RAGRetrievalEngine
    from system.rag.retrieval_config import RetrievalConfig
    HAVE_CHROMA = True
except ImportError as e:  # pragma: no cover
    HAVE_CHROMA = False
    _IMPORT_ERR = str(e)


def _make_engine(db_path, client=None, grade_mode=None):
    """
    构造一个只带 chroma_client + config 的引擎（跳过模型加载）。

    `client` 必须由 `_seed*` 传入并复用：**同一进程内不要对同一路径创建第二个
    PersistentClient**，否则第二个客户端的查询会间歇性失败（静默零召回）。
    见 RAGRetrievalEngine.__init__ 的说明。
    """
    engine = object.__new__(RAGRetrievalEngine)
    engine.chroma_client = client if client is not None else chromadb.PersistentClient(path=db_path)
    engine.config = RetrievalConfig.load()
    if grade_mode in ("soft", "hard"):
        engine.config.data.setdefault("routing", {})["grade_mode"] = grade_mode
    return engine


def _seed(db_path):
    client = chromadb.PersistentClient(path=db_path)
    spec = [
        ("neb_science_grade_10", {"subject": "science", "grade": "10"}),
        ("neb_science_grade_11", {"subject": "science", "grade": "11"}),
        # 人工导入、年级推断失败的教材会落在这种「无年级」集合里
        ("neb_science", {"subject": "science"}),
        ("neb_math_grade_10", {"subject": "math", "grade": "10"}),
        ("neb_readme", {"subject": "readme"}),
    ]
    for name, meta in spec:
        client.get_or_create_collection(name=name, metadata=dict(meta, **{"hnsw:space": "cosine"}))


def _seed_discipline(db_path):
    """science 家族按 discipline 拆开的情形（P2-5）。"""
    client = chromadb.PersistentClient(path=db_path)
    spec = [
        ("neb_physics_grade_10", {"subject": "science", "discipline": "physics", "grade": "10"}),
        ("neb_chemistry_grade_10", {"subject": "science", "discipline": "chemistry", "grade": "10"}),
        ("neb_biology_grade_10", {"subject": "science", "discipline": "biology", "grade": "10"}),
        ("neb_math_grade_10", {"subject": "math", "discipline": "mathematics", "grade": "10"}),
    ]
    for name, meta in spec:
        client.get_or_create_collection(name=name, metadata=dict(meta, **{"hnsw:space": "cosine"}))


def test_family_subject_returns_all_disciplines():
    """GUI 的「科学」必须仍能检索物理+化学+生物（拆集合不能拆掉学生入口）。"""
    tmp = tempfile.mkdtemp(prefix="mati_routing_")
    try:
        e = _make_engine(tmp, _seed_discipline(tmp))
        got = set(e._get_relevant_collections("Science", "10"))
        assert got == {"neb_physics_grade_10", "neb_chemistry_grade_10",
                       "neb_biology_grade_10"}, got
        # 不应串到数学
        assert "neb_math_grade_10" not in got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_discipline_narrows_to_single_subject():
    """指定 discipline 时只查该细分学科，不串科。"""
    tmp = tempfile.mkdtemp(prefix="mati_routing_")
    try:
        e = _make_engine(tmp, _seed_discipline(tmp))
        got = e._get_relevant_collections("Science", "10", "物理")
        assert got == ["neb_physics_grade_10"], got
        got_en = e._get_relevant_collections("Science", "10", "chemistry")
        assert got_en == ["neb_chemistry_grade_10"], got_en
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_discipline_falls_back_to_family_when_absent():
    """请求的细分学科没有集合时，退回家族检索而不是返回空。"""
    tmp = tempfile.mkdtemp(prefix="mati_routing_")
    try:
        # 只有家族级集合，没有 discipline metadata
        e = _make_engine(tmp, _seed(tmp))
        got = e._get_relevant_collections("Science", "10", "biology")
        assert "neb_science_grade_10" in got, got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_collection_name_from_discipline_is_mapped_back_to_family():
    """旧库/无名 metadata：集合名 neb_physics_grade_10 必须能反查回 science 家族。"""
    tmp = tempfile.mkdtemp(prefix="mati_routing_")
    try:
        client = chromadb.PersistentClient(path=tmp)
        # 故意不写 subject metadata，模拟旧集合
        client.get_or_create_collection(name="neb_physics_grade_10",
                                        metadata={"hnsw:space": "cosine"})
        e = _make_engine(tmp)
        disc = e._discover_collections()
        assert disc and disc[0]["subject"] == "science", disc
        assert "neb_physics_grade_10" in e._get_relevant_collections("Science", "10")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ungraded_collection_is_never_excluded_by_grade():
    """
    回归（P0 的「静默隐身」问题）：人工导入、年级推断失败的教材会落在
    `neb_science` 这类**无年级集合**里。它按定义覆盖全年级，
    因此**任何年级下都不能被排除**。

    P1 起年级默认是**软过滤**（`routing.grade_mode: soft`），
    因此"其他年级的集合也参与召回"是预期行为，不再是缺陷。
    这里对两种模式分别断言，核心不变的是：`neb_science` 必须始终在场。
    """
    tmp = tempfile.mkdtemp(prefix="mati_routing_")
    try:
        client = _seed(tmp)

        # 硬过滤模式：年级精确匹配 + 无年级
        e_hard = _make_engine(tmp, client, grade_mode="hard")
        got = e_hard._get_relevant_collections("Science", "10")
        assert "neb_science_grade_10" in got, got
        assert "neb_science" in got, f"无年级集合被排除（人工导入教材会隐身）：{got}"
        assert "neb_science_grade_11" not in got, got

        got11 = e_hard._get_relevant_collections("Science", "11")
        assert "neb_science_grade_11" in got11 and "neb_science" in got11, got11
        assert "neb_science_grade_10" not in got11, got11

        # 软过滤模式（P1 默认）：全年级都参与召回，但同年级排在前面。
        # 这是「高二学生问必修三内容」不被误判为"教材中未找到"的实现基础。
        e_soft = _make_engine(tmp, client, grade_mode="soft")
        soft10 = e_soft._get_relevant_collections("Science", "10")
        assert "neb_science" in soft10, soft10
        assert "neb_science_grade_11" in soft10, soft10
        assert soft10[0] == "neb_science_grade_10",             f"软过滤下同年级集合应排在前面：{soft10}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_grade_returns_whole_subject():
    tmp = tempfile.mkdtemp(prefix="mati_routing_")
    try:
        e = _make_engine(tmp, _seed(tmp))
        got = e._get_relevant_collections("Science", "")
        assert set(got) == {"neb_science_grade_10", "neb_science_grade_11", "neb_science"}, got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_grade_falls_back_to_all_when_no_exact_match():
    tmp = tempfile.mkdtemp(prefix="mati_routing_")
    try:
        e = _make_engine(tmp, _seed(tmp))
        # Math 只有 grade 10，请求 11 应回退到该学科全部（而不是返回空）
        got = e._get_relevant_collections("Math", "11")
        assert got == ["neb_math_grade_10"], got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_unknown_subject_returns_empty_not_cross_subject():
    tmp = tempfile.mkdtemp(prefix="mati_routing_")
    try:
        e = _make_engine(tmp, _seed(tmp))
        # 没有英语教材 -> 返回空（宁可拒答也不跨学科串答案）
        assert e._get_relevant_collections("English Grammar", "10") == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_readme_collection_is_filtered_out():
    tmp = tempfile.mkdtemp(prefix="mati_routing_")
    try:
        e = _make_engine(tmp, _seed(tmp))
        names = e._get_relevant_collections("Science", "10")
        assert "neb_readme" not in names, names
        # 也不应出现在集合发现结果里
        assert all(c["name"] != "neb_readme" for c in e._discover_collections())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
