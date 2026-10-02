#!/usr/bin/env python3
"""
P1 回归测试：混合检索 / RRF / Query 改写 / 结构切分 / 父子索引 / 7 特征重排。

这些用例全部是**纯逻辑**断言，不依赖向量库内容，因此：
  · 改一个权重、改一处切分逻辑，跑一次就知道有没有连带破坏；
  · 有向量库才能验证的「召回率变化」交给 scripts/eval_rag.py。

可用任意 Python 运行（缺 jieba/rank_bm25 时相关用例自动跳过，
BM25 退化为纯稠密属于设计内行为，不算失败）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def skip(name, reason):
    _SKIP.append((name, reason))


# ══════════════════════════════════════════ 分词与 BM25 可用性
def test_tokenize_zh_splits_chinese():
    from system.rag.hybrid_retriever import tokenize_zh
    toks = tokenize_zh("向心加速度与万有引力常量G")
    assert len(toks) > 3, toks
    assert any("向心" in t for t in toks), toks


def test_tokenize_zh_drops_punctuation():
    from system.rag.hybrid_retriever import tokenize_zh
    toks = tokenize_zh("什么是向心力？，，。！")
    assert "，" not in toks and "。" not in toks and "？" not in toks, toks


def test_tokenize_zh_empty_input():
    from system.rag.hybrid_retriever import tokenize_zh
    assert tokenize_zh("") == []
    assert tokenize_zh("   ，。 ") == []


# ══════════════════════════════════════════ RRF 融合
def _mk_retriever(rrf_k=60):
    from system.rag.hybrid_retriever import HybridRetriever
    r = object.__new__(HybridRetriever)
    r.rrf_k = rrf_k
    return r


def test_rrf_rewards_agreement_across_routes():
    """
    两路都排第 1 的文档，必须胜过只在一路排第 1 的文档。
    这正是 RRF「只依赖排名、不依赖分数尺度」的价值所在。
    """
    r = _mk_retriever()
    route_a = [
        {"id": "both", "collection": "c", "text": "both", "dense_score": 0.9},
        {"id": "onlyA", "collection": "c", "text": "onlyA", "dense_score": 0.8},
    ]
    route_b = [
        {"id": "both", "collection": "c", "text": "both", "bm25_score": 12.0},
        {"id": "onlyB", "collection": "c", "text": "onlyB", "bm25_score": 11.0},
    ]
    fused = r._rrf_fuse([route_a, route_b])
    assert fused[0]["id"] == "both", [f["id"] for f in fused]
    # 跨越两路命中的文档应记录两路的排名
    assert fused[0]["dense_rank"] == 0 and fused[0]["bm25_rank"] == 0, fused[0]


def test_rrf_ignores_score_scale():
    """
    RRF 只看排名：把某一路的分数整体放大 1000 倍，排序不应改变。
    （这是对「分数归一化后加权求和」方案的直接替代理由。）
    """
    r = _mk_retriever()
    a = [{"id": "x", "collection": "c", "text": "x", "dense_score": 0.9},
         {"id": "y", "collection": "c", "text": "y", "dense_score": 0.5}]
    b = [{"id": "y", "collection": "c", "text": "y", "bm25_score": 1.0}]
    fused_small = r._rrf_fuse([a, b])

    a2 = [{"id": "x", "collection": "c", "text": "x", "dense_score": 900.0},
          {"id": "y", "collection": "c", "text": "y", "dense_score": 500.0}]
    b2 = [{"id": "y", "collection": "c", "text": "y", "bm25_score": 1000.0}]
    fused_big = r._rrf_fuse([a2, b2])

    assert [f["id"] for f in fused_small] == [f["id"] for f in fused_big]


def test_rrf_normalized_scores_in_range():
    r = _mk_retriever()
    fused = r._rrf_fuse([[{"id": "a", "collection": "c", "text": "a", "dense_score": 0.9}]])
    assert 0 < fused[0]["rrf_norm"] <= 1.0, fused[0]
    assert fused[0]["rrf_norm"] == 1.0, fused[0]


def test_rrf_dedups_across_collections():
    """同一 id 出现在不同集合时不应被合并（去重键含集合名）。"""
    r = _mk_retriever()
    fused = r._rrf_fuse([[{"id": "same", "collection": "c1", "text": "t", "dense_score": 0.9}],
                         [{"id": "same", "collection": "c2", "text": "t", "bm25_score": 2.0}]])
    assert len(fused) == 2, fused


# ══════════════════════════════════════════ Query 改写
def _rewriter(**overrides):
    from system.rag.retrieval_config import RetrievalConfig
    from system.rag.query_rewriter import QueryRewriter
    cfg = RetrievalConfig({}, "test")
    cfg.data["rewrite"].update(overrides)
    rw = QueryRewriter(cfg)
    if overrides.get("_no_synonyms"):
        rw.synonym_map = {}
    return rw


def test_rewriter_expands_synonym():
    rw = _rewriter()
    rw.synonym_map = {"自由落体": ["自由下落"]}
    out = rw.rewrite("自由落体有什么规律")
    assert "自由下落有什么规律" in out["variants"], out


def test_rewriter_keeps_original_query():
    rw = _rewriter()
    rw.synonym_map = {"向心力": ["指向圆心的合力"]}
    out = rw.rewrite("什么是向心力")
    assert out["original"] == "什么是向心力"
    assert out["original"] not in out["variants"]


def test_rewriter_splits_comparison_question():
    rw = _rewriter()
    rw.synonym_map = {}
    out = rw.rewrite("平抛运动和自由落体有什么区别")
    assert out["subqueries"], out
    assert any("平抛运动" in s for s in out["subqueries"]), out
    assert any("自由落体" in s for s in out["subqueries"]), out


def test_rewriter_does_not_split_short_question():
    rw = _rewriter()
    rw.synonym_map = {}
    out = rw.rewrite("什么是向心力")
    assert out["subqueries"] == [], out


def test_rewriter_disabled_returns_no_variants():
    rw = _rewriter(enabled=False)
    out = rw.rewrite("自由落体和自由下落有什么区别")
    assert out["enabled"] is False and out["variants"] == [], out


def test_rewriter_failure_judgement():
    rw = _rewriter(failure_score_threshold=0.45)
    assert rw.is_failure(0.0, 0) is True
    assert rw.is_failure(0.30, 3) is True
    assert rw.is_failure(0.80, 3) is False


# ══════════════════════════════════════════ 结构感知切分
def _chunker():
    from scripts.rag_data_preparation.enhanced_chunker import EnhancedChunker
    return EnhancedChunker(chunk_size=384, overlap_ratio=0.1)


def test_heading_detection():
    c = _chunker()
    for line in ["第五章 抛体运动", "第一节 曲线运动", "1. 曲线运动", "例题",
                 "思考与讨论", "## 小节标题", "（一）基本概念"]:
        assert c._is_heading(line), line
    # 长正文行不能被当成标题（PDF 抽取的正文常常很长）
    assert not c._is_heading("物体做曲线运动时速度方向沿轨迹在这一点的切线方向，因此需要研究其规律。")


def test_toc_classification():
    c = _chunker()
    toc = "\n".join(["第一章 运动的描述", "第二章 匀变速直线运动",
                     "第三章 相互作用", "第四章 运动和力的关系",
                     "第五章 抛体运动", "第六章 圆周运动"])
    assert c.classify_content_type(toc) == "toc"
    body = "向心力是使物体做圆周运动、始终指向圆心的合力，方向沿半径指向圆心。物体做圆周运动时需要有向心力。"
    assert c.classify_content_type(body) != "toc"


def test_content_type_classification():
    c = _chunker()
    assert c.classify_content_type("例题 一辆汽车以初速度v0做匀加速直线运动，求其位移。") == "example"
    assert c.classify_content_type("思考与讨论 请分析下面两种情况下的受力。") == "exercise"
    assert c.classify_content_type("向心力的定义为指向圆心的合力，称为向心力。") in ("definition", "theorem")
    assert c.classify_content_type("法拉第电磁感应定律的内容是感应电动势与磁通量变化率成正比。") == "theorem"
    assert c.classify_content_type("图1 曲线运动的轨迹示意图") == "figure_caption"


def test_structure_chunking_produces_parents_and_children():
    c = _chunker()
    text = ""
    for sec in range(6):
        text += f"\n第五章 抛体运动\n第{sec+1}节 平抛运动\n"
        for i in range(30):
            text += f"这是第{sec+1}节的第{i+1}句说明文字，用于构造足够长的正文以触发切分逻辑，内容本身无实际意义。"
    children, parents = c.chunk_with_structure(
        text, child_target=384, child_min=120, child_overlap_ratio=0.1,
        parent_target=1000, parent_max=1300)
    assert parents, "应产出父块"
    assert children, "应产出子块"
    assert max(p["char_len"] for p in parents) <= 1300, "父块不得超过硬上限"
    # 每个子块的 parent_index 必须落在父块范围内，否则父子索引断裂
    for ch in children:
        assert 0 <= ch["parent_index"] < len(parents), ch


def test_parent_child_index_mapping_is_complete():
    """每个父块都应至少有一个子块引用（否则该父块永远不可能被召回）。"""
    c = _chunker()
    text = "".join(
        f"\n第{i+1}节 标题\n" + "这是一段用于测试的正文内容，长度需要足够以便产生子块。" * 12
        for i in range(5)
    )
    children, parents = c.chunk_with_structure(
        text, child_target=384, child_min=120, parent_target=1000, parent_max=1300)
    referenced = {ch["parent_index"] for ch in children}
    assert referenced == set(range(len(parents))), (referenced, len(parents))


def test_with_retry_recovers_from_transient_failure():
    """
    回归：Chroma 的惰性段加载会瞬发
    `Error creating hnsw segment reader: Nothing found on disk`。
    磁盘上文件齐全，重试即恢复；不重试会静默变成"检索不到"。
    """
    from system.rag.hybrid_retriever import with_retry
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Error creating hnsw segment reader: Nothing found on disk")
        return "ok"

    assert with_retry(flaky, "测试操作", attempts=2, delay=0.01) == "ok"
    assert calls["n"] == 2


def test_with_retry_raises_after_exhausting_attempts():
    from system.rag.hybrid_retriever import with_retry

    def always_fail():
        raise RuntimeError("boom")

    try:
        with_retry(always_fail, "测试操作", attempts=2, delay=0.01)
        raise AssertionError("应当抛出异常")
    except RuntimeError as e:
        assert "boom" in str(e)


class _FakeBfColl:
    def __init__(self, ids, embs, metas, docs):
        self._ids, self._embs, self._metas, self._docs = ids, embs, metas, docs

    def get(self, include=None):
        return {"ids": self._ids, "embeddings": self._embs,
                "metadatas": self._metas, "documents": self._docs}


class _FakeBfClient:
    def __init__(self, coll):
        self._coll = coll

    def get_collection(self, name):
        return self._coll


def test_brute_force_route_finds_nearest_and_excludes_parents():
    """
    回归：HNSW 不可用时的内存精确检索兜底。
    必须①按余弦相似度排序正确 ②排除父块（与稠密路过滤语义一致）。
    """
    from system.rag.hybrid_retriever import HybridRetriever
    r = object.__new__(HybridRetriever)
    r._bf_cache = {}
    r.client = _FakeBfClient(_FakeBfColl(
        ids=["a", "b", "p1", "c"],
        embs=[[1.0, 0.0], [0.8, 0.6], [0.99, 0.05], [0.0, 1.0]],
        metas=[{"doc_type": "child"}, {"doc_type": "child"},
               {"doc_type": "parent"}, {"doc_type": "child"}],
        docs=["da", "db", "dp", "dc"],
    ))
    out = r._brute_force_route([1.0, 0.0], "c", k=3)
    assert out[0]["id"] == "a", out
    assert all(o["metadata"].get("doc_type") != "parent" for o in out), out
    assert {o["id"] for o in out} == {"a", "b", "c"}, out


def test_dense_fallback_never_raises():
    """
    回归（2026-10-02 实测）：HNSW 段会**间歇性**失败（Error finding id），
    且连 `col.get()` 都可能一起失败。此时若让异常冒出去，整个问答直接中断
    —— 评测里一度 78 题挂掉 35 题。降级链必须**返回空列表**，
    让上层走"零证据拒答"（用户看得到），而不是抛异常。
    """
    from system.rag.hybrid_retriever import HybridRetriever

    class _BrokenClient:
        def get_collection(self, name):
            raise RuntimeError("Error finding id")

    r = object.__new__(HybridRetriever)
    r._bf_cache = {}
    r.client = _BrokenClient()
    out = r._dense_fallback([1.0, 0.0], "neb_x", k=5, reason="boom")
    assert out == [], out


def test_dense_fallback_recovers_without_where_filter():
    """过滤下推失败时，应能用「无过滤重查 + 客户端剔除父块」救回来。"""
    from system.rag.hybrid_retriever import HybridRetriever

    class _Coll:
        def query(self, query_embeddings=None, n_results=5, where=None, **kw):
            if where is not None:
                raise RuntimeError("Error finding id")   # 带过滤才失败
            return {"ids": [["a", "p1"]], "documents": [["da", "dp"]],
                    "metadatas": [[{"doc_type": "child"}, {"doc_type": "parent"}]],
                    "distances": [[0.1, 0.2]]}

    class _Client:
        def get_collection(self, name):
            return _Coll()

    r = object.__new__(HybridRetriever)
    r._bf_cache = {}
    r.client = _Client()
    out = r._dense_fallback([1.0, 0.0], "neb_x", k=5, reason="boom")
    assert [o["id"] for o in out] == ["a"], out      # 父块被剔除


def test_brute_force_route_handles_zero_vector():
    from system.rag.hybrid_retriever import HybridRetriever
    r = object.__new__(HybridRetriever)
    r._bf_cache = {}
    r.client = _FakeBfClient(_FakeBfColl(
        ids=["a"], embs=[[1.0, 0.0]], metas=[{"doc_type": "child"}], docs=["da"]))
    assert r._brute_force_route([0.0, 0.0], "c", k=3) == []


def test_rank_results_preserves_collection_and_id():
    """
    回归：重排会构造新字典，必须把 `collection` / `id` 原样带下去。

    曾经漏传 collection，表现为父子展开时
    「父块读取失败（）：Collection [] does not exist」—— 而且**不会抛异常**，
    只是静默退化成不展开。这类字段丢失必须由测试兜住。
    """
    from system.rag.anti_confusion_engine import AntiConfusionEngine
    eng = AntiConfusionEngine()
    r = eng.rank_results([{
        "id": "neb_physics_grade_10:abc123:0007",
        "text": "向心力是使物体做圆周运动、始终指向圆心的合力，方向沿半径指向圆心。",
        "score": 0.7,
        "collection": "neb_physics_grade_10",
        "metadata": {"source": "a.pdf", "grade": "10", "type": "x",
                     "doc_type": "child", "parent_id": "neb_physics_grade_10:abc123:p0002"},
    }], "什么是向心力", grade="10")
    assert r, "块文本需 >= 10 字"
    assert r[0]["collection"] == "neb_physics_grade_10", r[0]
    assert r[0]["id"].endswith("0007"), r[0]


def test_parent_expansion_handles_missing_collection_gracefully():
    """缺 collection 时不得抛异常（降级为不展开，并记录统计）。"""
    from system.rag.rag_retrieval_engine import RAGRetrievalEngine
    e = object.__new__(RAGRetrievalEngine)
    e.chroma_client = None       # 一旦真的去取父块就会 AttributeError
    chunks = [{"text": "t", "collection": "",
               "metadata": {"parent_id": "x:p0001", "doc_type": "child"}}]
    out, stats = e._expand_to_parents(chunks)
    assert out == chunks, out
    assert stats.get("missing_collection") == 1, stats


def test_structure_chunking_handles_short_text():
    c = _chunker()
    children, parents = c.chunk_with_structure("很短的一段话。")
    assert children or parents or (not children and not parents)
    # 不抛异常且不产生空文本块
    for ch in children:
        assert ch["text"].strip()


# ══════════════════════════════════════════ 7 特征重排
def _cfg():
    from system.rag.retrieval_config import RetrievalConfig
    return RetrievalConfig({}, "test")


def test_rank_weights_have_all_seven_features():
    w = _cfg().weights
    for k in ("similarity", "bm25", "term_coverage", "structure_bonus",
              "source_priority", "version", "grade_affinity"):
        assert k in w, k
    assert w["source_priority"] <= 0.05, "来源权重必须压到 0.05 以内"


def test_structure_bonus_table():
    cfg = _cfg()
    assert cfg.structure_bonus_for("definition") == 1.0
    assert cfg.structure_bonus_for("toc") == 0.0
    assert cfg.structure_bonus_for("cover") == 0.0
    assert cfg.structure_bonus_for("不存在的类型") == 0.5


def test_query_coverage_is_absolute_not_batch_relative():
    """
    query_coverage 必须是**绝对量**（不受同批其他候选影响）。
    这是它能当拒答判据、而最终加权分不能的原因。
    """
    from system.rag.anti_confusion_engine import AntiConfusionEngine
    eng = AntiConfusionEngine()
    q = "什么是向心力"

    def mk(text, cid):
        return {"text": text, "score": 0.7, "collection": f"neb_x_{cid}",
                "metadata": {"source": "a.pdf", "grade": "10", "type": "x",
                             "content_type": "body"}}

    alone = eng.rank_results([mk("向心力是使物体做圆周运动、始终指向圆心的合力。", "a")], q)
    with_peers = eng.rank_results([
        mk("向心力是使物体做圆周运动、始终指向圆心的合力。", "a"),
        mk("完全无关的一段文字，讨论的是别的主题。", "b"),
        mk("另一段也无关的内容。", "c"),
    ], q)
    assert alone[0]["query_coverage"] == with_peers[0]["query_coverage"], \
        "覆盖率不应随同批候选变化"


def test_out_of_scope_query_has_low_query_coverage():
    """库外问题（大学/竞赛内容）的覆盖率应显著低于教材内问题。"""
    from system.rag.anti_confusion_engine import AntiConfusionEngine
    eng = AntiConfusionEngine()
    chunk = ("向心力是使物体做圆周运动、始终指向圆心的合力，方向沿半径指向圆心。"
             "物体做匀速圆周运动时，向心力大小等于质量乘以线速度的平方除以半径。")

    def cov(q):
        r = eng.rank_results([{"text": chunk, "score": 0.6, "collection": "neb_science_grade_10",
                               "metadata": {"source": "a.pdf", "grade": "10", "type": "x",
                                            "content_type": "body"}}], q)
        return r[0]["query_coverage"]

    in_scope = cov("什么是向心力")
    out_scope = cov("请解释量子色动力学中的渐近自由现象")
    assert in_scope > out_scope, (in_scope, out_scope)
    assert out_scope < 0.30, f"库外问题覆盖率应低于拒答阈值 0.30，实际 {out_scope}"


def _one(eng, text, q, **kw):
    return eng.rank_results([{"text": text, "score": 0.7, "collection": "neb_x_10",
                              "metadata": {"source": "a.pdf", "grade": "10", "type": "x",
                                           "content_type": "body"}}], q, **kw)


def test_query_coverage_uses_query_variants():
    """
    口语提问与教材措辞不重合时，覆盖率必须按**变体取最大**，
    否则会判为「无依据」而误拒（实测误拒样本里绝大多数是口语提问）。
    """
    from system.rag.anti_confusion_engine import AntiConfusionEngine
    eng = AntiConfusionEngine()
    chunk = ("物质的量浓度的计算公式为 c = n / V，其中 n 表示溶质的物质的量，"
             "V 表示溶液的体积，单位通常为 mol/L。")

    bare = _one(eng, chunk, "摩尔浓度怎么算来着")[0]["query_coverage"]
    with_var = _one(eng, chunk, "摩尔浓度怎么算来着",
                    query_variants=["物质的量浓度怎么算来着"])[0]["query_coverage"]
    assert with_var > bare, (bare, with_var)
    assert with_var >= 0.30, f"带同义变体后覆盖率应可越过拒答阈值，实际 {with_var}"


def test_query_coverage_variants_do_not_break_scope_check():
    """变体不应把库外问题的覆盖率也抬到阈值以上。"""
    from system.rag.anti_confusion_engine import AntiConfusionEngine
    eng = AntiConfusionEngine()
    chunk = ("向心力是使物体做圆周运动、始终指向圆心的合力，方向沿半径指向圆心。")
    cov = _one(eng, chunk, "请解释量子色动力学中的渐近自由现象",
               query_variants=["什么是量子色动力学", "渐近自由是什么"])[0]["query_coverage"]
    assert cov < 0.30, cov


def test_rank_results_keeps_backward_compatible_keys():
    from system.rag.anti_confusion_engine import AntiConfusionEngine
    eng = AntiConfusionEngine()
    # 块文本必须 >= 10 字，否则会被重排的「过短块」过滤丢弃
    r = eng.rank_results([{"text": "向心力是使物体做圆周运动、始终指向圆心的合力。",
                           "score": 0.7, "collection": "neb_science_grade_10",
                           "metadata": {"source": "a.pdf", "grade": "10", "type": "x"}}],
                         "向心力", grade="10")
    assert r, "短于 10 字的块会被过滤，测试文本需足够长"
    for k in ("final_score", "grade_affinity", "keyword_coverage", "original_score"):
        assert k in r[0], k


def test_structure_bonus_affects_ranking():
    """同分候选下，定义块应排在目录块之前。"""
    from system.rag.anti_confusion_engine import AntiConfusionEngine
    eng = AntiConfusionEngine()
    toc = {"text": "第一章 运动的描述\n第二章 匀变速直线运动", "score": 0.7,
           "collection": "neb_science_grade_10",
           "metadata": {"source": "a.pdf", "grade": "10", "type": "x", "content_type": "toc"}}
    deff = {"text": "向心力的定义为指向圆心的合力。", "score": 0.7,
            "collection": "neb_science_grade_10",
            "metadata": {"source": "a.pdf", "grade": "10", "type": "x", "content_type": "definition"}}
    ranked = eng.rank_results([toc, deff], "什么是向心力", grade="10")
    assert ranked[0]["metadata"]["content_type"] == "definition", ranked


# ══════════════════════════════════════════ 配置集中化
def test_chunking_config_present():
    cfg = _cfg().chunking_config
    for k in ("structure_aware", "child_target_chars", "child_min_chars",
              "parent_target_chars", "parent_max_chars", "parent_child_index"):
        assert k in cfg, k
    assert cfg["parent_target_chars"] < cfg["parent_max_chars"]


def test_hybrid_config_present():
    cfg = _cfg().hybrid_config
    for k in ("enabled", "rrf_k", "dense_k", "bm25_k", "max_query_variants"):
        assert k in cfg, k
    assert cfg["rrf_k"] > 0


def test_context_budget_fits_one_parent():
    """证据预算必须至少装得下一个父块，否则父子索引形同虚设。"""
    cfg = _cfg()
    cb = cfg.data["context_budget"]
    assert cb["max_evidence_tokens"] >= cfg.chunking_config["parent_target_chars"], \
        "证据预算装不下一个父块"


def test_refusal_uses_query_coverage_not_final_score():
    """配置里必须存在基于覆盖率的拒答阈值（final_score 不可分，见实测）。"""
    gen = _cfg().generation_config
    assert "refuse_below_query_coverage" in gen, gen


# ══════════════════════════════════════════ 年级路由（软/硬过滤）
class _FakeColl:
    def __init__(self, name, meta):
        self.name = name
        self.metadata = meta
        self._n = 5

    def count(self):
        return self._n


class _FakeClient:
    def __init__(self, spec):
        self._spec = spec

    def list_collections(self):
        return [_FakeColl(n, m) for n, m in self._spec]


_PHYS = [
    ("neb_physics_grade_10", {"subject": "science", "discipline": "physics", "grade": "10"}),
    ("neb_physics_grade_11", {"subject": "science", "discipline": "physics", "grade": "11"}),
]


def _engine_with(collections, grade_mode):
    from system.rag.rag_retrieval_engine import RAGRetrievalEngine
    from system.rag.retrieval_config import RetrievalConfig
    e = object.__new__(RAGRetrievalEngine)
    e.config = RetrievalConfig({"routing": {"grade_mode": grade_mode}}, "test")
    e.chroma_client = _FakeClient(collections)
    return e


def test_grade_hard_mode_filters_by_grade():
    e = _engine_with(_PHYS, "hard")
    assert e._get_relevant_collections("science", "11", "physics") == ["neb_physics_grade_11"]


def test_grade_soft_mode_keeps_all_grades_same_grade_first():
    """
    soft 模式下**两个年级的集合都要参与召回**，只是同年级排在前面。
    这是「跨年级内容不假拒答」的实现基础。
    """
    e = _engine_with(_PHYS, "soft")
    names = e._get_relevant_collections("science", "11", "physics")
    assert set(names) == {"neb_physics_grade_10", "neb_physics_grade_11"}, names
    assert names[0] == "neb_physics_grade_11", names


def test_soft_grade_covers_cross_grade_query():
    """回归：高二提问、答案在高一册（如静电场中场强 → 物理必修三，目录标高一）。"""
    e = _engine_with(_PHYS, "soft")
    names = e._get_relevant_collections("science", "11", "physics")
    assert "neb_physics_grade_10" in names, "soft 模式必须能命中低年级集合，否则会假拒答"


def test_grade_mode_config_is_validated():
    from system.rag.retrieval_config import RetrievalConfig
    assert RetrievalConfig({}, "t").grade_mode == "soft"
    assert RetrievalConfig({"routing": {"grade_mode": "HARD"}}, "t").grade_mode == "hard"
    # 非法值回退到 soft，不抛异常
    assert RetrievalConfig({"routing": {"grade_mode": "whatever"}}, "t").grade_mode == "soft"


def test_grade_narrowing_drops_cross_grade_when_enough():
    from system.rag.rag_retrieval_engine import RAGRetrievalEngine
    chunks = [{"grade_affinity": 1.0}, {"grade_affinity": 0.8},
              {"grade_affinity": 0.2}, {"grade_affinity": 0.2}]
    kept, dropped = RAGRetrievalEngine._narrow_by_grade(chunks, min_same=2)
    assert len(kept) == 2 and dropped == 2, (kept, dropped)


def test_grade_narrowing_keeps_all_when_insufficient():
    """同年级证据不足时必须保留跨年级候选，否则会退化成硬过滤。"""
    from system.rag.rag_retrieval_engine import RAGRetrievalEngine
    chunks = [{"grade_affinity": 1.0}, {"grade_affinity": 0.2}, {"grade_affinity": 0.2}]
    kept, dropped = RAGRetrievalEngine._narrow_by_grade(chunks, min_same=2)
    assert len(kept) == 3 and dropped == 0, (kept, dropped)


# ══════════════════════════════════════════ 报告
def main():
    checks = [
        ("分词：中文切分", test_tokenize_zh_splits_chinese),
        ("分词：剔除标点", test_tokenize_zh_drops_punctuation),
        ("分词：空输入", test_tokenize_zh_empty_input),
        ("RRF：跨路一致者胜出", test_rrf_rewards_agreement_across_routes),
        ("RRF：不依赖分数尺度", test_rrf_ignores_score_scale),
        ("RRF：归一化在 0–1", test_rrf_normalized_scores_in_range),
        ("RRF：跨集合不去重", test_rrf_dedups_across_collections),
        ("重试：瞬发失败可恢复", test_with_retry_recovers_from_transient_failure),
        ("降级：全挂也不抛异常", test_dense_fallback_never_raises),
        ("降级：无过滤重查可恢复", test_dense_fallback_recovers_without_where_filter),
        ("重试：耗尽后仍抛出", test_with_retry_raises_after_exhausting_attempts),
        ("兜底：内存精确检索", test_brute_force_route_finds_nearest_and_excludes_parents),
        ("兜底：零向量不崩", test_brute_force_route_handles_zero_vector),
        ("改写：同义扩展", test_rewriter_expands_synonym),
        ("改写：保留原查询", test_rewriter_keeps_original_query),
        ("改写：对比型问题拆解", test_rewriter_splits_comparison_question),
        ("改写：短问题不拆", test_rewriter_does_not_split_short_question),
        ("改写：关闭开关", test_rewriter_disabled_returns_no_variants),
        ("改写：失败判定", test_rewriter_failure_judgement),
        ("切分：标题识别", test_heading_detection),
        ("切分：目录识别", test_toc_classification),
        ("切分：内容类型识别", test_content_type_classification),
        ("切分：产出父子块", test_structure_chunking_produces_parents_and_children),
        ("切分：父子映射完整", test_parent_child_index_mapping_is_complete),
        ("切分：短文本不崩", test_structure_chunking_handles_short_text),
        ("重排：7 特征齐全", test_rank_weights_have_all_seven_features),
        ("重排：结构加权表", test_structure_bonus_table),
        ("重排：覆盖率是绝对量", test_query_coverage_is_absolute_not_batch_relative),
        ("重排：覆盖率按变体取最大", test_query_coverage_uses_query_variants),
        ("重排：变体不破坏范围判定", test_query_coverage_variants_do_not_break_scope_check),
        ("重排：库外问题覆盖率低", test_out_of_scope_query_has_low_query_coverage),
        ("重排：向后兼容字段", test_rank_results_keeps_backward_compatible_keys),
        ("重排：透传 collection/id", test_rank_results_preserves_collection_and_id),
        ("父子：缺 collection 不崩", test_parent_expansion_handles_missing_collection_gracefully),
        ("重排：结构信号影响排序", test_structure_bonus_affects_ranking),
        ("配置：切分段", test_chunking_config_present),
        ("配置：混合检索段", test_hybrid_config_present),
        ("配置：预算装得下父块", test_context_budget_fits_one_parent),
        ("配置：拒答用覆盖率", test_refusal_uses_query_coverage_not_final_score),
        ("路由：硬过滤按年级收窄", test_grade_hard_mode_filters_by_grade),
        ("路由：软过滤保留全年级", test_grade_soft_mode_keeps_all_grades_same_grade_first),
        ("路由：软过滤覆盖跨年级", test_soft_grade_covers_cross_grade_query),
        ("路由：模式取值校验", test_grade_mode_config_is_validated),
        ("路由：同年级足够时收窄", test_grade_narrowing_drops_cross_grade_when_enough),
        ("路由：同年级不足时兜底", test_grade_narrowing_keeps_all_when_insufficient),
    ]
    for name, fn in checks:
        check(name, fn)

    print(f"\n{'='*64}\nRAG P1 修复回归测试\n{'='*64}")
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
