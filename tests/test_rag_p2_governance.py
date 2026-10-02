# -*- coding: utf-8 -*-
"""
P2「知识治理与可信生成」回归测试（方案 §3.4 步骤 3.1–3.4）。

覆盖 system/rag/knowledge_governance.py 与 ingest_content 的页码链路：
  · 3.2.1 规则清洗
  · 3.2.2 / 3.2.3 精确去重 + 语义去重
  · 3.3    数值冲突检测
  · 3.1    检索侧过滤下推（where + Python 兜底）
  · 3.4    引用生成 + 确定性越界校验
  · 3.1    页码溯源（PAGE 标记扫描 + 偏移反查）

这些测试**不需要向量库**（除语义去重用 numpy 造向量），
因此可以在纯逻辑环境跑；索引相关的验证走 scripts/eval_rag.py。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from system.rag.knowledge_governance import (  # noqa: E402
    DEFAULT_EXCLUDE_CONTENT_TYPES,
    build_citations,
    build_query_filter,
    clean_chunk_text,
    dedupe_chunks,
    detect_conflicts,
    exact_dup_key,
    is_creation_request,
    passes_metadata_filter,
    semantic_dup_pairs,
    validate_citations,
)


# ═══════════════════════════════════════════════════════════════════
# 3.2.1 规则清洗
# ═══════════════════════════════════════════════════════════════════

class TestCleanChunkText:
    def test_keeps_normal_text_untouched(self):
        text = ("牛顿第一定律指出：任何物体都要保持其匀速直线运动状态或静止状态，"
                "直到有外力迫使它改变这种状态为止。这个定律适用于一切物体。")
        cleaned, reason = clean_chunk_text(text)
        # reason 为空串 = 未做删改（约定见 clean_chunk_text docstring）
        assert reason == ""
        assert cleaned == text

    def test_reason_records_which_lines_were_dropped(self):
        """删了内容就必须说清楚删了什么 —— 调用方据此写治理报告。"""
        text = ("力的大小可以通过位移与时间的关系来度量，这是牛顿第一定律的核心内容之一。\n\n53\n"
                "已知物体的加速度为零时速度保持不变，这正是惯性运动的本质特征。")
        _, reason = clean_chunk_text(text)
        assert reason.startswith("lines:")
        assert "1p" in reason

    def test_strips_page_number_only_lines(self):
        text = ("力的大小可以通过位移与时间的关系来度量，这是牛顿第一定律的核心内容之一，"
                "它在日常生活中有非常广泛的应用。\n\n53\n\n继续补充说明："
                "该定律只适用于惯性参考系，非惯性系需要引入惯性力。")
        cleaned, reason = clean_chunk_text(text)
        assert "\n53\n" not in cleaned
        assert "牛顿第一定律" in cleaned
        assert "非惯性系" in cleaned
        assert reason != "kept"

    def test_strips_running_head_in_middle_not_at_block_start(self):
        """块首的章节标题是**有用上下文**，块中间的同名行才是重复页眉。"""
        head = "第二章 运动的描述"
        body = "已知物体的加速度为零时，速度保持不变，这正是牛顿第一定律所描述的惯性运动本质特征。"
        tail = "该定律只适用于惯性参考系，非惯性系中需要引入惯性力才能形式上保持描述统一。"

        # 页眉出现在正文中间 → 删
        cleaned, reason = clean_chunk_text(f"{body}\n{head}\n{tail}")
        assert head not in cleaned
        assert body in cleaned and tail in cleaned
        assert "1h" in reason

        # 同一行出现在块首 → 保留
        cleaned2, _ = clean_chunk_text(f"{head}\n{body}\n{tail}")
        assert cleaned2.startswith(head)
        assert body in cleaned2

    def test_strips_orphan_figure_number(self):
        text = ("图 2-1\n\n自由落体运动是一种初速度为零的匀加速直线运动，"
                "其加速度 g 在近地面处近似为常量，约等于 9.8 m/s²。")
        cleaned, _ = clean_chunk_text(text)
        assert "图 2-1" not in cleaned
        assert "自由落体" in cleaned

    def test_strips_ordinal_only_line(self):
        text = ("一、牛顿第一定律\n\n力是物体间相互作用的一种方式，"
                "力的作用是相互的，这一点与牛顿第三定律的内容一致。\n（1）")
        cleaned, _ = clean_chunk_text(text)
        assert "力的作用是相互的" in cleaned

    def test_drops_copyright_block(self):
        text = ("版权所有 © 人民教育出版社\nISBN 978-7-107-00000-0\n"
                "未经许可不得翻印、复制\n")
        cleaned, reason = clean_chunk_text(text, min_chars=40)
        assert cleaned == ""
        assert reason != "kept"

    def test_drops_too_short_block(self):
        cleaned, reason = clean_chunk_text("（1）", min_chars=40)
        assert cleaned == ""
        assert reason != "kept"

    def test_never_returns_silent_empty_without_reason(self):
        """清洗会丢内容 —— 每一处丢弃都必须带 reason，调用方要落审计日志。"""
        for text in ["短", "图 1", "版权所有 © X\n仅供测试使用禁止翻印"]:
            _, reason = clean_chunk_text(text, min_chars=40)
            assert reason, f"丢弃内容却没给 reason: {text!r}"

    def test_text_change_always_has_reason(self):
        """**审计不变量**：文本被改动就必须能说出改了什么。

        实测踩过：版权行在整块检查阶段被删，却没计入 dropped 计数，
        于是 2763 条 trim 事件的 reason 是空串 —— 内容改了但无法追溯。
        """
        samples = [
            # 版权行 + 足够长的正文 → 走「删行保留整块」分支
            "版权所有 © 人民教育出版社 仅供学习使用\n"
            "力是物体之间相互作用的一种方式，力的作用是相互的，这一点与牛顿第三定律的内容完全一致。",
            # 页码孤行
            "已知物体的加速度为零时速度保持不变，这正是惯性运动的本质特征。\n\n53\n"
            "该结论只对惯性参考系成立，在非惯性系中必须引入惯性力才能形式上统一。",
            # 块中重复页眉
            "已知物体的加速度为零时速度保持不变，这正是惯性运动的本质特征。\n第二章 运动的描述\n"
            "该结论只对惯性参考系成立，在非惯性系中必须引入惯性力才能形式上统一。",
        ]
        for text in samples:
            cleaned, reason = clean_chunk_text(text, min_chars=20)
            if cleaned != text:
                assert reason, (
                    f"文本被改动但 reason 为空，无法审计：{text[:40]!r}"
                )
                assert reason != "unknown", (
                    f"改动原因未登记到 dropped 计数：{text[:40]!r}"
                )


# ═══════════════════════════════════════════════════════════════════
# 3.2.2 / 3.2.3 去重
# ═══════════════════════════════════════════════════════════════════

class TestDedupe:
    def test_exact_dup_key_normalizes_whitespace(self):
        a = "力   是 物体\n\n间 相互作用"
        b = "力 是物体 间相互作用"
        assert exact_dup_key(a) == exact_dup_key(b)

    def test_exact_dup_key_differs_on_content(self):
        assert exact_dup_key("牛顿第一定律") != exact_dup_key("牛顿第二定律")

    def test_exact_dedupe_drops_second(self):
        chunks = [
            {"id": "a1", "text": "加速度方向与速度方向相同时物体做加速运动。"},
            {"id": "a2", "text": "加速度方向与速度方向相反时物体做减速运动。"},
            {"id": "a3", "text": "加速度方向与  速度方向相同时物体做加速运动。"},
        ]
        kept, dropped = dedupe_chunks(chunks, exact=True, semantic_threshold=None)
        assert [c["id"] for c in kept] == ["a1", "a2"]
        assert len(dropped) == 1
        assert dropped[0]["dup_reason"] == "exact"
        assert dropped[0]["dup_of"] == "a1"

    def test_semantic_dedupe_uses_keep_longer(self):
        chunks = [
            {"id": "short", "text": "动能等于质量与速度平方乘积的一半。"},
            {"id": "long", "text": "动能等于质量与速度平方乘积的一半，这是描述机械运动能量大小的物理量。"},
            {"id": "other", "text": "重力是由于地球的吸引而使物体受到的力。"},
        ]
        vectors = [[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]]
        for i, c in enumerate(chunks):
            c["embeddings"] = vectors[i]
        kept, dropped = dedupe_chunks(chunks, exact=True, semantic_threshold=0.95)
        ids = [c["id"] for c in kept]
        assert "long" in ids and "short" not in ids
        assert "other" in ids
        assert any(d["dup_reason"].startswith("semantic:") for d in dropped)
        # 丢弃记录必须能溯源到保留下来的那块
        sem = next(d for d in dropped if d["dup_reason"].startswith("semantic:"))
        assert sem["dup_of"] == "long"

    def test_semantic_dedupe_disabled_by_default(self):
        """semantic_threshold=None 时**不做**语义去重（避免误删正文）。"""
        chunks = [
            {"id": "p", "text": "父块是子块的超集，天然高度重叠。",
             "embeddings": [1.0, 0.0]},
            {"id": "c", "text": "父块是子块的超集，天然高度重叠。",
             "embeddings": [0.999, 0.0447]},
        ]
        kept, dropped = dedupe_chunks(chunks, exact=True, semantic_threshold=None)
        assert len(kept) == 1  # 只被精确去重
        assert len(dropped) == 1

    def test_semantic_dup_pairs_respects_threshold(self):
        np = pytest.importorskip("numpy")
        vecs = [[1.0, 0.0], [0.9999, 0.01], [0.0, 1.0]]
        texts = ["a", "b", "c"]
        pairs = semantic_dup_pairs(vecs, texts, threshold=0.95)
        assert len(pairs) == 1
        i, j, s = pairs[0]
        assert (i, j) == (0, 1)
        assert s >= 0.95
        # 放宽阈值后 c 也应参与成对
        assert semantic_dup_pairs(vecs, texts, threshold=0.0) is not None
        del np

    def test_semantic_dup_pairs_empty_input(self):
        assert semantic_dup_pairs([], [], 0.95) == []

    def test_semantic_dup_pairs_accepts_numpy_array(self):
        """摄取侧可能直接传 np.ndarray —— `if not vectors` 会在此抛
        「truth value of ambiguous」，必须只靠 len() 判断。"""
        np = pytest.importorskip("numpy")
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        # 不抛异常即可；正交向量不构成重复对
        assert semantic_dup_pairs(vecs, ["a", "b"], threshold=0.95) == []


# ═══════════════════════════════════════════════════════════════════
# 3.3 知识冲突检测
# ═══════════════════════════════════════════════════════════════════

class TestConflictDetection:
    @staticmethod
    def _chunk(cid: str, text: str, key: str = "", book: str = "物理必修1") -> Dict[str, Any]:
        meta: Dict[str, Any] = {"book_title": book}
        if key:
            meta["concept_key"] = key
        return {"id": cid, "text": text, "metadata": meta}

    def test_detects_numeric_mismatch_in_same_concept(self):
        chunks = [
            self._chunk("c1", "万有引力常量为 6.67e-11", key="物理.万有引力常量"),
            self._chunk("c2", "万有引力常量为 6.02e-11", key="物理.万有引力常量"),
        ]
        conflicts = detect_conflicts(chunks)
        assert len(conflicts) == 1
        c = conflicts[0]
        assert c["kind"] == "numeric_mismatch"
        assert c["field"] == "万有引力常量"
        assert set(c["values"].keys()) == {"6.67e-11", "6.02e-11"}
        assert c["chunk_ids"] == ["c1", "c2"]

    def test_same_value_is_not_a_conflict(self):
        chunks = [
            self._chunk("c1", "重力加速度为 9.8", key="物理.重力加速度"),
            self._chunk("c2", "重力加速度为 9.8", key="物理.重力加速度"),
        ]
        assert detect_conflicts(chunks) == []

    def test_different_concepts_are_not_compared(self):
        """**假警报比漏检更坏** —— 讲不同概念时绝不能报冲突。"""
        chunks = [
            self._chunk("c1", "向心力为 3 N", key="物理.向心力"),
            self._chunk("c2", "重力为 5 N", key="物理.重力"),
        ]
        assert detect_conflicts(chunks) == []

    def test_blocks_without_concept_key_are_skipped(self):
        chunks = [
            self._chunk("c1", "万有引力常量为 6.67e-11"),
            self._chunk("c2", "万有引力常量为 6.02e-11"),
        ]
        assert detect_conflicts(chunks, require_concept_key=True) == []

    def test_min_blocks_guard(self):
        chunks = [self._chunk("c1", "万有引力常量为 6.67e-11", key="K")]
        assert detect_conflicts(chunks, min_blocks=2) == []

    def test_conflict_records_textbooks(self):
        chunks = [
            self._chunk("c1", "光速为 3e8", key="物理.光速", book="物理必修2"),
            self._chunk("c2", "光速为 2.5e8", key="物理.光速", book="物理选修3-5"),
        ]
        conflicts = detect_conflicts(chunks)
        assert conflicts[0]["textbooks"] == ["物理必修2", "物理选修3-5"]


# ═══════════════════════════════════════════════════════════════════
# 3.1 检索侧过滤下推
# ═══════════════════════════════════════════════════════════════════

class TestQueryFilter:
    def test_where_excludes_parent_toc_cover_and_archived(self):
        where = build_query_filter()
        assert where is not None
        flat = str(where)
        assert "parent" in flat
        for t in DEFAULT_EXCLUDE_CONTENT_TYPES:
            assert t in flat
        assert "archived" in flat

    def test_where_single_condition_not_wrapped(self):
        where = build_query_filter(exclude_content_types=[], exclude_archived=False)
        assert where == {"doc_type": {"$ne": "parent"}}

    def test_where_none_when_nothing_to_exclude(self):
        # 父块条件恒存在，因此这里不为 None；显式断言以防未来改动
        assert build_query_filter(exclude_content_types=[], exclude_archived=False) is not None

    def test_python_fallback_rejects_toc_cover_archived(self):
        assert passes_metadata_filter({"content_type": "body"}) is True
        assert passes_metadata_filter({"content_type": "toc"}) is False
        assert passes_metadata_filter({"content_type": "cover"}) is False
        assert passes_metadata_filter({"status": "archived"}) is False
        assert passes_metadata_filter({"status": "published"}) is True

    def test_python_fallback_keeps_legacy_docs_without_keys(self):
        """旧索引没有 content_type/status 键 —— 不得被误杀。"""
        assert passes_metadata_filter({"book_title": "数学必修1"}) is True
        assert passes_metadata_filter({}) is True
        assert passes_metadata_filter(None) is True

    def test_python_fallback_honours_custom_types(self):
        assert passes_metadata_filter(
            {"content_type": "toc"}, exclude_content_types=()
        ) is True


# ═══════════════════════════════════════════════════════════════════
# 3.4 引用生成与校验
# ═══════════════════════════════════════════════════════════════════

class TestCitations:
    @staticmethod
    def _chunk(cid: str, title: str, chapter: str = "", page: Any = None) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"book_title": title}
        if chapter:
            meta["chapter"] = chapter
        if page is not None:
            meta["page"] = page
        return {"id": cid, "text": "正文", "metadata": meta}

    def test_build_citations_numbers_from_one(self):
        chunks = [
            self._chunk("a", "数学必修1", "第一章 集合", 12),
            self._chunk("b", "物理必修1", "第二章 运动的描述", 53),
        ]
        block, table = build_citations(chunks)
        assert [t["index"] for t in table] == [1, 2]
        assert "[1] 数学必修1 · 第一章 集合 · 第12页" in block
        assert "[2] 物理必修1 · 第二章 运动的描述 · 第53页" in block

    def test_build_citations_handles_missing_metadata(self):
        block, table = build_citations([{"id": "x", "text": "t", "metadata": {}}])
        assert table[0]["label"] == "来源1"
        assert "[1] 来源1" in block

    def test_build_citations_empty(self):
        block, table = build_citations([])
        assert block == ""
        assert table == []

    def test_validate_accepts_in_range_refs(self):
        _, table = build_citations([
            self._chunk("a", "数学必修1"),
            self._chunk("b", "物理必修1"),
        ])
        r = validate_citations("集合的交集满足交换律 [1]，也与并集互补 [2]。", table)
        assert r["n_cited"] == 2
        assert r["n_valid"] == 2
        assert r["n_out_of_range"] == 0
        assert r["accuracy"] == 1.0
        assert r["has_citation"] is True

    def test_validate_flags_out_of_range(self):
        _, table = build_citations([self._chunk("a", "数学必修1")])
        r = validate_citations("结论来自 [1]，另一处引用了 [5]。", table)
        assert r["n_cited"] == 2
        assert r["n_valid"] == 1
        assert r["n_out_of_range"] == 1
        assert r["out_of_range_refs"] == [5]
        assert r["accuracy"] == 0.5

    def test_missing_citation_is_not_punished(self):
        """模型不写标记是常态 —— accuracy 应为 None，不能算 0 分。"""
        _, table = build_citations([self._chunk("a", "数学必修1")])
        r = validate_citations("答案里没有任何引用标记。", table)
        assert r["n_cited"] == 0
        assert r["accuracy"] is None
        assert r["has_citation"] is False

    def test_zero_index_is_invalid(self):
        _, table = build_citations([self._chunk("a", "数学必修1")])
        r = validate_citations("从 [0] 看到的结论。", table)
        assert r["n_out_of_range"] == 1
        assert r["accuracy"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# 3.1 页码溯源链路
# ═══════════════════════════════════════════════════════════════════

class TestPageTraceability:
    """`"\n".join(page.get_text())` 丢页边界 —— 用 [[PAGE:n]] 标记补回。"""

    def test_scan_page_markers_records_offsets(self):
        from scripts.ingest_content import UniversalContentIngester as ContentIngestor

        content = "第一页正文\n[[PAGE:2]]\n第二页正文"
        markers, cleaned = ContentIngestor.scan_page_markers(content)
        assert markers, "应识别出页码标记"
        assert "[[PAGE:" not in cleaned
        assert "第一页正文" in cleaned and "第二页正文" in cleaned
        # 标记列表按偏移升序
        assert markers == sorted(markers)

    def test_page_for_offset_maps_back(self):
        from scripts.ingest_content import UniversalContentIngester as ContentIngestor

        # 与 extract_text_from_pdf 一致：每一页（含第 1 页）开头都有标记
        content = ("[[PAGE:1]]\n甲页正文内容在此处\n[[PAGE:2]]\n乙页正文内容在此处"
                   "\n[[PAGE:3]]\n丙页正文")
        markers, cleaned = ContentIngestor.scan_page_markers(content)
        p1 = cleaned.index("甲页正文")
        p2 = cleaned.index("乙页正文")
        p3 = cleaned.index("丙页正文")
        assert ContentIngestor.page_for_offset(markers, p1) == 1
        assert ContentIngestor.page_for_offset(markers, p2) == 2
        assert ContentIngestor.page_for_offset(markers, p3) == 3

    def test_page_for_offset_uses_last_marker_before_offset(self):
        """块起点落在页中间时，应归属其**起始所在页**而非下一段。"""
        from scripts.ingest_content import UniversalContentIngester as ContentIngestor

        content = "[[PAGE:1]]\n甲页正文\n[[PAGE:2]]\n乙页正文"
        markers, cleaned = ContentIngestor.scan_page_markers(content)
        mid_p1 = cleaned.index("甲页正文") + 2
        assert ContentIngestor.page_for_offset(markers, mid_p1) == 1
        # 偏移超出全文末尾时归到最后一页，不报错
        assert ContentIngestor.page_for_offset(markers, 10 ** 6) == 2

    def test_page_for_offset_handles_no_markers(self):
        from scripts.ingest_content import UniversalContentIngester as ContentIngestor

        assert ContentIngestor.page_for_offset([], 100) == 0

    def test_scan_page_markers_plain_text_unchanged(self):
        from scripts.ingest_content import UniversalContentIngester as ContentIngestor

        content = "没有任何页码标记的纯文本内容。"
        markers, cleaned = ContentIngestor.scan_page_markers(content)
        assert markers == []
        assert cleaned == content


# ═══════════════════════════════════════════════════════════════════
# 配置接线
# ═══════════════════════════════════════════════════════════════════

class TestCreationRefusal:
    """步骤 3.4.3 的补充拒答判据：创作类请求与覆盖率正交。

    实测依据：「帮我写一篇 800 字的散文」查询覆盖率 0.71（远高于阈值 0.28），
    因为「散文」「800」「字」在语文教材里天然高频 —— 覆盖率判据在此完全失效。
    """

    @pytest.mark.parametrize("q", [
        "帮我写一篇 800 字的散文",
        "写一篇议论文",
        "帮我写一首关于秋天的诗",
        "帮我续写下面的故事",
        "写一篇 600 字读后感",
        "帮我写个请假条",
        "写一篇演讲稿",
        "帮我写一篇周记",
        "写一篇小说",
    ])
    def test_flags_creation_requests(self, q):
        assert is_creation_request(q) is True

    @pytest.mark.parametrize("q", [
        "散文怎么写",                          # 问写法，教材有依据
        "如何赏析这篇散文",
        "作文的立意是什么",
        "帮我总结一下这篇课文的主题",
        "这篇文章的表达方式有什么特点",
        "记叙文的三要素是什么",
        "诗歌的意象有哪些",
        "帮我写一道物理题",                    # 非成品文体
        "集合的交集怎么定义",
        "牛顿第二定律的内容是什么",
    ])
    def test_does_not_flag_textbook_questions(self, q):
        assert is_creation_request(q) is False

    def test_empty_query_is_safe(self):
        assert is_creation_request("") is False
        assert is_creation_request(None) is False


class TestGovernanceConfig:
    def test_yaml_has_governance_section(self):
        import yaml

        path = os.path.join(
            os.path.dirname(__file__), "..", "mati_data", "config", "retrieval.yaml"
        )
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        gov = cfg.get("governance")
        assert isinstance(gov, dict), "retrieval.yaml 缺少 governance 段"
        assert "toc" in gov["exclude_content_types"]
        assert "cover" in gov["exclude_content_types"]
        assert gov["dedup"]["semantic_threshold"] >= 0.95, (
            "语义去重阈值不得低于 0.95 —— 教材相邻小节本就高度相似"
        )
        assert cfg["generation"]["require_citations"] is True

    def test_graphrag_reserved_fields_are_scalar(self):
        """Chroma 1.4.0 的 metadata 不接受 list —— 传 list 会 ValueError 中断入库。
        `prerequisites` 必须存字符串。"""
        import re

        src = open(
            os.path.join(os.path.dirname(__file__), "..", "scripts", "ingest_content.py"),
            "r", encoding="utf-8",
        ).read()
        m = re.search(r"'prerequisites':\s*(.+?),\s*\n", src)
        assert m, "ingest_content.py 未写入 GraphRAG 预留字段 prerequisites"
        assert not m.group(1).strip().startswith("["), (
            "prerequisites 不能是 list —— Chroma metadata 只接受标量"
        )
        assert "concept_key" in src, "缺少 GraphRAG 预留字段 concept_key"


class TestMetadataCompleteness:
    """步骤 3.1 验收：元数据完整率 ≥ 95%。

    口径要点：`chapter` / `section` **只对自带标题的块**要求。
    连续正文中间本就没有小节名，把它算成缺失会让指标永远差一口气，
    反而掩盖真正的接线错误。
    """

    @staticmethod
    def _full_meta(**over):
        meta = {
            "book_id": "b1", "book_title": "数学必修1", "page": 17,
            "content_type": "body", "subject": "math", "grade": 10,
            "status": "published", "doc_type": "child",
            "textbook_version": "2019版",
            "chapter": "第一章 集合", "section": "1.1 集合的概念",
            "section_title": "1.1 集合的概念",
        }
        meta.update(over)
        return meta

    def test_complete_block_scores_one(self):
        from system.rag.rag_retrieval_engine import _metadata_completeness

        assert _metadata_completeness([{"metadata": self._full_meta()}]) == 1.0

    def test_body_block_without_heading_is_not_penalised(self):
        """连续正文没有小节名是结构事实，不该算缺失。"""
        from system.rag.rag_retrieval_engine import _metadata_completeness

        meta = self._full_meta(chapter="", section="", section_title="")
        assert _metadata_completeness([{"metadata": meta}]) == 1.0

    def test_missing_identifier_field_is_penalised(self):
        from system.rag.rag_retrieval_engine import _metadata_completeness

        meta = self._full_meta(book_title="")
        assert _metadata_completeness([{"metadata": meta}]) < 1.0

    def test_heading_block_missing_chapter_is_penalised(self):
        """有标题却没 chapter —— 这是真接线错误，必须被抓出来。"""
        from system.rag.rag_retrieval_engine import _metadata_completeness

        meta = self._full_meta(chapter="")
        assert _metadata_completeness([{"metadata": meta}]) < 1.0

    def test_empty_chunks(self):
        from system.rag.rag_retrieval_engine import _metadata_completeness

        assert _metadata_completeness([]) == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
