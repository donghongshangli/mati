"""
RAG P0「止血」阶段修复的回归测试。

只覆盖**不依赖 chromadb / torch / sentence-transformers** 的纯逻辑，
因此可在无 ML 依赖的解释器下直接运行：
    python tests/test_rag_p0_fixes.py

覆盖的修复点：
  G3 中文置信度恒为 0.3
  E1/E2 双重截断 + break 丢弃高分块
  D4 来源权重（0.30）压过相关性
  F3 年级加权死分支
  G1 validate_grounding 中文失效
  A2/A3 学科别名归一化与集合过滤
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.rag.anti_confusion_engine import AntiConfusionEngine  # noqa: E402
from system.rag.context_budget import estimate_tokens, pack_evidence  # noqa: E402
from system.rag.retrieval_config import RetrievalConfig  # noqa: E402
from system.rag.text_utils import (  # noqa: E402
    cjk_bigrams,
    cjk_length,
    cjk_units,
    compute_confidence,
    content_overlap,
)


# ═══════════════════════════════════════ G3：中文置信度
def test_cjk_units_count_chinese_characters():
    assert cjk_length("向心力") == 3
    # 旧实现 answer.split() 对中文返回 1，这是置信度恒 0.3 的根因
    assert len("向心力是合力".split()) == 1
    assert cjk_length("向心力是合力") == 6


def test_confidence_is_not_constant_for_chinese():
    """回归：中文答案不得恒定返回 0.3。"""
    short = "向心力是合力。"
    medium = "向心力是使物体做圆周运动、始终指向圆心的合力。" * 3
    long = "向心力是使物体做圆周运动、始终指向圆心的合力。" * 12

    scores = [compute_confidence(a, "什么是向心力", []) for a in (short, medium, long)]
    assert len(set(scores)) > 1, f"置信度无区分度：{scores}"
    assert not all(s == 0.3 for s in scores), f"仍恒定 0.3：{scores}"
    assert compute_confidence("", "什么是向心力", []) == 0.0


def test_confidence_reflects_evidence_quality():
    answer = "向心力是使物体做圆周运动、始终指向圆心的合力。"
    low = compute_confidence(answer, "什么是向心力", [{"final_score": 0.3}])
    high = compute_confidence(answer, "什么是向心力", [{"final_score": 0.9}])
    assert high > low, f"证据质量未反映到置信度：low={low} high={high}"
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0


# ═══════════════════════════════════════ E1/E2：上下文预算
def test_estimate_tokens_direction():
    """CJK 每字符约 1 token；ASCII 约 0.3 token/字符（英文更省）。"""
    assert estimate_tokens("你好世界") == 4
    assert estimate_tokens("aa") == 1          # 2 * 0.3 = 0.6 -> 1
    assert estimate_tokens("a" * 100) == 30
    # 同样长度下中文 token 数应显著高于英文
    assert estimate_tokens("中" * 50) > estimate_tokens("a" * 50)


def test_pack_evidence_respects_token_budget():
    chunks = [{"text": "中" * 400, "final_score": 0.9},
              {"text": "中" * 400, "final_score": 0.8}]
    selected, ctx, stats = pack_evidence(chunks, max_tokens=500)
    assert stats["tokens_used"] <= 500, stats
    assert stats["tokens_used"] == sum(c["token_estimate"] for c in selected)
    assert ctx, "上下文不应为空"


def test_pack_evidence_does_not_lose_chunks_after_long_one():
    """
    回归（E2）：原实现用 `break`，一旦某块放不下就丢弃其后所有块，
    即使它们分数更高或体量更小。此处构造「大块在前」的场景验证修复。
    """
    chunks = [
        {"id": "A", "text": "A" * 500, "final_score": 0.95},  # 150 tokens，超预算
        {"id": "B", "text": "B" * 100, "final_score": 0.90},  # 30 tokens
        {"id": "C", "text": "C" * 60,  "final_score": 0.60},  # 18 tokens
        {"id": "D", "text": "D" * 60,  "final_score": 0.55},  # 18 tokens
    ]
    selected, _, stats = pack_evidence(chunks, max_tokens=100)
    got = {c["text"][0] for c in selected}
    assert "C" in got and "D" in got, f"高分/小块被丢弃：{got}"
    assert stats["tokens_used"] <= 100
    assert stats["skipped_too_large"] >= 1


def test_pack_evidence_output_sorted_by_score():
    chunks = [{"text": "中" * 30, "final_score": 0.3},
              {"text": "中" * 30, "final_score": 0.9},
              {"text": "中" * 30, "final_score": 0.6}]
    selected, _, _ = pack_evidence(chunks, max_tokens=200)
    scores = [c["final_score"] for c in selected]
    assert scores == sorted(scores, reverse=True), scores


def test_pack_evidence_never_returns_empty_when_budget_positive():
    """单块超出预算时也要截断保留，绝不返回空上下文，且不得超出预算。"""
    selected, ctx, stats = pack_evidence(
        [{"text": "中" * 5000, "final_score": 0.8}], max_tokens=100
    )
    assert len(selected) == 1 and ctx, stats
    assert stats["tokens_used"] <= 100, stats


# ═══════════════════════════════════════ D4/F3：排序权重
def _mk(text, score, collection, grade="10", source="unknown"):
    return {
        "text": text,
        "score": score,
        "collection": collection,
        "metadata": {"source": source, "grade": grade, "grade_span": grade, "type": "x"},
    }


def test_source_weight_no_longer_dominates_similarity():
    """
    回归（D4）：原实现来源权重占 0.30，一条相似度 0.65 的 NEB 块
    会压过相似度 0.75 但关键词更匹配的外部块。修复后相关性应胜出。
    """
    engine = AntiConfusionEngine()
    neb = _mk("本册教材第五章介绍曲线运动的基本规律与研究方法。", 0.65, "neb_science_grade_10")
    ext = _mk("向心力是使物体做圆周运动、始终指向圆心的合力。", 0.75, "ext_books",
              source="openstax")

    ranked = engine.rank_results([neb, ext], "向心力是什么", grade="10")
    assert ranked[0]["text"].startswith("向心力"), [
        (r["final_score"], r["text"][:12]) for r in ranked
    ]


def test_grade_affinity_is_no_longer_dead_branch():
    """回归（F3）：原分支依赖 query_text 中出现 grade 字符串，恒不命中。"""
    engine = AntiConfusionEngine()
    same = _mk("向心力是使物体做圆周运动、始终指向圆心的合力。", 0.6,
               "neb_science_grade_10", grade="10")
    other = _mk("向心力是使物体做圆周运动、始终指向圆心的合力。", 0.6,
                "neb_science_grade_11", grade="11")
    ranked = engine.rank_results([same, other], "什么是向心力", grade="10")
    assert len(ranked) == 2, ranked
    assert ranked[0]["metadata"]["grade"] == "10", ranked
    assert ranked[0]["grade_affinity"] > ranked[1]["grade_affinity"]


def test_grade_span_is_respected():
    engine = AntiConfusionEngine()
    span = _mk("向心力是使物体做圆周运动、始终指向圆心的合力。", 0.6,
               "neb_physics_grade_10", grade="10")
    span["metadata"]["grade_span"] = "10,11"
    assert engine._grade_affinity(span["metadata"], "11") == 0.8


# ═══════════════════════════════════════ G1：忠实度校验
CONTEXT = [
    "向心力是使物体做圆周运动、始终指向圆心的合力，方向沿半径指向圆心。",
    "物体做曲线运动时，速度方向沿轨迹在这一点的切线方向。",
]


def test_validate_grounding_accepts_grounded_chinese_answer():
    engine = AntiConfusionEngine()
    ok, reason = engine.validate_grounding(
        "向心力是使物体做圆周运动、始终指向圆心的合力，方向沿半径指向圆心。", CONTEXT
    )
    assert ok is True, reason


def test_validate_grounding_rejects_hallucinated_chinese_answer():
    """回归（G1）：原实现用英文 \\b\\w{5,}\\b，中文本上失效。"""
    engine = AntiConfusionEngine()
    ok, reason = engine.validate_grounding(
        "根据常识，向心力其实是由惯性产生的虚拟力，其数值等于质量与加速度平方根的乘积，"
        "在相对论框架下还需要引入洛伦兹因子进行修正，否则计算会完全失准。",
        CONTEXT,
    )
    assert ok is False, f"幻觉答案未被拦截：{reason}"


def test_validate_grounding_handles_empty_inputs():
    engine = AntiConfusionEngine()
    assert engine.validate_grounding("", CONTEXT)[0] is False
    assert engine.validate_grounding("任意答案", [])[0] is False


def test_check_grounding_severity_levels():
    """分级：模型自认无据=严重（加横幅）；覆盖度低=软（仅降置信度）。"""
    engine = AntiConfusionEngine()

    # 有据 -> grounded，不严重
    g = engine.check_grounding(
        "向心力是使物体做圆周运动、始终指向圆心的合力，方向沿半径指向圆心。", CONTEXT)
    assert g["status"] == "grounded" and g["severe"] is False, g

    # 模型自认无据 -> admitted，严重
    a = engine.check_grounding(
        "材料中未找到相关信息。", CONTEXT)
    assert a["status"] == "admitted" and a["severe"] is True, a

    # 无上下文
    n = engine.check_grounding("任意答案", [])
    assert n["status"] == "no_context" and n["severe"] is False, n


def test_check_grounding_low_overlap_is_soft_not_severe():
    """
    回归：内容覆盖度对中文区分力弱（有据答案也可能只有 ~17%），
    因此低覆盖度**不得**触发免责横幅，只能软降置信度。
    """
    engine = AntiConfusionEngine()
    paraphrased = ("导数的几何意义是曲线在某点处切线的斜率，"
                   "也就是函数在该点附近变化快慢的刻画，"
                   "可用于判断单调性并求极值。")
    res = engine.check_grounding(paraphrased, CONTEXT)
    assert res["severe"] is False, res
    assert res["status"] in ("grounded", "weak_overlap"), res


# ═══════════════════════════════════════ G1+：中文内容重叠度量
def test_cjk_bigrams_approximate_word_boundaries():
    assert cjk_bigrams("向心力") == {"向心", "心力"}
    assert cjk_bigrams("向心力是合力") == {"向心", "心力", "力是", "是合", "合力"}
    # 单字串退化为单字
    assert cjk_bigrams("力") == {"力"}
    # 英文按词，不切碎
    assert "vector" in cjk_bigrams("a vector space")


def test_char_level_overlap_is_useless_for_chinese():
    """
    说明为什么必须用二元组：任意两段不相关的中文，单汉字重叠率都很高。
    这是原实现（英文 \\b\\w{5,}\\b 或单汉字）在中文上失效的根本原因。
    """
    a = "量子色动力学中的渐近自由现象"
    unrelated = "向心力是使物体做圆周运动指向圆心的合力"
    char_units = cjk_units(a)
    char_hit = sum(1 for u in char_units if u in unrelated) / len(char_units)
    bigram_hit = content_overlap(a, unrelated)
    assert bigram_hit < char_hit, f"二元组未比单字更严格：{bigram_hit} vs {char_hit}"


def test_content_overlap_separates_grounded_from_hallucinated():
    ctx = "".join(CONTEXT)
    grounded = "向心力是使物体做圆周运动、始终指向圆心的合力，方向沿半径指向圆心。"
    hallucinated = ("量子色动力学中的渐近自由现象指夸克间相互作用随能量升高而减弱，"
                    "该效应由格点规范理论在二十世纪七十年代首次数值验证。")
    g = content_overlap(grounded, ctx)
    h = content_overlap(hallucinated, ctx)
    assert g > h, f"覆盖度未区分：grounded={g} hallucinated={h}"
    assert g >= 0.40, f"有据答案覆盖度偏低：{g}"
    assert h < 0.40, f"幻觉答案覆盖度过高：{h}"


# ═══════════════════════════════════════ A2/A3：学科路由配置
def test_canonical_subject_aliases():
    cfg = RetrievalConfig.load()
    cases = {
        "Science": "science", "物理": "science", "化学": "science",
        "Math": "math", "Mathematics": "math", "数学": "math",
        "Computer Science": "computer_science", "信息技术": "computer_science",
        "English Grammar": "english", "语文": "chinese",
        "Social Studies": "social_studies",
    }
    bad = {k: cfg.canonical_subject(k) for k, v in cases.items()
           if cfg.canonical_subject(k) != v}
    assert not bad, f"学科归一化不符：{bad}"


def test_collection_filter_excludes_readme():
    cfg = RetrievalConfig.load()
    assert cfg.is_collection_allowed("neb_science_grade_10") is True
    assert cfg.is_collection_allowed("neb_readme") is False
    assert cfg.is_collection_allowed("random_other") is False


def test_config_defaults_when_file_missing():
    cfg = RetrievalConfig.load("no/such/retrieval.yaml")
    assert cfg.candidate_k == 20
    assert cfg.min_final_score == 0.40
    assert cfg.weights["source_priority"] <= 0.20, "来源权重必须被压低"


def test_config_candidate_k_not_two():
    """回归（D1）：候选池不得再被压到 2。"""
    cfg = RetrievalConfig.load()
    assert cfg.candidate_k >= 10, cfg.candidate_k
    assert cfg.evidence_k >= 1


# ------------------------------------------------------------ 独立运行入口
def _main():
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
