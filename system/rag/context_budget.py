"""
Context budget manager for Mati RAG.

统一「检索层 → 生成层」的上下文预算，替代原先两处各自截断的做法：
  · rag_retrieval_engine.py 按 1600 **字符** 截断
  · qwen_handler._build_prompt 又按 850 **字符** 截断
导致召回证据有 45% 以上从未到达模型，且第二次截断无任何日志。

本模块按 **token** 预算组装证据，并按「单位 token 价值」贪心选块：
分数高且短的块优先，放不下的块跳过（continue）而不是终止（break），
避免因第 3 块过长而丢弃第 4、5 块更优的证据。
"""

import logging
from typing import Any, Dict, List, Tuple

from system.rag.text_utils import estimate_tokens, is_cjk  # noqa: F401  (re-export)

logger = logging.getLogger(__name__)


def pack_evidence(
    chunks: List[Dict[str, Any]],
    max_tokens: int,
    tokens_per_char_cjk: float = 1.0,
    tokens_per_char_ascii: float = 0.30,
    separator: str = "\n\n",
    max_items: int = None,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    在 token 预算内挑选证据并拼装上下文。

    Args:
        chunks: 候选块，需含 'text'，可选 'final_score'
        max_tokens: 证据部分的 token 预算
        tokens_per_char_cjk / tokens_per_char_ascii: 每字符约等于多少 token
        separator: 块之间的连接串
        max_items: 入选块数上限（对应配置的 evidence_k）。
            「按单位 token 价值贪心」在候选里有大量小块时会倾向于塞满很多块
            （小块的价值 = 分数/长度 天然偏高），这可能把一个 2000 token 的
            预算切成一堆碎片。max_items 是防止这种碎片化的安全阀。

    Returns:
        (selected_chunks, context_str, stats)
        selected_chunks 已按分数降序排列（供溯源展示）；
        stats 含候选数、入选数、token 估算、丢弃原因统计。
    """
    stats: Dict[str, Any] = {
        "candidates": len(chunks or []),
        "selected": 0,
        "tokens_used": 0,
        "tokens_budget": max_tokens,
        "skipped_too_large": 0,
        "skipped_empty": 0,
        "skipped_over_max_items": 0,
        "max_items": max_items,
    }
    if not chunks or max_tokens <= 0:
        return [], "", stats

    def tok(t: str) -> int:
        return estimate_tokens(t, tokens_per_char_cjk, tokens_per_char_ascii)

    # 预处理：算 token 与「单位 token 价值」
    candidates = []
    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            stats["skipped_empty"] += 1
            continue
        t = tok(text)
        if t <= 0:
            stats["skipped_empty"] += 1
            continue
        score = float(c.get("final_score", c.get("score", 0.0)) or 0.0)
        score = max(score, 1e-6)  # 避免负分导致价值为负
        candidates.append({"chunk": c, "text": text, "tokens": t, "score": score,
                           "value": score / t})

    if not candidates:
        return [], "", stats

    # 按单位 token 价值降序贪心；放不下的跳过（continue）而不是终止
    candidates.sort(key=lambda x: x["value"], reverse=True)
    selected, used = [], 0
    for cand in candidates:
        if max_items is not None and len(selected) >= int(max_items):
            stats["skipped_over_max_items"] += 1
            continue
        if used + cand["tokens"] <= max_tokens:
            selected.append(cand)
            used += cand["tokens"]
        else:
            stats["skipped_too_large"] += 1

    # 兜底：一个都放不下时，截取最高分块的开头，绝不让上下文为空。
    # 按 CJK 系数（最保守）反推字符数，保证**不会超出**预算。
    if not selected:
        top = max(candidates, key=lambda x: x["score"])
        keep_chars = max(1, int(max_tokens / max(1e-6, tokens_per_char_cjk)))
        truncated = top["text"][:keep_chars]
        # 尽量在句末对齐
        for sep in ("。", "！", "？", "；", ".", "!", "?"):
            idx = truncated.rfind(sep)
            if idx != -1:
                truncated = truncated[: idx + 1]
                break
        selected = [{"chunk": dict(top["chunk"], text=truncated),
                     "text": truncated, "tokens": tok(truncated), "score": top["score"],
                     "value": top["score"] / max(1, tok(truncated))}]
        used = selected[0]["tokens"]
        logger.warning(
            f"单块超出预算，已截断保留最高分块（{top['tokens']} -> {used} tokens）"
        )

    # 展示顺序按分数降序（与溯源列表一致）
    selected.sort(key=lambda x: x["score"], reverse=True)
    ordered_chunks = [dict(s["chunk"], text=s["text"], token_estimate=s["tokens"])
                      for s in selected]
    context_str = separator.join(s["text"] for s in selected)

    stats["selected"] = len(selected)
    stats["tokens_used"] = used
    return ordered_chunks, context_str, stats
