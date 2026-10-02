"""
Dependency-free text utilities for the Mati RAG layer.

只依赖标准库，便于单独测试（不引入 chromadb / torch / sentence-transformers）。

存在的意义：项目里原本有三处「中文分词/文本度量」的独立实现
（rag_retrieval_engine._calculate_confidence、qwen_handler._units、
anti_confusion_engine.validate_grounding），彼此口径不一致：
  · qwen_handler 用 CJK 感知的 _units（正确）
  · rag_retrieval_engine 的流式路径用 answer.split()（中文恒为 1 个词 → 置信度恒 0.3）
  · anti_confusion 用 \\b\\w{5,}\\b（中文本上失效）
此处统一为唯一实现，避免再次漂移。
"""

import re
from typing import Any, Dict, List, Optional, Set

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_UNIT_RE = re.compile(r"[\u4e00-\u9fff]|[a-z0-9_]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{1,}")


def is_cjk(ch: str) -> bool:
    """判断是否为中文/CJK 区字符（含中文标点与全角符号）。"""
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF        # CJK 统一表意
        or 0x3400 <= o <= 0x4DBF     # 扩展 A
        or 0x3000 <= o <= 0x303F     # CJK 标点
        or 0xFF00 <= o <= 0xFFEF     # 全角
    )


def cjk_units(text: str) -> Set[str]:
    """
    中英通用的分词单元。

    每个汉字算一个单元，每个英文/数字词算一个单元。
    这样 `len(cjk_units(x))` 在中文上近似「字数」，可安全用于长度与重叠率计算。

    ⚠️ 注意：**不要**用本函数做中文的「内容重叠/忠实度」判断 ——
    单个汉字太普遍，任意两段中文都会大量重合，导致判断失效。
    判断内容重叠请用 `cjk_bigrams`。
    """
    if not text:
        return set()
    return set(_UNIT_RE.findall(text.lower()))


def cjk_bigrams(text: str) -> Set[str]:
    """
    中文二元组（bigram）集合，英文按词。

    用途：判断「答案内容是否真的来自给定上下文」。
    单汉字重叠率对中文几乎没有区分度（任意两段中文汉字重合都很高），
    二元组能近似还原中文的词语边界，是无需 jieba 依赖的实用替代。

    例：
        "向心力" -> {"向心", "心力"}
        "向心力是合力" -> {"向心", "心力", "力是", "是合", "合力"}
    """
    if not text:
        return set()
    grams: Set[str] = set()
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            grams.add(run)
        else:
            for i in range(len(run) - 1):
                grams.add(run[i:i + 2])
    # 英文/数字按词（不切碎，避免噪声）
    for tok in _ASCII_WORD_RE.findall(text.lower()):
        if len(tok) >= 2:
            grams.add(tok)
    return grams


def content_overlap(answer: str, context: str) -> float:
    """
    答案内容被上下文覆盖的比例（0–1），基于中文二元组 / 英文词。

    这是 `validate_grounding` 的核心度量。
    """
    a = cjk_bigrams(answer)
    if not a:
        return 1.0
    c = cjk_bigrams(context)
    return len(a & c) / len(a)


def cjk_length(text: str) -> int:
    """
    按 CJK 感知方式计算长度（汉字各计 1，英文词各计 1）。

    注意：不能用 len(cjk_units(...))——那返回的是**去重后的集合**，
    重复字会被漏算（"向心力是合力" 会被算成 5 而不是 6）。
    """
    if not text:
        return 0
    return len(_UNIT_RE.findall(text.lower()))


def estimate_tokens(
    text: str,
    tokens_per_char_cjk: float = 1.0,
    tokens_per_char_ascii: float = 0.30,
) -> int:
    """
    轻量 token 估算（无需 tokenizer 依赖）。

    系数含义是「每个字符约等于多少 token」：
      中文 约 1 字符 ≈ 1.0 token
      英文 约 3.3 字符 ≈ 1 token → 0.30 token/字符
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if is_cjk(ch))
    other = len(text) - cjk
    est = cjk * tokens_per_char_cjk + other * tokens_per_char_ascii
    return int(est + 0.5)


def compute_confidence(
    answer: str,
    question: str,
    context_chunks: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """
    计算答案置信度。

    修复点：原实现用 len(answer.split()) 数词，中文没有空格 →
    整段中文被当成 1 个「词」→ 恒命中「过短」分支 → 置信度永远 0.3。
    这里改用 CJK 感知长度，并把各项改为加权平均
    （原实现直接相加再 min(1.0, ...)，会饱和失真）。

    Factors:
      - 长度项 0.40：30–200 个单元视为合适长度
      - 相关项 0.25：问题单元在答案中的覆盖率
      - 证据项 0.35：入选证据的 final_score 均值
    """
    if not answer:
        return 0.0

    length = cjk_length(answer)
    if length < 5:
        return 0.3

    if 30 <= length <= 200:
        length_score = 1.0
    elif 15 <= length < 30 or 200 < length <= 300:
        length_score = 0.7
    else:
        length_score = 0.4

    q_units = cjk_units(question)
    a_units = cjk_units(answer)
    relevance = len(q_units & a_units) / max(1, len(q_units))

    if context_chunks:
        evidence = sum(float(c.get("final_score", 0.0)) for c in context_chunks) / len(context_chunks)
    else:
        evidence = 0.0

    confidence = 0.40 * length_score + 0.25 * relevance + 0.35 * evidence
    return round(max(0.0, min(1.0, confidence)), 3)
