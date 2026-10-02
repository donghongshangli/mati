"""
知识治理：内容清洗、去重、冲突检测（P2 / 方案 §3.4 步骤 3.1–3.3）。

为什么需要它
────────────
P0/P1 把「检索得到的东西」变干净了（字形修复、中文断句、混合检索），
但**入库内容本身**仍然带着三类问题：

1. **噪声块**：页码孤行、重复页眉页脚、ISBN/版权声明、孤立图注编号。
   它们与真实正文共享学科词汇，语义距离近，容易被召回并占用
   仅 2000 token 的证据预算 —— 而目录/封面类块对回答问题**零贡献**。
2. **重复块**：同一页被相邻窗口重复覆盖、跨册教材的目录重复、
   同一文件被重复摄取。重复内容会稀释 RRF 排名（同一答案占多个位置），
   并让「证据条数」这个指标失真。
3. **知识冲突**：新旧教材版本对同一概念表述不一致时，两块会同时进证据，
   模型看到互相矛盾的说法只会含糊其辞 —— 这是业界 badcase 里占比最高的一类。

本模块的三部分都**只依赖标准库 + numpy（可选）**，可独立测试：

    clean_chunk_text()   规则清洗（步骤 3.2.1）
    dedupe_chunks()      哈希精确去重 + 向量语义去重（步骤 3.2.2/3.2.3）
    detect_conflicts()   同概念块的数值/表述矛盾检测（步骤 3.3.3）

设计上的两条自律
──────────────
- **不做看不见的删除**：所有清洗/去重都返回 `reason`，落盘成可审计的
  报告（`ingest_content.py` 会写 `mati_data/ingest_gov_report.jsonl`）。
  「悄悄丢内容」是这类治理最危险的地方 —— 方案 §6 已列为风险项。
- **阈值可配且偏保守**：宁可漏掉一些噪声，也不要误删正文。
  语义去重默认 0.95（方案给的是 0.92），实测本库几乎没有超过 0.95 的
  重复对，见 `docs/RAG优化-P2实施记录.md` §4.2。
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 一、内容清洗（步骤 3.2.1）
# ═══════════════════════════════════════════════════════════════════

# 版权/ISBN/出版发行块：整块几乎不含教学信息
_COPYRIGHT_RE = re.compile(
    r"(ISBN\s*[\d\-X]{10,17}"
    r"|CIP\s*数据核字"
    r"|版权所有[，,]?\s*(翻|译|编)"
    r"|人民教育出版社"
    r"|出版发行[:：]"
    r"|印刷[:：]"
    r"|责任编辑"
    r"|书\s*号[:：]\s*[A-Z]?\d"
    r"|开\s*本\s*\d"
    r"|字\s*数\s*[\d十百千万]"
    r"|印\s*张\s*\d"
    r"|定\s*价\s*[\d.]+\s*元"
    r"|印刷厂"
    r"|质量监督检测"
    r"|教育部(?:审定|批准)"
    r")"
)

# 页眉/页脚噪声：书名 + 出版社的重复行、纯页码行
_PAGE_NUM_ONLY_RE = re.compile(r"^[\s\-—–·\.]{0,6}\d{1,4}[\s\-—–·\.]{0,6}$")
# 页眉的最大长度：真实页眉都很短（书名/章名），超过就不是页眉而是正文标题
_RUNNING_HEAD_MAX = 24
_RUNNING_HEAD_RE = re.compile(
    r"^(?:第\s*[一二三四五六七八九十\d]+\s*[章节单元][^\n]{0,16}"
    r"|[\u4e00-\u9fff]{2,12}(?:选择性必修|必修)[^\n]{0,12}"
    r"|人教版?[^\n]{0,12}"
    r"|普通高中教科书[^\n]{0,10})$"
)

# 孤立图注编号：「图 3-1」「表 2-3」这类单独成行的编号
_FIGURE_NUM_ONLY_RE = re.compile(r"^(图|表)\s*[\d一二三四五六七八九十]+\s*[-－—]?\s*\d*\s*$")

# 习题答案区的机械编号：「1.」「(2)」「①」等纯序号行
_ORDINAL_ONLY_RE = re.compile(r"^\s*[（(]?\s*(?:\d{1,3}|[①-⑳])\s*[）).、]?\s*$")


def _norm_for_hash(text: str) -> str:
    """归一化文本用于哈希：去空白与标点，避免「换个空格就算新内容」。"""
    return re.sub(r"[\s\u3000]+", "", text or "")


def clean_chunk_text(text: str, *, min_chars: int = 40) -> Tuple[str, str]:
    """
    对单个块做规则清洗。

    Args:
        text: 原始块文本
        min_chars: 清洗后短于该长度则视为「噪声块」返回空串
                  （调用方据此丢弃，并在报告里记 reason）

    Returns:
        (cleaned_text, reason)。reason 为空串表示未做删改；
        否则是删除原因（`noise` / `copyright` / `too_short`），
        调用方**必须**把它记进治理报告 —— 不允许静默丢内容。

    为什么逐行处理而不是整体正则
    ────────────────────────────
    页码孤行、页眉、图注编号都只在**行粒度**上有意义：
    同一块里既有正文行也有页码行，只有删行不能删段。
    整体正则会误伤「（3）加速度的方向」这类正文。
    """
    if not text or not text.strip():
        return "", "noise"

    original = text

    # ① 版权/ISBN 块：整块命中即丢（这类块不含任何可回答问题的内容）
    n_copyright = 0
    if _COPYRIGHT_RE.search(text):
        # 但若正文占比很高（只是顺带提到了「人民教育出版社」），
        # 保留文本、只删掉命中的行，避免误删整页正文。
        lines = [ln for ln in text.split("\n") if not _COPYRIGHT_RE.search(ln)]
        n_copyright = len(text.split("\n")) - len(lines)
        kept = "\n".join(lines).strip()
        if not kept:
            return "", "copyright"
        if len(kept) < len(text) * 0.5:
            return "", "copyright"
        text = kept

    lines_out: List[str] = []
    dropped = {"page": 0, "head": 0, "fig": 0, "ord": 0, "copy": n_copyright}
    seen_content = False   # 是否已输出过正文行
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            lines_out.append("")
            continue
        if _PAGE_NUM_ONLY_RE.match(line):
            dropped["page"] += 1
            continue
        if _FIGURE_NUM_ONLY_RE.match(line):
            dropped["fig"] += 1
            continue
        if _ORDINAL_ONLY_RE.match(line):
            dropped["ord"] += 1
            continue
        if len(line) <= _RUNNING_HEAD_MAX and _RUNNING_HEAD_RE.match(line):
            # 「第二章 运动的描述」这类行有两种身份：
            #   · 出现在**块首** → 它是这一节真正的标题，是有用的上下文，保留；
            #   · 出现在正文中间 → 它是每页重复的页眉，纯噪声，删掉。
            # 不区分就会把每章第一块的标题删掉，白白损失检索上下文。
            if seen_content:
                dropped["head"] += 1
                continue
            seen_content = True
            lines_out.append(line)
            continue
        seen_content = True
        lines_out.append(line)

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines_out)).strip()

    reason = ""
    n_dropped = sum(dropped.values())
    if n_dropped and text:
        reason = (f"lines:{dropped['page']}p/{dropped['head']}h/"
                  f"{dropped['fig']}f/{dropped['ord']}o/{dropped['copy']}c")

    if len(_norm_for_hash(text)) < min_chars:
        return "", "too_short" if text else "noise"

    # 兜底自检：文本被改过却说不出改了哪儿，就是审计漏洞。
    # 实测踩过一次 —— 版权行在 ① 阶段被删，但没计入 dropped，
    # 于是 2763 条 trim 事件的 reason 全是空串，内容改了却无法追溯。
    # 这里让它变成显式的 unknown，而不是静默通过。
    if not reason and text != original:
        logger.warning("清洗改动了文本但未记录原因，已标记为 unknown（请补充 dropped 计数）")
        reason = "unknown"

    return text, reason


# ═══════════════════════════════════════════════════════════════════
# 二、去重（步骤 3.2.2 精确 / 3.2.3 语义）
# ═══════════════════════════════════════════════════════════════════

def exact_dup_key(text: str) -> str:
    """
    精确去重键：`sha256(归一化文本)`。

    归一化会去掉所有空白 —— PDF 抽取的换行/折行差异不该让同一段文字
    被当成两块。这也是方案 §3.4 步骤 3.2.2 说的 sha256 精确去重。
    """
    return hashlib.sha256(_norm_for_hash(text).encode("utf-8")).hexdigest()


def _cosine_matrix(vectors: Sequence[Sequence[float]]):
    """把向量列表归一化成矩阵（numpy 可选）。"""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy 是硬依赖，但保持可降级
        return None
    mat = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def semantic_dup_pairs(
    vectors: Sequence[Sequence[float]],
    texts: Sequence[str],
    threshold: float = 0.95,
) -> List[Tuple[int, int, float]]:
    """
    在已算好的嵌入向量上找语义重复对（步骤 3.2.3）。

    为什么不自己再编码
    ────────────────
    摄取时嵌入已经算过一次；再编码一遍 27 本教材要十几分钟。
    这里直接吃现成向量，成本只是一次矩阵乘法（1.3 万 × 512 维约 0.2 秒）。

    为什么阈值取 0.95 而不是方案写的 0.92
    ────────────────────────────────
    教材里大量段落**本就应该**高度相似：相邻小节的定义句式、
    例题的「解：」引导语、术语的重复定义。0.92 会把它们判成重复并删掉，
    造成正文损失。实测本库 >0.95 的重复对不足 0.5%（见实施记录 §4.2），
    阈值再往上就几乎没有可去重的了 —— 收益与风险的平衡点就在 0.95。

    父块/子块必须**分开比较**：父块是子块的超集，两者天然高度重叠，
    混在一起比较会把整本书的父块全部删光。
    """
    # 只用 len() 判断，不要写 `if not vectors`：
    # 传 numpy 数组时 `not array` 会抛「truth value is ambiguous」。
    if vectors is None or len(vectors) < 2:
        return []
    mat = _cosine_matrix(vectors)
    if mat is None:
        return []
    import numpy as np

    n = mat.shape[0]
    # 逐块计算，避免 n×n 矩阵在万级时占用过多内存（分块后峰值仅 chunk²）
    pairs: List[Tuple[int, int, float]] = []
    block = 512
    for start in range(0, n, block):
        end = min(start + block, n)
        # sim 的行对应全局下标 [start, end)，列对应全局下标 [0, n)
        sim = mat[start:end] @ mat.T
        for i in range(end - start):
            gi = start + i
            for j in range(gi + 1, n):
                # ⚠️ 列索引是**全局** j，不能写成 j - gi：
                # 那样取到的是 sim[i][i]（自己跟自己），恒等于 1.0，
                # 会让每一对相邻块都被判成重复 —— 语义去重变成「删掉大半正文」。
                s = float(sim[i, j])
                if s >= threshold:
                    pairs.append((gi, j, s))
    if not pairs:
        return []
    pairs.sort(key=lambda x: -x[2])
    logger.info(
        f"语义去重：{n} 块中找出 {len(pairs)} 对余弦 ≥ {threshold}"
    )
    return pairs


def dedupe_chunks(
    chunks: List[Dict[str, Any]],
    *,
    exact: bool = True,
    semantic_threshold: Optional[float] = None,
    keep: str = "longer",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    对块列表去重，返回 `(保留, 丢弃)`，每条丢弃都带 `dup_reason` 与 `dup_of`。

    Args:
        chunks: 至少含 `text`；若含 `embeddings`（numpy 数组或 list）则做语义去重
        exact: 是否做 sha256 精确去重
        semantic_threshold: 余弦阈值；None 表示跳过语义去重
        keep: 重复对里保留谁 —— "longer"（默认，留信息更全的）
              / "first"（留先出现的）

    顺序很重要：**先精确后语义**。精确去重零成本、零风险；
    语义去重有误删风险，不该承担「本可以用哈希解决的重复」。
    """
    if not chunks:
        return [], []

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    seen_exact: Dict[str, int] = {}

    for i, c in enumerate(chunks):
        txt = c.get("text") or ""
        if not txt.strip():
            dropped.append({**{k: c.get(k) for k in ("id", "doc_type")},
                            "dup_reason": "empty", "dup_of": ""})
            continue
        if exact:
            k = exact_dup_key(txt)
            if k in seen_exact:
                dropped.append({
                    "id": c.get("id", ""), "doc_type": c.get("doc_type", ""),
                    "dup_reason": "exact", "dup_of": chunks[seen_exact[k]].get("id", ""),
                })
                continue
            seen_exact[k] = i
        c.setdefault("id", "")
        c.setdefault("doc_type", "")
        kept.append(c)

    if semantic_threshold is None or len(kept) < 2:
        return kept, dropped

    # 父块与子块必须分开做语义去重（父块包含子块，混比会全删）
    by_kind: Dict[str, List[int]] = {}
    for idx, c in enumerate(kept):
        by_kind.setdefault(str(c.get("doc_type") or "child"), []).append(idx)

    remove: set = set()
    for kind, idxs in by_kind.items():
        if len(idxs) < 2:
            continue
        vecs = []
        for i in idxs:
            emb = kept[i].get("embeddings")
            if emb is None:
                vec = None
            else:
                try:
                    vec = [float(x) for x in emb]
                except Exception:  # noqa: BLE001
                    vec = None
            vecs.append(vec)
        if any(v is None for v in vecs):
            continue
        pairs = semantic_dup_pairs(vecs, [kept[i].get("text", "") for i in idxs],
                                   threshold=semantic_threshold)
        for gi, gj, sim in pairs:
            a, b = kept[idxs[gi]], kept[idxs[gj]]
            if a.get("id") in remove or b.get("id") in remove:
                continue
            if keep == "longer":
                # ⚠️ 三元表达式的两个分支顺序极易写反：
                # 目标是「留长的、删短的」。b 更长时应让 b 当 winner。
                loser, winner = (a, b) if len(b.get("text", "")) > len(a.get("text", "")) else (b, a)
            else:
                loser, winner = a, b
            remove.add(loser.get("id"))
            dropped.append({
                "id": loser.get("id", ""), "doc_type": kind,
                "dup_reason": f"semantic:{sim:.3f}", "dup_of": winner.get("id", ""),
            })

    if remove:
        kept = [c for c in kept if c.get("id") not in remove]
    return kept, dropped


# ═══════════════════════════════════════════════════════════════════
# 三、知识冲突检测（步骤 3.3）
# ═══════════════════════════════════════════════════════════════════

# 数值断言：「万有引力常量为 6.67×10⁻¹¹」这类
#
# 三种科学计数法写法都要认，否则同一个常量在不同教材里会抽出不同字符串，
# 冲突检测就漏报了：
#   6.67×10⁻¹¹（上标 Unicode，教材排版常见）
#   6.67*10^-11（ASCII，PDF 抽取后常见）
#   6.67e-11 / 6.67E-11（纯电子写法，题干与答案里常见）
# 统一归一化成「系数 + e + 指数」再比较。
_NUM_ASSERT_RE = re.compile(
    r"([\u4e00-\u9fff]{2,8})\s*(?:为|是|＝|=|约为|约)\s*"
    r"(\d+(?:\.\d+)?)"
    r"(?:\s*(?:[×xX*]\s*10|10\s*[×xX*])\s*[⁻⁻\-]?\s*(\d+)"
    r"|\s*[eE]\s*[-−]?\s*(\d+))?"
)
# 无量纲常数
_EXACT_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _numeric_claims(text: str) -> Dict[str, List[str]]:
    """
    抽取「概念 -> 数值」断言。

    只做**字面数值**层面的矛盾检测（不含公式语义推理）——
    方案 §3.4 步骤 3.3 明确要求「轻量」「用关键词/数值一致性规则」。
    在没有权威知识图谱的情况下，语义级冲突判定会产出大量假警报，
    而假警报会让人不再相信这份报告 —— 这是比漏检更坏的结果。
    """
    out: Dict[str, List[str]] = {}
    for m in _NUM_ASSERT_RE.finditer(text or ""):
        concept, base = m.group(1), m.group(2)
        exp_mul, exp_e = m.group(3), m.group(4)
        exp = exp_mul if exp_mul is not None else exp_e
        if exp is not None:
            value = f"{base}e{'-' + exp if exp.lstrip('-').isdigit() and not exp.startswith('-') else exp}"
        else:
            value = base
        out.setdefault(concept, [])
        if value not in out[concept]:
            out[concept].append(value)
    return out


def detect_conflicts(
    chunks: List[Dict[str, Any]],
    *,
    concept_key_field: str = "concept_key",
    min_blocks: int = 2,
    require_concept_key: bool = True,
) -> List[Dict[str, Any]]:
    """
    在**同一 concept_key** 的块之间检测数值矛盾。

    为什么按 concept_key 分组而不是全库两两比较
    ────────────────────────────────────────
    冲突的前提是「讲的是同一个概念」。没有 concept_key 时全库比较会把
    「向心力是 3 N」和「重力是 5 N」判成冲突 —— 这种假警报会让整个
    冲突治理失去可信度，而**没人相信的报告比没有报告更糟**。
    因此 require_concept_key=True 时，无 concept_key 的块一律不参与。

    Args:
        require_concept_key: True 时只检测带 concept_key 的块；
            False 时退化为「单文件内按数值断言聚合」的弱检查（仍不做全库两两比）。

    Returns:
        冲突记录列表；每条含 concept_key / kind / field / values / chunk_ids。
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        meta = c.get("metadata") or {}
        key = str(meta.get(concept_key_field) or c.get(concept_key_field) or "").strip()
        if key:
            groups.setdefault(key, []).append(c)

    if not require_concept_key and not groups:
        # 弱检查模式：把整批当作一个概念组，仅在**同字段名 + 不同数值**时报。
        # 仍然比全库两两比较克制得多。
        groups = {"__all__": list(chunks)}

    conflicts: List[Dict[str, Any]] = []
    for key, items in groups.items():
        if len(items) < min_blocks:
            continue
        claims: Dict[str, Dict[str, List[str]]] = {}
        for c in items:
            for concept, values in _numeric_claims(c.get("text", "")).items():
                for v in values:
                    claims.setdefault(concept, {}).setdefault(v, []).append(c.get("id", ""))
        for concept, by_value in claims.items():
            if len(by_value) < 2:
                continue
            conflicts.append({
                "concept_key": key,
                "kind": "numeric_mismatch",
                "field": concept,
                "values": {v: sorted(set(ids)) for v, ids in sorted(by_value.items())},
                "chunk_ids": sorted({i for ids in by_value.values() for i in ids}),
                "textbooks": sorted({str((c.get("metadata") or {}).get("book_title", ""))
                                     for c in items
                                     if (c.get("metadata") or {}).get("book_title")}),
            })
    if conflicts:
        logger.warning(f"检出 {len(conflicts)} 处疑似知识冲突（详见 conflicts.jsonl）")
    return conflicts


# ═══════════════════════════════════════════════════════════════════
# 四、溯源引用（步骤 3.4 的检索侧部分）
# ═══════════════════════════════════════════════════════════════════

# 检索侧默认排除的内容类型（方案附录 A「检索侧默认过滤条件」）。
# 目录/封面块结构分已经是 0，但**结构分只影响排序，不影响召回** ——
# 它们仍会占候选池与 BM25 语料的位置。因此必须在**查询层**排除。
DEFAULT_EXCLUDE_CONTENT_TYPES = ("toc", "cover")

# 已下架 / 归档块（步骤 3.3.2）
_ARCHIVED_STATUS = "archived"


def build_query_filter(
    exclude_content_types: Sequence[str] = DEFAULT_EXCLUDE_CONTENT_TYPES,
    exclude_archived: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    构造 Chroma `where` 过滤条件：排除父块、目录/封面、已下架块。

    与已有的 `{"doc_type": {"$ne": "parent"}}` 合并成一个 `$and`，
    避免多次查询。

    ⚠️ **对缺字段文档的语义（实测）**
    Chroma 1.4.0 的 `$ne` / `$nin` 对**没有该键**的文档 behave like「不匹配
    任何值」，因而会被**保留**。这对我们恰好是安全的：
    旧索引（清单导入的历史数据）没有 `content_type` / `status` 键，
    过滤不会把它们误杀。但这个行为不应被依赖 —— 因此 Python 侧
    仍会再做一次兜底过滤（见 `passes_metadata_filter`）。

    Returns:
        where dict；无条件可排除时返回 None。
    """
    conds: List[Dict[str, Any]] = [
        # 父块不参与召回：只用于「命中子块后取回」
        {"doc_type": {"$ne": "parent"}},
    ]
    types = [t for t in (exclude_content_types or []) if t]
    if types:
        conds.append({"content_type": {"$nin": list(types)}})
    if exclude_archived:
        conds.append({"status": {"$ne": _ARCHIVED_STATUS}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def passes_metadata_filter(
    metadata: Optional[Dict[str, Any]],
    exclude_content_types: Sequence[str] = DEFAULT_EXCLUDE_CONTENT_TYPES,
    exclude_archived: bool = True,
) -> bool:
    """
    Python 侧兜底过滤（不依赖 Chroma 的 `$ne`/`$nin` 语义）。

    两处必须做同一件事：
      · `HybridRetriever._dense_route` —— where 已过滤，这里再过滤一次
        是为了兜住「无过滤降级重查」那条路径（它不带 where）；
      · `HybridRetriever._corpus` —— BM25 语料是 Python 侧全量读的，
        根本不经过 where。

    **缺键视为通过**（与 Chroma 实测行为一致）：旧数据不该被新规则误杀。
    """
    meta = metadata or {}
    types = {t for t in (exclude_content_types or []) if t}
    if types and str(meta.get("content_type") or "") in types:
        return False
    if exclude_archived and str(meta.get("status") or "") == _ARCHIVED_STATUS:
        return False
    if str(meta.get("doc_type") or "") == "parent":
        return False
    return True


# ── 拒答策略补充判据（步骤 3.4.3）────────────────────────────────
# 为什么覆盖率单独不够
# ──────────────────────
# 拒答主判据是「查询覆盖率」（二元组绝对覆盖率），但它只回答
# 「问题里的词有没有出现在块里」，不回答「块里的内容是不是在回答这个问题」。
#
# 实测踩到的例子：「帮我写一篇 800 字的散文」
#   → 覆盖率 0.71（"散文""800""字"在语文教材里天然高频），远高于阈值 0.28；
#   → 于是被判为「教材里有依据」，模型拿着几段描写景色的课文开始写散文。
# 这不是覆盖率算错了，是**问题类型**根本不属于「教材里有答案」那一类。
#
# 因此加一条与覆盖率**正交**的判据：命中创作/写作类意图时直接走拒答。
# 保持克制：宁可漏判（继续当 RAG 问答）也不误判（把「作文怎么立意」这类
# 确有教材依据的问题挡掉），所以只匹配**明确的成品创作**动词，
# 不匹配「立意」「赏析」「表达方式」等需要教材依据的教学话题。
_CREATION_INTENT_RE = re.compile(
    r"(?:写|编|作|撰写|创作|续写|改写|仿写|誊写)"
    r"[^。？！?!]{0,12}?"
    r"(?:散文|诗歌|诗|议论文|记叙文|说明文|作文|文章|小说|故事|剧本|台词|标题|读后感|"
    r"倡议书|演讲稿|请假条|周记|日记|情书|誓词|发言稿|推荐信|倡议|书评|影评)"
)
# 明确**不**属于创作请求的补充词：这些问题确实要查教材
_CREATION_EXEMPT_RE = re.compile(
    r"(怎么|如何|为什么|为何|是什么|有哪些|区别|作用|意思|赏析|立意|表达方式|写法特点|"
    r"中心思想|主旨|修辞|词语|句子|段落|结构|思路|评分标准|要求)"
)


def is_creation_request(query: str) -> bool:
    """
    判断是否属于「写一篇成品」的创作类请求（步骤 3.4.3 的补充拒答判据）。

    与覆盖率正交：覆盖率再高也不改变「教材里没有现成答案」这个事实。
    命中时上层应走拒答（或专门的创作模式），而不是拿课文当素材硬凑。

    Returns:
        True 表示这是创作请求，应当拒答。
    """
    q = (query or "").strip()
    if not q or not _CREATION_INTENT_RE.search(q):
        return False
    # 出现教学型问法时视为「问写法」而非「要成品」，不拒答
    return not _CREATION_EXEMPT_RE.search(q)


def build_citations(chunks: Sequence[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    给证据块编号并生成「参考材料」里的来源清单。

    **编号与来源描述都由检索层生成，不由模型编造** —— 这是方案 §6
    风险表里对 1.5B 模型的对冲手段：模型只负责写 `[n]`，`[n]` 指向什么
    完全由我们决定。

    Returns:
        (sources_block, citation_table)
        sources_block 是追加到 context 末尾的来源清单文本；
        citation_table[i] = {index, book_title, chapter, page, ...}，
        供生成后做引用越界校验。
    """
    table: List[Dict[str, Any]] = []
    lines: List[str] = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata") or {}
        title = str(meta.get("book_title") or meta.get("source_file")
                    or meta.get("source") or "")
        chapter = str(meta.get("chapter") or meta.get("section_title") or "")
        page = meta.get("page")
        page_s = f"第{page}页" if page not in (None, "", 0) else ""
        bits = [b for b in (title, chapter, page_s) if b]
        label = " · ".join(bits) if bits else f"来源{i}"
        table.append({
            "index": i,
            "id": c.get("id", ""),
            "book_title": title,
            "chapter": chapter,
            "section_title": str(meta.get("section_title") or ""),
            "page": page,
            "textbook_version": str(meta.get("edition") or ""),
            "label": label,
        })
        lines.append(f"[{i}] {label}")
    if not lines:
        return "", table
    return "【来源清单】\n" + "\n".join(lines), table


_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def validate_citations(
    answer: str,
    citation_table: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    校验答案里的 `[n]` 引用标记（步骤 3.4.2，确定性规则）。

    做三件事，**都不依赖模型自觉**：
      1. 越界（n 超过实际证据数）→ 记为 `out_of_range`，是引用准确率的硬失败；
      2. 有效（n 落在范围内）→ 记为 `valid`，是引用准确率的分子；
      3. 完全没有标记 → 记为 `missing`。**不惩罚**：
         1.5B 模型不遵循编号指令是常态，方案 §6 已预判。
         硬要求它标注只会制造「看起来有引用其实没有」的假安全感。

    注意：这里**不移除**越界标记。理由与 P0 忠实度校验一致 ——
    忠实度覆盖度对中文区分力弱，误报的代价高于漏报；
    越界标记只是编号写错（如 5 而实际只有 3 条证据），
    学生看到「[5]」会疑惑但不会被误导内容。把越界率**报出来**即可。
    """
    n_available = len(citation_table or [])
    cited = [int(m.group(1)) for m in _CITATION_RE.finditer(answer or "")]
    valid = [n for n in cited if 1 <= n <= n_available]
    invalid = [n for n in cited if n < 1 or n > n_available]
    return {
        "n_cited": len(cited),
        "n_valid": len(valid),
        "n_out_of_range": len(invalid),
        "out_of_range_refs": sorted(set(invalid)),
        "used_indexes": sorted(set(valid)),
        # 引用准确率 = 有效标记 / 全部标记；无标记时 None（不是 0，也不是 1）
        "accuracy": (len(valid) / len(cited)) if cited else None,
        "has_citation": bool(cited),
    }
