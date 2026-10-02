"""
Query 改写：术语同义词扩展 + 子问题拆解 + 检索失败回流。

现状与动机
──────────
P0 之前的「改写」能力只有 `AdaptiveNormalizer`（纠错、口语归一、语种检测），
没有同义扩展、没有术语规范化、没有子问题拆解。后果是：

  · 学生说「自由下落」，教材写「自由落体」→ 术语不匹配，稠密向量召回受限；
  · 「平抛运动和斜抛有什么区别」其实是**两个问题**，当成一个查询去检索时，
    向量会被两个主题稀释，两边都召不全。

设计取舍（重要）
──────────────
1. **规则优先，不调模型**。语雀方法论提到"用小模型做子问题拆解"，但本地
   1.5B + CPU 上多一次生成意味着 3–10 秒延迟，与「交互式问答额外延迟上限
   1.5–2 s」的硬约束冲突。当前先用**确定性规则**（连接词 + 疑问尾缀）拆解，
   零延迟、可解释、可回归；模型兜底留作后续可选开关。
2. **变体不替换原查询，而是并行并查**。改写本身就是有损的：把「自由下落」
   改写成「自由落体」再检索，若改写错了就再也召不回原本能命中的块。
   因此改为**多路并查 + RRF 融合**（见 hybrid_retriever），原始查询始终
   是其中一路 —— 这比"要么改写要么不改"稳健得多。
3. **失败回流复用既有基础设施**。检索零命中/低分的问题写入
   `retrieval_failures.jsonl`，格式与 `pattern_miner.py` 消费的
   `low_confidence_cases.jsonl` 对齐，可直接复用已有点模式挖掘能力。
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML 已在 requirements 中
    yaml = None

logger = logging.getLogger(__name__)

# 连接词：出现这些词时，问题可能包含多个并列子问题
_CONNECTORS = ["以及", "分别", "和", "与", "跟", "、"]

# 结尾疑问模式：「……有什么区别」「……有什么关系」等
_TAIL_QUESTION_RE = re.compile(
    r"[，,]?\s*(?:有(?:什么|哪些)?|是(?:什么|怎样))?"
    r"(?:区别|不同|差异|关系|联系|异同|比较)\s*[？?。]?\s*$"
)

# 子查询片段的最小长度：**2 个字符**。
#
# 不能设得更大：中文术语常常很短，「自由落体」「栈」「队列」「导数」都是
# 2–4 字。设成 6 会让「平抛运动和自由落体有什么区别」拆出的「自由落体」
# 被当作碎片丢弃，整条拆解失效（这是实测踩到的坑）。
# 误拆的兜底不靠长度，而靠两点：① 整句话长度必须 >= subquery_min_chars；
# ② 变体只是**追加路**，原始查询始终参与检索，RRF 会压制噪声路。
_MIN_SUBQUERY_CHARS = 2


class QueryRewriter:
    """查询改写器：同义扩展 + 子问题拆解（规则版）。"""

    def __init__(self, config, synonyms_path: Optional[str] = None):
        """
        Args:
            config: system.rag.retrieval_config.RetrievalConfig
            synonyms_path: 同义词表路径；None 时用默认位置
        """
        self.config = config
        rw = dict(config.get("rewrite") or {})
        self.enabled = bool(rw.get("enabled", True))
        self.synonym_expansion = bool(rw.get("synonym_expansion", True))
        self.max_synonym_variants = int(rw.get("max_synonym_variants", 2))
        self.subquery_split = bool(rw.get("subquery_split", True))
        self.subquery_min_chars = int(rw.get("subquery_min_chars", 10))
        self.max_subqueries = int(rw.get("max_subqueries", 2))
        self.failure_log = rw.get("failure_log")
        self.failure_score_threshold = float(rw.get("failure_score_threshold", 0.45))

        self.synonym_map: Dict[str, List[str]] = {}
        self._load_synonyms(synonyms_path)
        self.last_info: Dict[str, Any] = {}

    # --------------------------------------------------------------- 同义词表
    @staticmethod
    def default_synonyms_path() -> str:
        root = Path(__file__).resolve().parents[2]
        return str(root / "mati_data" / "config" / "synonyms.yaml")

    def _load_synonyms(self, path: Optional[str]) -> None:
        p = Path(path or self.default_synonyms_path())
        if not p.exists():
            logger.warning(f"同义词表不存在（{p}），改写仅保留子问题拆解")
            return
        if yaml is None:
            logger.error("PyYAML 未安装，同义词表无法读取")
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:  # noqa: BLE001
            logger.error(f"同义词表解析失败（{p}）：{e}")
            return

        groups = data.get("groups") or []
        for grp in groups:
            terms = [str(t).strip() for t in (grp.get("terms") or []) if str(t).strip()]
            if len(terms) < 2:
                continue
            # 组内两两互为同义：以每个词为键，其余词为值
            for t in terms:
                bucket = self.synonym_map.setdefault(t, [])
                for other in terms:
                    if other != t and other not in bucket:
                        bucket.append(other)
        logger.info(
            f"同义词表已加载：{len(groups)} 组、{len(self.synonym_map)} 个可扩展术语（{p}）"
        )

    # ------------------------------------------------------------------- 改写
    def _synonym_variants(self, query: str) -> List[str]:
        """对查询中出现的术语做同义替换，产出变体查询（不修改原查询）。"""
        if not self.synonym_expansion or not self.synonym_map:
            return []
        variants: List[str] = []
        # 长词优先，避免「向心力」被「向心」先命中导致替换不完整
        for term in sorted(self.synonym_map, key=len, reverse=True):
            if term not in query:
                continue
            for syn in self.synonym_map[term]:
                if syn in query:
                    continue
                cand = query.replace(term, syn)
                if cand != query and cand not in variants:
                    variants.append(cand)
                    if len(variants) >= self.max_synonym_variants:
                        return variants
        return variants

    def _subqueries(self, query: str) -> List[str]:
        """规则型子问题拆解：去掉疑问尾缀后按连接词切分。"""
        if not self.subquery_split or len(query) < self.subquery_min_chars:
            return []

        core = _TAIL_QUESTION_RE.sub("", query).strip("，,。？? 、")
        if len(core) < _MIN_SUBQUERY_CHARS:
            return []

        parts: List[str] = []
        for conn in _CONNECTORS:
            if conn in core:
                parts = [p.strip() for p in core.split(conn)]
                break
        parts = [p for p in (parts or []) if len(p) >= _MIN_SUBQUERY_CHARS]
        if len(parts) < 2:
            return []

        # 拆出的子查询补齐主题语境，避免「自由落体」这种没有疑问词的碎片
        out = []
        for p in parts[: self.max_subqueries]:
            out.append(f"{p}是什么" if not re.search(r"[？?]", p) else p)
        return out

    def rewrite(
        self,
        query: str,
        subject: Optional[str] = None,
        discipline: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        产出用于多路检索的查询变体。

        Args:
            query: 已归一化的查询

        Returns:
            {
              "original": str,
              "variants": [str, ...],   # 追加路（原始查询不在其中）
              "synonyms": [...],        # 同义扩展
              "subqueries": [...],      # 子问题
              "enabled": bool,
            }
        """
        if not query:
            return {"original": "", "variants": [], "synonyms": [],
                    "subqueries": [], "enabled": False}

        if not self.enabled:
            self.last_info = {"enabled": False, "synonyms": [], "subqueries": []}
            return {"original": query, "variants": [], "synonyms": [],
                    "subqueries": [], "enabled": False}

        syn = self._synonym_variants(query)
        subs = self._subqueries(query)
        variants: List[str] = []
        # 子问题优先于同义变体：对比型问题（"A 和 B 有什么区别"）拆出的子查询
        # 信息增量远大于同义替换；而术语型问题（"万有引力常量"）拆不出子查询，
        # 自然由同义变体补位。两者都受 max_query_variants 约束。
        for v in subs + syn:
            if v and v != query and v not in variants:
                variants.append(v)

        self.last_info = {"enabled": True, "synonyms": syn, "subqueries": subs}
        if variants:
            logger.info(f"查询改写：原句 1 路 + 追加 {len(variants)} 路 → {variants}")
        return {"original": query, "variants": variants, "synonyms": syn,
                "subqueries": subs, "enabled": True}

    # --------------------------------------------------------------- 失败回流
    def is_failure(self, best_score: float, n_evidence: int) -> bool:
        """判断是否为一次「检索失败」（零证据 或 最高证据分过低）。"""
        if n_evidence <= 0:
            return True
        return float(best_score or 0.0) < self.failure_score_threshold

    def log_failure(
        self,
        query: str,
        subject: Optional[str] = None,
        grade: Optional[str] = None,
        discipline: Optional[str] = None,
        best_score: float = 0.0,
        collections: Optional[Sequence[str]] = None,
        variants: Optional[Sequence[str]] = None,
    ) -> None:
        """
        把检索失败的问题追加到 JSONL，供 pattern_miner 挖掘高频失败模式。

        字段与 `low_confidence_cases.jsonl` 风格对齐（含 query/notes/时间戳），
        方便复用既有挖掘流程。写入失败只记警告，绝不影响问答主链路。
        """
        if not self.failure_log or not query:
            return
        import datetime
        try:
            p = Path(self.failure_log)
            if not p.is_absolute():
                p = Path(__file__).resolve().parents[2] / p
            p.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "query": query,
                "subject": subject or "",
                "discipline": discipline or "",
                "grade": grade or "",
                "best_score": round(float(best_score or 0.0), 4),
                "collections": list(collections or []),
                "variants": list(variants or []),
                "source": "rag_retrieval",
            }
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"检索失败样本落盘失败（已忽略）：{e}")
