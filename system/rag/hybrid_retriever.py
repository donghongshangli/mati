"""
混合检索器：稠密向量路 + BM25 关键词路，用 RRF 融合。

为什么需要它
────────────
原实现是**纯稠密单路**检索。对本项目的具体危害是术语与符号密集的问法：
「向心加速度」「万有引力常量 G」「平抛运动」这类问题，512 维小模型做语义
匹配时容易与近邻概念混淆（向心力 vs 离心力、匀速圆周运动 vs 变速圆周运动）。
精确字面匹配恰恰是 BM25 的强项，而原实现完全缺位。

为什么用 RRF 而不是「分数归一化后加权求和」
────────────────────────────────────────
多路检索的分数尺度**不可比**：余弦相似度落在 [-1, 1] 且分布集中，
BM25 是未归一化的正数且长尾。P0 已经踩过一次同类坑（来源权重 0.30 压过
相关性、使阈值随来源漂移）。RRF 只依赖**排名**不依赖分数尺度：

    score(d) = Σ_routes 1 / (k + rank_route(d))        k = 60

天然规避了尺度不可比的问题，且无需标定。

设计约束
────────
    # jieba / rank_bm25 均为**可选依赖**：缺失时自动退化为纯稠密路，
    #  不让检索整体不可用（与 retrieval_config 的「配置缺失不崩」一致）。
    - BM25 语料按集合惰性构建并缓存；集合内容变化时用 invalidate() 失效。
    - `enabled=False`（或 dense_only=True）时行为与 P0 完全一致，
     便于做 A/B 消融，量化 BM25 的实际贡献。
    - 查询层统一过滤父块 / 目录 / 封面 / 已下架块（P2 知识治理），
    **不依赖**下游的 structure_bonus 降权 —— 降权只改排序，不改召回。
"""

import logging
import math
import time
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from system.rag.knowledge_governance import (
    DEFAULT_EXCLUDE_CONTENT_TYPES,
    build_query_filter,
    passes_metadata_filter,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- 可选依赖
# jieba 会在导入时通过 pkg_resources 触发弃用警告，这里一并静默。
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import jieba  # type: ignore
        jieba.setLogLevel(logging.WARNING)   # 关闭 "Building prefix dict" 之类的噪声
        _JIEBA_OK = True
    except Exception:  # noqa: BLE001 - 任何导入失败都退化为无 jieba
        jieba = None  # type: ignore
        _JIEBA_OK = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _BM25_OK = True
except Exception:  # noqa: BLE001
    BM25Okapi = None  # type: ignore
    _BM25_OK = False


# 中文/英文无关的标点与空白：分词后剔除，避免污染 BM25 词频
_PUNCT = set(
    " \t\r\n"
    "，。！？；：、（）〔〕【】《》〈〉「」『』“”‘’—…·～"
    ",.!?;:()[]{}<>\"'`~-_/\\|@#$%^&*+="
)


def bm25_available() -> bool:
    """BM25 路是否可用（两个依赖都在）。"""
    return _JIEBA_OK and _BM25_OK


def with_retry(fn, what: str, attempts: int = 3, delay: float = 0.3):
    """
    对可能瞬发失败的 Chroma 操作重试（退避递增）。

    背景（实测）：这个 Chroma 版本（1.4.0）的 HNSW 索引是**惰性落盘**的，
    个别集合的持久化索引可能是半成品（`data_level0.bin` 明显偏小、
    `index_metadata.pickle` 为 0 字节）。此时该集合的向量查询会间歇性抛
    `Error finding id` / `Error creating hnsw segment reader: Nothing found on disk`，
    磁盘上文件看起来是"在"的，重试往往能恢复（有时需要等 Chroma 在内存里
    重建完索引）。**不重试则表现为静默零召回** —— 整个集合查不到东西，
    上层只看到"没检索到相关内容"，既污染评测也会影响线上问答。

    用 3 次尝试、退避 0.3s：覆盖"内存重建需要一点时间"的窗口。

    Args:
        fn: 无参可调用对象
        what: 用于日志描述的操作名
        attempts: 总尝试次数（含首次）
        delay: 首次重试前的等待秒数（之后线性递增）

    Returns:
        fn() 的返回值

    Raises:
        最后一次异常（调用方自行决定是降级还是中断）
    """
    last_err = None
    for i in range(max(1, attempts)):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if i < attempts - 1:
                logger.warning(f"{what} 第 {i + 1} 次失败，将重试：{str(e)[:100]}")
                time.sleep(delay * (i + 1))
    raise last_err


def tokenize_zh(text: str) -> List[str]:
    """
    中文分词（BM25 用）。

    用 `cut_for_search`（搜索模式）：它会额外切出子词
    （如「向心加速度」→「向心」「心力」「加速度」），
    对术语型查询的召回明显优于精确模式。

    未安装 jieba 时退化为「单个汉字 + 英文数字词」，
    仍比 `str.split()` 在中文上完全失效要好。
    """
    if not text:
        return []
    if _JIEBA_OK:
        raw = jieba.cut_for_search(text)
    else:
        raw = _fallback_cut(text)
    out = []
    for t in raw:
        t = t.strip().lower()
        if not t or t in _PUNCT:
            continue
        # 剔除纯标点 token
        if all(ch in _PUNCT for ch in t):
            continue
        out.append(t)
    return out


def _fallback_cut(text: str) -> Iterable[str]:
    """无 jieba 时的兜底切分：汉字单字 + 连续 ASCII 词。"""
    buf_ascii: List[str] = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            if buf_ascii:
                yield "".join(buf_ascii)
                buf_ascii.clear()
            yield ch
        elif ch.isalnum():
            buf_ascii.append(ch)
        elif buf_ascii:
            yield "".join(buf_ascii)
            buf_ascii.clear()
    if buf_ascii:
        yield "".join(buf_ascii)


class _CollectionCorpus:
    """单个集合的 BM25 语料与索引（惰性构建）。"""

    __slots__ = ("tokenized", "docs", "metadatas", "ids", "bm25", "version")

    def __init__(self):
        self.tokenized: List[List[str]] = []
        self.docs: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.bm25 = None
        self.version = -1


class HybridRetriever:
    """稠密 + BM25 混合检索，RRF 融合。"""

    # 类级默认值：降级路径（_dense_fallback / _brute_force_route）可能在
    # 实例尚未跑完 __init__ 时被调用，缺属性会 AttributeError 让整轮查询无召回。
    # __init__ 会用配置值覆盖它们。
    exclude_content_types: tuple = DEFAULT_EXCLUDE_CONTENT_TYPES
    exclude_archived: bool = True
    where: Optional[Dict[str, Any]] = None

    def __init__(self, chroma_client, embedding_gen, config,
                 enabled: Optional[bool] = None):
        """
        Args:
            chroma_client: chromadb PersistentClient
            embedding_gen: scripts.rag_data_preparation.embedding_generator.EmbeddingGenerator
            config: system.rag.retrieval_config.RetrievalConfig
            enabled: 覆盖配置中的开关（None 时取配置）
        """
        self.client = chroma_client
        self.embedding_gen = embedding_gen
        self.config = config

        hybrid_cfg = dict(config.get("hybrid") or {})
        self.enabled = bool(hybrid_cfg.get("enabled", True)) if enabled is None else bool(enabled)
        self.rrf_k = int(hybrid_cfg.get("rrf_k", 60))
        self.dense_k = int(hybrid_cfg.get("dense_k", 20))
        self.bm25_k = int(hybrid_cfg.get("bm25_k", 20))
        self.max_variants = int(hybrid_cfg.get("max_query_variants", 3))

        # 依赖不可用则强制关闭 BM25（保留 dense-only，保证链路可用）
        self.bm25_enabled = self.enabled and bm25_available()
        if self.enabled and not bm25_available():
            logger.warning(
                "混合检索已配置开启，但 jieba / rank_bm25 不可用 → 退化为纯稠密检索。"
                "安装：pip install jieba rank_bm25"
            )

        self._corpora: Dict[str, _CollectionCorpus] = {}
        # 「HNSW 不可用」兜底用的缓存：集合名 -> (归一化向量矩阵, 元数据列表, id 列表)
        self._bf_cache: Dict[str, Tuple[Any, List[Dict[str, Any]], List[str], List[str]]] = {}
        self.last_stats: Dict[str, Any] = {}

        # ── 知识治理过滤（P2 / 方案附录 A「检索侧默认过滤条件」）
        # 目录/封面块此前只靠 structure_bonus=0 降权，**仍会被召回**，
        # 白占候选池与 BM25 语料的位置；status=archived 的下架机制更是
        # 完全没有实现。这里把过滤下推到查询层。
        gov = dict(config.get("governance") or {})
        types = gov.get("exclude_content_types")
        self.exclude_content_types = tuple(
            types if types is not None else DEFAULT_EXCLUDE_CONTENT_TYPES
        )
        self.exclude_archived = bool(gov.get("exclude_archived", True))
        self.where = build_query_filter(self.exclude_content_types, self.exclude_archived)
        logger.info(
            f"知识治理过滤：排除 content_type={list(self.exclude_content_types)}"
            f"{' + status=archived' if self.exclude_archived else ''}"
            f"（父块始终排除）"
        )

    # ---------------------------------------------------- HNSW 失效兜底
    def _brute_force_route(self, vec, collection_name: str, k: int,
                           exclude_parents: bool = True) -> List[Dict[str, Any]]:
        """
        HNSW 索引不可用时的**内存内精确余弦检索**兜底。

        为什么需要（实测）：这个 Chroma 版本的 HNSW 索引是惰性落盘的，
        个别集合的持久化索引可能是半成品（`data_level0.bin` 只覆盖前 ~100 条、
        `index_metadata.pickle` 为 0 字节）。此时该集合的向量查询会间歇性抛
        `Error finding id` / `Error creating hnsw segment reader`，
        表现为**整个集合静默零召回**。

        本库规模很小（1.3 万向量 × 512 维 = 27MB），做一次全量余弦扫描只要几毫秒
        —— 与其和 HNSW 的持久化较劲，不如在 HNSW 失败时退到精确检索：
        结果**确定、无近似误差**，代价是首次要读一次全部向量。

        注意：`col.get()` 走 sqlite，**不经过 HNSW**，所以在 HNSW 失效时依然可用
        （BM25 语料正是靠这一点在故障时照常工作）。
        """
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy 是硬依赖
            return []

        cache = self._bf_cache.get(collection_name)
        if cache is None:
            col = self.client.get_collection(collection_name)
            got = col.get(include=["embeddings", "documents", "metadatas"])
            ids = got.get("ids") or []
            embs = got.get("embeddings")
            docs = got.get("documents") or []
            metas = got.get("metadatas") or []
            if embs is None or not ids:
                return []
            mat = np.asarray(embs, dtype="float32")
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            cache = (mat, [m or {} for m in metas], list(ids), list(docs))
            self._bf_cache[collection_name] = cache
            logger.warning(
                f"HNSW 不可用，已为 {collection_name} 建立内存精确检索索引"
                f"（{len(ids)} 条，{mat.nbytes / 1e6:.1f} MB）"
            )

        mat, metas, ids, docs = cache
        q = np.asarray(vec, dtype="float32")
        n = float(np.linalg.norm(q))
        if n == 0:
            return []
        sims = mat @ (q / n)
        # 先按相似度全局排序，再流式过滤 —— 只取前 k 个有效项即可
        order = np.argsort(-sims)
        out: List[Dict[str, Any]] = []
        for idx in order:
            if not passes_metadata_filter(metas[idx],
                                          self.exclude_content_types,
                                          self.exclude_archived):
                continue
            out.append({
                "id": ids[idx],
                "text": docs[idx] if idx < len(docs) else "",
                "metadata": metas[idx],
                "collection": collection_name,
                "dense_score": float(sims[idx]),
                "route": "brute_force",
            })
            if len(out) >= max(1, int(k)):
                break
        return out

    def invalidate_brute_force(self, collection: Optional[str] = None) -> None:
        """失效内存精确检索缓存（集合内容变化或索引重建后调用）。"""
        if collection is None:
            self._bf_cache.clear()
        else:
            self._bf_cache.pop(collection, None)

    # ------------------------------------------------------------- 语料管理
    def invalidate(self, collection: Optional[str] = None) -> None:
        """失效 BM25 语料缓存（集合内容变化或索引重建后必须调用）。"""
        if collection is None:
            self._corpora.clear()
        else:
            self._corpora.pop(collection, None)

    def _corpus(self, collection_name: str) -> Optional[_CollectionCorpus]:
        """惰性构建并缓存某集合的 BM25 索引。"""
        if not self.bm25_enabled or not collection_name:
            return None
        col = self.client.get_collection(collection_name)
        n = col.count()
        cached = self._corpora.get(collection_name)
        if cached is not None and cached.version == n:
            return cached

        # 语料读取同样走重试：col.get() 也要建段读取器，会碰到同一类瞬发失败
        got = with_retry(
            lambda: col.get(include=["documents", "metadatas"]),
            f"BM25 语料读取（{collection_name}）",
        )
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        # 父块（doc_type=parent）不参与召回，避免与子块重复计分；
        # 目录/封面/已下架块同样排除（P2 知识治理）。
        triples = [
            (i, d, m) for i, d, m in zip(ids, docs, metas)
            if d and passes_metadata_filter(m, self.exclude_content_types,
                                            self.exclude_archived)
        ]
        corpus = _CollectionCorpus()
        corpus.ids = [t[0] for t in triples]
        corpus.docs = [t[1] for t in triples]
        corpus.metadatas = [t[2] or {} for t in triples]
        corpus.tokenized = [tokenize_zh(d) for d in corpus.docs]
        if corpus.tokenized and any(corpus.tokenized):
            corpus.bm25 = BM25Okapi(corpus.tokenized)
        corpus.version = n
        self._corpora[collection_name] = corpus
        logger.info(
            f"BM25 语料已构建：{collection_name}（{len(corpus.docs)} 篇）"
        )
        return corpus

    # --------------------------------------------------------------- 各路检索
    def _dense_fallback(self, vec, collection_name: str, k: int,
                        reason: str) -> List[Dict[str, Any]]:
        """
        HNSW 查询失败时的**逐级降级**，且任何一种降级都不允许让整个问答崩掉。

        实测（2026-10-02）：同一份索引、同一进程里连跑 78 个问题，会有
        若干个在 `Error finding id` 上失败，且**越往后越多** —— 典型的
        HNSW 段惰性加载/落盘竞态。更麻烦的是：单纯重试不够，连
        `col.get()` 都可能一起失败；此时若让异常冒出去，整个问答直接中断
        （评测里一度 78 题挂掉 35 题），比"这一路召回少一点"严重得多。

        降级顺序：
          1. 不带 where 过滤再查一次（过滤下推本身可能是失败源），
             父块在 Python 侧剔除；
          2. 内存内精确余弦检索（`col.get()` 走 sqlite，不经 HNSW）；
          3. 都失败则返回空列表 —— 会走入"零证据拒答"，**用户看得到**，
             不会是静默的错误答案。
        """
        # ① 无过滤重查
        try:
            res = with_retry(
                lambda: self.client.get_collection(collection_name).query(
                    query_embeddings=[vec], n_results=max(1, int(k))
                ),
                f"稠密检索（无过滤，{collection_name}）",
            )
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            ids = (res.get("ids") or [[]])[0]
            out = []
            for i in range(len(docs)):
                m = metas[i] or {}
                # 无过滤重查这条路不带 where，父块与噪声块必须在客户端剔除，
                # 否则降级反而会引入本该被排除的噪声。
                if not passes_metadata_filter(m, self.exclude_content_types,
                                              self.exclude_archived):
                    continue
                out.append({
                    "id": ids[i] if i < len(ids) else "",
                    "text": docs[i],
                    "metadata": m,
                    "collection": collection_name,
                    "dense_score": 1.0 - float(dists[i]),
                })
            if out:
                logger.warning(
                    f"降级成功（无过滤重查）：{collection_name} 取回 {len(out)} 条"
                )
            return out
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"无过滤重查仍失败（{collection_name}）：{str(e)[:100]}"
            )

        # ② 内存精确检索
        try:
            out = with_retry(
                lambda: self._brute_force_route(
                    vec, collection_name, k, exclude_parents=True),
                f"内存精确检索（{collection_name}）",
            )
            if out:
                logger.warning(
                    f"降级成功（内存精确检索）：{collection_name} 取回 {len(out)} 条"
                )
            return out
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"【严重】{collection_name} 全部降级路径均失败，本轮该集合无召回。"
                f"原始错误：{reason}；最终错误：{str(e)[:120]}。"
                f"该问题会表现为「教材中未找到」（显式拒答），不是错误答案。"
            )
            return []

    def _dense_route(self, query: str, collections: Sequence[str],
                     k: int, precomputed_embedding=None) -> List[Dict[str, Any]]:
        """
        稠密向量路：多集合并行查询后按余弦相似度合并成一个全局排名。

        Args:
            precomputed_embedding: 已算好的查询向量（仅对第一个查询传入）。
                RAG 引擎在调用检索前已为原始查询算过一次嵌入（缓存语义匹配需要），
                传入可避免重复编码 —— 小模型单次编码约 10–30 ms，
                多路检索下这个重复会被放大。
        """
        if not collections:
            return []
        if precomputed_embedding is not None:
            vec = precomputed_embedding.tolist() if hasattr(precomputed_embedding, "tolist") \
                else list(precomputed_embedding)
        else:
            emb = self.embedding_gen.generate_query_embeddings(query)
            vec = emb.tolist()
        merged: List[Dict[str, Any]] = []
        for name in collections:
            try:
                kwargs = {"query_embeddings": [vec], "n_results": max(1, int(k))}
                if self.where:
                    kwargs["where"] = self.where
                res = with_retry(
                    lambda n=name, kw=kwargs: self.client.get_collection(n).query(**kw),
                    f"稠密检索（{name}）",
                )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"稠密检索失败（{name}）：{str(e)[:120]}；逐级降级"
                )
                merged.extend(self._dense_fallback(vec, name, k, str(e)[:120]))
                continue
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            ids = (res.get("ids") or [[]])[0]
            for i in range(len(docs)):
                m = metas[i] or {}
                if not passes_metadata_filter(m, self.exclude_content_types,
                                              self.exclude_archived):
                    continue
                merged.append({
                    "id": ids[i] if i < len(ids) else "",
                    "text": docs[i],
                    "metadata": m,
                    "collection": name,
                    # 集合为 cosine 空间 → distance = 1 - cos → 1 - distance = 余弦相似度
                    "dense_score": 1.0 - float(dists[i]),
                })
        merged.sort(key=lambda x: x["dense_score"], reverse=True)
        return merged

    def _bm25_route(self, query: str, collections: Sequence[str],
                    k: int) -> List[Dict[str, Any]]:
        """BM25 关键词路。语料为「已路由集合」的并集。"""
        q_tokens = tokenize_zh(query)
        if not q_tokens:
            return []
        merged: List[Dict[str, Any]] = []
        for name in collections:
            try:
                corpus = self._corpus(name)
            except Exception as e:  # noqa: BLE001
                # 语料读取失败（同一类 HNSW/段读取故障）不应中断整条检索：
                # 稠密路与兜底仍可工作。
                logger.warning(f"BM25 语料不可用（{name}）：{str(e)[:100]}")
                continue
            if corpus is None or corpus.bm25 is None:
                continue
            try:
                scores = corpus.bm25.get_scores(q_tokens)
            except Exception as e:  # noqa: BLE001
                logger.error(f"BM25 打分失败（{name}）：{e}")
                continue
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(1, int(k))]
            for i in order:
                if scores[i] <= 0:
                    # BM25 为 0 表示该篇不含任何查询词 —— 计入召回只会稀释 RRF
                    continue
                merged.append({
                    "id": corpus.ids[i] if i < len(corpus.ids) else "",
                    "text": corpus.docs[i],
                    "metadata": corpus.metadatas[i],
                    "collection": name,
                    "bm25_score": float(scores[i]),
                })
        merged.sort(key=lambda x: x["bm25_score"], reverse=True)
        return merged

    # ----------------------------------------------------------------- 融合
    def _rrf_fuse(self, routes: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion。

        以 (collection, id) 作为跨路去重键 —— 单独用 id 在极端情况下可能
        跨集合重名；而 Chroma 的 id 已保证集合内唯一。
        """
        acc: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for route_idx, ranked in enumerate(routes):
            for rank, item in enumerate(ranked):
                key = (item.get("collection", ""), item.get("id", "") or item.get("text", "")[:60])
                slot = acc.get(key)
                if slot is None:
                    slot = dict(item)
                    slot["rrf_score"] = 0.0
                    slot["dense_rank"] = None
                    slot["bm25_rank"] = None
                    slot["route_hits"] = []
                    acc[key] = slot
                slot["rrf_score"] += 1.0 / (self.rrf_k + rank + 1)
                slot["route_hits"].append(route_idx)
                # 保留各路的最优分数与排名（供重排特征使用）
                if "dense_score" in item:
                    if slot.get("dense_rank") is None or rank < slot["dense_rank"]:
                        slot["dense_rank"] = rank
                    slot["dense_score"] = max(
                        float(slot.get("dense_score", 0.0)), float(item["dense_score"])
                    )
                if "bm25_score" in item:
                    if slot.get("bm25_rank") is None or rank < slot["bm25_rank"]:
                        slot["bm25_rank"] = rank
                    slot["bm25_score"] = max(
                        float(slot.get("bm25_score", 0.0)), float(item["bm25_score"])
                    )
        fused = list(acc.values())
        # RRF 同分时用稠密相似度做次序稳定，避免顺序抖动
        fused.sort(key=lambda x: (x["rrf_score"], x.get("dense_score", 0.0)), reverse=True)
        # 归一化 RRF 分到 0–1（便于与阈值体系共存；仅用于展示与特征）
        if fused:
            top = fused[0]["rrf_score"] or 1.0
            for x in fused:
                x["rrf_norm"] = round(x["rrf_score"] / top, 6)
                x["score"] = float(x.get("dense_score", 0.0))   # 兼容下游「相似度」语义
        return fused

    # ------------------------------------------------------------------ 入口
    def retrieve(
        self,
        query: str,
        collections: Sequence[str],
        k: Optional[int] = None,
        query_variants: Optional[Sequence[str]] = None,
        query_embedding=None,
    ) -> Dict[str, Any]:
        """
        执行混合检索。

        Args:
            query: 原始（已归一化的）查询
            collections: 已路由的集合名列表
            k: 总候选数上限；None 时取配置 rerank_k * 2 与 dense_k 的较大者
            query_variants: 同义/子问题改写后的额外查询。
                每一个变体作为**独立的一路**参与 RRF —— 这样「按变体并查」
                不需要人为决定各变体权重。
            query_embedding: 原始查询的预计算向量（避免重复编码，可选）。

        Returns:
            {
              "candidates": [...],        # 已按 RRF 排序，含各路排名与分数
              "routes": [ {...} ],        # 每路召回了多少条（可解释性）
              "stats": {...}
            }
        """
        collections = list(collections or [])
        top_k = int(k) if k else max(self.dense_k, self.config.rerank_k * 2)

        if not collections:
            self.last_stats = {"routes": [], "dense_n": 0, "bm25_n": 0, "fused_n": 0}
            return {"candidates": [], "routes": [], "stats": dict(self.last_stats)}

        queries = [query]
        for v in (query_variants or [])[: self.max_variants]:
            v = (v or "").strip()
            if v and v != query and v not in queries:
                queries.append(v)

        routes: List[List[Dict[str, Any]]] = []
        route_meta: List[Dict[str, Any]] = []

        for qi, q in enumerate(queries):
            dense = self._dense_route(
                q, collections, self.dense_k,
                precomputed_embedding=query_embedding if qi == 0 else None,
            )
            routes.append(dense)
            route_meta.append({"type": "dense", "query_index": qi, "n": len(dense)})

        if self.bm25_enabled:
            for qi, q in enumerate(queries):
                kw = self._bm25_route(q, collections, self.bm25_k)
                routes.append(kw)
                route_meta.append({"type": "bm25", "query_index": qi, "n": len(kw)})

        fused = self._rrf_fuse(routes)[:top_k]

        self.last_stats = {
            "routes": route_meta,
            "dense_n": sum(m["n"] for m in route_meta if m["type"] == "dense"),
            "bm25_n": sum(m["n"] for m in route_meta if m["type"] == "bm25"),
            "fused_n": len(fused),
            "queries": queries,
            "bm25_enabled": self.bm25_enabled,
        }
        logger.info(
            f"混合检索：查询 {len(queries)} 个（{queries[:3]}）→ "
            f"稠密 {self.last_stats['dense_n']} / BM25 {self.last_stats['bm25_n']} "
            f"→ RRF 融合 {len(fused)} 条候选"
        )
        return {"candidates": fused, "routes": route_meta, "stats": dict(self.last_stats)}
