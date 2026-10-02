"""
Lightweight RAG Retrieval Engine for Mati Learning System
"""

import logging
import time
from typing import Dict, List, Any, Optional
import chromadb

from system.rag.anti_confusion_engine import AntiConfusionEngine
from system.rag.ascii_diagram_library import ASCIIDiagramLibrary
from system.rag.user_edge_case_handler import UserEdgeCaseHandler
from system.rag.rag_cache import RAGCache
from system.rag.retrieval_config import RetrievalConfig
from system.rag.context_budget import pack_evidence, estimate_tokens
from system.rag.text_utils import compute_confidence
from system.rag.text_cleanup import clean_text, fix_pdf_glyphs, has_glyph_issues
from system.rag.hybrid_retriever import HybridRetriever, with_retry
from system.rag.query_rewriter import QueryRewriter
from system.rag.knowledge_governance import (
    build_citations,
    is_creation_request,
    validate_citations,
)
from system.input_processing.adaptive_normalizer import AdaptiveNormalizer
from scripts.rag_data_preparation.embedding_generator import EmbeddingGenerator
from ai_model.model_utils.model_handler import ModelHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 证据元数据完整度要检查的字段（P2 / 方案附录 A）。
# 只查「有值即可」的字段；`prerequisites`（GraphRAG 预留）本轮不要求。
#
# ⚠️ chapter / section 为什么不在必需字段里
# ────────────────────────────────────────
# 最初把它们算进去，实测完整率 95.4%，缺口里 3716 块是 section、2194 块是
# chapter。查下来发现这**不是缺陷而是结构事实**：连续正文中间没有标题行，
# 本来就没有小节名可填。把它算成「缺失」会让这个指标永远差一口气，
# 反而掩盖真正的接线错误（父块曾 100% 漏写 section，已修）。
#
# 拆分口径：
#   · 标识类字段必须齐全 —— 缺了就是真 bug（书名/学科/年级/页码/状态…）
#   · 结构类字段按「有标题的块才要求」统计 —— 见 _STRUCTURAL_FIELDS
_META_REQUIRED = ("book_id", "book_title", "page", "content_type",
                  "subject", "grade", "status", "doc_type", "textbook_version")
# 结构类字段：仅当块自身带标题（section_title 非空）时才要求 chapter/section 有值
_STRUCTURAL_FIELDS = ("chapter", "section")


def _metadata_completeness(chunks: List[Dict[str, Any]]) -> float:
    """
    入选证据的元数据完整率（0–1）。

    方案 §3.4 步骤 3.1 的验收标准是「抽查 30 个块，完整率 ≥ 95%」。
    这里做成**每次查询都自动统计**并进 stage_debug，而不是一次性人工抽查 ——
    人工抽查只能证明「当时没问题」，而索引会重建、元数据会退化。

    统计口径：见上方 _META_REQUIRED / _STRUCTURAL_FIELDS 的注释。
    """
    if not chunks:
        return 0.0
    filled = 0
    total = 0
    for c in chunks:
        meta = c.get("metadata") or {}
        for f in _META_REQUIRED:
            total += 1
            v = meta.get(f)
            if v not in (None, "", 0, "unknown"):
                filled += 1
        # 只有「本块确实有标题」的块才要求结构字段齐全
        if str(meta.get("section_title") or "").strip():
            for f in _STRUCTURAL_FIELDS:
                total += 1
                if meta.get(f) not in (None, "", 0, "unknown"):
                    filled += 1
    return filled / total if total else 0.0


class RAGRetrievalEngine:
    """RAG orchestrator for Mati."""

    def __init__(
        self,
        chroma_db_path: str = "mati_data/chroma_db",
        model_path: str = "mati_data/models/qwen2_5",
        llm_handler=None,
        load_llm: bool = True,
        hybrid_enabled: Optional[bool] = None,
        rewrite_enabled: Optional[bool] = None,
        grade_mode: Optional[str] = None,
        chroma_client=None,
    ):
        """
        Args:
            load_llm: False 时跳过 1GB GGUF 加载（仅验证检索层时用）
            hybrid_enabled: 覆盖配置的混合检索开关。置 False 可复现 P0 的
                纯稠密行为，用于 A/B 消融（量化 BM25 的实际贡献）。
            rewrite_enabled: 覆盖配置的 Query 改写开关，同样用于消融。
            grade_mode: 覆盖配置的年级过滤强度（"soft"/"hard"），用于消融。
            chroma_client: 复用外部已建好的 Chroma 客户端。

                ⚠️ **同一进程内不要对同一路径创建多个 PersistentClient**。
                实测（100% 复现）：先建 c1 读一遍元数据、再建 c2 去查询，
                c2 会有 1–2 个集合间歇性报
                `Error finding id` / `Error creating hnsw segment reader: Nothing found on disk`
                —— 磁盘上段文件完好，是 Rust 后端对同路径的段状态不一致所致。
                表现为**静默零召回**（部分集合整体查不到），极难从上层察觉。
                需要同时持有元数据扫描与检索能力时，把同一个 client 传进来。
        """
        logger.info("Initializing Mati RAG Engine...")

        self.chroma_db_path = chroma_db_path

        # 检索层配置（阈值/权重/预算集中管理，见 mati_data/config/retrieval.yaml）
        self.config = RetrievalConfig.load()
        if grade_mode in ("soft", "hard"):
            self.config.data.setdefault("routing", {})["grade_mode"] = grade_mode

        self.edge_case_handler = UserEdgeCaseHandler()
        self.anti_confusion = AntiConfusionEngine()
        self.diagram_library = ASCIIDiagramLibrary()
        _cache_cfg = self.config.cache_config
        self.cache = RAGCache(
            max_size=int(_cache_cfg["max_size"]),
            ttl_seconds=int(_cache_cfg["ttl_seconds"]),
        )
        self.input_normalizer = AdaptiveNormalizer(enable_spell_check=True)
        self.embedding_gen = EmbeddingGenerator(device='cpu')

        # ChromaDB init
        if chroma_client is not None:
            # 复用外部客户端：避免同进程内出现第二个 PersistentClient
            # （见 __init__ docstring 里记录的静默零召回问题）
            self.chroma_client = chroma_client
            logger.info("ChromaDB client reused (injected)")
        else:
            try:
                self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
                logger.info(f"ChromaDB connected at {chroma_db_path}")
            except Exception as e:
                logger.error(f"ChromaDB connection failed: {e}")
                self.chroma_client = None

        # 混合检索（稠密 + BM25 + RRF）与 Query 改写
        self.retriever = HybridRetriever(
            self.chroma_client, self.embedding_gen, self.config,
            enabled=hybrid_enabled,
        )
        self.rewriter = QueryRewriter(self.config)
        if rewrite_enabled is not None:
            self.rewriter.enabled = bool(rewrite_enabled)

        # LLM init
        if llm_handler:
            self.llm = llm_handler
            logger.info("LLM Model connected (pre-loaded)")
        elif not load_llm:
            # 仅验证检索层时跳过模型加载（避免白白加载 1GB GGUF）
            self.llm = None
            logger.info("LLM loading skipped (load_llm=False)")
        else:
            try:
                self.llm = ModelHandler(model_path=model_path)
                logger.info("LLM Model connected")
            except Exception as e:
                logger.error(f"LLM connection failed: {e}")
                self.llm = None
        
        logger.info("RAG Engine initialized")
    
    def warm_up(self):
        """
        Warms up RAG engine by pre-loading embedding generator and ChromaDB.
        This reduces first query latency.
        """
        logger.info("Warming up RAG engine...")
        try:
            # Warms up embedding generator with dummy query
            dummy_query = "测试查询 test query"
            self.embedding_gen.generate_query_embeddings(dummy_query)
            
            # Warms up ChromaDB by accessing a collection
            if self.chroma_client:
                collections = self.chroma_client.list_collections()
                if collections:
                    # Queries first collection with dummy to load indexes
                    test_coll = self.chroma_client.get_collection(collections[0].name)
                    test_embedding = self.embedding_gen.generate_query_embeddings(dummy_query)
                    test_coll.query(
                        query_embeddings=[test_embedding],
                        n_results=1
                    )
            
            logger.info("RAG engine warmed up!")
        except Exception as e:
            logger.warning(f"RAG warm-up failed (non-critical): {e}")

    @staticmethod
    def _replay_cached(answer: str, stream_callback) -> None:
        """
        缓存命中时的「伪流式」回放。

        原实现按空格切词再逐词推送，中文没有空格 → 整段答案被当作一个词，
        等于一次性全量输出（伪流式失效）。此处改为按小块推送。
        """
        if not stream_callback or not answer:
            return
        step = 12
        for i in range(0, len(answer), step):
            stream_callback(answer[i:i + step])

    def _evidence_token_budget(self, conversation=None, system_prompt: str = "",
                               max_output_tokens: int = None) -> int:
        """
        按**剩余上下文窗口**动态计算证据 token 预算。

        固定预算（如 900）在长历史多轮对话下会把 2048 的窗口挤爆：
            system_prompt + history + evidence + output 必须 <= n_ctx
        这里按实际历史长度扣除，并设上下限防止极端值。

        Returns:
            证据可用的 token 数（下限 200，上限取配置 max_evidence_tokens）
        """
        cb = self.config.data["context_budget"]
        n_ctx = int(cb.get("n_ctx", 2048))
        max_out = int(max_output_tokens or cb.get("max_output_tokens", 768))

        sys_tokens = estimate_tokens(system_prompt) if system_prompt else 300
        sys_tokens += 40  # ChatML 模板标记与流式解码误差的安全余量

        hist_tokens = 0
        if conversation:
            for q, a in list(conversation)[-3:]:
                hist_tokens += estimate_tokens(f"{q or ''}{a or ''}")

        budget = n_ctx - sys_tokens - hist_tokens - max_out
        ceiling = self.config.max_evidence_tokens
        clamped = max(200, min(budget, ceiling))
        logger.info(
            f"token 预算：n_ctx={n_ctx} - system={sys_tokens} - history={hist_tokens} "
            f"- output={max_out} = {budget} -> 证据预算 {clamped}"
        )
        return clamped

    def _discover_collections(self) -> List[Dict[str, Any]]:
        """
        枚举可用集合及其 subject / discipline / grade。

        优先读取集合自身的 metadata（由摄取侧写入 subject / discipline / grade）；
        metadata 缺失时（旧库）回退到解析集合名 `neb_{key}[_grade_{n}]`，
        并用 `discipline_to_subject` 把细分名（physics）反查回家族（science），
        否则按 `Science` 提问会匹配不到 `neb_physics_grade_10`。
        这样学科路由不再依赖硬编码的集合名清单。
        """
        if not self.chroma_client:
            return []
        found: List[Dict[str, Any]] = []
        for c in self.chroma_client.list_collections():
            name = c.name
            if not self.config.is_collection_allowed(name):
                continue
            meta = getattr(c, "metadata", None) or {}
            subject = meta.get("subject")
            discipline = meta.get("discipline")
            grade = meta.get("grade")
            if not subject:
                # 回退：解析 neb_{key}[_grade_{n}]
                body = name
                for p in (self.config.get("collection_whitelist_prefixes") or ["neb_"]):
                    if body.startswith(p):
                        body = body[len(p):]
                        break
                parts = body.split("_grade_")
                key = parts[0]
                if len(parts) > 1 and not grade:
                    grade = parts[1]
                discipline = discipline or self.config.canonical_discipline(key) or ""
                subject = self.config.subject_for_collection(key)
            found.append({
                "name": name,
                "subject": self.config.canonical_subject(subject),
                "discipline": str(discipline) if discipline else "",
                "grade": str(grade) if grade not in (None, "") else "",
            })
        return found

    def _get_relevant_collections(self, subject: str, grade: str = "",
                                  discipline: str = None) -> List[str]:
        """
        依据 subject / grade / discipline 选择集合。

        策略（修复点）：
        - 以集合**元数据中的 subject（家族）**为准做匹配，而非硬编码的集合名清单。
          原实现引用了 openstax_science / finemath / gsm8k 等 7 个库中并不存在的集合，
          导致除 Science 外所有学科检索返回空集合。
        - `discipline` 可选：给定时只收窄到该细分学科的集合
          （如 subject=Science + discipline=physics → 只查物理，不串化学/生物）。
          未给出时返回家族下**全部**细分集合，保证学生端「科学」选项仍然可用。
        - grade 精确匹配的集合 + **无年级标注的集合**（`grade` 为空）一并纳入：
          人工导入的教材常常推断不出年级（落在 `neb_{subject}`），若只返回精确匹配的
          有年级集合，这些书在学生选定年级后会**完全检索不到**（静默隐身）。
        - 若该学科一个精确匹配都没有，则回退到该学科全部集合。
        - 无匹配时返回空列表，**不做跨学科兜底**（宁可返回空也不串答案）。
        """
        canonical = self.config.canonical_subject(subject)
        want_discipline = self.config.canonical_discipline(discipline) if discipline else ""
        all_colls = self._discover_collections()
        if not all_colls:
            logger.warning("未发现任何可用集合（库为空或全部被过滤）")
            return []

        same_subject = [c for c in all_colls if c["subject"] == canonical]
        if not same_subject:
            logger.warning(
                f"学科 {subject!r}（归一化为 {canonical!r}）未匹配到任何集合；"
                f"库中现有学科：{sorted({c['subject'] for c in all_colls})}"
            )
            return []

        # 细分学科收窄（仅当该学科下确实存在这个 discipline 时生效）
        if want_discipline:
            narrowed = [c for c in same_subject if c.get("discipline") == want_discipline]
            if narrowed:
                same_subject = narrowed
                logger.info(f"细分收窄：discipline={want_discipline} → "
                            f"{[c['name'] for c in narrowed]}")
            else:
                logger.info(f"细分 {want_discipline!r} 无对应集合，按家族 {canonical!r} 检索")

        # 无年级标注 = 全年级适用，永不因年级被排除
        ungraded = [c for c in same_subject if c["grade"] in ("", "unknown")]
        graded = [c for c in same_subject if c not in ungraded]
        grade_mode = self.config.grade_mode

        if grade and grade_mode == "hard":
            exact = [c for c in graded if c["grade"] == str(grade)]
            if exact:
                chosen = exact + ungraded
                logger.info(
                    f"集合路由：subject={canonical} grade={grade}（硬过滤 "
                    f"{[c['name'] for c in exact]} + 无年级 "
                    f"{[c['name'] for c in ungraded]}）"
                )
                return [c["name"] for c in chosen]
            logger.info(f"集合路由：grade={grade} 无精确匹配，回退该学科全部集合")

        # ── soft（默认）：同家族**全部年级**的集合都参与召回，年级只决定排序。
        #
        # 为什么不能用硬过滤（实测暴露的问题）：
        #   「静电场中场强的定义式」属于物理必修第三册（目录标高一），
        #   但一个高二学生在 GUI 里选「高二」时，硬过滤只会搜 grade=11 的集合，
        #   结果把知识库里确实存在的教材挡在外面，最终判为「教材中未找到」——
        #   这是**假拒答**，且用户完全无法从界面看出原因。
        #   本项目的年级口径本就存在校际差异（物理必修三征订标高一、实际多在高二上），
        #   用硬过滤等于把口径差异变成"查不到"。
        #
        # 软过滤下，年级通过两条途径发挥作用：① 同年级集合排在前面（证据顺序）；
        # ② 重排的 grade_affinity 特征（同年级 1.0 / 区间内 0.8 / 其他 0.2）。
        # 代价是搜索空间变大（如物理同时覆盖高一与高二册），用配置可切回硬过滤。
        same_subject.sort(key=lambda c: (0 if grade and c["grade"] == str(grade) else 1,
                                         c["name"]))
        names = [c["name"] for c in same_subject]
        logger.info(f"集合路由：subject={canonical} grade={grade or '-'} "
                    f"模式={grade_mode} → {names}")
        return names

    @staticmethod
    def _narrow_by_grade(ranked_chunks: List[Dict], min_same: int = 2):
        """
        soft 模式的年级收窄：同年级候选足够时丢弃跨年级候选。

        Args:
            ranked_chunks: 已重排的候选（需含 grade_affinity）
            min_same: 触发收窄所需的同年级候选数

        Returns:
            (chunks, dropped_count)。同年级不足时原样返回、dropped=0 —— 
            保证跨年级问题（高二学生问必修三内容）依然能被兜住。
        """
        same = [c for c in ranked_chunks
                if float(c.get("grade_affinity", 0.0)) >= 0.8]
        if len(same) >= max(1, int(min_same)):
            return same, len(ranked_chunks) - len(same)
        return ranked_chunks, 0

    def _fetch_parents(self, parent_ids_by_collection: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        按集合批量取父块（一次 get 调用取多个 id，避免 N 次往返）。

        Returns:
            {parent_id: {"text": ..., "metadata": ...}}
        """
        out: Dict[str, Dict] = {}
        for coll_name, ids in parent_ids_by_collection.items():
            if not ids:
                continue
            try:
                # col.get 同样要建段读取器，会碰到与稠密检索同一类瞬发失败
                # （见 hybrid_retriever.with_retry 的说明）
                got = with_retry(
                    lambda n=coll_name, i=ids: self.chroma_client.get_collection(n).get(
                        ids=list(dict.fromkeys(i)), include=["documents", "metadatas"]),
                    f"父块读取（{coll_name}）",
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"父块读取失败（{coll_name}）：{str(e)[:100]}")
                continue
            for i, pid in enumerate(got.get("ids") or []):
                docs = got.get("documents") or []
                metas = got.get("metadatas") or []
                out[pid] = {
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": (metas[i] if i < len(metas) else {}) or {},
                }
        return out

    def _expand_to_parents(self, chunks: List[Dict]) -> tuple:
        """
        父子索引（P1）：把命中的**子块**替换为其**父块**文本，供生成阶段使用。

        为什么要把召回单元与生成单元解耦
        ────────────────────────────────
        子块短（~384 字符）→ 向量表达集中、召回精准；
        父块长（~1400 字符）→ 是一个完整小节，语义完整。
        用父块去召回会因为向量被稀释而降低相似度，用子块去生成则会
        把定理陈述、例题题干与解析拆散（正是语雀文档强调的「步骤 A/B 被拆分
        无法一起召回」）。

        返回:
            (chunks, stats)。stats 含请求父块数、命中数、去重后条数。
            父块缺失时**保留子块原文** —— 宁可给半截证据，也不要凭空丢证据。
        """
        stats = {"requested": 0, "found": 0, "deduped": 0}
        if not chunks:
            return chunks, stats

        by_coll: Dict[str, List[str]] = {}
        for c in chunks:
            meta = c.get("metadata") or {}
            pid = meta.get("parent_id")
            coll = c.get("collection", "")
            if not pid or not coll:
                # 缺 collection 时无法定位父块；静默跳过而不是抛错，
                # 但下面会对「有 parent_id 却没 collection」单独记一条警告，
                # 因为那说明上游把字段丢了（曾经真的发生过）。
                if pid and not coll:
                    stats.setdefault("missing_collection", 0)
                    stats["missing_collection"] += 1
                continue
            stats["requested"] += 1
            by_coll.setdefault(coll, []).append(pid)

        if not by_coll:
            if stats.get("missing_collection"):
                logger.warning(
                    f"父子展开被跳过：{stats['missing_collection']} 条候选带 parent_id "
                    f"但缺少 collection 字段（检查重排是否透传了 collection）"
                )
            return chunks, stats

        parents = self._fetch_parents(by_coll)
        stats["found"] = len(parents)

        seen_parents = set()
        expanded: List[Dict] = []
        for c in chunks:
            meta = c.get("metadata") or {}
            pid = meta.get("parent_id")
            if not pid or pid not in parents:
                expanded.append(c)
                continue
            # 多个子块命中同一父块时只保留一份，避免证据预算被重复内容吃掉
            if pid in seen_parents:
                continue
            seen_parents.add(pid)
            item = dict(c)
            item["text"] = parents[pid]["text"]
            # 合并元数据：父块的 content_type / chapter 等更完整，
            # 但 source / book_title 取子块（两者一致，子块更贴近命中位置）
            merged_meta = dict(parents[pid]["metadata"])
            merged_meta.update({k: v for k, v in meta.items() if v not in (None, "")})
            item["metadata"] = merged_meta
            item["expanded_from_child"] = c.get("id") or ""
            expanded.append(item)

        stats["deduped"] = len(expanded)
        logger.info(
            f"父子展开：{stats['requested']} 个子块请求父块，取到 {stats['found']} 个，"
            f"去重后证据 {len(chunks)} -> {len(expanded)} 条"
        )
        return expanded, stats

    def query(
        self,
        query_text: str,
        subject: str,
        grade: str = None,
        discipline: str = None,
        n_results: int = None,
        stream_callback=None,
        status_callback=None,
        conversation=None,
        already_normalized: bool = False,
        capture_stages: bool = False,
    ) -> Dict[str, Any]:
        """
        Main query method, with streaming support.

        Args:
            subject: 学科（GUI 键 / 中文名 / 规范值均可，内部做别名归一化）
            grade:   年级编码（"10"/"11"/"12"）。原实现完全没有该参数，
                     导致「按年级检索」在代码层面不存在；此处贯通到集合路由与缓存键。
                     年级只做**软加权与集合排序**，不做硬过滤，避免把学生实际
                     在学的书挡掉（如物理必修第三册征订标高一、实际多在高二上）。
            discipline: 细分学科（可选），如 "物理"/"physics"。
                     给出时只检索该细分学科的集合；不给则检索整个学科家族
                     （科学 = 物理 + 化学 + 生物）。
            n_results: 每集合召回的候选数；None 时取配置 candidate_k_per_collection。
            stream_callback: 接收最终答案 token 的回调（答案正文）。
            status_callback: 接收运行状态消息的回调（如“正在搜索…”），
                             与答案正文分离，避免状态流入答案框（方案A）。
            conversation:     多轮对话历史，[(问题, 回答), ...]，用于追问承接（方案C）。
            already_normalized: 调用方已完成归一化时置 True，避免同一问题被归一化两次
                             （GUI 已先做一次 AdaptiveNormalizer，原先引擎内会再做一次）。
            capture_stages: 置 True 时在返回结果里附带 `stage_debug`，
                             记录改名/路由/候选池/精排池/证据/生成六个阶段的中间产物，
                             供分阶段评测（scripts/eval_rag.py）与失败归因使用。
                             默认 False，避免正常问答携带大对象。
        """
        start_time = time.time()
        grade_key = str(grade) if grade not in (None, "") else ""

        # discipline 推导：GUI 的学科下拉已支持「物理/化学/生物」等细分学科，
        # 但只传 subject 时（subject_aliases 把「物理」映射到家族 science），
        # 路由会退化为整个学科家族，选中「物理」却查出化学内容。
        # 这里在未显式给 discipline 时，尝试从 subject 反推细分学科。
        if not discipline:
            discipline = self.config.canonical_discipline(subject) or None

        def _emit_status(msg):
            """状态消息优先发给 status_callback，否则退回 stream_callback。"""
            if status_callback:
                status_callback(msg)
            elif stream_callback:
                stream_callback(msg)

        # Edge case check
        edge_response = self.edge_case_handler.check_edge_cases(query_text)
        if edge_response:
            if stream_callback:
                stream_callback(edge_response)
            return {
                "answer": edge_response,
                "sources": [],
                "processing_time": time.time() - start_time,
                "type": "edge_case"
            }

        if not self.chroma_client:
            error_msg = "错误：数据库未连接。"
            if stream_callback:
                stream_callback(error_msg)
            return {"answer": error_msg, "type": "error"}
        
        if already_normalized:
            # 调用方（GUI）已完成归一化，此处不再重复处理，
            # 避免对已是改写结果的文本做二次改写引入偏差。
            clean_question = query_text
            normalization_result = {"clean_question": query_text, "notes": [],
                                    "intent": "", "confidence": 1.0}
        else:
            normalization_result = self.input_normalizer.normalize(query_text)
            clean_question = normalization_result["clean_question"]
        
        if normalization_result["notes"]:
            logger.info(f"Normalization applied: {normalization_result['notes']}")
            logger.info(f"Intent: {normalization_result['intent']}, Confidence: {normalization_result['confidence']:.2f}")
        
        effective_query = clean_question if clean_question else query_text

        # 多轮对话时不使用缓存（保证追问上下文不被旧缓存打断）
        if not conversation:
            cached_result = self.cache.get(query_text, subject, grade_key)
            if cached_result:
                logger.info("Cache HIT (exact)")
                self._replay_cached(cached_result.get("answer", ""), stream_callback)
                return {**cached_result, "processing_time": time.time() - start_time}

        _emit_status("🔍 正在搜索知识库...\n\n")
        _emit_status("📊 正在分析您的问题...\n\n")
        
        # BGE Chinese model: retrieval queries use the official query instruction.
        query_embedding = self.embedding_gen.generate_query_embeddings(effective_query)

        if not conversation:
            semantic_hit = self.cache.find_similar(
                query_embedding, subject, grade_key,
                threshold=float(self.config.cache_config["semantic_threshold"]),
            )
            if semantic_hit:
                logger.info("Cache HIT (semantic)")
                self._replay_cached(semantic_hit.get("answer", ""), stream_callback)
                return {**semantic_hit, "processing_time": time.time() - start_time}

        target_collections = self._get_relevant_collections(subject, grade or "", discipline)
        if not target_collections:
            logger.warning(f"No collections found for subject={subject!r} "
                           f"grade={grade!r} discipline={discipline!r}")

        _emit_status("🎯 正在寻找最佳匹配...\n\n")

        # ── Query 改写（P1）：术语同义扩展 + 子问题拆解，产出**追加路**而非替换原查询。
        # 改写是有损的，因此原始查询始终保留为其中一路，多路结果由 RRF 融合。
        rewrite_info = self.rewriter.rewrite(effective_query, subject, discipline)
        variants = rewrite_info["variants"]

        # 候选池大小：原实现用 min(2, n_results) 把每集合召回的块数硬压到 2，
        # 导致整个候选池最多 6 条，远小于业界 20–50 条的常规做法。
        candidate_k = int(n_results) if n_results else self.config.candidate_k

        # ── 混合检索（P1）：稠密向量路 + BM25 关键词路，RRF 融合。
        # retriever.last_stats 记录各路召回数，供分阶段评测（scripts/eval_rag.py）读取。
        retrieved = self.retriever.retrieve(
            effective_query, target_collections, k=candidate_k,
            query_variants=variants, query_embedding=query_embedding,
        )
        raw_results = retrieved["candidates"]
        retrieve_stats = retrieved["stats"]

        # 粗过滤：只对「纯稠密命中」施加相似度下限。
        # BM25 单独命中的候选 dense_score 可能很低（甚至趋近 0），但那正是
        # 术语精确匹配的价值所在 —— 若一律用相似度阈值过滤，等于白做混合检索。
        min_base = self.config.min_base_score
        filtered = []
        dropped_low = 0
        for c in raw_results:
            if c.get("bm25_rank") is None and float(c.get("dense_score", 0.0)) < min_base:
                dropped_low += 1
                continue
            filtered.append(c)
        logger.info(
            f"候选池：融合 {len(raw_results)} 条（稠密 {retrieve_stats.get('dense_n', 0)} / "
            f"BM25 {retrieve_stats.get('bm25_n', 0)}），相似度过滤丢弃 {dropped_low} 条，"
            f"余 {len(filtered)} 条"
        )

        ranked_chunks = self.anti_confusion.rank_results(
            filtered, effective_query, grade=grade, query_variants=variants
        )

        # ── 年级收窄（soft 模式配套规则）
        # 软过滤让全部年级参与召回，代价是证据里可能混入跨年级块。
        # 实测：问“什么是向心力”（高一必修二）时，高二选必二里讲
        # 「带电粒子在磁场中的圆周运动」的块也会被召回 —— 它是相关的，
        # 但对高一学生的直接问题而言不如必修二贴切，却会占掉证据预算。
        # 规则：**同年级候选足够时不引入跨年级噪声；不足时才扩展**。
        # 这样「跨年级能兜住」与「同年级更精准」两个目标可以同时成立。
        if grade and self.config.grade_mode == "soft":
            min_same = int((self.config.get("routing") or {}).get(
                "same_grade_min_evidence", 2))
            ranked_chunks, dropped = self._narrow_by_grade(ranked_chunks, min_same)
            if dropped:
                logger.info(f"年级收窄：丢弃跨年级候选 {dropped} 条，"
                            f"保留同年级 {len(ranked_chunks)} 条")
            else:
                logger.info("年级收窄：同年级候选不足，保留全部年级候选（跨年级兜底）")

        # 融合阈值（权重可配；来源权重已压到 0.05，不再随来源漂移阈值）
        min_final = self.config.min_final_score
        rank_pool = [c for c in ranked_chunks if c.get('final_score', 0) >= min_final]
        rank_pool = rank_pool[: self.config.rerank_k]

        # ── 父子索引（P1）：命中子块后取父块喂模型。
        # 子块短而精准（用于召回），父块是一个完整小节（用于生成）。
        # 这一步必须在 token 预算裁剪之前做，否则会按子块长度误判"放不下"。
        rank_pool, parent_stats = self._expand_to_parents(rank_pool)

        # ── 字形兜底清理
        # 索引重建后正文本身已经干净；这里再兜一次，覆盖「尚未重建的旧库」
        # 与「从外部内容包导入的历史数据」。
        # 为什么必须保证上下文干净：大模型会**照抄**上下文里的字符 ——
        # 学生在答案里看到 狓/狔，根因就是上下文里就是 狓/狔。
        glyph_cleaned = 0
        for c in rank_pool:
            t = c.get("text") or ""
            if has_glyph_issues(t):
                c["text"] = fix_pdf_glyphs(t)
                glyph_cleaned += 1
        if glyph_cleaned:
            logger.warning(
                f"证据字形兜底清理：{glyph_cleaned}/{len(rank_pool)} 条命中"
                f"（该集合可能尚未用新版抽取逻辑重建索引）"
            )

        # 统一上下文预算：只在这一处按 token 截断。
        # 原实现先在检索层按 1600 字符截断、qwen_handler 又按 850 字符截断，
        # 且循环用 break 而非 continue，导致高分证据被静默丢弃。
        budget = self.config.data["context_budget"]
        # 证据预算按剩余上下文窗口动态计算（多轮对话时历史会占用窗口）
        system_prompt = ""
        if self.llm is not None:
            try:
                system_prompt = getattr(self.llm.handler, "system_prompt", "") or ""
            except Exception:  # noqa: BLE001
                system_prompt = ""
        evidence_budget = self._evidence_token_budget(
            conversation=conversation, system_prompt=system_prompt
        )
        ordered_chunks, full_context_str, budget_stats = pack_evidence(
            rank_pool,
            max_tokens=evidence_budget,
            tokens_per_char_cjk=float(budget["tokens_per_char_cjk"]),
            tokens_per_char_ascii=float(budget["tokens_per_char_ascii"]),
            max_items=self.config.evidence_k,
        )
        logger.info(
            f"证据组装：精排池 {len(rank_pool)} 条 -> 入选 {budget_stats['selected']} 条"
            f"（上限 {self.config.evidence_k}），"
            f"token {budget_stats['tokens_used']}/{budget_stats['tokens_budget']}"
            f"（过长跳过 {budget_stats['skipped_too_large']}，"
            f"超数量上限跳过 {budget_stats.get('skipped_over_max_items', 0)}）"
        )
        context_texts = [c['text'] for c in ordered_chunks]

        gen_cfg = self.config.generation_config

        # ── 溯源引用编号（P2 / 步骤 3.4.1）
        # **编号与来源描述全部由检索层生成**，模型只写 [n]。
        # 这是对 1.5B 模型的对冲：它可以不听指令不写标记（我们只统计不惩罚），
        # 但无法编造「[7] 来自某本不存在的教材」—— 7 号根本不存在。
        citation_table: List[Dict[str, Any]] = []
        require_citations = bool(gen_cfg.get("require_citations", False))
        if ordered_chunks and require_citations:
            sources_block, citation_table = build_citations(ordered_chunks)
            if sources_block:
                # 来源清单进上下文，但**不计入证据预算**（它是几十个 token 的
                # 索引行，不是知识内容；且已经排在所有证据之后，不会挤掉正文）。
                full_context_str = full_context_str + "\n\n" + sources_block
                logger.info(
                    f"溯源引用：生成 {len(citation_table)} 条来源编号，"
                    f"要求模型标注={require_citations}"
                )

        best_score = max([c.get('final_score', 0.0) for c in ordered_chunks], default=0.0)
        # 拒答判据用**查询覆盖率**而不是 final_score。
        # 原因（实测）：final_score 里含多项「常量特征」（来源 0.05 + 结构 0.10 +
        # 版本 0.05 + 年级 0.05），给每个候选一个约 0.2 的下限；BM25 归一化又是
        # 批内相对值，每批必有一个候选拿满分。结果负样本（库外问题）的
        # final_score 最高 0.78，而正样本最低 0.46 —— 完全不可分，拿它拒答
        # 必然失效。中文二元组覆盖率是绝对量：问题里的「词」是否真的出现在块里。
        best_query_coverage = max(
            [float(c.get('query_coverage', 0.0)) for c in ordered_chunks], default=0.0
        )
        refuse_cov_threshold = float(gen_cfg.get("refuse_below_query_coverage", 0.30))
        no_evidence = (not ordered_chunks) or (best_query_coverage < refuse_cov_threshold)

        # 创作类请求的补充拒答判据（与覆盖率**正交**，见 knowledge_governance 注释）。
        # 实测：「帮我写一篇 800 字的散文」覆盖率 0.71 远高于阈值 —— 因为
        # 「散文」「800」「字」在语文教材里天然高频 —— 于是模型拿着几段
        # 描写景色的课文开始写散文。覆盖率回答的是「词有没有出现」，
        # 回答不了「教材里有没有现成答案」。
        creation_request = is_creation_request(effective_query or query_text)
        refuse_creation = creation_request and bool(
            gen_cfg.get("refuse_creation_requests", True))

        _emit_status("✨ Qwen 大模型正在生成答案...\n\n")

        answer = ""
        confidence = 0.0
        llm_used = False
        grounded = None

        # 零证据/证据不相关：明确拒答，而不是让模型凭参数化知识自由发挥
        # （教育场景下幻觉的代价高于拒答）
        if refuse_creation:
            answer = ("✍️ 这是写作/创作类请求，教材检索帮不上忙。\n\n"
                      "本系统按教材内容回答问题，不代写作文。"
                      "如果你想学写法，可以问「这类文章的结构是怎样的」"
                      "「这句话用了什么修辞」，我会结合教材给你讲。")
            logger.info("拒答：识别为创作类请求（教材中没有现成答案）")

        elif no_evidence and gen_cfg.get("refuse_when_no_evidence", True):
            answer = ("📚 教材中未找到与这个问题直接相关的内容。\n\n"
                      "可以试试：换一种问法、补充教材版本或章节，"
                      "或者确认该内容是否属于已收录的教材范围。")
            logger.info(
                f"拒答：入选证据 {len(ordered_chunks)} 条，最高查询覆盖率 "
                f"{best_query_coverage:.2%} < 阈值 {refuse_cov_threshold:.0%}"
            )

        elif self.llm:
            llm_used = True
            try:
                if stream_callback:
                    answer = ""
                    for token in self.llm.handler.get_answer_stream(
                        effective_query, full_context_str, history=conversation
                    ):
                        answer += token
                        stream_callback(token)

                    confidence = self._calculate_confidence(
                        answer, effective_query, ordered_chunks
                    )
                else:
                    answer, confidence = self.llm.simple_handler.get_answer(
                        effective_query, full_context_str, history=conversation
                    )
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                llm_used = False
                if stream_callback:
                    stream_callback("生成答案时出错，请重试。")
        else:
            # 方案B：模型未加载时明确提示，并附上检索结果摘要（不静默返回）
            if context_texts:
                snippet = context_texts[0][:300]
                answer = ("⚠️ 大模型未加载，当前仅返回检索结果，无法进行智能问答。\n\n"
                          "以下是与问题相关的知识库内容：\n" + snippet)
            else:
                answer = "⚠️ 大模型未加载，且未检索到相关资料。请重启应用确认模型加载正常。"

        # 输出后忠实度校验（原实现中 validate_grounding 从未被调用，形同死代码）
        # 分级处理：
        #   admitted      -> 模型自认无依据（硬信号）：追加免责提示 + 置信度上限 0.4
        #   weak_overlap  -> 内容覆盖度低（软信号，对中文区分力弱）：只降置信度，不加横幅
        grounding = None
        if gen_cfg.get("verify_grounding", True) and answer and llm_used:
            try:
                grounding = self.anti_confusion.check_grounding(answer, context_texts)
            except Exception as e:
                logger.warning(f"忠实度校验异常（已跳过）：{e}")
                grounding = None
            if grounding:
                if grounding["severe"]:
                    logger.warning(f"忠实度校验未通过：{grounding['reason']}")
                    answer = ("⚠️ 以下回答未能在教材中找到充分依据，仅供参考。\n\n" + answer)
                    confidence = min(confidence, 0.4)
                elif grounding["status"] == "weak_overlap":
                    # 仅软折扣，避免对有据但改写幅度大的答案误报
                    confidence = round(confidence * 0.7, 3)

        diagram = self.diagram_library.find_diagram_by_text(query_text)

        # ── 引用校验（P2 / 步骤 3.4.2，确定性规则，不依赖模型自觉）
        # 只统计、只降置信度，**不改答案**：
        # 编号写错（如写 [5] 而实际只有 3 条证据）不影响内容正确性，
        # 擅自删改模型输出反而会掩盖问题 —— 与忠实度校验同一原则。
        citation_check: Optional[Dict[str, Any]] = None
        if citation_table and answer and llm_used:
            try:
                citation_check = validate_citations(answer, citation_table)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"引用校验异常（已跳过）：{e}")
                citation_check = None
            if citation_check:
                logger.info(
                    f"引用校验：标记 {citation_check['n_cited']} 处，"
                    f"有效 {citation_check['n_valid']}，"
                    f"越界 {citation_check['n_out_of_range']}"
                    + (f"（{citation_check['out_of_range_refs']}）"
                       if citation_check["out_of_range_refs"] else "")
                )
                if citation_check["n_out_of_range"] > 0:
                    confidence = round(
                        confidence * float(gen_cfg.get("citation_penalty", 0.85)), 3)

        # ── 答案规整：GUI 的答案框是纯文本控件，渲染不了 LaTeX（学生会直接
        # 看到 \frac、\[ 这类反斜杠）；模型也可能把上下文里的错映射字符照抄出来。
        # 放在忠实度与引用校验**之后**：两者都针对模型原始输出，不被规整影响。
        clean_answer = clean_text(answer)
        if clean_answer != answer:
            logger.info("答案已规整（LaTeX 标记降级 / 字形残余清理）")
            answer = clean_answer

        result = {
            "answer": answer,
            "context_used": context_texts,
            "sources": [c['metadata'] for c in ordered_chunks],
            "diagram": diagram,
            "confidence": confidence,
            "processing_time": time.time() - start_time,
            "type": ("creation_refused" if refuse_creation
                     else "no_evidence" if (no_evidence
                                           and gen_cfg.get("refuse_when_no_evidence", True))
                     else "rag_response"),
            "llm_used": llm_used,
            "grounding": grounding,
            "grounded": None if not grounding else grounding["status"] == "grounded",
            # ── P2 溯源与引用（供 GUI 展示「依据」与评测统计引用准确率）
            "citations": citation_table,
            "citation_check": citation_check,
            "best_evidence_score": round(best_score, 4),
            "best_query_coverage": round(best_query_coverage, 4),
            "collections_used": target_collections,
            "context_stats": budget_stats,
        }

        # ── 检索失败回流（P1）：零证据或最高分过低的问题落盘，
        # 供 pattern_miner 挖掘高频失败模式、反哺同义词表。
        # 只记真实失败，不记「模型拒答」以外的生成层问题（那是另一类归因）。
        try:
            if self.rewriter.is_failure(best_score, len(ordered_chunks)):
                self.rewriter.log_failure(
                    effective_query, subject=subject, grade=grade,
                    discipline=discipline, best_score=best_score,
                    collections=target_collections, variants=variants,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"检索失败回流异常（已忽略）：{e}")

        if capture_stages:
            result["stage_debug"] = {
                "rewrite": rewrite_info,
                "routed_collections": list(target_collections),
                "retrieve": retrieve_stats,
                "candidate_pool": [
                    {
                        "rank": i,
                        "book_id": (c.get("metadata") or {}).get("book_id", ""),
                        "collection": c.get("collection", ""),
                        "dense_score": round(float(c.get("dense_score", 0.0)), 4),
                        "dense_rank": c.get("dense_rank"),
                        "bm25_rank": c.get("bm25_rank"),
                        "rrf_norm": c.get("rrf_norm"),
                    }
                    for i, c in enumerate(raw_results)
                ],
                "reranked": [
                    {
                        "rank": i,
                        "book_id": (c.get("metadata") or {}).get("book_id", ""),
                        "final_score": round(float(c.get("final_score", 0.0)), 4),
                        "features": c.get("features", {}),
                    }
                    for i, c in enumerate(ranked_chunks[: self.config.rerank_k])
                ],
                "parents": parent_stats,
                "evidence": {
                    "n": len(ordered_chunks),
                    "book_ids": [(c.get("metadata") or {}).get("book_id", "")
                                 for c in ordered_chunks],
                    "tokens_used": budget_stats.get("tokens_used", 0),
                    "tokens_budget": budget_stats.get("tokens_budget", 0),
                    # 精排池（裁剪前）的总 token —— 证据保留率 = used / pool
                    "pool_tokens": sum(
                        estimate_tokens(c.get("text", "")) for c in rank_pool
                    ),
                    "best_query_coverage": round(best_query_coverage, 4),
                    "refuse_threshold": refuse_cov_threshold,
                    # 最佳稠密相似度：与覆盖率互补的第二个判据，
                    # 供评测脚本做二维标定（单看覆盖率会把口语提问误拒）。
                    "best_dense": round(max(
                        [float(c.get("dense_score", 0.0)) for c in raw_results] or [0.0]
                    ), 4),
                    # ── P2 知识治理可观测项
                    # 噪声块是否真的被挡住了：若入选证据里出现 toc/cover，
                    # 说明检索侧过滤没生效（而不是「结构分降权就够了」）。
                    "noise_in_evidence": sum(
                        1 for c in ordered_chunks
                        if str((c.get("metadata") or {}).get("content_type") or "")
                        in ("toc", "cover")
                    ),
                    "archived_in_evidence": sum(
                        1 for c in ordered_chunks
                        if str((c.get("metadata") or {}).get("status") or "")
                        == "archived"
                    ),
                    # 证据的元数据完整度（方案 §3.4 步骤 3.1 验收：≥95%）
                    "meta_completeness": round(_metadata_completeness(ordered_chunks), 4),
                },
                "citations": {
                    "enabled": require_citations,
                    "n": len(citation_table),
                    "check": citation_check,
                },
            }

        # 错误/降级答案不入缓存（原实现会把错误文案也缓存 1 小时）
        #
        # ⚠️ 这段代码此前位于 `return result` **之后**，永远不可达 ——
        # 意味着 P0 以来缓存从未真正写入过，每次提问都重跑检索 + 1.5B 生成。
        # 「读缓存」那条路径一直在，只是命中率恒为 0，因此没有任何报错。
        # 属于典型的「不抛异常的功能缺失」，只能靠通读代码发现。
        cacheable = llm_used and bool(ordered_chunks)
        if not conversation and (cacheable or self.config.cache_config.get("cache_errors")):
            try:
                self.cache.set(query_text, subject, grade_key, result,
                               embedding=query_embedding)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"缓存写入失败（已忽略）：{e}")

        return result

    def _calculate_confidence(self, answer: str, question: str, context_chunks: list = None) -> float:
        """
        计算答案置信度（实现见 system/rag/text_utils.compute_confidence）。

        修复点：原实现用 len(answer.split()) 数词，中文没有空格 →
        整段中文被当成 1 个「词」→ 恒命中「过短」分支 → 置信度永远 0.3。
        """
        return compute_confidence(answer, question, context_chunks)