"""
Anti-Confusion Engine for Mati Learning System

5-Layer Defense System for high-quality, curriculum-aligned answers.

Layers:
1. Mandatory Filtering (Subject + Grade)
2. Source Prioritization (NEB Notes > HuggingFace)
3. Context Ranking (Relevance Scoring)
4. Strict Grounding (Hallucination Prevention)
5. Source Attribution (Transparency)
"""

import logging
from typing import List, Dict, Tuple, Optional
import re

from system.rag.text_utils import cjk_units as _cjk_units, content_overlap

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AntiConfusionEngine:
    """
    Filters, ranks, and verifies RAG context.
    Ensures quality answers for educational use.

    权重与优先级来自 mati_data/config/retrieval.yaml（配置缺失时用内置默认值）。
    """
    
    # 来源优先级（0–1）。原实现取值 0–100 且权重占 0.30，
    # 会让「来源」压过「相关性」，并使同一个相关性阈值随来源漂移。
    SOURCE_PRIORITY = {
        'neb': 1.0,
        'openstax': 0.7,
        'khanacademy': 0.65,
        'finemath': 0.6,
        'gsm8k': 0.6,
        'scienceqa': 0.6,
        'default': 0.5,
    }

    # 默认权重（与 retrieval.yaml 的 rank_weights 一致）。
    # P1 由 4 特征扩展为 7 特征：新增 bm25（关键词路）、structure_bonus（结构信号）、
    # version（教材版本），并把来源权重进一步压到 0.05。
    DEFAULT_WEIGHTS = {
        'similarity': 0.40,
        'bm25': 0.20,
        'term_coverage': 0.15,
        'structure_bonus': 0.10,
        'source_priority': 0.05,
        'version': 0.05,
        'grade_affinity': 0.05,
    }

    # 版本特征的年份参考区间（用于把「2019版」「2022版」映射到 0–1）
    _VERSION_YEAR_MIN = 2015
    _VERSION_YEAR_MAX = 2025
    
    # 模型自曝「无依据」的话术（中英双语）—— 这是**硬信号**：
    # 模型自己认为没找到依据时，应当明确提示而不是照常输出。
    FORBIDDEN_PHRASES = [
        "i don't have that information",
        "my knowledge cutoff",
        "unrelated to the context",
        "based on general knowledge",
        "i couldn't find",
        "not mentioned in the context",
        "根据常识",
        "根据一般常识",
        "我推测",
        "一般来说",
        "教材中没有",
        "材料中未找到",
    ]

    def __init__(self, config=None):
        """
        Args:
            config: system.rag.retrieval_config.RetrievalConfig 实例（可选）。
                    未提供时尝试自行加载，失败则用内置默认值。
        """
        self.config = config
        if self.config is None:
            try:
                from system.rag.retrieval_config import RetrievalConfig
                self.config = RetrievalConfig.load()
            except Exception as e:
                logger.warning(f"检索配置加载失败，使用内置默认值：{e}")
                self.config = None

        if self.config is not None:
            try:
                self.SOURCE_PRIORITY = dict(self.config.source_priority)
                pulled = dict(self.config.weights)
                # 配置可能来自旧版（P0 的 4 特征键名），缺键时用默认值补齐，
                # 避免 KeyError 让整条检索链路挂掉。
                missing = set(self.DEFAULT_WEIGHTS) - set(pulled)
                if missing:
                    logger.warning(
                        f"rank_weights 缺少键 {sorted(missing)}，已用内置默认值补齐"
                    )
                    merged = dict(self.DEFAULT_WEIGHTS)
                    merged.update(pulled)
                    pulled = merged
                self.weights = pulled
            except Exception:  # noqa: BLE001
                self.weights = dict(self.DEFAULT_WEIGHTS)
        else:
            self.weights = dict(self.DEFAULT_WEIGHTS)

        # 内容覆盖度软阈值（仅用于降低置信度，不触发免责横幅）
        self.min_content_overlap = 0.15
        if self.config is not None:
            try:
                self.min_content_overlap = float(
                    self.config.get("grounding", {}).get("min_content_overlap", 0.15)
                )
            except Exception:  # noqa: BLE001
                self.min_content_overlap = 0.15

        self.RELEVANCE_THRESHOLD = 0.6  # 保留：最低相似度参考值

    def calculate_priority_score(self, source_type: str) -> float:
        if not source_type:
            return self.SOURCE_PRIORITY.get('default', 0.5)
        
        source = source_type.lower()
        
        # Checks for NEB first (highest priority)
        if 'neb' in source:
            return self.SOURCE_PRIORITY.get('neb', 1.0)
        
        # Checks other sources
        for key, score in self.SOURCE_PRIORITY.items():
            if key in source:
                return score
        
        return self.SOURCE_PRIORITY.get('default', 0.5)

    def _grade_affinity(self, metadata: Dict, grade: Optional[str]) -> float:
        """
        年级亲和度（软加权，不做硬过滤）。

        原实现用 `str(metadata['grade']) in query_text`，而 grade 全为 "unknown"、
        用户提问也不会包含 "unknown"，该分支恒不命中（死分支）。
        这里改为比较请求年级与块所属年级/年级区间。
        """
        if not grade:
            return 0.5
        want = str(grade)
        block_grade = str(metadata.get('grade') or '')
        span = str(metadata.get('grade_span') or '')
        if block_grade == want:
            return 1.0
        if want in [s.strip() for s in span.split(',') if s.strip()]:
            return 0.8
        return 0.2

    def _version_feature(self, metadata: Dict) -> float:
        """
        教材版本新旧特征（0–1，新版更高）。

        数据来源：metadata 的 `edition`（如「2019版」）或 `book_volume`。
        解析不出年份时返回 0.5（中性）—— **不猜测**，避免凭版本号编造偏好。

        说明：当前库内教材版本集中在 2019/2022，该特征的信息量很低
        （几乎所有块同分）。保留它是为了将来引入新版教材时，
        版本冲突治理（方案 §3.4 步骤 3.3）能直接复用，不需要改结构。
        """
        raw = str(metadata.get('edition') or metadata.get('book_volume') or '')
        m = re.search(r'(19|20)\d{2}', raw)
        if not m:
            return 0.5
        year = int(m.group(0))
        lo, hi = self._VERSION_YEAR_MIN, self._VERSION_YEAR_MAX
        return max(0.0, min(1.0, (year - lo) / float(hi - lo)))

    def _structure_bonus(self, metadata: Dict) -> float:
        """内容类型 -> 结构加权分（定义/定理/例题优先于目录/封面/图注）。"""
        if self.config is not None:
            try:
                return float(self.config.structure_bonus_for(metadata.get('content_type')))
            except Exception:  # noqa: BLE001
                pass
        table = {'definition': 1.0, 'theorem': 1.0, 'example': 0.8, 'exercise': 0.6,
                 'body': 0.7, 'figure_caption': 0.3, 'toc': 0.0, 'cover': 0.0}
        return table.get(str(metadata.get('content_type') or '').lower(), 0.5)

    def rank_results(
        self, 
        results: List[Dict], 
        query_text: str,
        grade: Optional[str] = None,
        query_variants: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        特征加权重排（P1）。

        相较 P0 的 4 特征线性加权，新增三路特征并重新分配权重：

            final = w_sim      * 余弦相似度
                  + w_bm25     * BM25 分（批内归一化）
                  + w_term     * 问题词元在块中的覆盖率
                  + w_struct   * 结构信号（定义/定理/例题 > 目录/封面）
                  + w_source   * 来源优先级（≤ 0.05，仅轻微倾斜）
                  + w_version  * 教材版本新旧
                  + w_grade    * 年级亲和（软加权，不做硬过滤）

        **BM25 分必须做批内归一化**：rank_bm25 的原始分是无界正数，
        不归一化会让它凭绝对量级压过其余特征（与「来源权重污染阈值」同一类错误）。

        Args:
            query_variants: 改写后的同义/子问题变体。Query Coverage 取
                「原句与各变体」的**最大值** —— 学生用口语提问（"摩尔浓度怎么算来着"）
                时，原句与教材措辞不重合，但同义变体（"物质的量浓度"）会重合。
                只算原句会让这类问题被误判为无依据（实测误拒率 18%，其中
                绝大多数是口语化提问）。
        """
        if not results:
            return []

        variants = [query_text] if query_text else []
        for v in (query_variants or []):
            v = (v or "").strip()
            if v and v not in variants:
                variants.append(v)

        # BM25 批内归一化（按本批最大值）。全为 0 时该特征对排序无影响。
        bm25_raw = [float(r.get('bm25_score') or 0.0) for r in results]
        bm25_max = max(bm25_raw) if bm25_raw else 0.0

        ranked_chunks = []
        q_units = _cjk_units(query_text)
        w = self.weights
        
        for res in results:
            metadata = res.get('metadata', {})
            source = metadata.get('source', 'unknown')
            
            if (source.lower() == 'unknown' or not source) and 'seed_data' in metadata:
                source = metadata['seed_data']
                metadata['source'] = source
            
            current_type = metadata.get('type', 'unknown')
            if current_type == 'unknown':
                if source in ['openstax', 'khanacademy', 'fineweb_edu', 'finemath', 'scienceqa']:
                    metadata['type'] = 'General Knowledge'
                else:
                    metadata['type'] = 'Supplemental'
                
            chunk_text = res.get('text', '')
            base_score = res.get('score', 0.0)  # 余弦相似度（集合使用 cosine 空间）
            
            if not chunk_text or len(chunk_text.strip()) < 10:
                continue
            
            # 以来源集合名判定优先级：neb_* 集合（含导入教材）为最高优先级，
            # 不依赖文件名是否含 "neb"（中文教材文件名不含 neb）
            collection_name = res.get('collection', '')
            if 'neb' in collection_name.lower():
                priority_score = self.SOURCE_PRIORITY.get('neb', 1.0)
            else:
                priority_score = self.calculate_priority_score(source)

            # 关键词覆盖率：问题词元在块中的命中比例（对术语型问题尤其有效）
            if q_units:
                c_units = _cjk_units(chunk_text)
                term_coverage = len(q_units & c_units) / len(q_units)
            else:
                term_coverage = 0.0

            bm25_norm = (float(res.get('bm25_score') or 0.0) / bm25_max) if bm25_max > 0 else 0.0
            structure_bonus = self._structure_bonus(metadata)
            version = self._version_feature(metadata)
            grade_affinity = self._grade_affinity(metadata, grade)

            # 查询覆盖率（中文二元组）：问题有多少「词」真的出现在块里。
            #
            # 为什么用二元组而不是单字：
            #   · 单字覆盖率对中文几乎没有区分度 —— 任意中文块都能覆盖问题的
            #     「的/是/在」之类常用字，库外问题（"希格斯玻色子"）也能拿到
            #     不低的分数，导致拒答判据失效（实测 negative 最高 0.78，
            #     而 positive 最低 0.46，完全不可分）。
            #   · 二元组近似还原词的边界，"标准模型"→{标准,准模,模型}，
            #     且**绝对可比**（不像 BM25 需要在批内做最大值归一化，
            #     后者会让每批里总有一个候选拿到 1.0，负样本同样受益）。
            #
            # 取「原句 + 各改写变体」的最大值：口语提问与教材措辞不重合，
            # 但同义变体往往重合（"摩尔浓度" ↔ "物质的量浓度"）。
            query_coverage = 0.0
            for v in variants:
                query_coverage = max(query_coverage, content_overlap(v, chunk_text))

            features = {
                'similarity': float(base_score),
                'bm25': bm25_norm,
                'term_coverage': term_coverage,
                'query_coverage': query_coverage,
                'structure_bonus': structure_bonus,
                'source_priority': float(priority_score),
                'version': version,
                'grade_affinity': grade_affinity,
            }
            # 只对权重表里**存在对应特征**的键求和：配置多写一个键
            # （例如自定义特征）不应让整条检索链路 KeyError 挂掉。
            effective = [k for k in w if k in features]
            weighted_score = sum(w[k] * features[k] for k in effective) / \
                (sum(w[k] for k in effective) or 1.0)
            
            ranked_chunks.append({
                'id': res.get('id', ''),
                # `collection` 必须原样带下去：父子索引靠它去正确的集合取父块。
                # （曾经漏传过，表现为「父块读取失败（）：Collection [] does not exist」，
                #   父块展开静默失效 —— 这类字段丢失不会有异常冒泡，只会静默退化。）
                'collection': res.get('collection', ''),
                'text': chunk_text,
                'metadata': metadata,
                'original_score': base_score,
                'final_score': weighted_score,
                'keyword_coverage': round(term_coverage, 4),  # 旧名保留（兼容既有调用/测试）
                'term_coverage': round(term_coverage, 4),
                'query_coverage': round(query_coverage, 4),
                'bm25_normalized': round(bm25_norm, 4),
                'bm25_rank': res.get('bm25_rank'),
                'structure_bonus': round(structure_bonus, 4),
                'version_score': round(version, 4),
                'grade_affinity': grade_affinity,
                'features': features,
                'source_type': 'NEB' if priority_score >= self.SOURCE_PRIORITY.get('neb', 1.0)
                               else 'External'
            })
            
        ranked_chunks.sort(key=lambda x: x['final_score'], reverse=True)
        
        return ranked_chunks

    def check_grounding(
        self,
        answer: str,
        context_chunks: List[str],
    ) -> Dict:
        """
        Layer 4: 判定答案是否有证据支撑，返回结构化结果。

        实测标定（本库真实回答）：

            | 问题类型              | 内容覆盖度 | 是否应判为有据 |
            |----------------------|-----------|--------------|
            | 《乡土中国》核心观点   |   79.3%   |   是         |
            | 向心力（概念题）       |   33.0%   |   是         |
            | 导数的几何意义         |   17.3%   |   是         |
            | 量子色动力学（库外）   |   12.3%   |   否         |

        结论：**内容覆盖度对中文的区分力很弱** —— 模型会大量改写、且公式/符号
        不产生二元组，导致有据答案的覆盖度也可能很低（17%），与库外问题（12%）
        几乎重叠。因此本指标**只用作置信度的软折扣**，不用作"依据不足"横幅的判据；
        硬判据交给「模型自认无据」与「完全无证据」。

        Returns:
            {
              "status": "grounded" | "admitted" | "weak_overlap" | "no_context",
              "overlap": float,
              "reason": str,
              "severe": bool     # 是否应追加免责提示并压低置信度上限
            }
        """
        if not answer or not context_chunks:
            return {"status": "no_context", "overlap": 0.0,
                    "reason": "No answer or context", "severe": False}

        # 过短答案（如"是""不清楚"）无从校验，视为通过
        if len(_cjk_units(answer)) < 10:
            return {"status": "grounded", "overlap": 1.0,
                    "reason": "Answer too short to validate", "severe": False}

        full_context = "".join(context_chunks)
        overlap = content_overlap(answer, full_context)

        # 硬信号①：模型主动声明"材料中未找到"/"根据常识"等 —— 说明它自己认为没有依据
        answer_lower = answer.lower()
        for phrase in self.FORBIDDEN_PHRASES:
            if phrase in answer_lower:
                return {"status": "admitted", "overlap": overlap,
                        "reason": f"Model admitted lack of grounding: {phrase}",
                        "severe": True}

        # 软信号：内容覆盖度过低 —— 可能是幻觉，也可能只是改写幅度大，故只降置信度
        if overlap < self.min_content_overlap:
            logger.warning(f"Low content overlap: {overlap:.2%}（仅降置信度，不加横幅）")
            return {"status": "weak_overlap", "overlap": overlap,
                    "reason": f"Low content overlap ({overlap:.0%})", "severe": False}

        return {"status": "grounded", "overlap": overlap,
                "reason": "Grounded", "severe": False}

    def validate_grounding(
        self,
        answer: str,
        context_chunks: List[str]
    ) -> Tuple[bool, str]:
        """
        兼容入口：返回 (是否有据, 原因)。严重情形（模型自认无据）才返回 False。

        细粒度判定请用 `check_grounding`。
        """
        res = self.check_grounding(answer, context_chunks)
        ok = res["status"] in ("grounded", "weak_overlap")
        return ok, res["reason"]

    def resolve_conflicts(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []
        
        neb_chunks = [c for c in chunks if c.get('source_type') == 'NEB']
        other_chunks = [c for c in chunks if c.get('source_type') != 'NEB']
        
        return neb_chunks + other_chunks
    
    def filter_low_quality(self, chunks: List[Dict], min_score: float = 0.35) -> List[Dict]:
        return [c for c in chunks if c.get('final_score', 0) >= min_score]