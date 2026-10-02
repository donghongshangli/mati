"""
Retrieval layer configuration loader for Mati.

把原本散落在各模块的硬编码阈值/权重集中到 mati_data/config/retrieval.yaml，
并提供带默认值的读取接口（配置文件缺失时行为与内置默认一致，不会崩）。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML 已在 requirements 中
    yaml = None

logger = logging.getLogger(__name__)

# 与 retrieval.yaml 保持一致的默认值（配置缺失时的兜底）
DEFAULTS: Dict[str, Any] = {
    "subject_aliases": {
        "science": "science", "Science": "science", "科学": "science",
        "物理": "science", "化学": "science", "生物": "science",
        "math": "math", "Math": "math", "Mathematics": "math", "数学": "math",
        "chinese": "chinese", "Chinese": "chinese", "语文": "chinese",
        "english": "english", "English": "english", "English Grammar": "english",
        "英语": "english", "英语语法": "english",
        "computer_science": "computer_science", "Computer Science": "computer_science",
        "计算机": "computer_science", "计算机科学": "computer_science",
        "信息技术": "computer_science",
        "social_studies": "social_studies", "Social Studies": "social_studies",
        "社会科学": "social_studies", "政治": "social_studies",
        "历史": "social_studies", "地理": "social_studies",
    },
    "fallback_collections": [],
    "discipline_aliases": {
        "physics": "physics", "物理": "physics",
        "chemistry": "chemistry", "化学": "chemistry",
        "biology": "biology", "生物": "biology",
        "mathematics": "mathematics", "math": "mathematics", "数学": "mathematics",
        "chinese": "chinese", "语文": "chinese",
        "information_technology": "information_technology",
        "信息技术": "information_technology",
        "computer_science": "information_technology",
    },
    "discipline_to_subject": {
        "physics": "science", "chemistry": "science", "biology": "science",
        "mathematics": "math", "chinese": "chinese",
        "information_technology": "computer_science",
    },
    "collection_whitelist_prefixes": ["neb_"],
    "collection_blacklist": ["neb_readme"],
    "routing": {
        "grade_mode": "soft",
        "same_grade_min_evidence": 2,
    },
    "candidate_k_per_collection": 20,
    "rerank_k": 10,
    "evidence_k": 4,
    "min_base_score": 0.30,
    "min_final_score": 0.40,
    "rank_weights": {
        "similarity": 0.40, "bm25": 0.20, "term_coverage": 0.15,
        "structure_bonus": 0.10, "source_priority": 0.05,
        "version": 0.05, "grade_affinity": 0.05,
    },
    "structure_bonus": {
        "definition": 1.0, "theorem": 1.0, "example": 0.8, "exercise": 0.6,
        "body": 0.7, "figure_caption": 0.3, "toc": 0.0, "cover": 0.0,
        "default": 0.5,
    },
    "source_priority": {
        "neb": 1.0, "openstax": 0.7, "khanacademy": 0.65, "default": 0.5,
    },
    "hybrid": {
        "enabled": True, "rrf_k": 60, "dense_k": 20, "bm25_k": 20,
        "max_query_variants": 3,
    },
    "governance": {
        "exclude_content_types": ["toc", "cover"],
        "exclude_archived": True,
        "cleaning": {"enabled": True, "min_chars": 40, "drop_copyright": True},
        "dedup": {"enabled": True, "exact": True, "semantic_threshold": 0.95,
                  "keep": "longer"},
        "conflicts": {"enabled": True, "output": "mati_data/conflicts.jsonl",
                      "require_concept_key": True},
    },
    "rewrite": {
        "enabled": True, "synonym_expansion": True, "max_synonym_variants": 2,
        "subquery_split": True, "subquery_min_chars": 10, "max_subqueries": 2,
        "failure_log": "mati_data/retrieval_failures.jsonl",
        "failure_score_threshold": 0.45,
    },
    "chunking": {
        "structure_aware": True, "child_target_chars": 384, "child_min_chars": 120,
        "child_overlap_ratio": 0.1, "parent_target_chars": 1000,
        "parent_max_chars": 1300, "parent_child_index": True,
    },
    "context_budget": {
        "n_ctx": 4096,
        "max_output_tokens": 768,
        "max_evidence_tokens": 2000,
        "tokens_per_char_cjk": 1.0,
        "tokens_per_char_ascii": 0.30,
    },
    "grounding": {
        "min_content_overlap": 0.15,
    },
    "cache": {
        "max_size": 100, "ttl_seconds": 3600,
        "semantic_threshold": 0.92, "cache_errors": False,
    },
    "generation": {
        "require_citations": True, "citation_penalty": 0.85,
        "refuse_below_query_coverage": 0.28,
        "refuse_when_no_evidence": True, "verify_grounding": True,
        # 创作类请求拒答（P2）：与覆盖率正交的判据，理由见 retrieval.yaml 同名注释
        "refuse_creation_requests": True,
    },
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """递归合并：override 覆盖 base，未提供的键沿用 base。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class RetrievalConfig:
    """检索层配置访问器。"""

    def __init__(self, data: Optional[Dict] = None, source_path: Optional[str] = None):
        self.data = _deep_merge(DEFAULTS, data or {})
        self.source_path = source_path

    # ------------------------------------------------------------------ 加载
    @classmethod
    def load(cls, path: Optional[str] = None) -> "RetrievalConfig":
        if path is None:
            path = cls.default_path()
        p = Path(path)
        if not p.exists():
            logger.warning(f"检索配置不存在，使用内置默认值：{p}")
            return cls({}, str(p))
        if yaml is None:
            logger.error("PyYAML 未安装，检索配置无法读取；使用内置默认值")
            return cls({}, str(p))
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"检索配置解析失败（{p}）：{e}；使用内置默认值")
            return cls({}, str(p))
        logger.info(f"检索配置已加载：{p}")
        return cls(data, str(p))

    @staticmethod
    def default_path() -> str:
        project_root = Path(__file__).resolve().parents[2]
        return str(project_root / "mati_data" / "config" / "retrieval.yaml")

    # ------------------------------------------------------------------ 取值
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    # ------------------------------------------------------------- 便捷访问
    @property
    def candidate_k(self) -> int:
        return int(self.data["candidate_k_per_collection"])

    @property
    def rerank_k(self) -> int:
        return int(self.data["rerank_k"])

    @property
    def evidence_k(self) -> int:
        return int(self.data["evidence_k"])

    @property
    def min_base_score(self) -> float:
        return float(self.data["min_base_score"])

    @property
    def min_final_score(self) -> float:
        return float(self.data["min_final_score"])

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self.data["rank_weights"])

    @property
    def source_priority(self) -> Dict[str, float]:
        return dict(self.data["source_priority"])

    @property
    def max_evidence_tokens(self) -> int:
        return int(self.data["context_budget"]["max_evidence_tokens"])

    @property
    def cache_config(self) -> Dict[str, Any]:
        return dict(self.data["cache"])

    @property
    def generation_config(self) -> Dict[str, Any]:
        return dict(self.data["generation"])

    @property
    def hybrid_config(self) -> Dict[str, Any]:
        return dict(self.data["hybrid"])

    @property
    def grade_mode(self) -> str:
        """
        年级过滤强度："soft"（默认）或 "hard"。

        见 retrieval.yaml 的 routing 段说明：soft 让同家族全部年级参与召回、
        年级只做排序与软加权，避免跨年级内容的假拒答。
        """
        mode = str((self.data.get("routing") or {}).get("grade_mode", "soft")).strip().lower()
        return mode if mode in ("soft", "hard") else "soft"

    @property
    def rewrite_config(self) -> Dict[str, Any]:
        return dict(self.data["rewrite"])

    @property
    def chunking_config(self) -> Dict[str, Any]:
        return dict(self.data["chunking"])

    @property
    def governance_config(self) -> Dict[str, Any]:
        """知识治理配置（P2）：清洗 / 去重 / 冲突检测 / 检索侧排除条件。"""
        return dict(self.data.get("governance") or {})

    def structure_bonus_for(self, content_type: Optional[str]) -> float:
        """
        内容类型 → 结构加权分（0–1）。

        用于重排的 structure_bonus 特征：定义/定理/例题应当优先于
        目录、封面、图注这类噪声块 —— 后者会占用本就紧张的证据预算。
        """
        table = self.data.get("structure_bonus") or {}
        key = (content_type or "").strip().lower()
        if key in table:
            return float(table[key])
        return float(table.get("default", 0.5))

    # ------------------------------------------------------------- 学科归一
    def _lookup_alias(self, table: Dict[str, str], value: Optional[str]) -> str:
        """在别名表中查值：先精确匹配，再大小写不敏感匹配，最后保守归一。"""
        if not value:
            return ""
        s = str(value).strip()
        if s in table:
            return table[s]
        low = s.lower()
        for k, v in table.items():
            if k.lower() == low:
                return v
        return ""

    def canonical_subject(self, subject: Optional[str]) -> str:
        """
        把 GUI 学科键 / 中文名 / 细分学科 / 别名归一化成规范**家族** subject。

        若传入的是细分学科（如 physics / 物理），会通过 `discipline_to_subject`
        反查回家族（science）—— 这样即便集合按 discipline 拆开、或旧集合缺少
        metadata（只能从名字解析出 physics），路由依然能正确归到家族。

        未识别的取值做保守归一（小写 + 空格转下划线），保持向后兼容。
        """
        if not subject:
            return ""
        hit = self._lookup_alias(self.data["subject_aliases"], subject)
        if hit:
            return hit
        # 细分学科 -> 家族
        disc = self.canonical_discipline(subject)
        if disc:
            return self.data["discipline_to_subject"].get(disc, disc)
        return str(subject).strip().lower().replace(" ", "_")

    def canonical_discipline(self, value: Optional[str]) -> str:
        """
        把细分学科写法归一化为规范 discipline 键。

        未识别时返回空串（调用方据此决定是否退化为家族级别）。
        """
        if not value:
            return ""
        return self._lookup_alias(self.data["discipline_aliases"], value)

    def subject_for_collection(self, collection_key: str) -> str:
        """
        把「集合键」（可能是家族名 science，也可能是细分名 physics）
        解析为家族学科。供旧库/无名 metadata 的集合做路由回退。
        """
        key = (collection_key or "").strip().lower()
        family = self.data["discipline_to_subject"].get(key)
        if family:
            return family
        return self.canonical_subject(key) or key

    def is_collection_allowed(self, name: str) -> bool:
        """按白名单前缀 + 黑名单过滤集合（排除 README 等非教学集合）。"""
        if not name:
            return False
        if name in (self.data.get("collection_blacklist") or []):
            return False
        prefixes: List[str] = self.data.get("collection_whitelist_prefixes") or []
        if not prefixes:
            return True
        return any(name.startswith(p) for p in prefixes)
