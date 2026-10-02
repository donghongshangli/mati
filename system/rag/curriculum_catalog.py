"""
Curriculum Catalog Resolver for Mati

依据教材书名判定 subject / grade / grade_span 等元数据。

数据来源：mati_data/config/textbook_catalog.yaml（含网络核实依据与已知分歧说明）。

设计要点：
- 归一化后做「多组 AND、组内 OR」的关键词匹配，规则按列表顺序，先命中者胜。
- 未命中时回退到旧的「中文学科关键词 + 文件名数字」启发式，保证向后兼容。
- 纯标准库 + PyYAML，不引入向量库/模型依赖，便于单独测试。
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML 已在 requirements 中
    yaml = None

logger = logging.getLogger(__name__)

# 中文学科关键词 -> 英文学科（ChromaDB 集合名只允许 [a-zA-Z0-9._-]）
SUBJECT_KEYWORDS_CN = [
    ("物理", "science"), ("化学", "science"), ("生物", "science"), ("科学", "science"),
    ("地理", "social_studies"), ("历史", "social_studies"), ("政治", "social_studies"),
    ("数学", "math"), ("英语", "english"), ("英文", "english"), ("语法", "english"),
    ("计算机", "computer_science"), ("信息技术", "computer_science"),
]

# 英文关键词（回退路径用；按顺序匹配，较长/较具体的排在前面）
SUBJECT_KEYWORDS_EN = [
    ("physics", "science"), ("chemistry", "science"), ("biology", "science"),
    ("science", "science"),
    ("mathematics", "math"), ("math", "math"),
    ("chinese", "chinese"),
    ("english", "english"),
    ("computer_science", "computer_science"), ("computer", "computer_science"),
    ("information_technology", "computer_science"),
    ("social_studies", "social_studies"), ("history", "social_studies"),
    ("geography", "social_studies"),
]

GRADE_LABELS = {"10": "高一", "11": "高二", "12": "高三"}

# 归一化时剔除的装饰性字符（空白、间隔号、括号、连字符）
_STRIP_CHARS = re.compile(r"[\s·・．\.\-_—–()（）\[\]【】、,，:：;；'\"“”‘’]+")


def normalize_name(text: str) -> str:
    """归一化文件名：去装饰字符、转小写。用于关键词匹配。"""
    if not text:
        return ""
    return _STRIP_CHARS.sub("", text).lower()


def grade_label(grade: Optional[str]) -> str:
    """把 grade 编码转成中文标签；未知返回空串。"""
    if not grade:
        return ""
    return GRADE_LABELS.get(str(grade), "")


class CurriculumCatalog:
    """教材目录：书名 → 学科/年级/册次等元数据。"""

    # 需要按 discipline 拆集合的家族学科（可由 YAML 的 discipline_split_subjects 覆盖）
    DEFAULT_DISCIPLINE_SPLIT = {"science"}

    def __init__(self, books: Optional[List[Dict]] = None, source_path: Optional[str] = None,
                 discipline_split: Optional[set] = None):
        self.books: List[Dict] = list(books or [])
        self.source_path = source_path
        self.discipline_split = set(discipline_split or self.DEFAULT_DISCIPLINE_SPLIT)

    # ------------------------------------------------------------------ 加载
    @classmethod
    def load(cls, path: Optional[str] = None) -> "CurriculumCatalog":
        """
        从 YAML 加载目录。path 缺省时按「项目根/mati_data/config/textbook_catalog.yaml」查找。
        加载失败时返回空目录（调用方会走回退启发式），不抛异常。
        """
        if path is None:
            path = cls.default_path()
        p = Path(path)
        if not p.exists():
            logger.warning(f"教材目录不存在，将使用回退启发式：{p}")
            return cls([], str(p))
        if yaml is None:
            logger.error("PyYAML 未安装，无法读取教材目录；将使用回退启发式")
            return cls([], str(p))
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"教材目录解析失败（{p}）：{e}")
            return cls([], str(p))

        books = data.get("books") or []
        split = data.get("discipline_split_subjects")
        split_set = {str(s) for s in split} if split else None
        # 预归一化 match_all / exclude，避免每次匹配重复计算
        for b in books:
            b["_match_all_norm"] = [
                [normalize_name(alt) for alt in group] for group in (b.get("match_all") or [])
            ]
            b["_exclude_norm"] = [normalize_name(x) for x in (b.get("exclude") or [])]
            b["grade_span"] = [str(g) for g in (b.get("grade_span") or [])]
            if b.get("grade") is not None:
                b["grade"] = str(b["grade"])

        logger.info(f"教材目录已加载：{len(books)} 本（{p}）")
        return cls(books, str(p), discipline_split=split_set)

    @staticmethod
    def default_path() -> str:
        project_root = Path(__file__).resolve().parents[2]
        return str(project_root / "mati_data" / "config" / "textbook_catalog.yaml")

    # ------------------------------------------------------------------ 匹配
    def resolve(self, filename: str, path: Optional[str] = None) -> Dict:
        """
        解析教材元数据。

        Args:
            filename: 文件名（含或不含扩展名均可）
            path: 完整路径（可选）。目录未命中时，回退路径会从中提取年级
                  （支持 `textbooks/grade_10/xxx.pdf` 这类按年级分目录的组织方式）

        Returns:
            {
              "matched": bool,            # 是否命中目录
              "book_id", "subject", "discipline", "title", "volume",
              "publisher", "edition", "grade", "grade_span",
              "grade_label", "grade_note", "teaching_hint"
            }
            未命中时 subject/grade 由回退启发式给出，且 matched=False。
        """
        raw = Path(filename).stem if filename else ""
        norm = normalize_name(raw)

        for book in self.books:
            if self._matches(book, norm):
                span = book.get("grade_span") or ([book["grade"]] if book.get("grade") else [])
                return {
                    "matched": True,
                    "book_id": book.get("id"),
                    "subject": book.get("subject"),
                    "discipline": book.get("discipline"),
                    "title": book.get("title"),
                    "volume": book.get("volume"),
                    "publisher": book.get("publisher"),
                    "edition": book.get("edition"),
                    "grade": book.get("grade") or "unknown",
                    "grade_span": ",".join(span) if span else "",
                    "grade_label": grade_label(book.get("grade")),
                    "grade_note": book.get("grade_note") or "",
                    "teaching_hint": book.get("teaching_hint") or "",
                }

        return self._fallback(raw, path)

    @staticmethod
    def _matches(book: Dict, norm: str) -> bool:
        if not norm:
            return False
        for ex in book.get("_exclude_norm") or []:
            if ex and ex in norm:
                return False
        groups = book.get("_match_all_norm") or []
        if not groups:
            return False
        for alts in groups:
            if not any(alt and alt in norm for alt in alts):
                return False
        return True

    # ------------------------------------------------------------- 回退启发式
    @staticmethod
    def _fallback(stem: str, path: Optional[str] = None) -> Dict:
        """
        目录未命中时的兼容路径：
        1) 学科：中文学科关键词 -> 英文学科关键词 -> ASCII 前缀
        2) 年级：先看路径中的 `grade_10/` `class_10/` 目录（按文档约定的
           `textbooks/grade_10/` 组织方式），再看文件名中的 1–12 数字
        """
        subject = None
        for kw, en in SUBJECT_KEYWORDS_CN:
            if kw in stem:
                subject = en
                break
        if subject is None:
            low = stem.lower()
            for kw, en in SUBJECT_KEYWORDS_EN:
                if kw in low:
                    subject = en
                    break
        if subject is None:
            # 只保留 ASCII 字母数字：文件名可能整段是中文（如"调研笔记-..."），
            # 若照搬中文再去 sanitize 会塌缩成空前缀，生成名为 "neb" 的垃圾集合。
            cleaned = "".join(
                c for c in stem if (c.isalnum() and ord(c) < 128) or c == "_"
            ).lower()
            subject = cleaned.strip("_") or "general"

        grade = CurriculumCatalog._grade_from_path_or_stem(stem, path)

        return {
            "matched": False,
            "book_id": None,
            "subject": subject,
            "discipline": subject,
            "title": stem,
            "volume": "",
            "publisher": "",
            "edition": "",
            "grade": grade,
            "grade_span": grade if grade != "unknown" else "",
            "grade_label": grade_label(grade),
            "grade_note": "",
            "teaching_hint": "",
        }

    @staticmethod
    def _grade_from_path_or_stem(stem: str, path: Optional[str] = None) -> str:
        """
        从路径 / 文件名推断年级编码（10–12 视为高一–高三，1–9 为义务教育年级）。

        优先级：
          1. 路径中带显式标记的目录或文件名：`grade_10`、`class_3`
          2. 路径目录名中的裸数字（如 `textbooks/10/xxx.pdf`）
          3. 文件名中的裸数字（排除 4 位年份，如 2019/2025）
        """
        def _digits(s: str) -> str:
            return re.sub(r"\D", "", s)

        parts: List[str] = []
        if path:
            try:
                parts = [p for p in Path(path).parts if p not in ("", ".")]
            except Exception:  # noqa: BLE001
                parts = []
        # 目录部分（不含文件名），从最靠近文件的目录往前找
        dirs = parts[:-1] if len(parts) >= 2 else []

        # 1) 显式 grade_N / class_N（目录优先，再文件名）
        for candidate in list(reversed(dirs)) + [stem]:
            low = candidate.lower()
            for marker in ("grade_", "class_", "年级"):
                if marker in low:
                    d = _digits(low.split(marker)[-1])[:2]
                    if d and 1 <= int(d) <= 12:
                        return str(int(d))

        # 2) 目录名**恰好**是数字（如 textbooks/10/xxx.pdf）
        #    必须严格全数字：否则 "_p1_probe"、"grade_old_2" 这类目录名里的
        #    任意数字都会被当成年级（曾把沙箱目录 _p1_probe 判成「一年级」）。
        for candidate in reversed(dirs):
            c = candidate.strip()
            if c.isdigit() and len(c) <= 2 and 1 <= int(c) <= 12:
                return str(int(c))

        # 3) 文件名中的裸数字（排除年份）
        for d in re.findall(r"\d{1,2}", stem):
            if 1 <= int(d) <= 12:
                return str(int(d))

        return "unknown"


# ---------------------------------------------------------------- 集合命名
def collection_name_for(
    subject: str,
    grade: str = "unknown",
    discipline: str = None,
    split_subjects: Optional[set] = None,
    prefix: str = "neb",
) -> str:
    """
    生成合法的 ChromaDB 集合名（仅 [a-zA-Z0-9._-]，长度 >= 3）。

    命名规则：
      - 默认 `neb_{subject}_grade_{n}`（grade 未知则 `neb_{subject}`）
      - 当 `subject` 属于 `split_subjects`（默认 {"science"}）且提供了 `discipline` 时，
        用 `neb_{discipline}_grade_{n}` —— 让物理/化学/生物各占一个集合，
        避免「问生物召回物理」的串科问题。
      - math / chinese / english / computer_science / social_studies 只有一个细分学科，
        不拆集合，名字保持不变，避免无谓的全量重建。

    注意：集合 metadata 里始终写入**家族 subject**，检索层按家族路由，
    因此拆分不会让学生端的「科学」学科选择失效（discipline 仅用于进一步收窄）。

    若 subject 清洗后为空，会回退为 general —— 否则会生成只有前缀的退化集合名
    （如 "neb"），既难辨认又会与白名单前缀规则失配。
    """
    split_subjects = set(split_subjects if split_subjects is not None
                         else {"science"})

    def _clean(v: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]", "_", (v or "").strip().lower()).strip("._-")

    subject_key = _clean(subject) or "general"
    if discipline and subject_key in split_subjects:
        disc_key = _clean(discipline)
        if disc_key and disc_key != subject_key:
            subject_key = disc_key

    grade_key = _clean(str(grade or "unknown"))
    if grade_key in ("", "unknown"):
        name = f"{prefix}_{subject_key}"
    else:
        name = f"{prefix}_{subject_key}_grade_{grade_key}"
    if len(name) < 3:
        name = f"{prefix}_general"
    return name
