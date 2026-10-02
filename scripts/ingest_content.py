"""
Mati Universal Content Ingestion Script

Handles ALL content types:
- Regular PDFs (text-based)
- Scanned PDFs (OCR with Tesseract)
- Handwritten notes (OCR with EasyOCR)
- Text files (TXT, MD, JSONL)

Auto-detects content type and applies appropriate processing.

Usage:
    python scripts/ingest_content.py
    python scripts/ingest_content.py --input notes textbooks
    python scripts/ingest_content.py --ocr-mode auto  # auto-detect OCR need
    python scripts/ingest_content.py --ocr-mode force # force OCR on all PDFs
"""

import os
import re
import sys
import json
import hashlib
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple

# 离线优先环境准备（必须在 import huggingface_hub/transformers 之前执行，
# 否则 HF_HUB_OFFLINE/HF_ENDPOINT 因模块级常量缓存而不生效）。
_HF_MODEL = "BAAI/bge-small-zh-v1.5"
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / (
    "models--" + _HF_MODEL.replace("/", "--"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if _HF_CACHE.exists():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from tqdm import tqdm
import fitz  # PyMuPDF


def _prepare_hf_env(model_name: str) -> None:
    """离线优先：缓存存在则强制离线；缺失则用 hf-mirror 镜像下载。"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + model_name.replace("/", "--"))
    if cache_dir.exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.rag_data_preparation.enhanced_chunker import EnhancedChunker
from system.rag.curriculum_catalog import (
    CurriculumCatalog,
    collection_name_for,
    grade_label,
)
from system.rag.retrieval_config import RetrievalConfig
from system.rag.text_cleanup import fix_pdf_glyphs, glyph_stats
from system.rag.knowledge_governance import (
    clean_chunk_text,
    dedupe_chunks,
    detect_conflicts,
)


class OCREngineUnavailable(RuntimeError):
    """
    需要 OCR 但本机没有可用的 OCR 引擎。

    单独定义异常而不是返回空字符串，是为了让上层能区分
    「文件本身没文字」与「有文字但读不出来」—— 后者过去会静默变成 0 块，
    界面仍然报「导入完成」，属于最伤信任的一类失败。
    """


# 向量空间度量：cosine 使 Chroma 返回的 distance = 1 - cos，
# 上层可直接用 (1 - distance) 作为余弦相似度，避免 l2 空间下
# 「1 - 距离」被误当作相似度使用（阈值语义漂移）。
VECTOR_SPACE = "cosine"

# HNSW 参数：库规模仅数千块，提高 ef_search 可在延迟几乎不变的前提下提升召回。
HNSW_METADATA = {
    "hnsw:space": VECTOR_SPACE,
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 200,
    "hnsw:M": 32,
}

# PDF 页标记（P2）：由 extract_text_from_pdf 插入，scan_page_markers 消费。
# 形如独立一行，切分与清洗阶段都会被当作噪声行剔除，不会进入正文。
_PAGE_MARKER_RE = re.compile(r"\[\[PAGE:(\d{1,4})\]\]")

# 知识治理报告（P2）：清洗/去重的每一次删除都落盘，可审计。
# 「悄悄丢内容」是知识治理最危险的地方 —— 方案 §6 已把它列为风险项。
GOV_REPORT_PATH = "mati_data/ingest_gov_report.jsonl"
CONFLICTS_PATH = "mati_data/conflicts.jsonl"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TESSERACT_AVAILABLE = False
EASYOCR_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    logger.warning("Tesseract not available. Install: pip install pytesseract pillow")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    logger.warning("EasyOCR not available. Install: pip install easyocr")


class UniversalContentIngester:
    """Handles all content types with auto-detection."""
    
    def __init__(
        self,
        db_path: str,
        ocr_mode: str = "auto",
        force_subject: str = None,
        force_grade: str = None,
    ):
        """
        Args:
            db_path: Path to ChromaDB
            ocr_mode: "auto" (detect), "force" (always OCR), "never" (text only)
            force_subject: 强制学科（GUI 学科键 / 中文名 / 规范值均可），
                           None 表示按书名自动识别。用于人工导入时覆盖推断结果。
            force_grade: 强制年级编码（"10"/"11"/"12"），None 表示自动识别。
        """
        self.db_path = db_path
        self.ocr_mode = ocr_mode

        # 人工指定优先于自动推断（解决「自编讲义文件名推不出学科年级」的问题）
        self.retrieval_config = RetrievalConfig.load()
        self.force_subject = None
        self.force_discipline = None
        if force_subject:
            canonical = self.retrieval_config.canonical_subject(force_subject)
            if canonical:
                self.force_subject = canonical
                # 若用户选的是细分学科（物理/化学/生物），同时锁定 discipline，
                # 这样集合会落到 neb_physics_* 而不是家族级的 neb_science_*
                self.force_discipline = self.retrieval_config.canonical_discipline(force_subject) or None
                logger.info(f"强制学科：{force_subject!r} -> subject={canonical!r} "
                            f"discipline={self.force_discipline!r}")
        self.force_grade = None
        if force_grade not in (None, ""):
            g = str(force_grade).strip()
            if g.isdigit() and 1 <= int(g) <= 12:
                self.force_grade = str(int(g))
                logger.info(f"强制年级：{self.force_grade}")
            else:
                logger.warning(f"忽略非法年级参数：{force_grade!r}（应为 1–12）")

        # OCR readers are initialised lazily（见 _ocr_status）；未安装依赖时保持为 None
        self.easyocr_reader = None
        self._ocr_status_cache = None

        # Embedding model and ChromaDB are loaded lazily so that
        # --ingest-from-manifest --dry-run can validate packages without the
        # heavy ML dependencies. Normal ingestion paths behave as before.
        self.model = None
        self.client = None
        try:
            logger.info("Opening ChromaDB...")
            self.client = chromadb.PersistentClient(path=db_path)
        except Exception as e:
            logger.warning(f"ChromaDB unavailable (lazy retry later): {e}")

        # 教材目录表：书名 -> subject / grade / grade_span 等元数据
        self.catalog = CurriculumCatalog.load()

        # 分块参数统一由此处定义（此前 CLI 默认 0.2 与摄取侧 0.1 不一致）
        self.chunk_size = 512
        self.overlap_ratio = 0.1
        self.chunker = EnhancedChunker(
            chunk_size=self.chunk_size, overlap_ratio=self.overlap_ratio
        )

    # ------------------------------------------------------------ 集合管理
    def _get_or_create_collection(
        self, collection_name: str, subject: str = None, grade: str = None,
        discipline: str = None,
    ):
        """
        获取/创建集合，并统一指定向量空间为 cosine 与 HNSW 参数。

        同时把 subject / grade / discipline 写进集合 metadata，使检索层可以直接
        **读取集合元数据**做学科/年级路由，不必反向解析集合名
        （避免命名规则变更导致路由静默失效）。

        注意：`subject` 写的是**家族学科**（如 science），`discipline` 才是细分
        （如 physics）。集合可以按 discipline 拆开，但路由仍按家族匹配，
        因此拆分不会让学生端的「科学」选项失效。

        注意：Chroma 的 get_or_create_collection 不会更新已存在集合的
        metadata，因此改了度量/参数后必须重建集合（见 scripts/rebuild_index.py）。
        """
        client = self._ensure_client()
        meta = dict(HNSW_METADATA)
        if subject:
            meta["subject"] = str(subject)
        if grade:
            meta["grade"] = str(grade)
        if discipline:
            meta["discipline"] = str(discipline)
        try:
            return client.get_or_create_collection(name=collection_name, metadata=meta)
        except Exception as e:
            logger.warning(f"指定集合参数失败，回退默认创建（{collection_name}）：{e}")
            return client.get_or_create_collection(name=collection_name)

    def _ensure_model(self):
        """Load the embedding model on first use (offline-first)."""
        if self.model is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is not installed. Install it with: "
                    "pip install sentence-transformers"
                )
            # 离线优先：缓存存在则强制离线，缺失则用 hf-mirror 镜像下载
            _prepare_hf_env('BAAI/bge-small-zh-v1.5')
            logger.info("Loading Embedding Model...")
            # Chinese-optimized embedding model (512-dim, ~95MB).
            self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5', device='cpu')
        return self.model

    def _ensure_client(self):
        """(Re)connect to ChromaDB on first use."""
        if self.client is None:
            if chromadb is None:
                raise RuntimeError(
                    "chromadb is not installed. Install it with: pip install chromadb"
                )
            logger.info("Opening ChromaDB...")
            self.client = chromadb.PersistentClient(path=self.db_path)
        return self.client

    # ------------------------------------------------------------ OCR 能力
    def _ocr_status(self) -> Dict:
        """
        探测本机 OCR 能力（结果缓存）。

        注意：`import pytesseract` 成功**不等于**可用 —— 它只是 Python 封装，
        真正需要 `tesseract` 可执行文件在 PATH 上。过去因为只判断 import 结果，
        扫描件会走到一个必然抛异常的路径，最后静默返回空字符串。

        Returns:
            {"easyocr": bool, "tesseract": bool, "tesseract_langs": [...],
             "available": bool, "detail": str}
        """
        if self._ocr_status_cache is not None:
            return self._ocr_status_cache

        status = {"easyocr": False, "tesseract": False,
                  "tesseract_langs": [], "available": False, "detail": ""}
        details = []

        if self.ocr_mode == "never":
            status["detail"] = "ocr_mode=never（已禁用 OCR）"
            self._ocr_status_cache = status
            return status

        if EASYOCR_AVAILABLE:
            status["easyocr"] = True
        else:
            details.append("easyocr 未安装")

        if TESSERACT_AVAILABLE:
            try:
                import pytesseract
                status["tesseract_langs"] = list(pytesseract.get_languages(config="") or [])
                status["tesseract"] = True
            except Exception as e:  # noqa: BLE001
                details.append(f"tesseract 可执行文件不可用（{type(e).__name__}）")
        else:
            details.append("pytesseract 未安装")

        status["available"] = status["easyocr"] or status["tesseract"]
        status["detail"] = "；".join(details)
        if status["tesseract"] and "chi_sim" not in status["tesseract_langs"]:
            details.append("tesseract 缺少中文语言包 chi_sim")
            status["detail"] = "；".join(details)
        self._ocr_status_cache = status
        return status

    def detect_pdf_type(self, pdf_path: Path) -> str:
        """
        判定 PDF 是文本型还是扫描/图片型。

        改进点（P1-4）：原实现取**前 3 页**的文字总量与固定阈值 100 比较，
        于是「封面 + 目录页」这类正文很少的书会被误判为扫描件
        （封面版权页可能只有几十个字符），进而在没有 OCR 时被整本丢弃。

        现改为在全书范围内**等距抽样**若干页，取**每页文字数的中位数**：
        中位数对封面/插图页这类离群值不敏感，判定更稳。

        Returns:
            "text" | "scanned" | "handwritten"
        """
        try:
            with fitz.open(pdf_path) as doc:
                n = len(doc)
                if n == 0:
                    return "text"
                sample_n = min(6, n)
                step = max(1, n // sample_n)
                counts = []
                for page_num in range(0, n, step):
                    counts.append(len(doc[page_num].get_text().strip()))
                    if len(counts) >= sample_n:
                        break
        except Exception as e:
            logger.error(f"PDF detection error: {e}")
            return "text"  # 检测失败时按文本处理，让后续抽取自己报错

        if not counts:
            return "text"
        counts.sort()
        median = counts[len(counts) // 2]
        # 扫描件：整本中位文字量极低（正常教材页在数百字符量级）
        return "text" if median >= 80 else "scanned"
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from regular PDF.

        P2：在每页开头插入页标记 `[[PAGE:n]]`（n 为 1-based 物理页码）。
        为什么需要
        ──────────
        方案 §3.4 步骤 3.1 要求元数据补 `page`，用于「依据：p.XX」的溯源展示。
        此前 `page` 一直填不上并不是「PDF 解析不出页码」，而是
        `"\n".join(page.get_text() ...)` **把页边界信息丢掉了** ——
        拼起来之后无法反查某段文字来自第几页。
        插入标记后，切分出的每个块带上全文绝对偏移，即可反查页码
        （见 `page_for_offset`）。

        标记形如独立一行，块清洗阶段会作为噪声行剔除，不会进入正文。
        """
        try:
            with fitz.open(pdf_path) as doc:
                parts: List[str] = []
                for i, page in enumerate(doc):
                    parts.append(f"\n[[PAGE:{i + 1}]]\n")
                    parts.append(page.get_text())
                return "".join(parts)
        except Exception as e:
            logger.error(f"PDF read error: {e}")
            return ""

    @staticmethod
    def page_for_offset(markers: List[Tuple[int, int]], offset: int) -> int:
        """
        由块在全文中的偏移反查页码（配合 extract_text_from_pdf 的页标记）。

        Args:
            markers: [(页码, 该页起始偏移), ...]，按偏移升序
            offset: 块的绝对起始偏移

        Returns:
            1-based 页码；越界/无标记时返回 0（表示未知）。
        """
        if not markers or offset is None:
            return 0
        page = markers[0][0]
        for pg, start in markers:
            if start <= offset:
                page = pg
            else:
                break
        return page

    @staticmethod
    def scan_page_markers(content: str) -> Tuple[List[Tuple[int, int]], str]:
        """
        扫描并移除页标记。

        Returns:
            (markers, cleaned_text)
            markers = [(页码, 该页在**清洗后文本**中的起始偏移), ...]

        偏移是在**移除标记的同时**累加已保留文本的长度算出来的，
        因此天然与返回的 cleaned_text 对齐 —— 调用方必须用这对
        (markers, cleaned_text)，不能拿原始带标记文本的偏移来反查。
        """
        markers: List[Tuple[int, int]] = []
        out: List[str] = []
        pos = 0
        for m in _PAGE_MARKER_RE.finditer(content or ""):
            out.append(content[pos:m.start()])
            markers.append((int(m.group(1)), sum(len(x) for x in out)))
            pos = m.end()
        out.append(content[pos:] if content else "")
        return markers, "".join(out)
    
    def extract_text_with_tesseract(self, pdf_path: Path) -> str:
        """
        用 Tesseract 抽取扫描件文字。

        语言按实际安装的语言包动态选择：优先 `chi_sim+eng`（中文教材必须），
        缺失时退回 `eng` 并在日志中明确告知 —— 旧实现写死 `lang='eng'`，
        中文教材会得到近似乱码的结果，比直接失败更难排查。
        """
        if not TESSERACT_AVAILABLE:
            logger.error("Tesseract not available!")
            return ""

        langs = (self._ocr_status().get("tesseract_langs") or [])
        if "chi_sim" in langs:
            lang = "chi_sim+eng" if "eng" in langs else "chi_sim"
        else:
            lang = "eng"
            logger.warning("   Tesseract 缺少中文语言包 chi_sim，将用 eng 识别（中文结果不可靠）")

        try:
            text_parts = []
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    # Convert page to image
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    # OCR
                    text = pytesseract.image_to_string(img, lang=lang)
                    text_parts.append(text)

                    logger.info(f"   OCR page {page_num + 1}/{len(doc)}")

            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return ""
    
    def extract_text_with_easyocr(self, pdf_path: Path) -> str:
        """Extract text from handwritten notes using EasyOCR."""
        if not self.easyocr_reader:
            logger.error("EasyOCR not available!")
            return ""
        
        try:
            text_parts = []
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    # Convert page to image
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")
                    
                    # Save temp image
                    temp_img = f"temp_page_{page_num}.png"
                    with open(temp_img, 'wb') as f:
                        f.write(img_bytes)
                    
                    # OCR
                    results = self.easyocr_reader.readtext(temp_img)
                    text = "\n".join([r[1] for r in results])
                    text_parts.append(text)
                    
                    # Cleanup
                    os.remove(temp_img)
                    
                    logger.info(f"   EasyOCR page {page_num + 1}/{len(doc)}")
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return ""
    
    @staticmethod
    def _extract_json_text(path: Path) -> str:
        """Flatten Mati structured content JSON into plain text lines."""
        import hashlib  # noqa: F401  (kept import local for readability)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        lines: List[str] = []

        def walk(node):
            if isinstance(node, dict):
                for key in ("subject", "grade", "name", "summary", "question",
                            "answer", "acceptable_answers", "hint", "hints",
                            "description", "example", "formula", "explanation"):
                    if key in node and isinstance(node[key], (str, int, float)):
                        lines.append(f"{key}: {node[key]}")
                for key in ("acceptable_answers", "hints"):
                    if key in node and isinstance(node[key], list):
                        for item in node[key]:
                            if isinstance(item, str):
                                lines.append(f"{key[:-1]}: {item}")
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data)
        return "\n".join(lines)

    def process_file(self, file_path: Path) -> Optional[str]:
        """
        Process any file type and return extracted text.
        Auto-detects best extraction method.
        """
        logger.info(f"📄 Processing: {file_path.name}")
        
        # Text files
        if file_path.suffix.lower() in ['.txt', '.md', '.jsonl']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Text file error: {e}")
                return None
        
        # Structured content JSON (subject/units/QA), e.g. from
        # scripts/data_collection/data/content/*.json
        if file_path.suffix.lower() == '.json':
            try:
                return self._extract_json_text(file_path)
            except Exception as e:
                logger.error(f"JSON content error: {e}")
                return None
        
        # PDFs
        if file_path.suffix.lower() == '.pdf':
            # Auto-detect or force OCR
            if self.ocr_mode == "force":
                pdf_type = "scanned"
            elif self.ocr_mode == "never":
                pdf_type = "text"
            else:  # auto
                pdf_type = self.detect_pdf_type(file_path)
            
            logger.info(f"   Detected type: {pdf_type}")
            
            if pdf_type == "text":
                return self.extract_text_from_pdf(file_path)
            elif pdf_type == "scanned":
                return self._extract_with_ocr(file_path)
        
        logger.warning(f"Unsupported file type: {file_path.suffix}")
        return None

    def _extract_with_ocr(self, pdf_path: Path) -> str:
        """
        用可用的 OCR 引擎抽取扫描/图片型 PDF。

        与旧实现的区别：不再「判断 import 成功就当可用」—— pytesseract 只是封装，
        真正需要 tesseract 可执行文件。无可用引擎时抛 `OCREngineUnavailable`，
        让上层把「读不出来」如实报给用户，而不是静默变成 0 块。
        """
        status = self._ocr_status()
        if not status["available"]:
            raise OCREngineUnavailable(
                f"检测为扫描/图片型 PDF，但本机无可用 OCR 引擎（{status['detail']}）"
            )

        if status["easyocr"]:
            text = self.extract_text_with_easyocr(pdf_path)
            if text and text.strip():
                return text

        if status["tesseract"]:
            text = self.extract_text_with_tesseract(pdf_path)
            if text and text.strip():
                return text

        return ""  # 引擎跑过但没识别出文字，由上层如实报告
    
    def extract_metadata(self, file_path: Path) -> Dict[str, str]:
        """
        依据教材目录表判定 subject / grade / grade_span 等元数据。

        优先级：
          1. 调用方**强制指定**的 subject / grade（人工导入时在界面上选定）
          2. `mati_data/config/textbook_catalog.yaml` 书名精确匹配
          3. 回退启发式（学科关键词 + 路径/文件名数字；传入完整路径以支持
             `textbooks/grade_10/xxx.pdf` 这类按年级分目录的组织方式）

        始终返回合法的 ChromaDB 集合名（仅 [a-zA-Z0-9._-]）。
        """
        resolved = self.catalog.resolve(file_path.name, path=str(file_path))

        if self.force_subject:
            resolved["subject"] = self.force_subject
            # 用户只选到家族（如「科学」）时 discipline 退化为家族名，
            # 集合落在 neb_science_*；选到细分（如「生物」）则落在 neb_biology_*。
            resolved["discipline"] = self.force_discipline or self.force_subject
            resolved["forced_subject"] = True
        if self.force_grade:
            resolved["grade"] = self.force_grade
            resolved["grade_span"] = self.force_grade
            resolved["grade_label"] = grade_label(self.force_grade)
            resolved["forced_grade"] = True

        resolved["collection_name"] = collection_name_for(
            resolved["subject"], resolved["grade"],
            discipline=resolved.get("discipline"),
            split_subjects=self.catalog.discipline_split,
        )
        resolved["file_name"] = file_path.name
        return resolved

    def ingest_directory(self, input_dir: str, progress: bool = True) -> Dict[str, Any]:
        """
        摄取一个目录下的全部受支持文件。

        Returns:
            {
              "input": str,
              "found": int,          # 发现的候选文件数
              "ok": int,             # 成功入库的文件数
              "skipped": int,        # 被跳过的文件数
              "failed": int,         # 异常的文件数
              "chunks": int,         # 写入的块总数
              "glyphs_fixed": int,   # 修复的 PDF 字形错映射处数
              "files": [ {file, status, chunks, collection, subject, grade,
                          grade_label, discipline, catalog_matched, reason}, … ]
            }

        调用方**必须**检查返回值：`chunks == 0` 或存在非 ok 状态的条目时，
        不能向用户报告「导入成功」—— 这是本项目此前最伤信任的一类失败
        （扫描件 0 块入库却显示 ✅）。
        """
        summary = {"input": str(input_dir), "found": 0, "ok": 0,
                   "skipped": 0, "failed": 0, "chunks": 0,
                   "glyphs_fixed": 0, "cleaned": 0, "conflicts": 0,
                   "files": []}
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.warning(f"Directory not found: {input_dir}")
            summary["files"].append({
                "file": str(input_dir), "status": "error", "chunks": 0,
                "collection": "", "subject": "", "grade": "",
                "grade_label": "", "discipline": "", "catalog_matched": False,
                "reason": "目录不存在",
            })
            summary["failed"] = 1
            return summary

        logger.info(f"\n Ingesting: {input_dir}")

        # Find all supported files
        files = (
            list(input_path.rglob("*.pdf")) +
            list(input_path.rglob("*.txt")) +
            list(input_path.rglob("*.md")) +
            list(input_path.rglob("*.jsonl")) +
            list(input_path.rglob("*.json"))
        )
        # 排除目录说明文件（README）
        skipped_readme = [f for f in files if f.name.lower() in SKIP_FILENAMES]
        files = [f for f in files if f.name.lower() not in SKIP_FILENAMES]
        if skipped_readme:
            logger.info(f"跳过 {len(skipped_readme)} 个目录说明文件："
                        f"{[f.name for f in skipped_readme]}")
        summary["found"] = len(files)
        logger.info(f"Found {len(files)} files")

        iterator = tqdm(files, desc="Processing files") if progress else files
        for file_path in iterator:
            # 单个文件失败不应中断整个目录（process_file 也可能抛异常）
            try:
                result = self._ingest_one_file(file_path)
            except OCREngineUnavailable as e:
                logger.error(f"  {file_path.name}: {e}")
                result = self._result(file_path, "skipped", reason=str(e))
            except Exception as e:  # noqa: BLE001
                logger.error(f"  Failed on {file_path.name}: {e}", exc_info=True)
                result = self._result(file_path, "error",
                                      reason=f"{type(e).__name__}: {e}")

            summary["files"].append(result)
            summary["chunks"] += result["chunks"]
            summary["glyphs_fixed"] += result.get("glyphs_fixed", 0)
            summary["cleaned"] += result.get("cleaned", 0)
            summary["conflicts"] += result.get("conflicts", 0)
            if result["status"] == "ok":
                summary["ok"] += 1
            elif result["status"] == "error":
                summary["failed"] += 1
            else:
                summary["skipped"] += 1

        if summary["ok"] == 0 and summary["found"] > 0:
            logger.error(
                f"⚠️  {input_dir}: 发现 {summary['found']} 个文件，"
                f"但没有任何内容成功入库（跳过 {summary['skipped']}，失败 {summary['failed']}）"
            )
        return summary

    @staticmethod
    def _result(file_path: Path, status: str, chunks: int = 0,
                metadata: Dict = None, collection: str = "",
                reason: str = "", parents: int = 0,
                content_types: Dict[str, int] = None,
                glyphs_fixed: int = 0, cleaned: int = 0,
                conflicts: int = 0) -> Dict[str, Any]:
        """构造单文件处理结果（供 ingest_directory 汇总与界面展示）。"""
        metadata = metadata or {}
        return {
            "file": file_path.name,
            "path": str(file_path),
            "status": status,              # ok | skipped | error
            "chunks": int(chunks),         # 子块数（= 可召回单元数）
            "parents": int(parents),       # 父块数（父子索引，供生成阶段）
            "content_types": dict(content_types or {}),
            "glyphs_fixed": int(glyphs_fixed),   # PDF 字形错映射修复处数
            # ── 知识治理（P2）：清洗/去重丢弃数与检出的冲突数。
            # 如实报出，让「删了多少」可见 —— 静默删除无法审计。
            "cleaned": int(cleaned),
            "conflicts": int(conflicts),
            "collection": collection or metadata.get("collection_name", ""),
            "subject": metadata.get("subject", ""),
            "discipline": metadata.get("discipline", ""),
            "grade": metadata.get("grade", ""),
            "grade_label": metadata.get("grade_label", ""),
            "book_title": metadata.get("title", ""),
            "catalog_matched": bool(metadata.get("matched")),
            "reason": reason,
        }

    def _ingest_one_file(self, file_path: Path) -> Dict[str, Any]:
        """
        处理单个文件：抽取 → 分块 → 嵌入 → 写入。

        Returns:
            结构化结果 dict（见 `_result`）。**不再静默返回 0**
            ——每一种失败都带可读原因，供上层如实汇报。
        """
        # Extract content
        try:
            content = self.process_file(file_path)
        except OCREngineUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            return self._result(file_path, "error", reason=f"抽取失败：{e}")

        if not content or not content.strip():
            # 区分「文件本身没内容」与「是扫描件/图片型读不出文字」
            if file_path.suffix.lower() == ".pdf":
                try:
                    ptype = self.detect_pdf_type(file_path)
                except Exception:  # noqa: BLE001
                    ptype = "unknown"
                if ptype == "scanned":
                    ocr = self._ocr_status()
                    if not ocr["available"]:
                        return self._result(
                            file_path, "skipped",
                            reason=f"扫描/图片型 PDF，本机无可用 OCR 引擎（{ocr['detail']}）")
                    return self._result(
                        file_path, "skipped",
                        reason="扫描/图片型 PDF，OCR 未识别出文字")
                return self._result(file_path, "skipped", reason="PDF 抽取不到文本")
            return self._result(file_path, "skipped", reason="文件无有效文本内容")

        # ── 字形修复（必须在分块/嵌入之前）
        # 这批人教版电子教材 PDF 用的字体没有正确的 ToUnicode 映射，
        # 数学斜体字母会被错映射成生僻汉字（x→狓、y→狔……实测约 3 万处，
        # 集中在数学教材），另有约 8 千个私用区符号与 4 千处异体文字噪声。
        # 不在这里修，这些字符就会一路经过检索、重排、大模型上下文，
        # 最后原样出现在学生看到的答案里 —— 模型只会照抄，不会猜出正确字母。
        # 映射表的判据见 system/rag/text_cleanup.py。
        glyph_map, glyph_noise = glyph_stats(content)
        glyph_fixed = sum(glyph_map.values()) + glyph_noise
        if glyph_fixed:
            content = fix_pdf_glyphs(content)
            top = "、".join(f"{k}×{n}" for k, n in
                           sorted(glyph_map.items(), key=lambda x: -x[1])[:5])
            logger.info(
                f"   字形修复：{glyph_fixed} 处"
                f"（{len(glyph_map)} 种字形 + 噪声 {glyph_noise}）主要：{top}"
            )

        # ── 页标记扫描与移除（P2）：必须在切分之前
        # 偏移是相对**移除后**的文本，因此先移除再切分，
        # 切分得到的 start_pos 才能直接用来反查页码。
        page_markers, content = self.scan_page_markers(content)

        # Extract metadata（书名 → subject / grade / grade_span …）
        metadata = self.extract_metadata(file_path)
        collection_name = metadata["collection_name"]
        logger.info(
            f"   → Collection: {collection_name} "
            f"[{metadata.get('grade_label') or metadata['grade']}]"
        )

        gov = self.retrieval_config.governance_config
        clean_cfg = dict(gov.get("cleaning") or {})
        dedup_cfg = dict(gov.get("dedup") or {})
        clean_enabled = bool(clean_cfg.get("enabled", True))
        min_chars = int(clean_cfg.get("min_chars", 40))

        # ── Chunk：结构感知切分 + 父子索引（P1）
        # 子块（短）用于召回，父块（一个完整小节）用于喂给生成模型。
        # 参数统一来自 retrieval.yaml 的 chunking 段（此前散落三处且不一致）。
        ck = self.retrieval_config.chunking_config
        children: List[Dict] = []
        parents: List[Dict] = []
        if ck.get("structure_aware", True):
            children, parents = self.chunker.chunk_with_structure(
                content,
                child_target=int(ck.get("child_target_chars", 384)),
                child_min=int(ck.get("child_min_chars", 120)),
                child_overlap_ratio=float(ck.get("child_overlap_ratio", 0.1)),
                parent_target=int(ck.get("parent_target_chars", 1400)),
                parent_max=int(ck.get("parent_max_chars", 1800)),
            )
        if not children:
            # 结构切分失效（例如极短文本）时回退到定长切分，保证不丢内容
            chunks = self.chunker.smart_chunk_with_overlap(content)
            children = [{
                "text": c["text"], "chunk_id": c["chunk_id"], "parent_index": None,
                "start_pos": c.get("start_pos", 0), "end_pos": c.get("end_pos", 0),
                "content_type": "body", "chapter": "", "section_title": "",
            } for c in chunks]
        if not ck.get("parent_child_index", True):
            parents = []

        # ── 逐块清洗（P2 / 步骤 3.2.1）
        # 在**入库前**清，因此索引本身不含版权页/页码孤行/孤立图注。
        gov_events: List[Dict[str, Any]] = []
        if clean_enabled:
            for group in (parents, children):
                kept = []
                for c in group:
                    cleaned, reason = clean_chunk_text(c.get("text", ""),
                                                       min_chars=min_chars)
                    if not cleaned:
                        gov_events.append({
                            "file": file_path.name,
                            "stage": "clean",
                            "action": "drop",
                            "reason": reason or "empty",
                            "text_head": (c.get("text", "") or "")[:60],
                        })
                        continue
                    if cleaned != c.get("text", ""):
                        gov_events.append({
                            "file": file_path.name, "stage": "clean",
                            "action": "trim", "reason": reason,
                            "doc_type": c.get("parent_index") is not None and "parent"
                                        or "child",
                        })
                    c["text"] = cleaned
                    kept.append(c)
                if group is parents:
                    parents = kept
                else:
                    children = kept
        if gov_events:
            n_drop = sum(1 for e in gov_events if e["action"] == "drop")
            logger.info(f"   知识治理·清洗：丢弃 {n_drop} 个噪声块，"
                        f"改写 {len(gov_events) - n_drop} 个")
        if not children:
            return self._result(file_path, "skipped", metadata=metadata,
                                reason="清洗后无有效内容块")

        collection = self._get_or_create_collection(
            collection_name, subject=metadata["subject"], grade=metadata["grade"],
            discipline=metadata.get("discipline"),
        )

        # 父块与子块一起嵌入（父块只在「命中子块后取回」，不参与召回，
        # 但 Chroma 的 add/upsert 需要向量，故一并编码）。
        file_key = hashlib.sha256(file_path.name.encode("utf-8")).hexdigest()[:10]
        content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

        def _parent_id(idx: int) -> str:
            return f"{collection_name}:{file_key}:p{idx:04d}"

        # 父子重建索引：子块 id 依赖 parent_index，清洗/去重会改变下标，
        # 因此**先给父块最终 id**，再据此重算子块的 parent_id。
        parent_ids: List[str] = []
        for i in range(len(parents)):
            pid = _parent_id(i)
            parent_ids.append(pid)
            parents[i]["_id"] = pid
        for c in children:
            pi = c.get("parent_index")
            c["_id"] = (f"{collection_name}:{file_key}:{int(c['chunk_id']):04d}")

        base_common = {
            # 向后兼容：anti_confusion / 检索层读取 source、type、grade
            'source': file_path.name,
            'source_file': file_path.name,
            'type': 'neb_curriculum',
            'subject': metadata['subject'],
            'grade': metadata['grade'],
            # 结构化字段
            'grade_span': metadata.get('grade_span', ''),
            'grade_label': metadata.get('grade_label', ''),
            'discipline': metadata.get('discipline') or metadata['subject'],
            'book_id': metadata.get('book_id') or '',
            'book_title': metadata.get('title') or '',
            'book_volume': metadata.get('volume') or '',
            'publisher': metadata.get('publisher') or '',
            'edition': metadata.get('edition') or '',
            'catalog_matched': bool(metadata.get('matched')),
            'content_sha256': content_sha256,
            'source_key': file_key,
            'section_title': '',
            # ── P2 元数据补全（方案附录 A）
            # status：默认 published。设 archived 即为「下架」，
            # 检索侧 governance.exclude_archived 会把它过滤掉（步骤 3.3.2）。
            'status': 'published',
            # textbook_version 独立于 edition：版本治理与冲突消解按它排序，
            # 不去猜「2019版」与「第2册」的相对新旧。
            'textbook_version': metadata.get('edition') or '',
            'updated_at': datetime.now().strftime("%Y-%m-%d"),
            # ── GraphRAG 预留字段（方案 §3.4 步骤 3.5：阶段三**不实施** GraphRAG，
            #    仅留位）。这里显式写空值而不是省掉，是为了让「字段已预留、
            #    值为空」与「字段不存在」在数据上可区分 —— 将来上概念图时
            #    无需再改 schema，只需回填。
            # 现状：本库没有概念级标注体系，因此全为空。
            # 依赖它的地方只有一处 —— detect_conflicts 的 require_concept_key，
            # 空值即「不参与冲突检测」，这正是我们想要的克制行为。
            #
            # ⚠️ prerequisites 存 JSON 字符串而非 list：Chroma 1.4.0 的
            # metadata 只接受 str/int/float/bool/None，传 list 会抛
            # ValueError 直接中断入库（已实测）。
            'concept_key': '',
            'prerequisites': '',
        }

        # ── 组装待写入记录（先建 dict 列表，去重需要 doc_type 区分）
        records: List[Dict[str, Any]] = []
        content_type_counter: Dict[str, int] = {}

        for p in parents:
            md = dict(base_common)
            chapter = str(p.get("chapter") or "")
            page = self.page_for_offset(page_markers, int(p.get("start_pos", 0)))
            md.update({
                'doc_type': 'parent',
                'parent_id': '',
                'chunk_index': int(p["parent_index"]),
                'content_type': p.get("content_type") or 'body',
                'section_title': p.get("title") or '',
                # section 与 section_title 同源。父块此前**只写** section_title，
                # 导致 section 在父块上 100% 为空（元数据完整率被拉到 95.4%，
                # 缺口 3716 块里绝大部分是父块）。两个字段都写，口径一致。
                'section': p.get("title") or '',
                'chapter': chapter,
                'page': page,
                'char_len': int(p.get("char_len", 0)),
            })
            records.append({"id": p["_id"], "text": p["text"], "metadata": md,
                            "doc_type": "parent"})

        for c in children:
            md = dict(base_common)
            pi = c.get("parent_index")
            page = self.page_for_offset(page_markers, int(c.get("start_pos", 0)))
            md.update({
                'doc_type': 'child',
                'parent_id': (parent_ids[int(pi)]
                              if pi is not None and 0 <= int(pi) < len(parent_ids) else ''),
                'chunk_index': int(c["chunk_id"]),
                'content_type': c.get("content_type") or 'body',
                'chapter': str(c.get("chapter") or ''),
                'section': str(c.get("section_title") or ''),
                'section_title': str(c.get("section_title") or ''),
                'page': page,
            })
            records.append({"id": c["_id"], "text": c["text"], "metadata": md,
                            "doc_type": "child"})

        for r in records:
            ct = r["metadata"]["content_type"]
            content_type_counter[ct] = content_type_counter.get(ct, 0) + 1

        # ── 去重（P2 / 步骤 3.2）
        # 顺序：先精确（sha256，零风险）后语义（余弦，有误删风险）。
        # 父块与子块分开比较 —— 父块是子块的超集，混在一起会把整本书删光。
        if dedup_cfg.get("enabled", True):
            keep_ids = {r["id"] for r in records}
            kept_records, dropped = dedupe_chunks(
                [{"id": r["id"], "text": r["text"], "doc_type": r["doc_type"]}
                 for r in records],
                exact=bool(dedup_cfg.get("exact", True)),
                semantic_threshold=None,   # 语义去重需要向量，见下方嵌入后
                keep=str(dedup_cfg.get("keep", "longer")),
            )
            keep_ids = {k["id"] for k in kept_records}
            for d in dropped:
                gov_events.append({"file": file_path.name, "stage": "dedup",
                                   "action": "drop", **d})
            if dropped:
                logger.info(f"   知识治理·精确去重：丢弃 {len(dropped)} 个重复块")
            records = [r for r in records if r["id"] in keep_ids]

        if not records:
            return self._result(file_path, "skipped", metadata=metadata,
                                reason="去重后无有效内容块")

        embeddings = self._ensure_model().encode(
            [r["text"] for r in records], convert_to_numpy=True
        ).tolist()
        for r, emb in zip(records, embeddings):
            r["embeddings"] = emb

        # ── 语义去重（P2 / 步骤 3.2.3）：需要向量，故在嵌入之后做
        if dedup_cfg.get("enabled", True) and dedup_cfg.get("semantic_threshold"):
            kept_records, dropped = dedupe_chunks(
                [{"id": r["id"], "text": r["text"], "doc_type": r["doc_type"],
                  "embeddings": r["embeddings"]} for r in records],
                exact=False,   # 精确去重已在上一步做完
                semantic_threshold=float(dedup_cfg["semantic_threshold"]),
                keep=str(dedup_cfg.get("keep", "longer")),
            )
            if dropped:
                keep_ids = {k["id"] for k in kept_records}
                for d in dropped:
                    gov_events.append({"file": file_path.name, "stage": "dedup",
                                       "action": "drop", **d})
                logger.info(f"   知识治理·语义去重：丢弃 {len(dropped)} 个近重复块"
                            f"（余弦 ≥ {dedup_cfg['semantic_threshold']}）")
                records = [r for r in records if r["id"] in keep_ids]

        # ── 冲突检测（P2 / 步骤 3.3.3）
        # 只在**显式带 concept_key** 的块之间做。当前教材目录没有概念级标注，
        # 因此这里预期产出为空 —— 如实记录跳过的块数，不做无根据的「冲突」。
        conflicts: List[Dict[str, Any]] = []
        n_scanned = 0
        conf_cfg = dict(gov.get("conflicts") or {})
        if conf_cfg.get("enabled", True):
            candidates = [r for r in records
                          if r["metadata"].get("doc_type") == "child"]
            n_scanned = len(candidates)
            conflicts = detect_conflicts(
                [{"id": r["id"], "text": r["text"], "metadata": r["metadata"]}
                 for r in candidates],
                require_concept_key=bool(conf_cfg.get("require_concept_key", True)),
            )
            if conflicts:
                self._append_conflicts(conflicts, file_path.name)
                logger.warning(f"   知识治理·冲突检测：{len(conflicts)} 处，"
                               f"已写入 {conf_cfg.get('output', CONFLICTS_PATH)}")
            else:
                logger.info(f"   知识治理·冲突检测：扫描 {n_scanned} 个子块，"
                            f"无同概念数值矛盾"
                            + ("（当前无 concept_key 标注，仅能做单文件内检查）"
                               if conf_cfg.get("require_concept_key", True) else ""))
            # 「查过了、0 冲突」必须与「没查」可区分：
            # conflicts.jsonl 只在真有冲突时才创建，若不记录扫描事实，
            # 事后看到文件不存在无法判断是「干净」还是「没跑」。
            gov_events.append({
                "file": file_path.name, "stage": "conflict", "action": "scan",
                "scanned": n_scanned, "found": len(conflicts),
                "require_concept_key": bool(conf_cfg.get("require_concept_key", True)),
            })

        # 幂等：先删除该文件上一次入库的**全部**块（含父块），
        # 否则块数变少时会留下孤儿块（父块尤其容易残留）。
        try:
            existing = collection.get(where={'source_file': file_path.name}, include=[])
            old_ids = (existing or {}).get('ids') or []
            if old_ids:
                collection.delete(ids=old_ids)
                logger.info(f"   替换旧块 {len(old_ids)} 个")
        except Exception as e:
            logger.warning(f"   旧块清理失败（继续写入）：{e}")

        ids = [r["id"] for r in records]
        texts = [r["text"] for r in records]
        metadatas = [r["metadata"] for r in records]
        embeddings = [r["embeddings"] for r in records]

        for i in range(0, len(ids), 100):
            end = min(i + 100, len(ids))
            collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end]
            )

        # 治理报告落盘：每一次删除都可审计（方案 §6 风险项「语义去重误删有效内容」）
        if gov_events:
            self._append_gov_report(file_path.name, gov_events)

        n_parents = sum(1 for r in records if r["doc_type"] == "parent")
        n_children = len(records) - n_parents
        logger.info(f"   Ingested {n_children} 子块 + {n_parents} 父块")
        return self._result(file_path, "ok", chunks=n_children,
                            parents=n_parents, metadata=metadata,
                            collection=collection_name,
                            content_types=content_type_counter,
                            glyphs_fixed=glyph_fixed,
                            cleaned=sum(1 for e in gov_events
                                        if e.get("action") == "drop"),
                            conflicts=len(conflicts))

    def _append_gov_report(self, file_name: str, events: List[Dict[str, Any]]) -> None:
        """把知识治理事件追加到 jsonl 报告（失败不阻断摄取）。"""
        try:
            p = Path(GOV_REPORT_PATH)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                for e in events:
                    f.write(json.dumps({**e, "ts": datetime.now().isoformat(timespec="seconds")},
                                       ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"治理报告写入失败（已忽略）：{e}")

    def _append_conflicts(self, conflicts: List[Dict[str, Any]], file_name: str) -> None:
        """把检出的知识冲突追加到 conflicts.jsonl 供人工复核。"""
        try:
            out = self.retrieval_config.governance_config.get(
                "conflicts", {}).get("output", CONFLICTS_PATH)
            p = Path(out)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                for c in conflicts:
                    f.write(json.dumps({**c, "detected_in": file_name,
                                        "ts": datetime.now().isoformat(timespec="seconds")},
                                       ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"冲突报告写入失败（已忽略）：{e}")

    def ingest_manifest(self, pkg_dir: Path, dry_run: bool = False) -> Dict[str, int]:
        """
        Incrementally import a synced content package (downloaded by
        tools/sync_client.py) into ChromaDB using its manifest.json.

        Compatible with the existing pipeline: chunk ids and collection names
        from the manifest are preserved, so imported content merges cleanly
        with content ingested by `ingest_directory` without rewriting existing
        ids.

        Args:
            pkg_dir: Package directory containing manifest.json + chunks/.
            dry_run: Validate hashes and report, but do not write to DB.
        """
        manifest_path = pkg_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error(f"No manifest.json found in {pkg_dir}")
            return {"packages": 0, "chunks": 0}

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        package_id = manifest.get("package_id", pkg_dir.name)
        chunks = manifest.get("chunks", [])
        logger.info(f"Importing package {package_id}: {len(chunks)} chunks "
                    f"(dry_run={dry_run})")

        # Verify every chunk file exists and matches its content-address hash.
        compression = manifest.get("compression", "none")
        verified: List[Dict] = []
        for c in chunks:
            cf = pkg_dir / "chunks" / f"{c['hash']}.chunk"
            if not cf.exists():
                logger.error(f"  Missing chunk file: {cf.name}")
                continue
            payload = cf.read_bytes()
            if compression == "zlib":
                import zlib
                try:
                    raw = zlib.decompress(payload)
                except zlib.error:
                    raw = payload  # tolerate uncompressed legacy chunks
            else:
                raw = payload
            import hashlib
            if hashlib.sha256(raw).hexdigest() != c["hash"]:
                logger.error(f"  Hash mismatch: {cf.name}")
                continue
            verified.append({"chunk": c, "text": raw.decode("utf-8")})

        if dry_run:
            logger.info(f"[dry-run] {package_id}: {len(verified)}/{len(chunks)} "
                        "chunks verified; no DB write performed")
            return {"packages": 1, "chunks": len(verified)}

        # Group verified chunks by collection and upsert in batches.
        by_collection = defaultdict(list)
        for item in verified:
            by_collection[item["chunk"]["collection"]].append(item)

        total = 0
        for collection_name, items in by_collection.items():
            collection = self._get_or_create_collection(collection_name)
            ids = [it["chunk"]["id"] for it in items]
            texts = [it["text"] for it in items]
            embeddings = self._ensure_model().encode(texts, convert_to_numpy=True).tolist()
            metadatas = [
                dict(it["chunk"].get("metadata") or {}) for it in items
            ]
            for i in range(0, len(ids), 100):
                end = min(i + 100, len(ids))
                collection.upsert(
                    ids=ids[i:end],
                    embeddings=embeddings[i:end],
                    documents=texts[i:end],
                    metadatas=metadatas[i:end],
                )
            total += len(items)
            logger.info(f"  -> upserted {len(items)} chunks into {collection_name}")

        logger.info(f"Imported {total} chunks from package {package_id}")
        return {"packages": 1, "chunks": total}


# 目录说明文件：属于「目录的使用说明」，不是教学内容，一律不入库。
# 过去 `textbooks/README.md` 会被当成教材摄取（生成 neb_readme 集合），
# 既占索引也在早期版本里污染过检索结果。
SKIP_FILENAMES = {"readme.md", "readme.txt", "readme.json", "readme"}


def format_ingest_summary(summary: Dict[str, Any]) -> str:
    """
    把 `ingest_directory` 的返回值渲染成可读汇总（CLI 与 GUI 共用口径）。

    关键语义：只要出现 skipped / error，或有文件 0 块，就必须显式标注，
    不能笼统地说「导入完成」。
    """
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"摄取汇总：{summary.get('input', '')}")
    lines.append("=" * 78)
    lines.append(f"{'文件':<40}{'状态':<10}{'块数':>6}  {'集合':<28}")
    lines.append("-" * 78)
    for r in summary.get("files", []):
        status_cn = {"ok": "成功", "skipped": "跳过", "error": "失败"}.get(r["status"], r["status"])
        lines.append(f"{r['file'][:38]:<40}{status_cn:<10}{r['chunks']:>6}  {r['collection'][:26]:<28}")
        if r.get("reason"):
            lines.append(f"    └─ {r['reason']}")
    lines.append("-" * 78)
    lines.append(
        f"发现 {summary.get('found', 0)} 个文件：成功 {summary.get('ok', 0)}，"
        f"跳过 {summary.get('skipped', 0)}，失败 {summary.get('failed', 0)}；"
        f"共写入 {summary.get('chunks', 0)} 块"
    )
    if summary.get("glyphs_fixed"):
        lines.append(
            f"字形修复：共修正 {summary['glyphs_fixed']} 处 PDF 字形错映射"
            f"（斜体字母被错映射成生僻汉字等）。"
        )
    # 知识治理（P2）：删了多少必须可见 —— 静默删除无法审计，
    # 出了问题也无法定位是哪一批内容被吃掉了。
    if summary.get("cleaned"):
        lines.append(
            f"知识治理：清洗/去重丢弃 {summary['cleaned']} 个噪声或重复块"
            f"（明细见 {GOV_REPORT_PATH}）。"
        )
    if summary.get("conflicts"):
        lines.append(
            f"⚠️  检出 {summary['conflicts']} 处疑似知识冲突，"
            f"请复核 {CONFLICTS_PATH}。"
        )
    if summary.get("found", 0) > 0 and summary.get("ok", 0) == 0:
        lines.append("⚠️  没有任何文件成功入库 —— 请检查上方的跳过/失败原因，"
                     "不要把它当作「导入完成」。")
    elif summary.get("skipped", 0) or summary.get("failed", 0):
        lines.append("⚠️  部分文件未入库，学生将无法检索到这些内容。")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Universal Content Ingestion")
    parser.add_argument("--input", nargs='*', help="Folders to ingest")
    parser.add_argument("--ocr-mode", choices=['auto', 'force', 'never'], default='auto',
                        help="OCR mode: auto (detect), force (always), never (text only)")
    parser.add_argument("--subject", default=None,
                        help="强制学科（覆盖按书名推断），如 Science / 物理 / physics")
    parser.add_argument("--grade", default=None,
                        help="强制年级（覆盖按书名推断），1–12，如 10 / 11 / 12")
    
    # Auto-detect DB path
    auto_db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mati_data", "chroma_db"
    )
    parser.add_argument("--db", default=auto_db_path, help="ChromaDB path")
    parser.add_argument(
        "--ingest-from-manifest",
        metavar="DIR",
        help="Import synced packages from DIR (contains packages/*/manifest.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests/chunks without writing to ChromaDB"
    )
    
    args = parser.parse_args()

    # Manifest import path (from tools/sync_client.py output)
    if args.ingest_from_manifest:
        src_dir = Path(args.ingest_from_manifest)
        packages_root = src_dir / "packages"
        if not packages_root.exists():
            logger.error(f"No packages/ directory found under {src_dir}")
            sys.exit(1)
        ingester = UniversalContentIngester(args.db, args.ocr_mode,
                                            force_subject=args.subject,
                                            force_grade=args.grade)
        manifests = sorted(packages_root.glob("*/manifest.json"))
        if not manifests:
            logger.warning(f"No manifests found under {packages_root}")
            sys.exit(0)
        total = 0
        for mp in manifests:
            res = ingester.ingest_manifest(mp.parent, dry_run=args.dry_run)
            total += res["chunks"]
        logger.info(f"\n Imported {total} chunks from {len(manifests)} packages "
                    f"(dry_run={args.dry_run})")
        sys.exit(0)
    
    # Determine input directories
    dirs_to_process = []
    if args.input:
        dirs_to_process = args.input
    else:
        # Default: textbooks + 结构化内容 JSON。
        # 注意：`notes/` 目前只存放项目开发笔记（非高中教学内容），故不列入默认范围；
        # 教师笔记就位后可用 --input notes 显式指定。
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for d in ["textbooks", "scripts/data_collection/data/content"]:
            p = os.path.join(project_root, d)
            if os.path.exists(p):
                dirs_to_process.append(p)
        
        if not dirs_to_process:
            logger.warning("No input provided and no default folders found.")
            sys.exit(0)
    
    # Run ingestion
    ingester = UniversalContentIngester(args.db, args.ocr_mode,
                                        force_subject=args.subject,
                                        force_grade=args.grade)
    total_ok = total_found = 0
    for dir_path in dirs_to_process:
        summary = ingester.ingest_directory(dir_path)
        print(format_ingest_summary(summary))
        total_ok += summary["ok"]
        total_found += summary["found"]

    if total_found and not total_ok:
        logger.error("摄取结束：没有任何文件成功入库。")
        sys.exit(2)
    logger.info("\n摄取结束。")


if __name__ == "__main__":
    main()
