#!/usr/bin/env python3
"""
PDF → Qwen → content JSON 教师工具

教师只需提供课本 PDF，本脚本：
  1. 用 PyMuPDF 提取 PDF 文本（按章节/页分组）
  2. 调用 Qwen2.5-1.5B（llama-cpp，ChatML）为每章生成结构化学习内容
  3. 按 Mati 内容格式（subject → topics → subtopics → concepts → questions）
     输出 content JSON，放到内容目录后 GUI 即可浏览与做题。

依赖（项目已有）：pymupdf、llama-cpp-python
模型：mati_data/models/qwen2_5/*.gguf（Qwen2.5-1.5B-Instruct Q4_K_M）

用法：
    # 单个 PDF（推荐显式指定学科与年级）
    python tools/pdf_to_content.py --pdf 课本.pdf --subject Science --grade 10

    # 从文件名推断（如 "grade_7_science.pdf" 或 "七年级科学.pdf"）
    python tools/pdf_to_content.py --pdf 课本.pdf --out scripts/data_collection/data/content

    # 测试：只处理前 2 页（不写文件，打印章节划分）
    python tools/pdf_to_content.py --pdf 课本.pdf --sample 2 --no-write

    # 无模型测试：不调用 Qwen，仅验证 PDF 提取与分章
    python tools/pdf_to_content.py --pdf 课本.pdf --mock
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pdf_to_content")

# ---------------------------------------------------------------------------
# 学科名：中文 -> 英文数据键（与 UI 映射一致）
# ---------------------------------------------------------------------------
_SUBJECT_CN_TO_EN = {
    "科学": "Science", "数学": "Math", "英语语法": "English Grammar",
    "英语": "English", "计算机科学": "Computer Science", "社会科学": "Social Studies",
}

_SYSTEM_PROMPT = (
    "你是 Mati 智能教学系统的教材结构化编辑助手。"
    "请根据提供的教材片段，提取该章节的学习内容，并严格按照要求的 JSON 格式输出。"
    "只输出 JSON，不要输出任何解释或多余文字。"
)

def _build_user_prompt(chunk_text: str) -> str:
    return (
        "请阅读下面的教材片段，生成该章节的结构化学习内容。只输出 JSON：\n\n"
        '{\n'
        '  "name": "章节主题名称（简短）",\n'
        '  "concepts": [\n'
        '    {\n'
        '      "name": "概念名称（简短）",\n'
        '      "summary": "概念概述（2-3 句）",\n'
        '      "steps": ["学习要点 1", "学习要点 2", "学习要点 3"],\n'
        '      "questions": [\n'
        '        {"question": "一道简答题（基于本段内容）", '
        '"acceptable_answers": ["参考答案（1-2 句）"], "hints": ["提示"]},\n'
        '        {"question": "第二道简答题", '
        '"acceptable_answers": ["参考答案"], "hints": []}\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        "要求：\n"
        "1. concepts 生成 2-4 个，每个概念 2-4 道简答题；\n"
        "2. 题目必须基于教材内容，并给出明确参考答案；\n"
        "3. 使用中文，可保留必要的英文术语。\n\n"
        "教材片段：\n" + chunk_text
    )


# ---------------------------------------------------------------------------
# PDF 文本提取与分章
# ---------------------------------------------------------------------------
def extract_pdf_text(pdf_path: Path) -> List[str]:
    """按页返回 PDF 文本列表（每页一个字符串）。"""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        logger.error("需要安装 pymupdf：pip install pymupdf")
        raise SystemExit(1) from e

    pages: List[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pages.append(page.get_text())
    return pages


def group_pages_into_topics(pages: List[str], pages_per_topic: int) -> List[Dict[str, Any]]:
    """把页文本按 pages_per_topic 分组为章节块。"""
    topics: List[Dict[str, Any]] = []
    n = len(pages)
    for start in range(0, n, pages_per_topic):
        end = min(start + pages_per_topic, n)
        text = "\n".join(pages[start:end]).strip()
        if text:
            topics.append({"pages": (start + 1, end), "text": text})
    return topics


def truncate_text(text: str, max_chars: int = 1500) -> str:
    """按字符截断到合理长度（尽量在句末断）。"""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # 优先在句号/问号处断句
    m = max(cut.rfind("。"), cut.rfind("."), cut.rfind("？"), cut.rfind("?"))
    if m > max_chars * 0.5:
        cut = cut[: m + 1]
    return cut


# ---------------------------------------------------------------------------
# Qwen 生成与 JSON 解析
# ---------------------------------------------------------------------------
class QwenGenerator:
    def __init__(self, model_path: Path, n_threads: int = 4):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            logger.error("需要安装 llama-cpp-python：pip install llama-cpp-python")
            raise SystemExit(1) from e
        logger.info("加载 Qwen 模型：%s", model_path)
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_batch=96,
            n_threads=n_threads,
            use_mmap=True,
            use_mlock=False,
            f16_kv=False,
            verbose=False,
        )
        self.stop = ["<|im_end|>", "<|endoftext|>", "</s>"]

    def _prompt(self, user_text: str) -> str:
        return (
            "<|im_start|>system\n" + _SYSTEM_PROMPT + "<|im_end|>\n"
            "<|im_start|>user\n" + user_text + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate_json(self, chunk_text: str, max_tokens: int = 1400) -> Optional[Dict[str, Any]]:
        prompt = self._prompt(_build_user_prompt(chunk_text))
        try:
            out = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=self.stop,
                stream=False,
            )
            raw = out["choices"][0]["text"].strip()
        except Exception as e:
            logger.error("生成失败：%s", e)
            return None
        return extract_json(raw)


def extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """从模型输出中鲁棒地提取 JSON 对象。"""
    if not raw:
        return None
    # 1) ```json ... ```
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # 2) 第一个 { 到最后一个 }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    logger.warning("未能从模型输出解析出 JSON（前 120 字：%s）", raw[:120].replace("\n", " "))
    return None


# ---------------------------------------------------------------------------
# 组装 content JSON
# ---------------------------------------------------------------------------
def _normalize_concepts(concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """规范化 LLM 输出的概念：补齐缺省字段，统一 steps/questions 结构。"""
    out: List[Dict[str, Any]] = []
    for c in concepts or []:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        # steps：对象 -> 字符串
        steps = []
        for s in c.get("steps", []) or []:
            if isinstance(s, dict):
                s = s.get("step") or s.get("text") or ""
            if isinstance(s, str) and s.strip():
                steps.append(s.strip())
        # questions：确保 question/acceptable_answers/hints 齐全
        qs = []
        for q in c.get("questions", []) or []:
            if not isinstance(q, dict) or not q.get("question"):
                continue
            qs.append({
                "question": str(q["question"]).strip(),
                "acceptable_answers": [
                    str(a).strip() for a in (q.get("acceptable_answers") or [])
                    if str(a).strip()
                ] or [""],
                "hints": [
                    str(h).strip() for h in (q.get("hints") or [])
                    if str(h).strip()
                ] or [],
            })
        out.append({
            "name": str(c["name"]).strip(),
            "summary": str(c.get("summary") or "").strip(),
            "steps": steps,
            "questions": qs,
        })
    return out


def assemble_content(subject: str, grade: int, topics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把每个章节块（{name, concepts}）包装成 topic -> subtopics 结构，并规范化。"""
    out_topics = []
    for t in topics:
        name = (t.get("name") or "未命名章节").strip()
        concepts = _normalize_concepts(t.get("concepts") or [])
        out_topics.append({
            "name": name,
            "subtopics": [
                {"name": name, "concepts": concepts}
            ],
        })
    return {
        "subject": subject,
        "grade": grade,
        "topics": out_topics,
    }


def infer_subject_grade(pdf_path: Path, subject_arg: Optional[str], grade_arg: Optional[int]):
    """从参数或文件名推断 subject(英文key)/grade。"""
    subject = subject_arg
    grade = grade_arg
    stem = pdf_path.stem
    if grade is None:
        m = re.search(r"(?:grade|class|年级)[_\-]?(\d{1,2})", stem, re.I)
        if m:
            grade = int(m.group(1))
    if subject is None:
        # 中文学科名
        for cn, en in _SUBJECT_CN_TO_EN.items():
            if cn in stem:
                subject = en
                break
        if subject is None:
            for en in ("Science", "Math", "Mathematics", "English", "Computer", "Social"):
                if en.lower() in stem.lower():
                    subject = en if en != "Mathematics" else "Math"
                    break
    if subject is None:
        logger.warning("无法从文件名推断学科，请用 --subject 指定（中文或英文均可）")
        subject = "Science"
    if grade is None:
        logger.warning("无法从文件名推断年级，请用 --grade 指定")
        grade = 10
    if subject in _SUBJECT_CN_TO_EN.values():
        pass
    elif subject in _SUBJECT_CN_TO_EN:
        subject = _SUBJECT_CN_TO_EN[subject]
    return subject, grade


def generate_from_pdf(
    pdf_path: Path,
    subject: Optional[str] = None,
    grade: Optional[int] = None,
    out_dir: str = "scripts/data_collection/data/content",
    model_arg: str = "mati_data/models/qwen2_5",
    pages_per_topic: int = 8,
    sample: int = 0,
    no_write: bool = False,
    mock: bool = False,
    n_threads: int = 4,
) -> Dict[str, Any]:
    """
    PDF → Qwen → content JSON 核心流程（供教师工作台 GUI 与命令行工具共用）。

    返回：
        {"ok": bool, "out_file": Optional[str], "topics": int,
         "concepts": int, "questions": int, "error": Optional[str]}
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return {"ok": False, "error": f"找不到 PDF：{pdf_path}"}

    # 1) 提取文本
    logger.info("提取 PDF 文本：%s", pdf_path.name)
    pages = extract_pdf_text(pdf_path)
    if not pages or not any(p.strip() for p in pages):
        return {"ok": False, "error": "PDF 无可提取文本（可能是扫描件，请先 OCR）。"}
    if sample and sample > 0:
        pages = pages[:sample]
    logger.info("共 %d 页", len(pages))

    # 2) 分章
    chunks = group_pages_into_topics(pages, pages_per_topic)
    logger.info("划分为 %d 个章节块", len(chunks))
    if no_write:
        for c in chunks:
            logger.info("  章节（第 %d-%d 页）：%s",
                        c["pages"][0], c["pages"][1], truncate_text(c["text"], 80))

    # 3) 生成
    subject, grade = infer_subject_grade(pdf_path, subject, grade)
    logger.info("学科：%s，年级：%d", subject, grade)

    topics: List[Dict[str, Any]] = []
    if mock:
        for i, c in enumerate(chunks, 1):
            topics.append({
                "name": f"章节 {i}",
                "concepts": [{
                    "name": f"概念 {i}",
                    "summary": truncate_text(c["text"], 200),
                    "steps": ["示例学习要点"],
                    "questions": [{"question": f"本章的核心内容是什么？",
                                   "acceptable_answers": ["（占位答案，请使用非 mock 模式生成）"],
                                   "hints": []}],
                }],
            })
    else:
        model = QwenGenerator(_resolve_model(model_arg), n_threads)
        for i, c in enumerate(chunks, 1):
            logger.info("生成章节 %d/%d（第 %d-%d 页）...",
                        i, len(chunks), c["pages"][0], c["pages"][1])
            data = model.generate_json(truncate_text(c["text"]))
            if not data or not data.get("name"):
                logger.warning("章节 %d 生成失败，跳过", i)
                continue
            topics.append(data)

    if not topics:
        return {"ok": False, "error": "未能生成任何章节内容（请检查模型或教材文本）。"}

    content = assemble_content(subject, grade, topics)
    n_concepts = sum(len(t.get("concepts") or []) for t in topics)
    n_questions = sum(
        len(c.get("questions") or []) for t in topics for c in (t.get("concepts") or [])
    )
    logger.info("生成完成：%d 章 / %d 概念 / %d 题", len(topics), n_concepts, n_questions)

    if no_write:
        return {"ok": True, "out_file": None, "topics": len(topics),
                "concepts": n_concepts, "questions": n_questions, "error": None}

    # 4) 写文件
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{subject.lower().replace(' ', '_')}_grade_{grade}_auto.json"
    # 避免覆盖同名 subject 的现有文件
    existing = [p for p in out_dir.glob("*.json") if p.name != out_file.name]
    for p in existing:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("subject") == subject:
                logger.warning("内容目录已存在学科 %s 的数据（%s），生成文件可能覆盖其在应用中的显示。",
                               subject, p.name)
        except Exception:
            pass
    out_file.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写入：%s", out_file)
    return {"ok": True, "out_file": str(out_file), "topics": len(topics),
            "concepts": n_concepts, "questions": n_questions, "error": None}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PDF → Qwen → content JSON 教师工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pdf", required=True, help="课本 PDF 路径")
    parser.add_argument("--subject", help="学科（中文或英文，如 Science/科学）")
    parser.add_argument("--grade", type=int, help="年级（8-12）")
    parser.add_argument("--out", default="scripts/data_collection/data/content",
                        help="输出目录（ContentManager 加载目录）")
    parser.add_argument("--model", default="mati_data/models/qwen2_5",
                        help="Qwen GGUF 目录或文件")
    parser.add_argument("--pages-per-topic", type=int, default=8,
                        help="每多少页划分一个章节主题")
    parser.add_argument("--sample", type=int, default=0,
                        help="只处理前 N 页（测试用）")
    parser.add_argument("--no-write", action="store_true",
                        help="不写文件，仅打印章节划分与统计")
    parser.add_argument("--mock", action="store_true",
                        help="不调用 Qwen，输出占位章节（仅验证管道）")
    parser.add_argument("--n-threads", type=int, default=4, help="模型线程数")
    args = parser.parse_args()

    result = generate_from_pdf(
        pdf_path=Path(args.pdf),
        subject=args.subject,
        grade=args.grade,
        out_dir=args.out,
        model_arg=args.model,
        pages_per_topic=args.pages_per_topic,
        sample=args.sample,
        no_write=args.no_write,
        mock=args.mock,
        n_threads=args.n_threads,
    )
    if not result["ok"]:
        logger.error("处理失败：%s", result.get("error"))
        return 1
    if result.get("out_file"):
        logger.info("已写入：%s", result["out_file"])
    return 0


def _resolve_model(model_arg: str) -> Path:
    p = Path(model_arg)
    if p.is_dir():
        ggs = list(p.glob("*.gguf"))
        if not ggs:
            logger.error("目录 %s 下没有 .gguf 模型文件。请先下载 Qwen2.5-1.5B GGUF。", p)
            raise SystemExit(1)
        return ggs[0]
    if p.is_file():
        return p
    logger.error("模型不存在：%s", p)
    raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
