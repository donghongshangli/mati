#!/usr/bin/env python3
"""
Enhanced Chunker for Mati Learning System

Math-aware chunking with overlap to prevent formula splitting.

Features:
- Overlap between chunks（摄取侧 0.1，CLI 默认 0.2）
- LaTeX/formula preservation
- Sentence-boundary splitting for both Chinese and English
  （中文句末标点「。！？；…」后通常无空白，需单独规则；顿号「、」不切分）
- Math marker detection ($, $$, \\[, \\])
- Preserves code blocks
- Metadata tagging
"""

import os
import re
import logging
from typing import List, Dict, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ──────────────────────── 结构识别正则（P1：结构感知切分）
# 结构标题：Markdown 标题 / 第X章节课篇 / 1.2 编号 / （一）式编号 / 教材固定栏目名
_HEADING_PATTERNS = [
    re.compile(r"^\s*#{1,6}\s+\S"),
    re.compile(r"^\s*第\s*[一二三四五六七八九十百零\d]+\s*[章节单元课篇讲]"),
    re.compile(r"^\s*\d{1,2}\s*\.\s*\d{1,2}\s+\S"),      # 1.2 xxx
    re.compile(r"^\s*\d{1,2}\s*[.、]\s*\S"),               # 1. xxx / 12、xxx
    re.compile(r"^\s*[（(]\s*[一二三四五六七八九十\d]+\s*[)）]"),
    re.compile(r"^\s*(例题|练习|习题|思考与讨论|做一做|实验|本节小结|章末|"
               r"复习|阅读|拓展|探究|活动|科学漫步|STSE)"),
]

# 目录条目：短、以「第X章/第X节」或「1.2」开头、且不以句号结尾
_TOC_LINE_RE = re.compile(
    r"^(第\s*[一二三四五六七八九十百零\d]+\s*[章节单元课篇]|"
    r"\d{1,2}(\.\d{1,2})*\s*[.、]?\s*\S)"
)

# 封面/版权页特征
_COVER_RE = re.compile(r"(ISBN|版权所有|人民教育出版社|出版发行|定价|教材\s*$|"
                       r"普通高中教科书)")

# 内容类型判定
_CT_EXAMPLE_RE = re.compile(r"(例题|【例|^\s*例\s*\d|^\s*例\d)")
_CT_EXERCISE_RE = re.compile(r"(练习|习题|思考与讨论|做一做|章末|复习题|本节小结|"
                             r"课后作业)")
_CT_FIG_RE = re.compile(r"^(图|表)\s*[\d一二三四五六七八九十]")
_CT_DEF_RE = re.compile(r"(定义|叫做|称为|是指|定义为|的含义是|概念)")
_CT_THEOREM_RE = re.compile(r"(定律|定理|原理|公式|规律|推论)")


class EnhancedChunker:
    """
    Math-aware text chunking with overlap.
    
    Prevents splitting of mathematical formulas and code blocks.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap_ratio: float = 0.2,
        min_chunk_size: int = 100
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            overlap_ratio: Overlap between chunks (0.0-0.5)
            min_chunk_size: Minimum chunk size
        """
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.overlap_size = int(chunk_size * overlap_ratio)
        self.min_chunk_size = min_chunk_size
        
        logger.info(f"Enhanced chunker initialized:")
        logger.info(f"   Chunk size: {chunk_size} chars")
        logger.info(f"   Overlap: {overlap_ratio*100:.0f}% ({self.overlap_size} chars)")
    
    def find_math_regions(self, text: str) -> List[Tuple[int, int]]:
        """
        Find all math regions in text (LaTeX, inline math, etc.).
        
        Args:
            text: Input text
            
        Returns:
            List of (start, end) positions for math regions
        """
        math_regions = []
        
        # LaTeX display math: $$...$$
        for match in re.finditer(r'\$\$.*?\$\$', text, re.DOTALL):
            math_regions.append((match.start(), match.end()))
        
        # LaTeX inline math: $...$
        for match in re.finditer(r'\$[^\$]+?\$', text):
            math_regions.append((match.start(), match.end()))
        
        # LaTeX brackets: \[...\]
        for match in re.finditer(r'\\\[.*?\\\]', text, re.DOTALL):
            math_regions.append((match.start(), match.end()))
        
        # LaTeX parentheses: \(...\)
        for match in re.finditer(r'\\\(.*?\\\)', text):
            math_regions.append((match.start(), match.end()))
        
        # Common math expressions (e.g., "E = mc²", "x² + 5x + 6 = 0")
        for match in re.finditer(r'[a-zA-Z]\s*[²³⁴⁵⁶⁷⁸⁹⁰₁₂₃₄₅₆₇₈₉₀\^]\s*[+\-=]', text):
            # Extend to full expression
            start = match.start()
            end = match.end()
            
            # Extend backwards to start of expression
            while start > 0 and text[start-1] not in ['.', '\n', '?', '!']:
                start -= 1
            
            # Extend forwards to end of expression
            while end < len(text) and text[end] not in ['.', '\n', '?', '!']:
                end += 1
            
            math_regions.append((start, end))
        
        # Merge overlapping regions
        if math_regions:
            math_regions.sort()
            merged = [math_regions[0]]
            
            for start, end in math_regions[1:]:
                if start <= merged[-1][1]:
                    # Overlapping - merge
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            
            return merged
        
        return []
    
    def find_code_blocks(self, text: str) -> List[Tuple[int, int]]:
        """
        Find code blocks in text.
        
        Args:
            text: Input text
            
        Returns:
            List of (start, end) positions for code blocks
        """
        code_regions = []
        
        # Markdown code blocks: ```...```
        for match in re.finditer(r'```.*?```', text, re.DOTALL):
            code_regions.append((match.start(), match.end()))
        
        # Inline code: `...`
        for match in re.finditer(r'`[^`]+?`', text):
            code_regions.append((match.start(), match.end()))
        
        return code_regions
    
    def is_in_protected_region(
        self,
        position: int,
        protected_regions: List[Tuple[int, int]]
    ) -> bool:
        """
        Check if position is inside a protected region.
        
        Args:
            position: Character position
            protected_regions: List of (start, end) tuples
            
        Returns:
            True if position is protected
        """
        for start, end in protected_regions:
            if start <= position <= end:
                return True
        return False
    
    def find_sentence_boundaries(self, text: str) -> List[int]:
        """
        Find sentence boundaries (safe split points).

        中文必须单独处理：中文行文的句末标点后通常**没有空白字符**，
        因此不能用英文的 `[.!?]\\s+` 规则（会导致中文文档完全识别不到句界，
        退化为按固定字符数硬切）。顿号「、」是列举分隔符而非句界，不参与切分。

        Args:
            text: Input text
            
        Returns:
            List of character positions for sentence boundaries
        """
        boundaries = []

        # 中文句末标点：无需后续空白；允许后接右引号/右括号
        for match in re.finditer(r'[。！？；…]+[”’」』）)]*', text):
            boundaries.append(match.end())

        # 英文句末标点：要求后续空白，避免误切小数、缩写、域名
        for match in re.finditer(r'[.!?]\s+', text):
            boundaries.append(match.end())

        # 段落分隔（空行；PDF 抽取的单个换行多为折行，不作为句界）
        for match in re.finditer(r'\n\s*\n+', text):
            boundaries.append(match.start())

        return sorted(set(boundaries))

    # ══════════════════════════════ 结构感知切分（P1）
    #
    # 背景：原实现把 PDF 抽取的文本直接按固定字符数切块，章节标题、例题标记、
    # 图注都不是结构信号。后果是无法做父子索引 —— 召回单元与生成单元被绑死
    # 在同一个粒度上：块小则语义不完整（定理陈述与例题解析被拆散），
    # 块大则向量被稀释、相似度下降。
    #
    # 这里补上结构信号：先按标题把文本切成语义段，再组装成
    #   父块（完整小节，供生成）+ 子块（语义段，供召回）
    # 两级粒度。

    def _is_heading(self, line: str) -> bool:
        """
        判断一行是否为结构标题。

        约束：标题行必须**短**（PDF 抽取的正文行可能很长），
        否则「1. 曲线运动」这种编号会被误判为正文里的枚举项。
        """
        s = (line or "").strip()
        if not s or len(s) > 40:
            return False
        for pat in _HEADING_PATTERNS:
            if pat.match(s):
                return True
        return False

    def looks_like_toc(self, text: str) -> bool:
        """
        判断一段文本是否为**目录页**。

        目录页的典型特征：行短、多为「第X章/第X节」「1.2 标题」形式的条目、
        且条目行末尾**没有句号**（正文的编号列举通常带句末标点）。
        """
        lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
        if len(lines) < 5:
            return False
        hits = 0
        for l in lines:
            if len(l) > 24 or l.endswith("。"):
                continue
            if _TOC_LINE_RE.match(l):
                hits += 1
        return hits / len(lines) >= 0.6

    def classify_content_type(self, text: str, is_first: bool = False) -> str:
        """
        内容类型判定（供重排的 structure_bonus 特征与检索侧噪声过滤使用）。

        取值与 retrieval.yaml 的 structure_bonus 表对应：
        definition / theorem / example / exercise / body / figure_caption / toc / cover
        """
        if not text or not text.strip():
            return "body"
        head = text[:80]
        sample = text[:300]
        if self.looks_like_toc(text):
            return "toc"
        # 封面/版权页：出现在最前面、很短、含出版信息
        if is_first and len(text) < 300 and _COVER_RE.search(sample):
            return "cover"
        if _CT_EXAMPLE_RE.search(head):
            return "example"
        if _CT_EXERCISE_RE.search(head):
            return "exercise"
        if _CT_FIG_RE.match(head.strip()):
            return "figure_caption"
        if _CT_DEF_RE.search(sample):
            return "definition"
        if _CT_THEOREM_RE.search(sample):
            return "theorem"
        return "body"

    def _split_sections(self, text: str) -> List[Tuple[str, str, str, int]]:
        """
        按标题行把文本切成语义段。

        Returns:
            [(标题, 正文, 所属章标题, 段在**全文**中的起始偏移), ...]

        为什么带上「章标题」与「起始偏移」（P2）
        ──────────────────────────────────────
        - 章标题：方案 §3.4 步骤 3.1 要求元数据补 `chapter`。教材的
          `第X章` 标题与 `X.X` 小节标题是**两个层级**，只记小节会丢掉章。
        - 起始偏移：P2 要把 `page`（页码）落到每个块上。PDF 抽取时会在
          每页开头插入页标记，块在全文中的位置反查页标记即可得页码。
          此前 `start_pos` 是**段内相对偏移**，无法反查 —— 这是
          `chapter`/`page` 一直填不上的根因，不是「PDF 解析不了」。
        """
        sections: List[Tuple[str, str, str, int]] = []
        cur_title: str = ""
        cur_chapter: str = ""
        cur_start = 0
        cur_lines: List[str] = []
        pos = 0
        for line in (text or "").split("\n"):
            line_start = pos
            pos += len(line) + 1        # +1 for the '\n' we split on
            if self._is_heading(line):
                if cur_title or any(l.strip() for l in cur_lines):
                    sections.append((cur_title, "\n".join(cur_lines),
                                     cur_chapter, cur_start))
                cur_title = line.strip()
                chapter = self._as_chapter(cur_title)
                if chapter:
                    cur_chapter = chapter
                cur_start = line_start
                cur_lines = []
            else:
                cur_lines.append(line)
        if cur_title or any(l.strip() for l in cur_lines):
            sections.append((cur_title, "\n".join(cur_lines), cur_chapter, cur_start))
        return sections

    @staticmethod
    def _as_chapter(title: str) -> str:
        """
        从标题行里识别「第X章」，识别不出返回空串。

        只认「章」这一层：小节（`第X节` / `1.2`）另存 `section_title`。
        """
        m = re.match(r"^\s*(第\s*[一二三四五六七八九十百零\d]+\s*章[^\n]{0,20})", title or "")
        return m.group(1).strip() if m else ""

    def _parent_windows(self, text: str, target: int, max_chars: int,
                        base_offset: int = 0) -> List[Tuple[str, int]]:
        """
        把一段文本切成父块窗口。

        短于 max_chars 的整段作为一个父块（完整小节）；
        超出时按句子边界切成 ~target 大小的窗口，绝不从句中切断。

        Returns:
            [(窗口文本, 窗口在**全文**中的起始偏移), ...]
            base_offset 是本段在全文中的起始偏移（P2：用于反查页码）。
        """
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= max_chars:
            return [(text, base_offset)]

        boundaries = [b for b in self.find_sentence_boundaries(text) if 0 < b < len(text)]
        windows: List[Tuple[str, int]] = []
        start = 0
        guard = 0
        while start < len(text) and guard < len(text) + 100:
            guard += 1
            end = min(start + target, len(text))
            if end >= len(text):
                windows.append((text[start:], base_offset + start))
                break
            safe = None
            for b in reversed(boundaries):
                if start < b <= end:
                    safe = b
                    break
            if safe is None or safe <= start:
                safe = end          # 找不到句界就硬切，保证前进
            windows.append((text[start:safe], base_offset + start))
            start = safe
        return [(w.strip(), off) for w, off in windows if w.strip()]

    def chunk_with_structure(
        self,
        text: str,
        child_target: int = 384,
        child_min: int = 120,
        child_overlap_ratio: float = 0.1,
        parent_target: int = 1400,
        parent_max: int = 1800,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        结构感知切分 + 父子索引（P1）。

        Returns:
            (children, parents)

            children[i] = {
                'text', 'chunk_id', 'parent_index', 'start_pos', 'end_pos',
                'content_type', 'chapter', 'section_title'
            }
            parents[j] = {
                'parent_index', 'text', 'title', 'content_type', 'char_len',
                'chapter', 'start_pos'
            }

            `start_pos` 是块在**全文**中的绝对偏移（此前是段内相对偏移），
            摄取侧据此反查页码，写入元数据 `page`（方案 §3.4 步骤 3.1）。
            `chapter` / `section_title` 对应元数据的 `chapter` / `section`。

        为什么返回两级而不是一级
        ────────────────────────
        子块（~384 字符）用于**召回**：向量表达集中，相似度更高；
        父块（~1400 字符，一个完整小节）用于**生成**：语义完整，
        定理陈述与例题解析不会被拆散。
        这正是语雀方法论里的「父子索引」，也是本项目此前缺失的一环。
        """
        if not text or not text.strip():
            return [], []

        sections = self._split_sections(text)

        # 把语义段按 parent_target 累积成「父块组」，避免出现大量零碎父块
        groups: List[Tuple[str, str, str, str, int]] = []
        buf_text, buf_title, buf_type = "", "", "body"
        buf_chapter, buf_abs = "", 0
        for title, body, chapter, abs_start in sections:
            t = ((title + "\n") if title else "") + body
            t = t.strip()
            if not t:
                continue
            ctype = self.classify_content_type(t, is_first=(not groups and not buf_text))
            if buf_text and len(buf_text) + len(t) > parent_target:
                groups.append((buf_title, buf_text, buf_type, buf_chapter, buf_abs))
                buf_text, buf_title, buf_type = t, title, ctype
                buf_chapter, buf_abs = chapter, abs_start
            elif not buf_text:
                buf_text, buf_title, buf_type = t, title, ctype
                buf_chapter, buf_abs = chapter, abs_start
            else:
                buf_text = buf_text + "\n" + t
        if buf_text:
            groups.append((buf_title, buf_text, buf_type, buf_chapter, buf_abs))

        children: List[Dict] = []
        parents: List[Dict] = []
        for title, gtext, gtype, chapter, gabs in groups:
            for win, win_abs in self._parent_windows(gtext, parent_target,
                                                      parent_max, base_offset=gabs):
                pidx = len(parents)
                wtype = gtype if gtype != "body" else self.classify_content_type(win)
                parents.append({
                    "parent_index": pidx,
                    "text": win,
                    "title": title or "",
                    "content_type": wtype,
                    "char_len": len(win),
                    "chapter": chapter,
                    "start_pos": win_abs,
                })
                subs = self.smart_chunk_with_overlap(
                    win, chunk_size=child_target, overlap_ratio=child_overlap_ratio
                )
                for c in subs:
                    ct = (c.get("text") or "").strip()
                    if len(ct) < child_min:
                        # 太短的子块单独召回没有意义（向量噪声大），
                        # 其内容已包含在父块里，父块被命中时依然会喂给模型，
                        # 因此丢弃不会丢信息。
                        continue
                    children.append({
                        "text": ct,
                        "chunk_id": len(children),
                        "parent_index": pidx,
                        # 子块偏移 = 窗口偏移 + 窗口内相对偏移
                        "start_pos": win_abs + int(c.get("start_pos", 0)),
                        "end_pos": win_abs + int(c.get("end_pos", 0)),
                        "content_type": (wtype if wtype != "body"
                                         else self.classify_content_type(ct)),
                        "chapter": chapter,
                        "section_title": title or "",
                        "metadata": {},
                    })

        logger.info(
            f"结构切分：{len(text)} 字符 -> {len(parents)} 个父块 / {len(children)} 个子块"
        )
        return children, parents
    
    def smart_chunk_with_overlap(
        self,
        text: str,
        metadata: Dict = None,
        chunk_size: int = None,
        overlap_ratio: float = None,
    ) -> List[Dict]:
        """
        Chunk text with math-aware splitting and overlap.
        
        Args:
            text: Input text
            metadata: Optional metadata to attach to chunks
            chunk_size: 覆盖实例默认块长（父子索引里子块比主块短）
            overlap_ratio: 覆盖实例默认重叠比
            
        Returns:
            List of chunk dictionaries
        """
        size = int(chunk_size or self.chunk_size)
        overlap = int(size * (self.overlap_ratio if overlap_ratio is None else overlap_ratio))
        if not text or len(text) < self.min_chunk_size:
            return [{
                'text': text,
                'chunk_id': 0,
                'start_pos': 0,
                'end_pos': len(text),
                'metadata': metadata or {}
            }]
        
        # Find protected regions (math and code)
        math_regions = self.find_math_regions(text)
        code_regions = self.find_code_blocks(text)
        protected_regions = sorted(math_regions + code_regions)
        
        # Find sentence boundaries
        sentence_boundaries = self.find_sentence_boundaries(text)
        
        chunks = []
        current_pos = 0
        chunk_id = 0
        loop_counter = 0
        
        while current_pos < len(text):
            # Safety check for infinite loops
            if loop_counter > len(text) + 1000: # Generous buffer
                logger.error(f"Infinite loop detected in chunking! Breaking safely.")
                break
            loop_counter += 1

            # Calculate target end position
            target_end = current_pos + size
            
            if target_end >= len(text):
                # Last chunk
                chunk_text = text[current_pos:]
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'start_pos': current_pos,
                    'end_pos': len(text),
                    'metadata': metadata or {}
                })
                break
            
            # Find safe split point near target_end
            # Priority: sentence boundary > paragraph > word boundary
            
            # Check if target_end is in protected region
            if self.is_in_protected_region(target_end, protected_regions):
                # Find end of protected region
                for start, end in protected_regions:
                    if start <= target_end <= end:
                        target_end = end
                        break
            
            # Find nearest sentence boundary before target_end
            safe_split = target_end
            for boundary in reversed(sentence_boundaries):
                if boundary <= target_end and boundary > current_pos:
                    # Make sure boundary is not in protected region
                    if not self.is_in_protected_region(boundary, protected_regions):
                        safe_split = boundary
                        break
            
            # If no good sentence boundary, try word boundary
            if safe_split == target_end:
                # Find last space before target_end
                last_space = text.rfind(' ', current_pos, target_end)
                if last_space > current_pos:
                    safe_split = last_space
            
            # Create chunk
            chunk_text = text[current_pos:safe_split].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'start_pos': current_pos,
                    'end_pos': safe_split,
                    'metadata': metadata or {}
                })
                chunk_id += 1
            
            # Ensure we made progress
            if safe_split <= current_pos:
                safe_split = current_pos + size

            # Move to next position with overlap
            next_pos = safe_split - overlap
            
            # Make sure we're strictly moving forward
            if chunks:
                last_start = chunks[-1]['start_pos']
                if next_pos <= last_start:
                    next_pos = safe_split # Give up overlap to ensure progress
            else:
                 if next_pos <= 0:
                     next_pos = safe_split

            # Absolute final safety: if we haven't moved, force move
            if next_pos <= current_pos:
                next_pos = current_pos + max(1, int(size * 0.5))

            current_pos = next_pos
        
        logger.info(f"Created {len(chunks)} chunks from {len(text)} chars")
        logger.info(f"   Math regions protected: {len(math_regions)}")
        logger.info(f"   Code blocks protected: {len(code_regions)}")
        
        return chunks
    
    def chunk_markdown_file(
        self,
        markdown_file: str,
        output_dir: str = None
    ) -> List[Dict]:
        """
        Chunk a markdown file and optionally save chunks.
        
        Args:
            markdown_file: Path to markdown file
            output_dir: Optional directory to save chunks
            
        Returns:
            List of chunks
        """
        logger.info(f"Chunking: {markdown_file}")
        
        # Read file
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata from YAML frontmatter (if present)
        metadata = {}
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                # Parse YAML frontmatter
                frontmatter = parts[1]
                for line in frontmatter.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip().strip('"\'')
                
                # Use content after frontmatter
                content = parts[2].strip()
        
        # Add source file to metadata
        metadata['source_file'] = os.path.basename(markdown_file)
        
        # Chunk content
        chunks = self.smart_chunk_with_overlap(content, metadata)
        
        # Save chunks if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(markdown_file).stem
            
            for chunk in chunks:
                chunk_file = output_path / f"{base_name}_chunk_{chunk['chunk_id']:03d}.txt"
                
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    # Write metadata as comments
                    f.write(f"# Chunk {chunk['chunk_id']}\n")
                    f.write(f"# Source: {metadata.get('source_file', 'unknown')}\n")
                    f.write(f"# Grade: {metadata.get('grade', 'unknown')}\n")
                    f.write(f"# Subject: {metadata.get('subject', 'unknown')}\n")
                    f.write(f"# Chapter: {metadata.get('chapter', 'unknown')}\n")
                    f.write(f"# Position: {chunk['start_pos']}-{chunk['end_pos']}\n")
                    f.write("\n")
                    f.write(chunk['text'])
            
            logger.info(f"Saved {len(chunks)} chunks to {output_dir}")
        
        return chunks
    
    def batch_chunk_directory(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*.md"
    ) -> Dict:
        """
        Batch chunk all files in a directory.
        
        Args:
            input_dir: Directory containing markdown files
            output_dir: Directory to save chunks
            file_pattern: File pattern to match (default: *.md)
            
        Returns:
            Processing results
        """
        input_path = Path(input_dir)
        files = list(input_path.glob(file_pattern))
        
        logger.info(f"Found {len(files)} files to chunk")
        
        results = {
            'processed': 0,
            'total_chunks': 0,
            'failed': []
        }
        
        for i, file in enumerate(files, 1):
            try:
                logger.info(f"\n[{i}/{len(files)}] Processing {file.name}...")
                
                chunks = self.chunk_markdown_file(str(file), output_dir)
                
                results['processed'] += 1
                results['total_chunks'] += len(chunks)
                
            except Exception as e:
                logger.error(f"Failed to chunk {file.name}: {e}")
                results['failed'].append({
                    'file': str(file),
                    'error': str(e)
                })
        
        logger.info(f"\nChunking complete!")
        logger.info(f"   Files processed: {results['processed']}")
        logger.info(f"   Total chunks: {results['total_chunks']}")
        logger.info(f"   Failed: {len(results['failed'])}")
        
        return results


def main():
    """CLI interface for enhanced chunker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Math-aware chunking for educational content")
    parser.add_argument('input_dir', help="Directory containing markdown files")
    parser.add_argument('output_dir', help="Directory to save chunks")
    parser.add_argument('--chunk-size', type=int, default=512, help="Target chunk size (default: 512)")
    parser.add_argument('--overlap', type=float, default=0.2, help="Overlap ratio (default: 0.2)")
    parser.add_argument('--pattern', default="*.md", help="File pattern (default: *.md)")
    
    args = parser.parse_args()
    
    # Initialize chunker
    chunker = EnhancedChunker(
        chunk_size=args.chunk_size,
        overlap_ratio=args.overlap
    )
    
    # Batch process
    results = chunker.batch_chunk_directory(
        args.input_dir,
        args.output_dir,
        args.pattern
    )
    
    # Print summary
    print(f"\n✅ Processing complete!")
    print(f"Files processed: {results['processed']}")
    print(f"Total chunks: {results['total_chunks']}")
    print(f"Failed: {len(results['failed'])}")


if __name__ == "__main__":
    main()
