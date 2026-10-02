# 内容处理指南

## 快速开始

**用一条命令处理所有内容：**

```bash
python scripts/ingest_content.py
```

这将处理 `textbooks/` 与 `notes/` 目录下的教材与笔记。

---

## 文件夹结构

```
Mati/
├── textbooks/                    # 教材 PDF
│   ├── grade_10/
│   │   ├── computer_science.pdf
│   │   ├── english.pdf
│   │   └── science.pdf
│   └── README.md
│
├── notes/                        # 教师笔记
│   ├── grade_10/
│   │   ├── cs_notes.pdf
│   │   ├── english_summary.md
│   │   └── science_revision.txt
│   └── README.md
│
└── mati_data/
    └── chroma_db/               # 生成的 ChromaDB 集合
        ├── neb_computer_science_grade_10/
        ├── neb_english_grade_10/
        └── neb_science_grade_10/
```

---

## 教材 vs 笔记

| 方面 | 教材 | 笔记 |
|------|------|------|
| **位置** | `textbooks/grade_10/` | `notes/grade_10/` |
| **用途** | 主要参考资料 | 补充材料 |
| **集合名** | `neb_{subject}_grade_10` | `neb_{subject}_notes_grade_10` |
| **命名** | 灵活 | 灵活 |
| **格式** | PDF、TXT、MD | PDF、TXT、MD |

---

## 处理选项

### 处理所有内容（推荐）

```bash
python scripts/ingest_content.py
```

### 仅处理教材

```bash
python scripts/ingest_content.py --input textbooks
```

### 仅处理笔记

```bash
python scripts/ingest_content.py --input notes
```

### OCR 模式

**自动检测（推荐）：**
```bash
python scripts/ingest_content.py --ocr-mode auto
```

**对所有 PDF 强制 OCR：**
```bash
python scripts/ingest_content.py --ocr-mode force
```

**从不使用 OCR（仅文本）：**
```bash
python scripts/ingest_content.py --ocr-mode never
```

---

## 会被处理什么

### 支持的文件类型

- **PDF** — 文本型或扫描型（带 OCR）
- **TXT** — 纯文本文件
- **MD** — Markdown 文件
- **JSONL** — 结构化数据
- **JSON** — 结构化内容（学科/单元/问答）

### 自动检测

脚本会自动：
- 检测内容类型（文本 PDF、扫描 PDF、手写件）
- 从文件夹结构提取年级（`grade_10/`）
- 从文件名推断学科
- 应用相应的处理方式（PyMuPDF、Tesseract OCR 或 EasyOCR——简体中文 + 英文）

---

## 命名约定

**灵活命名** — 系统自动检测学科与年级：

**教材：**
- `computer_science.pdf` ✅
- `science_grade_10.pdf` ✅
- `english.pdf` ✅

**笔记：**
- `cs_notes.pdf` ✅
- `english_summary.md` ✅
- `science_revision.txt` ✅

> **注意：** 年级从文件夹名（`grade_10/`）提取，而非文件名。

---

## 验证

检查已创建的内容：

```python
import chromadb

client = chromadb.PersistentClient(path='mati_data/chroma_db')
collections = client.list_collections()

for c in collections:
    print(f"{c.name}: {c.count()} chunks")
```

**预期输出：**
```
neb_computer_science_grade_10: 1234 chunks
neb_computer_science_notes_grade_10: 567 chunks
neb_english_grade_10: 2345 chunks
neb_english_notes_grade_10: 890 chunks
neb_science_grade_10: 3456 chunks
neb_science_notes_grade_10: 1234 chunks
```

---

## 处理细节

### 处理流程

1. **内容检测** — 识别文件类型与内容
2. **提取** — 文本用 PyMuPDF，扫描/手写件用 OCR
3. **分块** — 创建 512-token 分块，10% 重叠
4. **嵌入** — 使用 `BAAI/bge-small-zh-v1.5`（512 维，中文优化）生成嵌入
5. **存储** — 连同元数据存入 ChromaDB

### 元数据

每个分块包含：
```python
{
    "source": "filename.pdf",
    "type": "neb_curriculum",
    "grade": "10",
    "subject": "computer_science"
}
```

---

## 故障排查

### "找不到文件"

**解决方案：**
- 确认文件位于 `textbooks/grade_10/` 或 `notes/grade_10/`
- 检查文件扩展名（.pdf、.txt、.md）
- 确保文件可读

### "OCR 失败"

**解决方案：**
- 安装 OCR 依赖：
  ```bash
  pip install pytesseract pillow easyocr
  ```
- 确认图像质量（建议 300 DPI）
- 尝试 `--ocr-mode force`

### "集合已存在"

**注意：** 这是正常现象——集合是可累加的。

**如需重建：**
```python
import chromadb

client = chromadb.PersistentClient(path='mati_data/chroma_db')
client.delete_collection('neb_computer_science_grade_10')

# 然后重新运行摄取
```

### "处理缓慢"

**优化建议：**
- 尽量使用文本型 PDF 而非扫描件
- 分批处理
- 关闭其他应用程序

---

## 最佳实践

1. **按年级组织** — 将特定年级的内容放在对应文件夹中
2. **文件名清晰** — 使用描述性名称
3. **格式一致** — 优先使用 PDF 以获得兼容性
4. **定期更新** — 内容变化时重新摄取
5. **验证摄取** — 处理后检查集合
6. **备份数据库** — 定期备份 `chroma_db/`

---

## 其他资源

- **详细流水线文档：** `scripts/rag_data_preparation/README.md`
- **快速入门指南：** `scripts/rag_data_preparation/QUICK_START.md`
- **笔记策略：** `scripts/rag_data_preparation/NOTES_GUIDE.md`
- **教材 README：** `textbooks/README.md`
- **笔记 README：** `notes/README.md`

---

## 完成清单

运行 `ingest_content.py` 之后，你应该拥有：

- [ ] 已创建 ChromaDB 集合
- [ ] 教材与笔记均可检索
- [ ] 元数据标记正确
- [ ] 已通过数量检查验证集合

**你的 RAG 系统现已就绪！**
