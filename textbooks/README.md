# Textbooks Directory

This directory contains **textbook PDFs** that will be processed into book chunks for the RAG system.

## 📁 Directory Structure

```
textbooks/
├── grade_10/              # Grade 10 textbooks
│   ├── computer_science_grade_10.pdf
│   ├── english_grade_10.pdf
│   └── science_grade_10.pdf
└── README.md             # This file
```

## 📝 Naming Convention

For automatic processing, name your PDF files using this format:

```
{subject}_grade_{grade}.pdf
```

### Examples:
- `computer_science_grade_10.pdf`
- `english_grade_10.pdf`
- `science_grade_10.pdf`

### Alternative Formats (also supported):
- `computer_science_grade10.pdf`
- `computer_science_10.pdf`
- `computer_science.pdf`

## 🚀 How to Process Textbooks

### Option 1: In the GUI (Recommended)

The easiest path — no terminal needed:

1. Start the app: `python -m student_app.gui_app.main_window`
2. Open **🧑‍🏫 教师工作台**
3. Click **选择教材文件**, pick the PDFs, then click **导入知识库**

### Option 2: Batch ingestion script

The GUI calls this same non-interactive script:

```bash
# Textbooks and notes together (default)
python scripts/ingest_content.py

# Textbooks only
python scripts/ingest_content.py --input textbooks

# Force OCR for scanned PDFs
python scripts/ingest_content.py --input textbooks --ocr-mode force
```

### Option 3: Embedding pipeline module (advanced)

`scripts/rag_data_preparation/embedding_generator.py` exposes the embedding layer
used by ingestion:

```python
from scripts.rag_data_preparation.embedding_generator import EmbeddingGenerator

gen = EmbeddingGenerator()          # defaults to BAAI/bge-small-zh-v1.5 (512-dim)
vectors = gen.generate_embeddings(["什么是计算机网络？"])
```

> Instantiating `EmbeddingGenerator` downloads the embedding model (~95 MB) on
> first use. For normal ingestion you do not need to call this directly —
> `scripts/ingest_content.py` handles it.

## 📚 What Happens During Processing?

1. **Content detection** — identifies digital text, scanned pages, or handwriting
2. **Extraction** — `PyMuPDF` for clean text; Tesseract / EasyOCR for scans and handwriting
3. **Chunking** — 512-token chunks with 10% overlap
4. **Embedding** — `BAAI/bge-small-zh-v1.5` (512-dim, Chinese-optimized)
5. **Storage** — written to ChromaDB with metadata (source, type, grade, subject)

## 💡 Tips

- **Quality Matters**: Use high-quality PDFs for better text extraction
- **OCR Support**: Scanned PDFs need OCR (see `--ocr-mode`)
- **File Size**: Large PDFs may take longer to process
- **Backup**: Keep original PDFs safe

## 🔍 Where Are Processed Chunks Stored?

Directly in the local vector database:

```
mati_data/chroma_db/       # ChromaDB collections (chroma.sqlite3 + index segments)
```

There is no intermediate `processed_data_new/` directory — the ingestion pipeline
goes straight from PDF to ChromaDB.

## 📊 ChromaDB Collections

After processing, chunks are stored in collections named:

- `neb_computer_science_grade_10`
- `neb_english_grade_10`
- `neb_science_grade_10`

Notes use the `neb_{subject}_notes_grade_{grade}` pattern.

These collections contain comprehensive textbook content for RAG retrieval.

> **Rebuilding:** if you change the embedding model, the old vector space becomes
> incompatible. Run `python scripts/rebuild_index.py` to back up and rebuild.
