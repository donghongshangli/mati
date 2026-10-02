# Mati Teacher Guide

This guide explains how teachers manage educational content for Mati. **All teacher workflows are performed inside the graphical application (GUI)** — Mati has no terminal/command-line mode.

## 1. Introduction

Mati ships a **Teacher Workbench (教师工作台)** built into the GUI. It lets you:

- Import textbook files (PDF / TXT / MD) into the RAG knowledge base so students can ask questions about them.
- Generate chapters, concepts and short-answer questions from a textbook PDF using the local Qwen2.5 model.

## 2. Installation

Refer to the main [Installation Guide](https://github.com/aa-sikkkk/mati/blob/master/readme.md#installation) for setup steps. The Teacher Workbench is part of the standard installation — no extra tools are needed.

## 3. Opening the Teacher Workbench

1. Start the application (`MatiGUI.exe`, or `python -m student_app.gui_app.main_window` from source).
2. Enter your username on the welcome screen.
3. In the sidebar, click **🧑‍🏫 教师工作台**.

The workbench has two sections and a live status log at the bottom.

## 4. Importing Textbooks into the Knowledge Base

Use this when you want students to be able to ask questions about a book.

1. In the **📚 导入教材到知识库** section, click **选择教材文件** and pick one or more `.pdf` / `.txt` / `.md` files.
2. Click **导入知识库**.
3. The selected files are copied into `textbooks/` and ingested into ChromaDB.
4. Watch the status log for progress. When it reports success, students can immediately ask questions about the material.

> **Note:** First-time ingestion downloads the Chinese embedding model (`BAAI/bge-small-zh-v1.5`, ~95 MB) and requires `chromadb` plus `sentence-transformers`.

## 5. Generating Chapters and Test Questions from a PDF

Use this when you only have a textbook PDF and want structured lessons plus practice questions.

1. In the **🧩 从 PDF 生成章节与测试题** section, click **选择 PDF**.
2. Choose the **学科** (subject) and **年级** (grade) from the dropdowns.
3. Click **生成测试题**. Generation runs on the local model and typically takes a few minutes.
4. When it finishes, the status log reports the number of chapters, concepts and questions written, plus the output file path.
5. Go to **📚 浏览学科** to review the new chapters and solve the questions.

> **Prerequisite:** the Qwen model must be present at `mati_data/models/qwen2_5/*.gguf`.

### Output location and naming

Generated content is written to `scripts/data_collection/data/content/` as
`{subject}_grade_{grade}_auto.json` and is auto-loaded on the next content refresh.

> **Caution:** if a content file for the same subject already exists, the newly
> generated file may take precedence in the app. Back up existing content files
> before generating.

## 6. Editing Content Files Directly

Content is plain JSON under `scripts/data_collection/data/content/`. You can also
edit these files with any text editor; the structure is described in
[Content Structure and Management Guide](content-explanation.md) and must follow
[Project Standards](PROJECT_STANDARDS.md).

After editing files by hand, restart the app (or reopen **浏览学科**) so the
content manager reloads from disk.

## 7. Student Progress Data

Per-student progress is stored locally on each device under
`student_app/progress/` as `progress_<username>.json`.

Students manage their own progress from the **⚙️ 进度管理** page
(export / import / reset). Imported progress files are validated before being accepted.

To review progress centrally, collect the progress JSON files from student
devices and place them into the `student_app/progress/` directory of the machine
that needs to read them.

> **Note:** the previous command-line analytics and content-editor tools
> (`teacher_tools/`) were removed — Mati is GUI-only. Aggregate class reporting
> therefore has to be derived from the collected progress JSON files.

## 8. Content Validation and Quality Control

Run the standards validator to check code style, documentation, naming
conventions and content structure against the rules in
`docs/PROJECT_STANDARDS.md`:

```bash
python -m scripts.validation.validate_standards
```

This is a non-interactive batch check, not an interactive tool.

## 9. Community Contribution

Mati content is community-editable. Improve the JSON content, then submit your
changes through version control (Git).

## 10. Troubleshooting

If you encounter issues, refer to the main README troubleshooting section or the
[Technical Implementation Guide](TECHNICAL_IMPLEMENTATION.md#troubleshooting).

Common cases:

| Symptom | Likely cause | Fix |
|---|---|---|
| 导入知识库 fails | `chromadb` / `sentence-transformers` missing | Install them; first run downloads the embedding model |
| 生成测试题 fails | Qwen model missing | Put a `.gguf` file in `mati_data/models/qwen2_5/` |
| Text extraction empty | Scanned PDF without a text layer | Convert with OCR (e.g. EasyOCR) first |
