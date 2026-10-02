# Mati Project Overview: An Offline-First, RAG-Powered AI Learning Companion for Nepali students

## Executive Summary

Mati is a pioneering educational technology initiative aimed at bridging educational disparities for students in Nepal, particularly in areas with limited or no internet connectivity. This project delivers an **offline-first, RAG-powered learning companion** designed to cover the Computer Science, Science, and English curricula. By leveraging a **single lightweight Qwen2.5-1.5B-Instruct model** and **intelligent content discovery through RAG (Retrieval-Augmented Generation)**, Mati provides interactive learning experiences and content-authoring tools for both students and teachers, all operating efficiently on **low-end hardware**. Every interaction happens inside a **desktop graphical interface (GUI)** — there is no terminal/command-line mode.

## Project Vision

Our vision is to empower the next generation of Nepali students by providing accessible, high-quality educational resources regardless of their geographical location or access to online infrastructure. We aim to democratize learning through **intelligent, RAG-enhanced technology** and foster a collaborative environment for content development and continuous improvement, ultimately contributing to enhanced educational outcomes across Nepal.

## Key Objectives

The Mati project is guided by the following core objectives:

1.  **Maximize Accessibility:** Ensure the learning companion is fully functional offline and requires minimal system resources to run on commonly available computers. Provide comprehensive support for both English and Nepali languages to cater to diverse student needs and promote inclusivity.
2.  **Deliver High-Quality Education:** Offer comprehensive coverage of the Grade 10 curriculum with engaging, interactive content designed for effective learning. Implement **RAG-enhanced content discovery** and robust progress tracking mechanisms to personalize the learning experience and address individual student needs.
3.  **Empower Teachers:** Provide an intuitive graphical **Teacher Workbench** so teachers can import textbooks into the knowledge base and auto-generate chapters and practice questions without touching a terminal, and review student progress from the locally stored progress files.
4.  **Foster Community Collaboration:** Establish a transparent and streamlined platform and workflow for community members, particularly subject matter experts and educators, to contribute to, rigorously validate, and continuously update the educational content, ensuring its accuracy, relevance, and pedagogical soundness.

## Target Users

### Primary Users
-   **Grade 9-12 Students in Nepal:** The core beneficiaries, utilizing the **GUI application** for self-paced, interactive learning, **RAG-enhanced Q&A**, and practice exercises in an entirely offline setting.
-   **Subject Teachers:** Using the built-in **Teacher Workbench** to import textbooks and generate structured content, and reviewing student progress from the progress JSON files collected from student devices.
-   **School Administrators:** Potentially utilizing aggregated student performance data and system usage metrics for educational planning, resource allocation, and assessing the impact of the learning companion within their institutions.

### Secondary Users
-   **Parents:** Interested in monitoring their child's learning journey and academic progress as recorded by the local logging system.
-   **Educational Content Creators/Experts:** Contributing their expertise to expand and refine the educational content database by editing the structured JSON content files.
-   **System Administrators:** Responsible for the technical deployment, maintenance, updates, and troubleshooting of the Mati system on local school or community center infrastructure.

## Core Features

The Mati system is architected around several key features designed to provide a comprehensive and effective learning ecosystem:

### 1. Student-Facing Features (GUI Learning Application)

*   **Offline Content and Functionality:** Guarantees uninterrupted access to all educational materials and AI functionalities without any requirement for internet connectivity, crucial for low-connectivity regions.
*   **RAG-Enhanced AI-Powered Learning Assistance:**
    *   **Intelligent Content Discovery:** Uses RAG (Retrieval-Augmented Generation) to find the most relevant study materials for any question, even with vague or incomplete queries.
    *   **Contextual Question Answering (Q&A):** Provides accurate and relevant answers to student queries using the **Qwen2.5-1.5B-Instruct model** with retrieved RAG context for enhanced accuracy.
    *   **Streaming Answers:** Tokens stream into the answer panel as they are generated, with a live status line and a confidence indicator.
    *   **AI Answer Grading:** Free-text answers are graded by the model against the reference answer, with a short explanation returned in Chinese.
    *   **Follow-up Questions:** Recent turns are kept in context so students can ask follow-up questions naturally.
*   **Smart Text Handling:** Automatically normalizes uppercase, lowercase, and mixed case input for better AI understanding and response quality.
*   **Interactive and Structured Content Delivery:** Presents the curriculum (grade → subject → topic → subtopic → concept) in a clear, navigable, and engaging interface, making self-directed learning intuitive.
*   **Diagram Support:** Pre-authored ASCII diagrams from the grade-aware diagram library are rendered next to answers when they help explain the concept.
*   **Detailed Progress Tracking:** Systematically records student performance data (attempts and correct counts per question), allowing students and teachers to monitor learning progress.
*   **Graphical User Interface (GUI):** A CustomTkinter desktop application is the **only** interaction method — no terminal mode exists.

### 2. Teacher-Facing Features (Teacher Workbench, inside the GUI)

*   **Textbook Ingestion:** Copy PDF / TXT / MD textbooks into `textbooks/` and ingest them into ChromaDB so students can ask questions about them.
*   **Automated Content Generation:** Turn a textbook PDF into structured chapters, concepts and short-answer questions using the local Qwen2.5 model, with subject/grade selected from dropdowns.
*   **Automated Content Validation:** Content files are validated against the JSON schema when the content manager loads them, and `scripts/validation/validate_standards.py` performs a non-interactive batch check of project standards.
*   **Support for Collaborative Content Development:** Content lives as version-controlled JSON files, so contributions and review can happen through the usual Git workflow.

## Technical Architecture

The technical architecture of Mati is fundamentally designed for **offline operation, efficiency on low-resource hardware, and intelligent content discovery**. Key architectural components include:

### **RAG (Retrieval-Augmented Generation) System**
- **PDF Content Processing**: structure-aware chunking into parent (~1000 chars) and child (~384 chars) blocks; PDF glyph repair; rule-based cleaning; exact (sha256) + semantic (cosine ≥ 0.95) deduplication — every drop is written to an audit log
- **Hybrid Recall**: dense vectors + BM25 (jieba tokenization) merged by RRF (k=60), then 7-feature interpretable reranking
- **Page Traceability**: every evidence chunk carries `book_title · chapter · page`; citation numbers are generated by the retrieval layer, so the model cannot fabricate sources
- **Vector Embeddings**: `BAAI/bge-small-zh-v1.5`, 512-dim, cosine space
- **ChromaDB**: local vector database — 11 collections / ~13k vectors (grades 10–11; subjects: math, chinese, science, computer science)
- **Query-layer Governance Filters**: parent chunks, table-of-contents/cover blocks and archived chunks are excluded at recall time

### **Single AI Model Architecture**
- **Qwen2.5-1.5B-Instruct**: One lightweight model handles all AI tasks (Q&A, answer grading, content generation)
- **GGUF Format (Q4_K_M)**: Optimized for low-end hardware, CPU-only inference; context window **4096 tokens** (`n_ctx` is defined in `qwen_handler.py` *and* `retrieval.yaml → context_budget`, and the two must match)
- **Text Normalization**: Handles various input formats intelligently
- **Streaming Generation**: ChatML-formatted streaming output for low perceived latency; math is rendered as plain-text Unicode (the GUI answer box cannot render LaTeX)
- **Refusal Policy**: when evidence coverage is below the calibrated threshold, or the request is a creative-writing task, the system refuses explicitly instead of letting the model improvise

### **Content Management System**
- **Structured JSON Content**: Hierarchical organization of educational materials
- **RAG + Structured Fallback**: Intelligent content discovery with traditional content backup
- **Schema Validation**: JSON-schema checks on load, plus a batch standards validator

(For an in-depth examination of the system architecture, including the interconnections between components and data flow, please refer to `docs/TECHNICAL_IMPLEMENTATION.md`.)

## Implementation Strategy

The development and implementation of Mati adhere to a structured, phased approach to ensure all critical components are developed and integrated effectively. The strategy involves:

1. **Core Infrastructure**: RAG system setup, ChromaDB integration, and Qwen2.5-1.5B model optimization
2. **AI Integration**: Single model architecture with streaming output and text normalization
3. **User Applications**: A single GUI application (student views + teacher workbench) with RAG capabilities
4. **Content Pipeline**: PDF processing and embedding generation for intelligent search
5. **Testing & Optimization**: Performance validation for low-end hardware
6. **Community Framework**: GUI teacher tools and JSON content management

## Quality Assurance

A stringent quality assurance process is integrated throughout the development lifecycle. This includes:

- **Extensive Testing**: Unit and integration testing for functional correctness (`tests/`). The RAG-related suites run **199 tests, all passing**; the full suite currently reports 368 passed / 9 failed, where the 9 failures are long-standing legacy issues documented in the test section of `docs/TECHNICAL_IMPLEMENTATION.md`
- **Retrieval Evaluation**: a 78-question evaluation set with staged metrics and ablation (`scripts/eval_rag.py`) — Recall@10 100%, routing accuracy 100%
- **Citation Verification**: end-to-end citation-chain check with the real model (`scripts/verify_citations.py`) — 0 out-of-range citations
- **Index Health Check**: `scripts/verify_index.py` must pass before any evaluation (HNSW self-healing)
- **Performance Profiling**: Memory and speed optimization for low-resource systems (`system/performance/`)
- **Content Validation**: Automated schema validation and quality checks
- **Expert Review**: Educational content accuracy and pedagogical effectiveness

## Deployment and Maintenance

Mati is architected for straightforward local deployment on target machines without requiring complex server setups. The installation procedure involves:

1. **Obtaining Project Files**: Clone the repository or download the student package
2. **Installing Dependencies**: Python dependencies managed via `requirements.txt`
3. **Downloading the Qwen2.5 Model**: Lightweight GGUF model (~1 GB) for AI capabilities
4. **Running the Application**:
    - **Option A (Recommended)**: Use the pre-built `MatiGUI.exe` for immediate graphical access.
    - **Option B (Dev Mode)**: `python -m student_app.gui_app.main_window` for source execution.
5. **Building a Distribution**: `scripts/release/build_nuitka.ps1` (uses `scripts/release/bootstrap.py` as the Nuitka entry point)

Maintenance activities include applying updates to the application and content, potential model optimizations, and addressing any reported issues, facilitated by the version-controlled content repository and comprehensive documentation.

## Future Roadmap

Potential future developments for Mati include:

- **Enhanced RAG Capabilities**: Multi-modal content understanding (text + images)
- **Advanced Analytics**: Learning pattern recognition and adaptive recommendations
- **Multi-Language Support**: Enhanced Nepali language processing
- **Mobile Application**: Offline-capable mobile version
- **Community Features**: Student collaboration and peer learning tools
- **Content Expansion**: Additional subjects and grade levels

## Support and Resources

Comprehensive documentation, encompassing user guides, detailed technical specifications (`docs/TECHNICAL_IMPLEMENTATION.md`), content structure and management guidelines (`docs/content-explanation.md`), and project standards (`docs/PROJECT_STANDARDS.md`), is available within the `docs/` directory. Support channels will be established to assist users with installation, usage, and troubleshooting, and to facilitate the reporting of bugs and submission of feature requests.

## Acknowledgments

We gratefully acknowledge **Alibaba Qwen** for the **Qwen2.5-1.5B-Instruct** model that powers our AI capabilities, **BAAI** for the `bge-small-zh-v1.5` Chinese embedding model, **ChromaDB** for enabling intelligent content discovery, and the **llama-cpp-python** community for efficient model inference. Special thanks to [readersnepal](https://readersnepal.com/) and [CDC](http://lib.moecdc.gov.np/) for the notes and resources necessary for the Dataset gathering.

---

*Pioneering accessible, intelligent AI education in Nepal with community power and RAG technology.*
