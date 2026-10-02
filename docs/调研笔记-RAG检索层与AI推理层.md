# Mati RAG 检索层 & AI 推理层 代码调研笔记

> 调研对象：`D:\mati-master`（Windows）。所有事实均标注 `文件:行号`。
> 无法从代码/数据确认的内容一律标注「未找到」。
> 本机 Python 环境未安装 `chromadb` / `sentence-transformers` / `llama_cpp` / `easyocr`（实测 import 全部 ModuleNotFoundError），因此**端到端运行延迟未实测**；但 ChromaDB 持久化文件（`mati_data/chroma_db/chroma.sqlite3`）与静态代码可完整核对。

> **2026-09 更新（GUI-only 改造后）**：终端界面（TUI）已整体移除——`student_app/main.py`、`student_app/interface/`（`cli_interface.py`、`cli_renderer.py`）、`teacher_tools/` 均已删除，`prompt_toolkit`/`wcwidth` 已从 `requirements.txt` 移除。
> 影响：① 下文中所有 `cli_interface.py` / `cli_renderer.py` / `main.py` 的行号引用**已随文件删除而失效**，已就地删除或标注；② `student_app/gui_app/main_window.py` 因新增安全/性能接线而**整体下移了若干行**，本文中的 `main_window.py:NNN` 已按改造后状态重新标注；③ 检索/模型层的结论不受影响。

---

## 1. 端到端数据流（提问 → 答案）

### 1.1 调用入口

| 入口 | 代码位置 | 关键点 |
|---|---|---|
| GUI 提问 | `student_app/gui_app/main_window.py:811-816` | `rag_engine.query(query_text=..., subject=self.current_subject_filter, stream_callback=on_token, status_callback=on_status, conversation=history)` |
| GUI 线程模型 | `main_window.py:764`, `main_window.py:864` | 提问在 `worker()`（`:771`）内执行，`threading.Thread(target=worker, daemon=True).start()` |
| GUI 多轮历史 | `main_window.py:808`, `main_window.py:839-841` | 最近 6 轮，仅当 `llm_used` 且 `answer` 非空才入历史 |
| GUI 学科过滤值 | `main_window.py:44-55`, `main_window.py:112` | 下拉框中文→英文键：Science / Math / English Grammar / Computer Science / Social Studies；默认 `"Science"` |
| 引擎入口签名 | `system/rag/rag_retrieval_engine.py:168-176` | `query(query_text, subject, n_results=3, stream_callback=None, status_callback=None, conversation=None)`，**没有 grade 参数** |

### 1.2 逐步流程（函数名 + 行号）

1. **边缘用例短路**：`rag_retrieval_engine.py:196-205` → `UserEdgeCaseHandler.check_edge_cases`（`system/rag/user_edge_case_handler.py:41-72`）：空串→提示、长度 <4→提示、寒暄/感谢/道歉精确匹配→固定话术；命中即返回 `type="edge_case"`，不检索不生成。
2. **连接/模型可用性检查**：`rag_retrieval_engine.py:207-211`（ChromaDB 为 None → 返回「错误：数据库未连接。」）；模型在 `__init__` 阶段注入或自建（`:58-67`）。
3. **输入归一化**：`rag_retrieval_engine.py:213-220` → `AdaptiveNormalizer.normalize`（`system/input_processing/adaptive_normalizer.py:57-98`，含拼写检查缓存与日志）→ `InputNormalizer.normalize`（`system/input_processing/input_normalizer.py:69-107`）：转正式（`:78`）、去噪（`:81`）、缩写展开（`:84`）、句面规范化（`:87`）、上下文扩展（`:90`）、意图分类（`:93`）、归一化置信度（`:94`）。产出 `clean_question/intent/confidence/notes`；`effective_query = clean_question or query_text`（`rag_retrieval_engine.py:220`）。
4. **精确缓存命中**：`rag_retrieval_engine.py:222-233` → `RAGCache.get`（`system/rag/rag_cache.py:51-75`）。key = `md5(f"{query.lower().strip()}|{subject.lower()}|{grade}")`（`rag_cache.py:46-49`）。命中后把答案按空格切分逐词回放（`rag_retrieval_engine.py:229-232`）。**有 conversation 时跳过缓存**（`:222-223`）。
5. **状态消息（与答案分离）**：`rag_retrieval_engine.py:188-193` 的 `_emit_status`，优先 `status_callback`，回退 `stream_callback`；调用点 `:235-236`、`:256`、`:306`。
6. **查询向量化**：`rag_retrieval_engine.py:239` → `EmbeddingGenerator.generate_query_embeddings`（`scripts/rag_data_preparation/embedding_generator.py:135-148`），会拼接 BGE 官方中文检索指令 `"为这个句子生成表示以用于检索相关文章："`（`embedding_generator.py:67`，`:147-148`），再走 `generate_embeddings`（`:150-186`，`normalize_embeddings=True` 于 `:171`）。
7. **语义缓存命中**：`rag_retrieval_engine.py:241-250` → `RAGCache.find_similar`（`rag_cache.py:77-141`），余弦相似度，调用点阈值 `0.88`（`rag_retrieval_engine.py:242`）。
8. **集合选择**：`rag_retrieval_engine.py:252` → `_get_relevant_collections(subject, "")`（`:98-166`）。**注意 grade 恒传空串**。
9. **向量检索**：`rag_retrieval_engine.py:260-284`。每个集合 `coll.query(query_embeddings=[...], n_results=min(2, n_results))`（`:263-266`），得分 `score = 1.0 - distance`（`:273`），并发 `ThreadPoolExecutor(max_workers=2)`（`:281-284`）。
10. **过滤/重排**：`rag_retrieval_engine.py:286` → `AntiConfusionEngine.rank_results`（`system/rag/anti_confusion_engine.py:70-126`）：`weighted = base*0.7 + (priority/100)*0.3`（`:109`），grade 出现在问题文本中 +0.1（`:111-114`），按 `final_score` 降序（`:124`）。
11. **上下文组装**：`rag_retrieval_engine.py:287-301`：丢弃 `final_score < 0.35`（`:292`），累计长度 < 1600 字符且首块必留（`:296`）；`:302` → `resolve_conflicts`（`anti_confusion_engine.py:181-188`，NEB 块前置）；`:304` 以 `"\n\n"` 拼接。
12. **生成**：`rag_retrieval_engine.py:312-330`。流式走 `self.llm.handler.get_answer_stream`（`ai_model/model_utils/qwen_handler.py:179-211`）；非流式走 `self.llm.simple_handler.get_answer`（`ai_model/model_utils/model_handler.py:34-40` → `qwen_handler.py:213-242`）。模型未加载分支 `:331-338`。
13. **后处理**：非流式路径执行 `QwenHandler._clean_answer`（`qwen_handler.py:129-153`：去 `Q:/A:` 前缀、截断练习题标记、压缩空白、补句号）；**流式路径完全不调用 `_clean_answer`**（`:179-211` 无该调用）。ASCII 图：`rag_retrieval_engine.py:340` → `ASCIIDiagramLibrary.find_diagram_by_text`（`system/rag/ascii_diagram_library.py:119-128`）。GUI 侧另用 LLM/规则生成图：`ask_question_view.py:235-247` → `system/diagrams/diagram_service.py:75-117`。
14. **置信度**：流式路径 `rag_retrieval_engine.py:323` → `_calculate_confidence`（`:358-388`）；非流式路径用 Qwen 侧置信度（`qwen_handler.py:236` → `:165-177`）。
15. **写缓存**：`rag_retrieval_engine.py:353-354`（仅无 conversation 时）。
16. **展示**：GUI `main_window.py:819-845`（finalize）+ `student_app/gui_app/views/ask_question_view.py:348-367`（<0.7 显示「中等/低置信度」，<0.3 或 None 不显示）。原 CLI 的置信度着色已随文件删除。

---

## 2. 检索关键参数

### 2.1 top_k / 阈值

| 参数 | 值 | 位置 |
|---|---|---|
| `n_results` 默认 | 3 | `rag_retrieval_engine.py:172` |
| 每集合实际取回数 | `min(2, n_results)` → 实际 2 | `rag_retrieval_engine.py:265` |
| 集合数上限 | 3（`collections[:3]`） | `rag_retrieval_engine.py:166`；HF 分支内部提前 break 于 `:152-153` |
| 上下文块得分阈值 | `final_score >= 0.35` | `rag_retrieval_engine.py:292` |
| 上下文长度上限 | 1600 字符（首块豁免） | `rag_retrieval_engine.py:289`, `:296` |
| 语义缓存阈值 | 调用点 0.88；`RAGCache.find_similar` 默认 0.92 | `rag_retrieval_engine.py:242`；`rag_cache.py:82` |
| `RELEVANCE_THRESHOLD=0.6` | 定义但**无任何读取点** | `anti_confusion_engine.py:51`（全仓 grep 仅定义处） |
| `MIN_CONTEXT_OVERLAP=0.3` | 仅被未被调用的 `validate_grounding` 使用 | `anti_confusion_engine.py:50`, `:162` |
| 来源优先级 | neb=100, openstax=70, khanacademy=65, finemath/gsm8k/scienceqa=60, default=50 | `anti_confusion_engine.py:39-47` |

### 2.2 collection 命名规则

- 写入侧：`ingest_content.py:371-377`
  - 有年级：`neb_{subject}_grade_{grade}`；无年级：`neb_{subject}`；
  - 非 `[a-zA-Z0-9._-]` 字符替换为 `_`（`:374`），首尾 `._-` 去除（`:375`），长度 <3 → `neb_general`（`:376-377`）。
- 中文学科→英文学科映射：`ingest_content.py:329-335`（物理/化学/生物/科学→science；地理/历史/政治→social_studies；数学→math；英语/英文/语法→english；计算机/信息技术→computer_science），命中文件名 stem 则采用（`:361-364`）；否则从文件名清洗推断（`:365-368`）。
- 选择侧：`rag_retrieval_engine.py:98-166`
  - NEB：`neb_key = neb_{subject}_grade_{grade}` 或 `low.startswith(f"neb_{subject}")`（`:116-123`），**该循环无数量上限**；
  - HuggingFace：硬编码 `subject_map`（`:127-143`），仅按学科、忽略年级（`:145-153`）；
  - 兜底：`khanacademy_pedagogy`、`fineweb_edu`（`:156-163`），最多 2 个；
  - 最终 `collections[:3]`（`:166`）。
- 实际 DB 只有两个集合（`mati_data/chroma_db/chroma.sqlite3` → `collections` 表）：`neb_science`（512 维，226 块）与 `neb_readme`（512 维，5 块，来源 `README.md`）。

### 2.3 metadata 过滤字段

- 写入字段（仅 4 个）：`source`（文件名）、`type`（固定 `neb_curriculum`）、`grade`、`subject` — `ingest_content.py:430-435`；manifest 路径直接沿用包内 metadata（`:519-521`）。
- 查询侧**没有任何 `where=` 元数据过滤**：`coll.query` 只传 `query_embeddings` 与 `n_results`（`rag_retrieval_engine.py:263-266`；全仓 grep `where=` 无匹配）。metadata 仅用于：
  - 结果字典中的 `sources`（`:345`）——**当前无任何 UI 展示点**：`AskQuestionView.finalize_answer`（`ask_question_view.py:171`）不接收 `sources`，原 CLI 的来源列表已随文件删除；
  - `anti_confusion_engine.py:111-114` 的 grade 字符串包含判断（+0.1 权重）；
  - `anti_confusion_engine.py:82-93` 的 `source`/`type` 兜底赋值。
- 实测 DB 内 metadata 分布：`grade` 全为 `"unknown"`；`type` 全为 `"neb_curriculum"`；`subject` ∈ {`science`(226), `readme`(5)}。

### 2.4 缓存键与过期策略

- 键：`md5(f"{query.lower().strip()}|{subject.lower()}|{grade}")`（`rag_cache.py:46-49`），grade 由引擎传 `""`（`rag_retrieval_engine.py:224`, `:354`）。
- 容量/过期：`max_size=100`、`ttl_seconds=3600`（`rag_retrieval_engine.py:45`；`rag_cache.py:33`）；TTL 为**惰性过期**，在 `get`（`:71-73`）与 `find_similar` 遍历（`:111-113`）以及 `stats`（`:189-194`）时清理。
- 淘汰：`set` 中当 `len(cache) >= max_size` 时删除**时间戳最小者**（`rag_cache.py:164-167`）——是 FIFO，不更新访问时间，非严格 LRU（类注释 `:26-31` 称 LRU）。
- 语义检索：逐条余弦（`rag_cache.py:132`），要求 `meta['subject'] == subject and meta['grade'] == grade`（`:116`）、向量形状一致（`:125`）；`np.linalg.norm == 0` 直接跳过（`:107-108`, `:129-130`）。
- 无持久化、无跨进程共享（纯内存 dict，`rag_cache.py:44`）；索引重建后旧缓存仍存活至进程退出。
- 多轮对话时完全绕过缓存（`rag_retrieval_engine.py:222-223`、`:241`、`:353`）——注意 `conversation=[]` 为空列表时仍走缓存。

### 2.5 降级 / fallback 分支

| 场景 | 行为 | 位置 |
|---|---|---|
| ChromaDB 连接失败 | `chroma_client=None`，query 返回「错误：数据库未连接。」 | `rag_retrieval_engine.py:50-55`, `:207-211` |
| 未选到集合 | 仅 warning，继续（上下文为空） | `:253-254` |
| 单集合查询异常 | 吞异常返回 `[]` | `:277-279` |
| LLM 生成异常 | 回调「生成答案时出错，请重试。」并置 `llm_used=False` | `:326-330` |
| LLM 未加载 | 返回「⚠️ 大模型未加载…」+ 首块前 300 字 | `:331-338` |
| warm_up 失败 | 仅 `logger.warning` | `:95-96` |
| 归一化失败 | `InputNormalizer._fallback_result` | `input_normalizer.py:105-107` |
| 2 秒超时降级 | `rag_helper.get_context_with_timeout(timeout_seconds=2.0)` 返回 `("", "AI Knowledge (timeout)")` | `system/rag/rag_helper.py:67-138`——**全仓无任何导入者（grep 无匹配），实际未生效** |

---

## 3. 模型层

### 3.1 GGUF 加载参数（实际值）

- 模型解析：`ModelHandler.__init__` 用 `Path(model_dir).glob("*.gguf")` 取**第一个**文件（`ai_model/model_utils/model_handler.py:52-57`）；目录默认 `mati_data/models/qwen2_5`（`:47-48`）；无 gguf 抛 `FileNotFoundError`（`:54-55`）；加载或 warm-up 失败 re-raise（`:63-70`）。
- 实际文件：`mati_data/models/qwen2_5/qwen2.5-1.5b-instruct-q4_k_m.gguf`，1,117,320,736 字节（≈1.04 GB）。
- `Llama(...)` 参数（`qwen_handler.py:54-63`）：`n_ctx=2048`、`n_batch=96`、`n_threads=4`、`use_mmap=True`、`use_mlock=False`、`f16_kv=False`、`verbose=False`。
- 预热：`warm_up` 用 `"Instruct: What is 2+2?\nOutput: Answer:"` 做 `max_tokens=10, temperature=0.1` 的假推理（`qwen_handler.py:66-85`）。
- 生成参数：
  - `get_answer_stream`（`qwen_handler.py:191-202`）与 `get_answer`（`:223-232`）：`max_tokens=768`、`temperature=0.6`、`top_p=0.9`、`repeat_penalty=1.1`、`stop=STOP_SEQUENCES`；
  - `STOP_SEQUENCES = ["<|im_end|>", "<|endoftext|>", "</s>"]`（`:28`）；
  - `generate_response`（判题/工具链用）：`max_tokens` 默认 512、`temperature=0.3`、额外 stop `["Student Answer:", "Question:"]`（`:244-258`）。
  - 官方推荐值被显式下调的注释见 `:194-196`（Qwen2.5 官方 0.7/0.8/1.05 → 实际 0.6/0.9/1.1）。
- `answer_length` 参数在 `model_handler.py:74`/`:81` 声明但从未使用；`main_window.py:764`（`on_ask_submit` 形参）接收后也未转发（`:811-816` 的 `query()` 调用无该参数）。
- `get_model_info`（`model_handler.py:96-105`）返回 `context_size: "2048 tokens"`、`prompt_format: "ChatML"`、`optimized_for: "i3_cpu"`。

### 3.2 提示词模板结构

- 手工拼 ChatML（`qwen_handler.py:121-127`）：`<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n`。
- system（`:37-47`）：角色「Mati，中文教师，中学教育场景」；5 条约束——简体中文；优先依据「参考材料」并用其术语、材料不足时先说「材料中未找到相关信息」再简答；先结论后分点、可举例；长度 150–250 字、追问需结合「对话历史」与「参考材料」；不输出练习题、不复述问题。
- user（`_build_prompt`，`:87-119`）四段拼接：
  1. 固定指令「请用中文回答用户的问题…」（`:103`）；
  2. `【对话历史】`：最近 **3** 轮，格式 `学生：…\n老师：…`（`:91-100`）；
  3. `参考材料：\n{context}`，且 context 超过 **850 字符**时按 `。.！？` 句末截断（`:106-117`）；
  4. `问题：{question}`（`:118`）。
- **是否含引用来源**：不含。prompt 中无 chunk 编号、文件名或出处标注；`sources` 仅存在于结果字典（`rag_retrieval_engine.py:345`），当前没有任何 UI 展示点（原 CLI 的来源列表已随文件删除，GUI 未实现对应展示）。

### 3.3 流式输出实现

- `get_answer_stream` 为生成器，逐 chunk 取 `chunk["choices"][0]["text"]` 后 `yield`（`qwen_handler.py:191-207`）。
- 引擎侧累加并回调：`answer += token; stream_callback(token)`（`rag_retrieval_engine.py:318-321`）。
- GUI：`on_token` → `self.after(0, display)` 投递到 Tk 主线程（`main_window.py:793-799`），首 token 时关闭 loading（`:794-798`），`append_answer_token` 追加（`ask_question_view.py:140-152`）。
- 缓存命中时的「伪流式」：按空格切词逐个回调（`rag_retrieval_engine.py:229-232`、`:246-249`），中文答案无空格 → 实际一次性输出。

### 3.4 超时与异常处理

- **无任何生成超时**（未找到 `timeout` 出现在 `qwen_handler.py` / `rag_retrieval_engine.py` 的生成调用上）。
- 异常兜底三层重复：
  1. `qwen_handler.py:209-211`（流式）与 `:240-242`（非流式）→ 「生成答案时出错，请重试。」；
  2. `model_handler.py:81-86` / `:74-79` → 「我在理解您的问题时遇到了困难，请重试。」；
  3. `rag_retrieval_engine.py:326-330` → 「生成答案时出错，请重试。」。
- 空/极短问题保护：`len(question.strip()) < 3` → 「请输入有效的问题。」（`qwen_handler.py:183-185`, `:217-218`）。
- `cleanup()` 释放模型（`qwen_handler.py:265-269`；`model_handler.py:107-112`）。

---

## 4. 摄取链路

### 4.1 内容类型自动检测

`process_file`（`scripts/ingest_content.py:276-326`）：
- `.txt/.md/.jsonl` → 直接 UTF-8 读取（`:284-290`）；
- `.json` → `_extract_json_text` 递归扁平化（`:294-299`；实现 `:247-274`，抽取 subject/grade/name/summary/question/answer/acceptable_answers/hint/description/example/formula/explanation 等键）；
- `.pdf` → 依 `ocr_mode` 决定（`:302-323`）：`force`→强制 scanned、`never`→text、`auto`→`detect_pdf_type`；
- 其它后缀 → warning 并返回 None（`:325-326`）。

PDF 判定：`detect_pdf_type`（`:157-177`）取**前 3 页**文本总字符数，`< 100` 判为 `"scanned"`，否则 `"text"`；异常兜底 `"text"`（`:175-177`）。docstring 声称可返回 `"handwritten"`（`:158-161`）但代码**从不返回该值**。

### 4.2 OCR 分支条件（重要缺陷）

- 代码意图（`:315-323`）：scanned → 若 `self.easyocr_reader` 存在走 EasyOCR，否则若 `TESSERACT_AVAILABLE` 走 Tesseract，两者皆无则报错。
- Tesseract：`pytesseract.image_to_string(img, lang='eng')`，页面 `dpi=300`（`:197-204`）。
- EasyOCR：`easyocr.Reader(['ch_sim', 'en'], gpu=False)`（`:155`），逐页转 PNG 临时文件后 `readtext`（`:221-240`）。
- **实测缺陷**：`self.easyocr_reader` 从未被赋值——初始化代码位于 `_ensure_client` 的 `return self.client`（`:148`）之后（`:150-155`），为不可达代码，且其中引用的是未定义的 `ocr_mode`（应为 `self.ocr_mode`）。字节码核验：`_ensure_client` 的 `co_names` 含 `easyocr_reader`，但反汇编中无对应 STORE（不可达）。实测对一个空白 PDF 调用 `process_file`：
  - `detect_pdf_type` 返回 `"scanned"`；
  - `process_file` 抛 `AttributeError: 'UniversalContentIngester' object has no attribute 'easyocr_reader'`；
  - 因此 Tesseract 分支（`:319`）实际不可达。
- `requirements.txt` 未包含 `easyocr`（仅 `pytesseract==0.3.13`、`pillow==12.1.0`、`PyMuPDF==1.26.7`），`HOW_TO_PROCESS.md:195` 却要求用户 `pip install easyocr`。

### 4.3 分块大小与重叠

- 摄取侧实例化：`EnhancedChunker(chunk_size=512, overlap_ratio=0.1)`（`ingest_content.py:122`）。
- 语义为**字符**而非 token：`chunk_size: Target chunk size in characters`（`enhanced_chunker.py:55`）；`overlap_size = int(512*0.1) = 51`（`:61`）；`min_chunk_size=100`（`:49`）。
- 保护区域：LaTeX（`$$..$$`、`$..$`、`\[..\]`、`\(..\)`、上/下标表达式）与代码块（```…```、`…`）内不允许切分（`:68-148`），切点落在保护区内时延后到区域结束（`:255-260`）→ 可能出现 >512 的块。
- 边界优先级：句末（`.!?` + 空白）> 段落（`\n\n+`）> 单词空格（`:170-190`, `:262-276`）。
- 防死循环：`loop_counter > len(text)+1000` 强制 break（`:231-234`），并有多重「必须前进」保护（`:291-311`）。
- 实测入库块长：`neb_science` 226 块，min 101 / max 512 / 平均 387.8 字符，无 >512 块；`neb_readme` 5 块，min 164 / max **1046** / 平均 572.2（3 块 >512，符合保护区域延后切分的预期）。
- 文档与代码不一致：`readme.md:183`、`readme.md:257`、`HOW_TO_PROCESS.md:163` 均写「512 token、10% 重叠」，代码是 512 **字符**。

### 4.4 嵌入维度与模型名

- 模型：`BAAI/bge-small-zh-v1.5`（`embedding_generator.py:99`、`ingest_content.py:136`），CPU（`ingest_content.py:136`；引擎侧 `RAGRetrievalEngine` 传 `device='cpu'`，`rag_retrieval_engine.py:47`）。
- 维度：运行时从模型读取（`embedding_generator.py:132`）；DB `collections.dimension = 512`（两集合均为 512）。
- 离线优先：import 前设置 `HF_ENDPOINT=https://hf-mirror.com`，缓存存在则 `HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1`（`ingest_content.py:38-48`, `:63-73`；`embedding_generator.py:39-48`, `:70-88`）。
- 批量与线程：`batch_size=16`（`embedding_generator.py:101`），`torch.set_num_threads(2)`（`:107`）。
- 检索/摄取不一致：摄取路径直接 `SentenceTransformer.encode(texts, convert_to_numpy=True)`（`ingest_content.py:427`, `:518`），未传 `normalize_embeddings`；检索路径经 `EmbeddingGenerator.generate_embeddings`（`normalize_embeddings=True`，`embedding_generator.py:171`）并加 BGE 查询指令（`:147`）。模型自带 `2_Normalize` 模块（HF 缓存 `modules.json`），故两者实际都近似单位向量。

### 4.5 幂等 / 去重

- 集合：`get_or_create_collection(name=collection_name)`（`ingest_content.py:424`, `:515`）。
- 写入：`collection.upsert(...)`，批大小 100（`:438-446`；manifest 路径 `:522-529`）。
- chunk id：`f"{collection_name}_{i}"`（`:429`），**不含文件名/内容哈希** → 同一集合内不同文件的同序号块会互相覆盖；文件内容变化导致块数减少时旧 id 残留。
- manifest 路径使用包内 `chunk["id"]`（`:516`），并对每个 chunk 文件校验 `sha256(raw)`（`:497-500`），zlib 解压失败时容错为原始字节（`:489-495`）。
- 无全局去重、无「内容未变则跳过」的判断（未找到）。

### 4.6 manifest 机制

- `ingest_manifest(pkg_dir, dry_run)`（`ingest_content.py:453-534`）：
  - 读 `packages/<package_id>/manifest.json`（`:467-475`）；
  - 逐 chunk 校验存在性与内容寻址哈希（`:483-501`），不一致则跳过并 `logger.error`；
  - `--dry-run` 只校验不写库（`:503-506`）；
  - 按 `collection` 分组后批量 upsert（`:508-531`）。
- CLI：`--ingest-from-manifest DIR` 遍历 `DIR/packages/*/manifest.json`（`:564-581`）。
- 包生成侧：`tools/package_builder.py`——sha256 内容寻址（`tools/common.py:44-51`）、zlib level 6（`common.py:64-69`）、单包上限 50 MB（`package_builder.py:148`, `:172-178`）、manifest 字段 format_version/package_id/total_chunks/total_size_bytes/collections/chunks（`:200-212`）；下载侧 `tools/sync_client.py:117-233` 支持增量与 Range 续传。

### 4.7 CLI 参数清单

`scripts/ingest_content.py:537-604`：
- `--input [DIR ...]`：待摄取目录（`:540`）；缺省时取 `textbooks`、`notes`（`:588-597`）；
- `--ocr-mode {auto,force,never}`（默认 `auto`，`:541-542`）；
- `--db`：ChromaDB 路径，默认 `<项目根>/mati_data/chroma_db`（`:545-549`）；
- `--ingest-from-manifest DIR`（`:550-554`）；
- `--dry-run`（`:555-559`）。

`scripts/rebuild_index.py:26-76`：
- `--content [DIR ...]`（默认 `textbooks`、`notes`、`scripts/data_collection/data/content`，`:62-65`）；
- `--db-path`（默认 `mati_data/chroma_db`，`:32-33`）；
- `--no-backup`：直接删除旧索引，否则重命名为 `<db>_backup_old_embedding`（`:42-52`）；
- `--ocr-mode {auto,force,never}`（`:36`）。
- 动机注释：嵌入模型从 `all-MiniLM-L6-v2`(384 维) 换为 `bge-small-zh-v1.5`(512 维)，向量空间不兼容必须重建（`:3-8`）。

`scripts/rag_data_preparation/embedding_generator.py:243-266`：位置参数 `input`、`--output`、`--batch-size`（默认 16）、`--model`（默认 `BAAI/bge-small-zh-v1.5`）。

`scripts/rag_data_preparation/enhanced_chunker.py:438-468`：位置参数 `input_dir`、`output_dir`、`--chunk-size`(512)、`--overlap`(0.2)、`--pattern`(`*.md`)——注意 CLI 默认重叠 0.2 与摄取侧 0.1 不同。

上游内容生成（非检索链路）：`tools/pdf_to_content.py:394-414`（`--pdf` 必填、`--subject`、`--grade`、`--out`、`--model`、`--pages-per-topic` 默认 8、`--sample`、`--no-write`、`--mock`、`--n-threads` 默认 4），内部 `Llama(n_ctx=2048, n_batch=96, n_threads, use_mmap=True, f16_kv=False)`（`:135-144`）、生成 `temperature=0.3 / top_p=0.9 / max_tokens=1400`（`:154-165`）、JSON 鲁棒解析（```json 围栏 → 首 `{` 到末 `}`，`:173-195`）。

### 4.8 本机索引实际状态（可核对）

- `mati_data/chroma_db/chroma.sqlite3`（7,655,424 B）：集合 `neb_science`（226 块，来自 `2025人教版物理必修一.pdf` 171 块 + `2025人教版物理必修三.pdf` 55 块）、`neb_readme`（5 块，来自 `textbooks/README.md`）。
- `textbooks/` 下实际有 6 个物理 PDF + README.md，`notes/` 仅 README.md → 说明本轮摄取只完成 2 个 PDF（远未覆盖 6 本教材）。
- 向量索引 schema：`"space":"l2"`，hnsw `ef_construction=100, max_neighbors=16, ef_search=100`（`collections.schema_str`）；`collection_metadata` 表为空（无自定义配置）。

---

## 5. 可靠性 / 风险控制实现

### 5.1 置信度计算

- 流式主路径（GUI/CLI 默认）：`rag_retrieval_engine.py:358-388`
  - `answer.split() < 5` → **直接 0.3**（`:367-368`）；
  - 词数 50–150 → +0.2；30–50 或 150–200 → +0.1（`:372-376`）；
  - 问题词（长度 >3）与答案词的交集/问题词数 × 0.2（`:378-382`）；
  - 上下文平均 `final_score` × 0.1（`:384-386`）；上限 1.0（`:388`）。
  - **中文失效**：按空白切分，中文答案通常只有 1 个 token。实测把该方法源码抽出单独执行，一段 67 字中文答案（含良好上下文 `final_score=0.9`）返回 `0.3`；同一逻辑下带 5+ 空格才会进入 0.5 档。→ GUI 几乎恒显示「⚠️ 低置信度 - 请核实」（`ask_question_view.py:350-358`）。
- 非流式/工具链路径：`qwen_handler.py:165-177` 使用 CJK 感知的 `_units`（`:155-163`：每个汉字算一个单位、拉丁词算一个），`0.5 + relevance*0.5`；**流式路径不使用它**。
- 归一化置信度（另一条线）：`input_normalizer.py:296+`，<0.7 会被 `AdaptiveNormalizer._flag_for_review` 落盘（`adaptive_normalizer.py:235-236`）。

### 5.2 防答非所问 / 防幻觉机制

**实际生效的：**
1. 输入归一化 + 意图分类（`input_normalizer.py:69-107`；意图模式 `:51-62`）。
2. 边缘用例短路（`user_edge_case_handler.py:41-72`）——省算力。
3. 检索分数阈值 0.35 + 上下文 1600 字符上限（`rag_retrieval_engine.py:289-300`）。
4. 来源优先级加权与 NEB 前置（`anti_confusion_engine.py:104-109`, `:181-188`）。
5. Prompt 层接地约束：system 第 2 条与 user 指令均要求「优先依据参考材料…材料不足先说明『材料中未找到相关信息』」（`qwen_handler.py:41-42`, `:116`）。
6. UI 层低置信提示（`ask_question_view.py:348-367`；`main_window.py:819-845`）。

**已实现但未接线的（死代码）：**
- `AntiConfusionEngine.validate_grounding`（`anti_confusion_engine.py:128-179`）：术语重叠 <30% 判为不接地、命中 6 条禁止语（`i don't have that information` 等）判为「模型承认不知道」——**全仓无调用点**（grep 仅定义处）。
- `filter_low_quality`（`:190-191`）无调用（引擎内联了同样的 0.35 过滤）。
- `rag_helper.should_use_rag` / `get_context_with_timeout` / `validate_context_relevance`（`rag_helper.py:21-169`）——整个模块无导入者。
- 注意：禁止语列表只有英文，即便接线也抓不到中文的「材料中未找到相关信息」。

### 5.3 异常兜底

见 2.5 表格；额外观察：三处 LLM 异常文案重复（`qwen_handler.py:211`, `:242`；`model_handler.py:86`；`rag_retrieval_engine.py:330`），且引擎在异常后仍会返回 `llm_used=False` 的结果并写入缓存（`:326-330`, `:353-354`）——错误文案会被缓存 1 小时。

### 5.4 日志埋点

- 文件日志：`mati.log`（`system/data_manager/content_manager.py:30` 的 `logging.FileHandler('mati.log')`，相对路径 = 进程 CWD；原 `main.py`/`cli_interface.py` 的配置点已随文件删除）；性能日志 `system/performance/performance.log`（`system/performance/performance_utils.py:23-28` 用 `basicConfig(filename=...)`，但被前者抢先占位，实际不会生成）。
- 耗时/资源：`@timeit` 记录函数耗时（`performance_utils.py:31-47`）；`log_resource_usage` 记录 RSS(MB) 与 CPU%（`:50-59`）。【已更新】GUI 侧调用点：`main_window.py:706`（`@timeit` 装饰 `_grade_answer_with_ai`）、`:360`（初始化完成）、`:843`（回答完成）；原 CLI 调用点（含被注释的两处）已随文件删除。
- 归一化埋点：`AdaptiveNormalizer._log_normalization`（`adaptive_normalizer.py:215-239`）→ 内存 `feedback_db` 每 100 条落盘 `mati_data/normalization_logs/feedback_db.jsonl`（`:238-239`）；置信度 <0.7 立即写 `low_confidence_cases.jsonl`（`:241-248`）；拼写缓存 `spell_cache.json` 上限 10000 条（`:141-148`）。
- 坏例闭环：`PatternMiner.mine_new_patterns` 从 feedback 日志挖 2–4 gram（`pattern_miner.py:99-109`），双阈值 `min_frequency`/`min_confidence`（`:26`，脚本里改为 5/0.6，`scripts/run_pattern_mining.py:33`），人工确认后回写归一化器（`:76-84`）。
- **实测**：`mati_data/normalization_logs/` 为空目录（本机从未产生日志）；`scripts/run_pattern_mining.py` 直接运行报 `ModuleNotFoundError: No module named 'system'`——`project_root = Path(__file__).parent.parent.parent` 多算了一层（`:18`）。
- **未找到**：请求级 trace id、检索/生成耗时分解（只有整段 `processing_time`，`rag_retrieval_engine.py:186`, `:348`）、缓存命中率统计（`rag_cache.py:200` 直接写 `"hit_rate": "N/A"`）、生成阶段的超时或取消机制。

---

## 6. 技术债与风险点（带证据）

1. **OCR 分支实际不可用**：`self.easyocr_reader` 从未赋值（`ingest_content.py:148` 之后 `:150-155` 不可达 + 引用未定义 `ocr_mode`），扫描 PDF 会抛 `AttributeError`（实测），Tesseract 兜底（`:319`）不可达；`requirements.txt` 无 easyocr。
2. **摄取中断风险**：`ingest_directory` 的 try 只包住「建集合→嵌入→upsert」（`:423-451`），`process_file` 调用在 try 之外（`:407`）→ 单个扫描件异常会中断整个目录摄取。
3. **集合选择与真实索引严重不匹配**（实测用真实逻辑 + 当前集合列表复现）：
   - `""` → `['neb_science', 'neb_readme']`（`neb_` 前缀命中所有集合，README 噪声可进入上下文；原 CLI 正是传空串触发该分支，该调用点已随文件删除，但空 subject 分支在引擎内仍然存在）；
   - `'Science'` → `['neb_science']`；
   - `'Math'`/`'Mathematics'`/`'Computer Science'`/`'English Grammar'`/`'Social Studies'`/`'physics'` → `[]`。
   - 原因：`subject_map` 中的 `openstax_science`/`scienceqa`/`finemath`/`gsm8k`/`cs_stanford`/`fineweb_edu`/`khanacademy_pedagogy` 在 DB 中都不存在；GUI 学科键（`main_window.py:38-44`）与 map 键也不完全对齐（如 `English Grammar` vs `english`）。数学/计算机/英语问题将完全无检索上下文，退化为纯参数化问答。
4. **相似度语义错误**：`score = 1.0 - distance`（`rag_retrieval_engine.py:273`）被注释与字段名当作「余弦相似度」（`anti_confusion_engine.py:96`），但集合 schema 是 `"space":"l2"`（`chroma.sqlite3` 的 `collections.schema_str`）。由于向量近似单位长度，`score ≈ 2cos−1`；`final_score = 0.7*base+0.3`（`anti_confusion_engine.py:109`）下 0.35 阈值实际约等价 `cos ≥ 0.54`。建议显式 `hnsw:space=cosine` 或改 `1 - d/2` 并重新标定阈值。
5. **grade 维度形同虚设**：`query()` 无 grade 参数（`rag_retrieval_engine.py:168-176`），内部固定传 `""`（`:252`）；摄取时年级解析常失败（文件名「2025人教版物理必修一.pdf」→ `unknown`，`ingest_content.py:347-357`），DB 内 grade 全为 `unknown`；查询也无 `where=` 过滤 → 文档宣称的「按年级过滤」在代码中不存在。
6. **硬编码**：0.35 / 0.88 / 1600（`rag_retrieval_engine.py:292`, `:242`, `:289`）、850 / 768 / 0.6 / 0.9 / 1.1（`qwen_handler.py:107`, `:193-199`）、2048 / 96 / 4（`:56-58`）、chunk 512/0.1（`ingest_content.py:122`）、模型与 DB 路径（`model_handler.py:48`、`rag_retrieval_engine.py:35-36`）、`subject_map`（`:127-143`）。无配置层/环境变量覆盖。
7. **异常吞噬**：`query_collection` 吞掉单集合异常（`:277-279`）；`warm_up` 吞异常（`:95-96`）；`rag_helper` 全模块 `logger.debug` 级吞异常且未被引用；`ModelHandler` 初始化失败在 GUI 侧被降级为界面错误文案（`main_window.py:366-370` → `:387-391`）。
8. **缓存一致性**：进程内、无持久化（`rag_cache.py:44`）；「LRU」实为最旧写入淘汰（`:164-167`）且不更新访问时间；语义阈值调用点 0.88 与默认 0.92 不一致；索引重建（`rebuild_index.py`）后旧答案仍在内存；错误文案也会被缓存 1 小时（`rag_retrieval_engine.py:326-330`, `:353-354`）；中文缓存回放按空格切词导致「伪流式」（`:229-232`）。
9. **并发与资源**：`ThreadPoolExecutor(max_workers=2)`（`:281`）与最多 3 个集合不匹配；`RAGCache` 无锁，而 GUI 在 worker 线程调用（`main_window.py:864`）；`SentenceTransformer` 在引擎 `__init__` 即加载（`rag_retrieval_engine.py:47`）→ 启动即常驻内存；`torch.set_num_threads(2)`（`embedding_generator.py:107`）与 llama.cpp `n_threads=4`（`qwen_handler.py:58`）在低端 CPU 上争抢；模型文件 1.04 GB + Chroma 7.6 MB。
10. **文档与代码不一致**：「512 token / 10% 重叠」（`readme.md:183`, `readme.md:257`, `HOW_TO_PROCESS.md:163`）实为 512 字符（`enhanced_chunker.py:55`）；`HOW_TO_PROCESS.md:48` 宣称 notes 集合名 `neb_{subject}_notes_grade_{grade}`，代码无此规则（`ingest_content.py:371-373`）；`HOW_TO_PROCESS.md:195` 要求安装 easyocr 而该分支不可用。
11. **其它死代码 / 小 bug**：`rag_helper.py` 整模块无引用（且 `:170` 在文件末尾 `import threading`）；`anti_confusion.validate_grounding`/`filter_low_quality`/`RELEVANCE_THRESHOLD` 未使用；`user_edge_case_handler.is_math_query` 未使用（`:74-77`）、`:56-58` 为空循环 `pass`（本意应是处理 "hello xxx"）；`ascii_diagram_library.list_available_diagrams` 未使用（`:130-132`）；`scripts/run_pattern_mining.py:18` 路径算错（实测报错）；`system/rag/` 无 `__init__.py`（依赖隐式命名空间包，与 `system/input_processing/`、`system/diagrams/` 不一致）。
12. **日志配置副作用**：多个模块在 import 时执行 `logging.basicConfig`（`rag_retrieval_engine.py:27`、`anti_confusion_engine.py:25-28`、`embedding_generator.py:51-54`、`performance_utils.py:24-28`），最终生效取决于 import 顺序；`performance_utils` 的 `basicConfig(filename=...)` 可能把后续 root 日志重定向到 performance.log。
13. **未找到**：RAG 层的并发/多用户隔离、检索结果缓存持久化、生成超时/取消、token 级限流、prompt 注入防护、检索-生成一致性的自动化校验（唯一相关实现是死代码 `validate_grounding`）。

---

## 7. 最值得在面试里讲的 3 个技术点（≤200 字）

1. **把「检索分数」算错但被阈值掩盖**：代码用 `1 - l2_distance` 当余弦相似度（`rag_retrieval_engine.py:273`），而集合 schema 是 `space="l2"`；我通过 `chroma.sqlite3` 的 `schema_str` 反推真实几何关系 `score=2cos−1`，再回算 0.35 阈值的真实语义，并给出 `hnsw:space=cosine` 的修复与重标定方案。
2. **中文置信度恒为 0.3 的根因定位**：流式路径用 `answer.split()` 数词（`:367`），中文无空格 → 永远命中「过短」分支；我把方法源码抽出单测复现，并指出仓库里已有 CJK 感知的 `_units`（`qwen_handler.py:155`）却没被流式路径复用。
3. **检索层与索引实际状态脱节**：用真实集合列表跑 `_get_relevant_collections`，证明除 `Science` 外所有 GUI 学科都返回空集合，而空 subject 会命中全部集合（含 README 噪声），说明「按学科/年级检索」是文档承诺而非代码事实。
