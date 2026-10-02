# Mati 内容数据治理与自动化校验/分发链路 —— 代码调研笔记

工作目录：`D:\mati-master`（Windows）。所有结论标注 `文件:行号`；标 `【实测】` 的为本机实际运行/脚本验证结果；标 `【未找到】` 的为全仓检索无结果。

> **2026-09 更新（GUI-only 改造后）**：终端界面（TUI）已整体移除。
> 删除内容：`student_app/main.py`、`student_app/interface/`（`cli_interface.py`、`cli_renderer.py`）、`teacher_tools/`（含 `analytics_cli.py`、`analytics_utils.py`、`content_editor_cli.py`、`content_editor_utils.py`），并从 `requirements.txt` 移除 `prompt_toolkit` / `wcwidth`。
> 影响：① §2.4「教师内容编辑器」、§5.1 编辑链路、§5.3 分析工具等章节所依据的文件**已不存在**，相关内容改为“已移除”标注，仅作历史记录；② §2.5、§5.1 中“学生端唯一校验入口是 CLI”的结论**已失效**，最新状态是 GUI 侧接入 `security_utils`；③ `docs/TECHNICAL_IMPLEMENTATION.md` / `docs/PROJECT_OVERVIEW.md` 中指向 `run_cli.*` / `build_offline_bundle.py` 的失效链接已同步清理。

---

## 1. 内容数据模型

### 1.1 目录与文件命名

| 事实 | 证据 |
|---|---|
| 默认内容目录 `scripts/data_collection/data/content` | `system/data_manager/content_manager.py:144` |
| 仓库中该目录实际只有 **1 个** 文件：`science_grade_10_auto.json`（16333 字节） | 【实测】目录列举 |
| 文档声明的目录含 `computer_science.json` / `English_Grammar.json` / `science.json` | `docs/content-explanation.md:248-254` |
| 上述三个文件在内容目录中不存在，曾存在于手工备份目录 `_backup_20260803_035623/content/`（该目录已于 2026-10-01 移除） | 【实测】目录列举 |
| 文档命名规范：描述性、小写下划线、`.json` 后缀 | `docs/content-explanation.md:256-259` |
| 实际自动生成命名：`{subject.lower().replace(' ','_')}_grade_{grade}_auto.json` | `tools/pdf_to_content.py:377` |
| 打包器识别的后缀：`.txt .md .jsonl .json` | `tools/package_builder.py:56` |

### 1.2 JSON schema 层级（subject → topics → subtopics → concepts → questions）

严格 schema 定义在 `content_manager.py`（**不是**文档里的那份）：

- **Root（CONTENT_SCHEMA）** `content_manager.py:102-133`
  - required：`subject`、`grade`、`topics`（`:104`）
  - `additionalProperties: False`（`:132`）
  - `grade`：`anyOf` 整数 1–12 或字符串 `^(1[0-2]|[1-9])$`（`:110-115`）→ 字符串 `"10"` 合法【实测 PASS】
- **Topic** `:116-130`：required `name` + `subtopics`；`additionalProperties: False`（`:128`）
- **Subtopic（递归 3 层）** `create_subtopic_schema(max_depth=3)` `:75-100`
  - required `name`（`:78`）；`anyOf: [required concepts, required subtopics]`（`:93-96`）
  - 第 3 层后 `subtopics.items` 退化为 `{"type":"object"}`（`:87`）
- **Concept（CONCEPT_SCHEMA）** `:56-72`：required `name`/`summary`/`steps`/`questions`；`additionalProperties: False`（`:71`）；`name`、`summary` 均 `minLength: 1`（`:60-61`）
- **Question（QUESTION_SCHEMA）** `:37-53`：required `question`/`acceptable_answers`/`hints`；`question` `minLength:1`；`acceptable_answers` 数组 `minItems: 1` 但**元素无 minLength**（`:42-46`）；`additionalProperties: False`（`:52`）

**文档与实现冲突（实测）**：

| 文档声明 | 实测结果 |
|---|---|
| “Only `name` is required at each level” `docs/content-explanation.md:747` | concept 只有 `name` → `FAIL: 'summary' is a required property` |
| topic/subtopic 层 `concepts` 可选 `docs/content-explanation.md:89-91` | topic 只有 `concepts`、无 `subtopics` → `FAIL: 'subtopics' is a required property` |
| “Questions can be objects … or simple strings” `docs/content-explanation.md:107` | question 为字符串 → `FAIL: '…' is not of type 'object'` |
| 文档内自述的 Concept Schema 为 `required:["name"]` + `additionalProperties: True` `docs/content-explanation.md:505-516` | 与 `content_manager.py:56-72` 完全相反 |
| `docs/PROJECT_STANDARDS.md:66-136` 的 JSON Schema | 与 `content_manager.py` 大体一致（但缺 `additionalProperties`、缺 `anyOf` 细节） |

### 1.3 必填字段与 ID 规则

- **没有显式 ID 字段**。唯一标识即 `name` 字符串：
  - subject key = `content['subject']` 或文件名（扁平布局）`content_manager.py:180`；旧式子目录布局 = 目录名 `:196`
  - topic/concept/question 全部按 `name`/`question` 文本匹配 `:269`、`:291`、`:305`
- 内容 JSON 的全部键（对 `science_grade_10_auto.json` 递归统计）【实测】：
  `subject, grade, topics, name, subtopics, concepts, summary, steps, questions, question, acceptable_answers, hints`
  → 无 `id` / `version` / `schema_version` / `language` / `source` / `author` / `review_status`
- 结构规模【实测】：7 topics / 7 subtopics / 16 concepts / 30 questions

### 1.4 版本与元数据字段

- 内容 JSON 层：**无任何版本/审核/来源字段**；且**无法新增**——根级与各层 `additionalProperties: False`（`:52, :71, :90, :128, :132`），实测加 `schema_version` + `review_status` 被拒。
- 唯一的“版本”机制是文件备份：`update_content` 覆盖前复制到 `<subject>/backups/content_YYYYMMDD_HHMMSS.json`（`content_manager.py:349-354`），文档同步声明 `docs/content-explanation.md:268-271`。
- 打包格式有版本：`FORMAT_VERSION = 2`（`tools/common.py:20`）、manifest `format_version`/`version`（`tools/package_builder.py:201-203`），但 `version` 默认恒为 `"1.0.0"` 且 `build_from_sources` 未传值（`:146`、`:236`）→ 无区分度。
- 学科词表**三方不一致**：
  - ContentManager：只要非空字符串，无枚举 `content_manager.py:106-109`
  - 校验脚本白名单 `{"Computer Science","Science","English"}` `scripts/validation/validate_standards.py:104`
  - `docs/PROJECT_STANDARDS.md:74` 同集合
  - 教师前端/生成器：`Science / Math / English Grammar / Computer Science / Social Studies` `student_app/gui_app/views/teacher_view.py:27-33`、`tools/pdf_to_content.py:43-46`
  - 文档示例还出现 “English Grammar” `docs/content-explanation.md:74`
  - ⇒ 由前端生成的 `Math` / `English Grammar` / `Social Studies` 内容若跑 standards 校验会被白名单判错（`validate_standards.py:104-108`）

---

## 2. 校验自动化

### 2.1 `scripts/validation/validate_standards.py` 逐条规则

入口：`--input-file`（`:265`），`main()` 返回 `0 if valid else 1`（`:285`），`exit(main())`（`:288`）。

**分派**：文件不存在 → error（`:36-38`）；仅 `.py` / `.json`，其它后缀 → error（`:42-48`）。

**Python 规则**
| 规则 | 行号 |
|---|---|
| 文件名必须 `^[a-z][a-z0-9_]*\.py$` | `:53-55` |
| 必须存在文件 docstring（正则 `^""".*?"""`，DOTALL） | `:60-62` |
| `ast.parse` 语法检查 | `:64-68` |
| 类名 `^[A-Z][a-zA-Z0-9]*$` | `:71-74` |
| 每个函数必须有 docstring | `:76-79` |
| 无返回类型注解 → **warning**（不影响退出码） | `:81-82` |

**JSON 规则**
| 规则 | 行号 |
|---|---|
| JSON 可解析（`JSONDecodeError`） | `:89-94` |
| 根必填 `subject/grade/topics` | `:96-101` |
| `subject` ∈ `{"Computer Science","Science","English"}` | `:104-108` |
| `grade` 可 `int()` 且 1–12 | `:110-123` |
| `topics` 必须 list | `:125-128` |
| topic 必填 `name` + `subtopics` | `:136-146` |
| subtopic 必有 `name`，且至少有 `concepts` 或 `subtopics` | `:155-170` |
| concept 必填 `name/summary/steps/questions` | `:184-194` |
| `steps` 为 list 且每项非空字符串 | `:197-205` |
| 每个 question 必须 dict | `:213-217` |
| question 必填 `question/acceptable_answers/hints`；缺则**提前 return False**（后续不再检查） | `:224-231` |
| `question` 非空；`acceptable_answers` 非空 list 且每项非空；`hints` 每项非空 | `:234-259` |

**不检查**（实测/阅读确认）：`additionalProperties`、concept `summary` 非空、重复 name、跨文件 ID/名称冲突、subject 与文件名一致性、grade 与目录一致性、内容语义/引用/事实正确性。

### 2.2 退出码与实测行为

- 内容文件：`python scripts/validation/validate_standards.py --input-file scripts/data_collection/data/content/science_grade_10_auto.json` → **rc=0**，输出 `Validation passed successfully!`【实测】
- 脚本检查自身：`--input-file scripts/validation/validate_standards.py` → **rc=1**，报 9 个 `Missing function docstring`【实测】
- 缺参数：直接运行（无 `--input-file`）→ **rc=2**（argparse 报错）【实测】
- **warnings 不影响退出码**：`valid` 只在 errors 时置 False（`:74`、`:82`、`:285`）

### 2.3 CI / 预提交集成

- **【未找到】任何 CI 或预提交配置**：仓库根目录仅 `.gitignore / CODE_OF_CONDUCT.md / HOW_TO_PROCESS.md / LICENSE / readme.md / requirements.txt / mati.log / USER_GUIDE.md`【实测】；无 `.github/`、无 `.pre-commit-config.yaml`、无 `pyproject.toml`/`setup.cfg`/`Makefile`【实测】
- `validate_standards.py` 只被文档引用：
  - `docs/CONTRIBUTING.md:116`：`python scripts/validation/validate_standards.py`
  - `docs/TEACHER_GUIDE.md:55`、`:60`：`python -m scripts.validation.validate_standards`
  - **两处命令都缺 `--input-file`，照抄执行 rc=2**【实测】
- 单次只能校验**一个文件**（`--input-file` 单值，`:265`），无目录/批量模式

### 2.4 教师内容编辑器（content editor）的校验点

> 【已移除】原“教师内容编辑器（CLI）”整体删除（`content_editor_cli.py`、`content_editor_utils.py`），本节历史结论如下，**不再构成当前风险**：
> - `content_editor_utils.validate_content()` 封装 `content_manager._validate_content()`，全仓无调用点 ⇒ 死代码（已随删除消失）。
> - 真正生效的校验在保存路径：`content_editor_cli.py` 保存 → `content_manager.update_content` → `_validate_content`（`content_manager.py:342`、`:226-241`）。
> - 两个编辑器操作产出的内容**必然被 schema 拒绝**【实测】：Add Question 只构造 `{"question": text}` → `'acceptable_answers' is a required property`；Add Concept 允许空 summary → `'' should be non-empty`。
> - 保存失败时只提供“另存为 `<subject>_emergency_save.json`” ⇒ 形成**绕过校验的旁路文件**。

- 现存写法：教师现在只能直接编辑 `scripts/data_collection/data/content/*.json`，或通过 GUI 教师工作台生成；加载时由 `ContentManager._validate_content`（`content_manager.py:226-241`）把关。
- 【已更新】输入体量护栏：`system/security/security_utils.py:61-73` `validate_content_input()` 限制 <100KB，现由 GUI 进度导入调用（`student_app/gui_app/main_window.py:960`），原 `content_editor_utils.py` 的调用点已删除。

### 2.5 其它校验关口

| 关口 | 证据 | 是否校验内容 schema |
|---|---|---|
| 加载期（`ContentManager._load_content`） | `content_manager.py:179`、`:195` | 是（失败即丢弃整个文件） |
| 写入期（`update_content`） | `content_manager.py:342` | 是 |
| AI 生成期（`pdf_to_content.py`） | `tools/pdf_to_content.py:388` 直接 `write_text` | **否**（仅对同学科已有文件打 warning `:379-387`） |
| 同步摄取期（`ingest_content --ingest-from-manifest`） | `scripts/ingest_content.py:480-501` | **否**（只校验 chunk 存在性 + sha256） |
| standards 校验脚本 | `scripts/validation/validate_standards.py` | 是（人工手动，单文件） |

**两套校验器规则相反（实测）**：

| 场景 | `content_manager` CONTENT_SCHEMA | `validate_standards.py` |
|---|---|---|
| `acceptable_answers: [""]` | **通过**（`:42-46` 元素无 minLength） | **拒绝** rc=1（`:246-249`） |
| `summary: ""` | **拒绝**（`:61` minLength 1） | **通过**（无 summary 规则，只报答案错） |
| 根级多余字段 `schema_version`+`review_status` | **拒绝**（`:132`） | **通过** rc=0（无 additionalProperties 检查） |

---

## 3. 内容加载与容错

### 3.1 路径解析与回退

`content_manager.py:151-165`：
1. 相对路径先 `abspath`（`:153-156`）
2. 若 `isdir` → 直接使用（`:157-158`）
3. 否则用 `__file__` 上溯两级得项目根，拼 `scripts/data_collection/data/content`（`:161-163`）
4. 该候选目录也不存在 → 退回第 1 步的绝对路径（`:164`）

【实测】CWD=`D:\mati-master\tools` 时，默认相对路径不存在 → 解析为 `D:\mati-master\scripts\data_collection\data\content`，加载到 `['Science']`。

**导入期副作用（实测）**：模块顶部 `logging.basicConfig(handlers=[FileHandler('mati.log'), NullHandler()])`（`:27-33`）——CWD 不可写时 **import 直接抛 `PermissionError`**（在 `C:\` 下实测）。

### 3.2 发现机制

两轮 `os.listdir`（无 glob、无递归、无索引文件）：
- 扁平 `*.json`（`:173-184`）：`subject_name = content.get('subject') or 文件名`（`:180`）
- 旧式子目录 `<subject>/content.json`（`:187-199`）：subject key = 目录名（`:196`）

### 3.3 坏 JSON / 缺字段处理

- `_validate_content` 失败 `raise`（`:241`），外层 `except Exception` 捕获 → `logger.error` + `continue`（`:182-184`、`:197-199`）；顶层再兜一层，注释明说 “Don't raise here, allow partial loading”（`:201-203`）
- 日志**只写 `mati.log`**（`:27-33` 只挂了 FileHandler + NullHandler，无 StreamHandler）→ 控制台静默
- 【实测】构造 `broken.json`（截断 JSON）、`invalid_schema.json`（缺 `grade`）与合法文件同目录：坏文件被跳过，合法文件正常加载；`mati.log` 记录
  `Error loading file broken.json: Expecting value: line 1 column 47` / `Content validation error: 'grade' is a required property`
- ⇒ **全有或全无**：任一必填字段缺失 → 整个文件（整个学科）被丢弃，且只在文件日志里留痕

### 3.4 合并语义（`_merge_subject`，`:205-218`）

- 同 key 学科：按 `topic.name` 建 dict 覆盖合并（`:214-218`）
- 加载顺序固定为“先扁平文件、后子目录” → **子目录版本覆盖扁平版本**（`_merge_subject` 后写者胜）
- 【实测】扁平 `good.json` + 子目录 `Good Subject/content.json` 同 key：reload 后该 topic 只有一条，且为子目录版本；`good.json` 内旧值仍在磁盘上
- ⇒ 无时间戳/版本比较，纯覆盖；同名 topic 静默丢失旧内容

### 3.5 缓存与刷新

- **无缓存层、无 mtime 检查、无 LRU**。文档 `docs/PROJECT_STANDARDS.md:309-318` 示例里写的 `JSONSchemaValidator` / `LRUCache` / `GitVersionControl` 成员在真实 `ContentManager` 中**不存在**【实测：类中无这些属性】
- `reload()` 清空 `self.subjects` 后全量重载（`:220-224`）
- 唯一调用方：`student_app/gui_app/main_window.py:1007`（教师导入/生成后由 `teacher_view.py:144-150` 回调触发），提示文案 `main_window.py:1008`

### 3.6 写入与来源可追踪性

`update_content`（`:332-360`）：
1. 先 schema 校验（`:342`）
2. 更新内存（`:343`）
3. 固定写到 `<content_dir>/<subject_key>/content.json`（`:345-347`）
4. 覆盖前备份到 `<subject_key>/backups/content_YYYYMMDD_HHMMSS.json`（`:349-354`）

【实测】对**扁平来源**学科 `Good Subject` 保存两次后：
```
Good Subject\content.json
Good Subject\backups\content_20260908_215712.json
good.json            <-- 原始文件未变，仍是旧 summary
```
⇒ 写路径不记录源文件，产生第二份内容；无作者/时间/理由字段。

### 3.7 读取 API（供 UI/RAG）

`get_subject` `:243-253`、`get_topic` `:255-271`、`get_concept`（递归 subtopics）`:273-312`、`get_question` `:314-330`、`get_all_*` `:378-433`、`get_subject_structure` `:435-448`、`list_browseable_topics` `:680-745`、`get_concepts_at_path` `:747-796`、`search_content` `:548-678`（difflib `ratio > 0.6` + 关键词回退 `:628-678`）、`get_default_context` `:798-818`、`suggest_next_concept` `:450-483`、`get_weak_concepts` `:485-517`（阈值 attempts≥2 且 correct==0 `:510`）。

注意 `get_concept`/`get_all_concepts`/`list_browseable_topics` 都支持 **topic 级 `concepts`**（`:303-310`、`:427-428`、`:726`），而 schema 禁止（`subtopics` 必填 + `additionalProperties:False`）——代码能力与校验规则不一致。

---

## 4. 分发与同步

### 4.1 package_builder 打包产物

布局（`tools/package_builder.py:11-15`、`tools/README.md:48-58`）：
```
dist/
├── catalog.json
└── packages/<package_id>/
    ├── manifest.json
    └── chunks/<sha256>.chunk
```
常量：`FORMAT_VERSION=2`、`HASH_ALGO=sha256`、`CHUNK_EXT=.chunk`、`NETWORK_BLOCK_SIZE=64KB`、`MAX_CHUNK_BYTES=256KB`、`STATE_DIR=.mati_sync`、`COMPRESSION_ALGO=zlib`（`tools/common.py:20-38`、`:64`）

- **哈希基于未压缩字节**，存储/传输为 zlib 压缩载荷（`package_builder.py:167-170`；`common.py:62-64` 注释）；写盘用 `write_bytes` 避免 Windows `\n→\r\n` 破坏载荷（`package_builder.py:182-185`）
- manifest 字段（`:200-211`）：`format_version, package_id, version, description, compression, total_chunks, total_size_bytes(压缩后), total_source_bytes, collections[], chunks[{id,hash,size,compressed_size,compression,collection,metadata}]`
- catalog（`:276-301`）：`format_version, generated_at, package_count, packages{id:{version,total_chunks,total_size_bytes,collections,manifest(相对路径)}}`
  - **`generated_at` 是 `os.urandom(8).hex()` 随机值，不是时间戳**（`:296`）
  - catalog **不含** manifest 摘要/签名 → 客户端无法检测 manifest 被替换
- 来源→文本：`extract_json_content` 按键白名单扁平化（`:62-91`：`subject/grade/name/summary/question/answer/acceptable_answers/hint/hints/description/example/formula/explanation`）；jsonl 取 `text/content/question`（`:100-114`）
- 元数据推断：路径含 `grade_/class_` 推年级、文件名首段推学科、collection `neb_{subject}_grade_{grade}`（`:122-136`）；无法识别时 `grade="unknown"`、collection `neb_{stem}`
- `package_id = f"{collection_name}_{stem}"` 再做字符规范化（`:232-233`）
- 分块：`simple_chunk` 段落 → 句子 → 硬切，按 UTF-8 字节（`common.py:125-184`）
- 自校验：`validate_dist` 逐 chunk 存在性 + 解压后 sha256（`:304-328`），容忍旧未压缩 chunk（`:322-323`）；CLI `--validate` → rc 0/1（`:349-350`）；超 50MB 预算只 warning（`:362-371`）
- `--demo` 生成合成数据并显式声明“不得作为教学内容发布”（`:242-249`）

### 4.2 sync_server 协议与校验

`tools/sync_server.py:8-13`、`:49-72`：
| 端点 | 行为 |
|---|---|
| `GET /` | `{"service":"mati-sync","status":"ok"}`（`:54-55`） |
| `GET /catalog` | 返回 `catalog.json`（`:77-81`） |
| `GET /manifest/<id>` | 返回该包 manifest（`:83-87`） |
| `GET /chunk/<id>/<hash>` | 单 chunk，支持 `Range`，206/416（`:89-131`） |
| `GET /diff?package=&have=` | 返回缺失 chunk hash 列表（`:133-153`） |

- 无鉴权、无 HTTPS 强制、无速率限制（全文件无相关代码）
- **路径未净化【实测漏洞】**：`GET /manifest/..%2F..%2Fsecret` 成功返回服务根目录之外的 `manifest.json`（响应 `{"SECRET":"outside-root"}`）。原因：`package_dir()` 直接拼接未校验的 `package_id`（`common.py:118-119`，`sync_server.py:83-87`）。chunk 端点同样未净化，但受 `.chunk` 后缀限制（`common.py:106-107`，代码推导）
- 【已更新】`system/security/security_utils.py:37-51` 的 `sanitize_filepath` 现由 `student_app/progress/progress_manager.py:39` 使用（保护进度文件路径）；**`sync_server` 仍未使用**（grep 确认）。原 CLI 调用点已随文件删除。

### 4.3 sync_client 协议与校验

`tools/sync_client.py`：
- 拉取 `catalog` / `manifest/<id>`（`:114-118`）
- **增量靠本地 `.mati_sync/<pkg>/*.done` 标记**（`:126-133`、`:205-209`）——**不调用服务端 `/diff`**（grep 确认）；而 `tools/README.md:68` 声称增量同步“（`/diff` 接口）” ⇒ 文档与实现不一致
- 续传：`.part` + `Range: bytes=offset-end`，64KB 一块（`:162-178`）
- 完整性：下载完 → 解压 + sha256 比对（`common.verify_chunk_bytes:77-82`）→ 通过才 `part.replace(final)` 并 `touch .done`（`:180-190`）；失败丢弃 `.part`（`:181-184`）
- 重试：任何异常都重试，指数退避 `min(0.5*2^(n-1), 8s)`，默认 5 次（`:98-106`、`:57`）
- 并发：`ThreadPoolExecutor`，默认 4（`:57-58`、`:218-229`）
- 测试注入：`--simulate-packet-loss`（`:72`、`:84-87`），RNG 固定种子 `20260801`（`:79`）
- 健壮性缺口（代码推导）：`while offset < expected_size` 无“零进展”保护（`:169-178`）→ 服务端返回空 body 时可能死循环；404 也会重试（`:98-105`）

### 4.4 端到端验证【实测】

`python tools/test_sync_e2e.py` → **rc=0**，输出：
```
[1] Built & validated 100 demo packages (each <= 50.0MB)
[2/3] Full sync of all packages completed with 0 failures   (300/300 chunks)
[2/3] All downloaded chunks verified on disk
[4] Under simulated 5% loss: success rate 100.0% (>= 95%)
[5] Resume-from-partial verified for package demo_computer_science_grade_10_unit_007
[6] Incremental re-sync fetched 0 chunks (all cached)
ALL END-TO-END CHECKS PASSED
```
对应断言：100 包 + 每包 ≤50MB + 逐 chunk 校验（`tools/test_sync_e2e.py:37-48`）、全量同步 0 失败（`:78-81`）、独立磁盘哈希复核（`:84-91`）、5% 丢包成功率 ≥95%（`:93-103`）、`.part` 续传（`:105-118`）、二次同步 0 chunk（`:120-123`）。
注意：demo chunk 仅约 200 字节（日志 `OK chunk … (201 bytes compressed)`）——量级远小于真实教材内容。

另有 `tools/test_ingest_manifest.py`：起本地服务 → 同步 3 个 demo 包 → 跑 `ingest_content.py --ingest-from-manifest <dest> --dry-run` 并断言 rc=0（`:27-61`）。

### 4.5 摄取与离线 bundle

- 摄取：`scripts/ingest_content.py:453-534` `ingest_manifest`——逐 chunk 存在性 + 解压 + sha256（`:480-501`），失败仅记录并跳过（`:485-500`）；`--dry-run` 只校验不写库（`:503-506`）；真实导入按 collection 分组 upsert，保留 manifest 的 id/collection/metadata（`:508-531`）；嵌入模型 `BAAI/bge-small-zh-v1.5`（`:133-136`）
- 离线分发 = **Nuitka 单文件打包**：`scripts/release/build_nuitka.ps1` `--onefile`（`:84`）、把 `mati_data` 与 `scripts\data_collection\data\content` 打进产物（`:93-94`）、入口 `--main=bootstrap.py`（`:120`）
- `scripts/release/bootstrap.py`：定位 `app_dir`（`NUITKA_ONEFILE_PARENT` → `sys.frozen` → `__file__`，`:34-39`）、`os.chdir(app_dir)`（`:51`）、启动 GUI（`:53-57`）、失败时写 `mati_startup_error.log` + `MessageBoxW`（`:26-32`、`:59-64`）
- **【已修正】** 文档所述的 `scripts/release/build_offline_bundle.py` 不存在（`scripts/release/` 实际只有 `bootstrap.py`、`build_nuitka.ps1`）；该失效引用已从 `docs/` 中清理。
- **【已修正】** 文档所述的 `scripts/release/run_cli.sh|bat`、`run_gui.sh|bat` 从未存在；随着 TUI 移除，`run_cli.*` 已无意义，相关描述已从 `docs/` 中删除（启动方式统一为 `python -m student_app.gui_app.main_window`）。

---

## 5. 教师 / 社区治理流程

### 5.1 现状链路（无审核环节）

```
教师工作台/AI 生成                  编辑/写盘                       学生可见
teacher_view.py  ──►  tools/pdf_to_content.py:377-388  ──►  main_window.py reload
（原 content_editor_cli.py 编辑路径已随 TUI 移除而删除）
```
- 生成路径：`teacher_view.py` 调 `pdf_to_content.generate_from_pdf`，输出目录直接是 `CONTENT_DIR`（`teacher_view.py:23`、`:242`），完成后只 `_notify_content_changed()`（`:251`）
- 【已移除】原“编辑路径” `content_editor_cli.py:253` → `update_content`（含 schema 校验 + 备份）已随文件删除；现在唯一的写盘路径是生成脚本，`update_content` 已无 GUI 调用点。
- 两条路径都**没有**：校验门禁 → 审核人 → 发布状态 → 学生端灰度
- 【已更新】学生端带输入校验的入口已迁移到 GUI：进度导入在 `main_window.py:950-961`（JSON 可解析性 + `validate_content_input` + `log_security_event`）；进度文件路径由 `progress_manager.py:39`（`sanitize_filepath`）保护。原 CLI 的 `sanitize_filepath`/`validate_content_input` 调用点已删除。

### 5.2 自动生成内容的合规风险（实测）

`tools/pdf_to_content.py`：
- 写盘前**无任何 schema 校验**（`:388`）
- 同学科已有文件只 warning、仍写新文件（`:379-387`）→ 同学科多文件，由 `_merge_subject` 按 topic 名覆盖（`content_manager.py:205-218`）
- `_normalize_concepts` 会把缺失答案补成 `[""]`（`:221-224`）→ ContentManager **接受**，standards 校验**拒绝**（实测 rc=1，报 `Answer 1 must be a non-empty string`）
- `summary` 可能为空串（`:232`）→ ContentManager **拒绝**（实测 `'' should be non-empty`）→ 整文件静默丢弃
- 这解释了内容目录里出现 `science_grade_10_auto.json`（文件名带 `_auto`，`pdf_to_content.py:377`）这类未经人工复核的内容

### 5.3 suggestions 机制

- 定义：`content_manager.py:362-376`，写 `<content_dir>/<subject>/suggestions/suggestion_<username>_<timestamp>.json`（`:370-375`），注释称 “saved for review”（`:364`）
- 文档：`docs/content-explanation.md:355-371`（含目录/命名示例）、`:499` 列为 “Community suggestions” 特性
- **【未找到】任何读取/审核/合并该目录的代码**（全仓 grep `suggestions` 仅命中定义处与文档）→ 写入即终点，无批准/拒绝/合流逻辑
- 【已更新】`username` 校验现状：`security_utils.validate_username:27-35` 现由 GUI 登录页调用（`student_app/gui_app/views/welcome_view.py:158`、`:184-185`），且失败会写安全事件；`progress_manager.sanitize_filepath`（`:39`）提供第二层路径防护。原 CLI 调用点已删除。

### 5.4 权限与版本控制痕迹

- **【未找到】** 内容链路上的角色/权限/审批/审计代码（grep `review|approv|publish|moderation|role|permission|is_admin` 在内容链路无命中；命中的是输入规范化侧 `system/input_processing/adaptive_normalizer.py:236-248` 的低置信度复核与 `scripts/run_pattern_mining.py:59-89` 的交互式 approve，与内容治理无关）
- **【未找到】** 工作副本没有 `.git`（实测）→ 无法核对分支/PR/提交；文档仍声明 Git flow 与 PR 评审：`docs/PROJECT_STANDARDS.md:26-42`、`docs/CONTRIBUTING.md:70-77`、`docs/TEACHER_GUIDE.md:67-69`
- 仓库里能看到的“版本”痕迹只有两类：
  1. 手工备份目录 `_backup_20260803_035623/`（含 `chroma_db/` 与 `content/{computer_science,English_Grammar,science,science_grade_10_auto}.json`）【实测】—— 该目录已于 2026-10-01 按主人要求移除（项目外留有安全副本）
  2. `ContentManager` 的时间戳备份 `backups/content_YYYYMMDD_HHMMSS.json`（`:349-354`）
- 内容质量维度（清晰/年龄适配/文化敏感/事实准确）仅存在于文档：`docs/PROJECT_STANDARDS.md:138-142`、`docs/content-explanation.md:201-239`；**无任何自动化检查**
- ~~其它可验证的实现瑕疵：`teacher_tools/analytics/analytics_cli.py:205` 使用 `csv.writer` 但该模块未 `import csv` → 班级报告 CSV 导出会 `NameError`~~ → **已消除**：`teacher_tools/` 整体删除（该缺陷随代码消失）。

---

## 6. 可改进点（均带证据）

1. **无 schema 版本号，且 schema 不可扩展**：各层 `additionalProperties: False`（`content_manager.py:52, 71, 90, 128, 132`）→ 无法加 `schema_version`/`review_status`/`source`；实测加这两个字段即被拒。而 standards 校验却放行（rc=0）→ 两套校验器对同一文件结论相反。
2. **两套校验器规则互相矛盾**（见 2.5 表）：空答案、空 summary、额外字段三类场景结论完全相反；学生端/AI 生成端到底以谁为准没有定义。
3. **校验不是门禁**：无 CI、无预提交、无批量校验（无 `.github/`、无 pre-commit 配置，`validate_standards.py:265` 只收单文件）；文档给的命令还缺参数（`docs/CONTRIBUTING.md:116`、`docs/TEACHER_GUIDE.md:60`，实测 rc=2）；该脚本甚至过不了自己的规则（实测 rc=1，`:60-62` 的 docstring 正则）。
4. **生成链路绕过校验**：`pdf_to_content.py:388` 无校验直接落盘。【已更新】原 `content_editor_cli.py` 的“产出必被拒 + 紧急另存旁路”问题已随 `teacher_tools/` 删除而消失；现存唯一风险面是生成脚本写盘前不过 schema 门禁。
5. **无签名/来源完整性保护**：manifest 与 catalog 无摘要、无签名（`package_builder.py:200-211`、`:294-299`）；`catalog.generated_at` 是随机 hex 而非时间（`:296`）→ 无法校验来源与构建时间；客户端只校验 chunk↔manifest 一致（`sync_client.py:196-203`），manifest 被替换无法察觉。
6. **分发端无鉴权、路径未净化**：`sync_server.py:49-72` 无 token/HTTPS；实测 `/manifest/..%2F..%2Fsecret` 越权读取服务根外文件；`security_utils.sanitize_filepath`（`:37-51`）未在 sync_server 使用。
7. **加载容错“静默 + 全有或全无”**：单字段错误 → 整文件丢弃（`content_manager.py:179, 182-184, 237-241`），日志只进 `mati.log`（`:27-33` 无控制台 handler）→ 运营侧无告警、无校验报告、无失败清单。
8. **标识与合并不可靠**：以 `name` 为唯一键（`:180, 269, 291`），无稳定 ID；同学科按加载顺序覆盖（`:205-218`），子目录覆盖扁平（实测）；subject key 在两种布局下语义不同（`:180` vs `:196`；实测 `get_subject("Legacy Subject")` → `None`，只能用目录名 `legacy_subject` 取到）。
9. **写入不可溯源**：`update_content` 固定写 `<subject_key>/content.json`（`:345-347`），不记录源文件 → 实测扁平来源学科保存后多出一份内容、原文件未更新；内容 JSON 无作者/时间/变更理由字段。
10. **suggestions 无闭环**：写后无消费方（`:362-376`，grep 无读取）；`username` 未校验。
11. **客户端健壮性**：无零进展保护可能死循环（`sync_client.py:169-178`）；404 也重试（`:98-105`）；服务端 `/diff` 接口无客户端使用（`tools/README.md:68` 与实现不符）。
12. **学科词表三方不一致**（见 1.4）：standards 白名单还会把前端可选的 `Math`/`English Grammar`/`Social Studies` 判为非法。
13. **文档与代码脱节**（可核验清单）：~~`build_offline_bundle.py` 不存在；`run_cli/run_gui` 脚本不存在~~（**已修正**：两处失效引用已从 `docs/` 清理）；`LRUCache/JSONSchemaValidator/GitVersionControl` 不存在（`docs/PROJECT_STANDARDS.md:309-318`）；文档说增量用 `/diff`（`tools/README.md:68`）但客户端用 `.done` 标记；`docs/content-explanation.md` 的“only name required / questions 可为字符串 / topic concepts 可选”与实现相反。
14. **e2e 测试量级偏小**：demo chunk ≈200 字节（实测日志），未覆盖真实教材的 256KB 分块边界与压缩比；`tools/README.md:6` 宣称实测 3.5x 压缩率，仓库内无可复现的压缩比测试。

---

## 最值得在面试里讲的 3 个治理/自动化点（≤200 字）

1. **内容分发是可验证的内容寻址链路**：zlib 压缩 + sha256（基于未压缩字节）+ 64KB Range 续传 + `.done` 增量，实测 100 包全同步 0 失败、5% 丢包成功率 100%、二次同步 0 chunk（`tools/package_builder.py:167-185`、`tools/sync_client.py:162-190`、`tools/test_sync_e2e.py`）。
2. **治理缺口最有讲点**：存在两套规则相反、且都不是门禁的校验器（空答案/空 summary/额外字段结论相反，无 CI 与预提交），AI 生成与教师编辑器都能绕过校验直写线上内容目录。
3. **分发侧完整性/安全需补**：manifest 无签名、`catalog.generated_at` 是随机值（`package_builder.py:296`），且 sync_server 未净化路径——实测可越权读取服务根外文件。
