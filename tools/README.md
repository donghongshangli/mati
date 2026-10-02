# Mati 资源打包与低带宽同步工具

一套**独立于主程序**的轻量化工具链，用于构建「AI 教育资源包」并在低带宽网络下分发：

- `package_builder.py` — 把内容打包成 ≤50MB 的资源包（含 `manifest.json`，逐 chunk 记录 id/hash/size/compressed_size/collection）。
- **真实压缩**：chunk 以 **zlib**（标准库，零额外依赖）压缩后存储/传输，哈希基于**未压缩内容**（保证内容去重与校验语义不变）。真实教材内容实测压缩约 **3.5x**（节省约 70%）。
- `sync_server.py` — 标准库实现的轻量 HTTP 服务（catalog / manifest / chunk(Range) / diff）。
- `sync_client.py` — 断点续传、增量同步、逐块 SHA-256 校验、并发下载与指数退避重试。
- `ingest_content.py --ingest-from-manifest` — 把同步下来的包增量导入 ChromaDB（保留原 collection/id，兼容现有流程）。
- `test_sync_e2e.py` — 端到端验证：生成 100 个包、模拟 5% 丢包、验证成功率 ≥95%。
- `test_ingest_manifest.py` — 验证 `--ingest-from-manifest --dry-run`。

> ⚠️ `--demo` 生成的包是**合成测试数据**，仅用于验证工具链，不代表真实教学内容。

## 为什么这样做

- **不破坏现有系统**：所有新代码在 `tools/`，主程序默认行为完全不变；对 `scripts/ingest_content.py` 的改动仅是「新增可选参数 + 惰性加载」，正常路径等价。
- **轻依赖**：工具链只用 Python 标准库，可在 4GB 低配机上运行。
- **指标可验证**：丢包、断点续传、增量、50MB 限制都有自动化测试覆盖。

## 快速开始

```powershell
# 1) 打包（用 demo 生成 100 个示例包；或用 --src 打包真实内容）
python tools/package_builder.py --demo 100 --out dist
# python tools/package_builder.py --src scripts/data_collection/data/content --out dist

# 2) 启动同步服务器
python tools/sync_server.py --root dist --host 127.0.0.1 --port 8000

# 3) 在另一终端：客户端同步全部（支持断点续传/增量）
python tools/sync_client.py --server http://127.0.0.1:8000 --dest mati_data/content

# 4) 校验后导入知识库（dry-run 先行；真实导入需要 chromadb + sentence-transformers）
python scripts/ingest_content.py --ingest-from-manifest mati_data/content --dry-run
python scripts/ingest_content.py --ingest-from-manifest mati_data/content
```

## 端到端测试

```powershell
python tools/test_sync_e2e.py      # 100 包 + 5% 丢包 + 断点续传 + 增量
python tools/test_ingest_manifest.py
```

期望输出：`[4] Under simulated 5% loss: success rate 100.0% (>= 95%)` 与 `ALL END-TO-END CHECKS PASSED`。

## 包结构（dist/）

```
dist/
├── catalog.json                  # 总目录：包 id -> 版本/大小/chunks/collections
└── packages/
    └── <package_id>/
        ├── manifest.json         # 逐 chunk: {id, hash(sha256 of raw), size, compressed_size, compression, collection, metadata}
        └── chunks/
            └── <sha256>.chunk    # 每个内容单元一个文件（zlib 压缩后的内容寻址载荷）
```

## 如何满足目标指标

| 指标 | 实现机制 |
|---|---|
| 资源包 ≤50MB | `--max-package-mb`（默认 50），按**压缩后字节**逐包校验 |
| 轻量化（压缩/去重） | zlib 压缩存储/传输（实测 ~3.5x，省 ~70%）；哈希基于原文 → 相同内容自动去重 |
| ≥100 个资源包 | `--demo 100` 生成测试包；真实内容按「学科×年级×单元」拆分 |
| 断点续传 | 每个 chunk 用 HTTP Range 分 64KB 块续传（`.part` 文件），不从头重来 |
| 增量同步 | 内容寻址（sha256），客户端只拉缺失 chunk（`/diff` 接口） |
| 丢包率≤5% 时端到端成功率≥95% | 块级重试（指数退避）+ 逐块哈希校验；测试模拟 5% 丢包成功率 100% |

## 常用 CLI 参数

**package_builder.py**
- `--src DIR` 打包真实内容（txt/md/jsonl/json）
- `--demo N` 生成 N 个合成测试包
- `--max-package-mb N` 单包上限（默认 50）
- `--validate dist` 校验已有 dist（逐 chunk 哈希）

**sync_client.py**
- `--package ID` 只同步单个包（默认全部）
- `--concurrency N` 并发下载（默认 4）
- `--max-retries N` 每请求最大重试（默认 5）
- `--simulate-packet-loss PCT` 模拟丢包（**仅测试**）

**ingest_content.py（新增）**
- `--ingest-from-manifest DIR` 导入 DIR 下 `packages/*/manifest.json`
- `--dry-run` 只校验不写库

## 兼容性与注意事项

- 包内 chunk 的 `id`/`collection` 与 `ingest_content.py` 现有命名规则一致，导入为**增量 upsert**，不会改写已有索引。
- 更换 embedding 模型会改变向量空间，需要重建索引（本工具不自动覆盖旧索引）。
- `--simulate-packet-loss` 与 `--demo` 仅供测试/演示，生产环境勿用。
- 模型（Phi-1.5 GGUF / 嵌入模型）是运行时依赖，**一次安装**，不属于 ≤50MB 内容包范畴。

## 目录

- `common.py` — 共享常量/哈希/manifest 读写/分块
- `package_builder.py` — 打包器
- `sync_server.py` — 同步服务器
- `sync_client.py` — 同步客户端
- `test_sync_e2e.py` — 端到端测试
- `test_ingest_manifest.py` — 导入 dry-run 测试
- `requirements-sync.txt` — 可选依赖说明
