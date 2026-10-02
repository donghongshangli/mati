#!/usr/bin/env python3
"""
Rebuild the ChromaDB index for the Chinese-optimized embedding model.

必须重建的场景（任一触发就要重建，因为 HNSW 不支持原地改度量/维度）：
  1. 嵌入模型变更（如 all-MiniLM-L6-v2 384 维 → bge-small-zh-v1.5 512 维），
     旧向量处于不同向量空间，不兼容；
  2. 向量空间度量变更（本次由 l2 改为 cosine，使 `1 - distance` 直接等于
     余弦相似度，消除「用 1-l2 距离冒充余弦」导致的阈值语义漂移）；
  3. 切分策略变更（本次修复了中文句子边界识别，切分结果整体变化）；
  4. 元数据 schema 变更（本次新增 grade/grade_span/book_id/content_type 等字段）。

This script:
  1. Backs up (or removes) the old ChromaDB directory.
  2. Re-ingests content with the new embedding model via scripts/ingest_content.py.
  3. Prints a verification summary (collection / subject / grade / chunk count).

Usage:
    python scripts/rebuild_index.py
    python scripts/rebuild_index.py --content textbooks notes scripts/data_collection/data/content
    python scripts/rebuild_index.py --db-path mati_data/chroma_db --no-backup
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def print_summary(db_path: str, client=None) -> None:
    """
    重建后自检：列出集合、子块/父块数与向量检索可用性。

    ⚠️ `client` 必须由调用方传入（摄取器已持有的那个）。同一进程内对同一路径
    再建一个 PersistentClient，会让**新建的那个**间歇性查询失败
    （`Error finding id` / `Nothing found on disk`，实测 100% 复现），
    自检会误报"索引损坏"。
    """
    try:
        if client is None:
            import chromadb
            client = chromadb.PersistentClient(path=db_path)
        colls = client.list_collections()
    except Exception as e:  # noqa: BLE001
        print(f"[rebuild] 自检跳过（无法打开索引：{e}）")
        return

    if not colls:
        print("[rebuild][WARN] 索引为空！请检查 content 目录与摄取日志。")
        return

    print("\n[rebuild] ===== 索引自检 =====")
    print(f"{'集合':<32}{'学科':<12}{'细分':<15}{'年级':<6}{'子块':>7}{'父块':>7}")
    print("-" * 86)
    total_child = total_parent = 0
    subjects, disciplines, grades = set(), set(), set()
    empty_collections = []
    for c in sorted(colls, key=lambda x: x.name):
        meta = getattr(c, "metadata", None) or {}
        n = c.count()
        # 分别统计子块（可召回）与父块（供生成）—— 父子索引后两者必须分开看
        n_child = n_parent = 0
        try:
            got = c.get(include=["metadatas"])
            for m in got.get("metadatas") or []:
                if (m or {}).get("doc_type") == "parent":
                    n_parent += 1
                else:
                    n_child += 1
        except Exception:  # noqa: BLE001
            n_child = n
        total_child += n_child
        total_parent += n_parent
        if n == 0:
            empty_collections.append(c.name)
        subj = str(meta.get("subject", "-"))
        disc = str(meta.get("discipline", "-"))
        grd = str(meta.get("grade", "-"))
        subjects.add(subj)
        disciplines.add(disc)
        grades.add(grd)
        print(f"{c.name:<32}{subj:<12}{disc:<15}{grd:<6}{n_child:>7}{n_parent:>7}")
    print("-" * 86)
    print(f"{'合计':<32}{len(subjects)} 学科{'':<5}{len(disciplines)} 细分"
          f"{'':<7}{'':<6}{total_child:>7}{total_parent:>7}")
    print(f"[rebuild] 学科家族：{sorted(subjects)}")
    print(f"[rebuild] 细分学科：{sorted(disciplines)}")
    print(f"[rebuild] 年级取值：{sorted(grades)}")
    # 空集合 = 上次重建的残留，会污染集合枚举与路由，必须报出来
    if empty_collections:
        print(f"[rebuild][WARN] 发现 {len(empty_collections)} 个空集合（建议删除）："
              f"{empty_collections}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild ChromaDB index for BAAI/bge-small-zh-v1.5 (512-dim)")
    parser.add_argument("--content", nargs="*", default=None,
                        help="Content directories to ingest (default: textbooks, "
                             "scripts/data_collection/data/content)")
    parser.add_argument("--db-path", default="mati_data/chroma_db",
                        help="ChromaDB directory (default: mati_data/chroma_db)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Delete the old index instead of backing it up")
    parser.add_argument("--ocr-mode", choices=["auto", "force", "never"], default="auto")
    parser.add_argument("--subject", default=None,
                        help="强制学科（覆盖按书名推断）")
    parser.add_argument("--grade", default=None,
                        help="强制年级（覆盖按书名推断），1–12")
    args = parser.parse_args()

    db = Path(args.db_path)

    # 1) 处理旧索引：**一律用「重命名」而不是「删除」**。
    #
    # 原实现会先 rmtree 同名备份目录再改名。两个问题：
    #   ① 在大目录上 rmtree 会被环境的批量删除保护拦下（实测 88 个文件即触发
    #      SAFE_DELETE_BULK_CONFIRM_REQUIRED），导致重建整体失败；
    #   ② 历史备份被静默覆盖 —— 旧索引是唯一的回滚手段，不该被悄悄丢掉。
    # 改为：已有备份就加时间戳挪到一边，即保留又多份。
    if db.exists():
        if args.no_backup:
            shutil.rmtree(db, ignore_errors=True)
            print(f"[rebuild] Removed old index: {db}")
        else:
            backup = db.with_name(db.name + "_backup_old_embedding")
            if backup.exists():
                import time as _t
                stamped = db.with_name(
                    f"{backup.name}_{_t.strftime('%Y%m%d_%H%M%S')}")
                backup.rename(stamped)
                print(f"[rebuild] 已有备份先挪走（保留多份可回滚）：{stamped.name}")
            db.rename(backup)
            print(f"[rebuild] Backed up old index to: {backup}")
    else:
        print(f"[rebuild] No existing index at {db}; creating fresh.")

    # 2) Re-ingest with the new embedding model.
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from scripts.ingest_content import UniversalContentIngester

    ingester = UniversalContentIngester(str(db), args.ocr_mode,
                                        force_subject=args.subject,
                                        force_grade=args.grade)

    if args.content:
        dirs = args.content
    else:
        # 注意：`notes/` 目前只存放项目开发笔记（非高中教学内容），
        # 因此不列入默认摄取范围；教师笔记就位后可用 --content 显式指定。
        dirs = ["textbooks", "scripts/data_collection/data/content"]

    from scripts.ingest_content import format_ingest_summary

    total_ok = total_found = 0
    for d in dirs:
        p = project_root / d if not Path(d).is_absolute() else Path(d)
        if p.exists():
            summary = ingester.ingest_directory(str(p))
            print(format_ingest_summary(summary))
            total_ok += summary["ok"]
            total_found += summary["found"]
        else:
            print(f"[rebuild] Skipping missing directory: {d}")

    # 复用摄取器已持有的客户端（同进程内不能再建第二个，见 print_summary 说明）
    print_summary(str(db), client=ingester.client)

    # ⚠️ 关键：**先释放摄取器的客户端再校验**。
    # verify() 每轮都会新建 PersistentClient，若此时摄取器的客户端仍存活，
    # 同进程内就出现了两个指向同一路径的客户端 —— 会让新客户端的查询间歇性失败，
    # 甚至让 HNSW 索引写不完整（实测 chemistry 集合的 data_level0.bin 只剩连接表）。
    ingester.client = None
    ingester.model = None
    del ingester
    import gc
    gc.collect()

    # 关键收尾：刚写完的索引可能有个别集合的 HNSW 尚未完整落盘，
    # 表现为整个集合静默零召回（详见 scripts/verify_index.py 的说明）。
    # 这里多轮重开客户端做探针，触发 Chroma 自愈，确保重建产物立刻可用。
    print("\n[rebuild] ===== 索引健康校验（自愈重试）=====")
    from scripts.verify_index import verify
    index_healthy = verify(str(db), rounds=3, delay=1.5, repair=True)
    if not index_healthy:
        print("[rebuild][ERROR] 索引校验未通过 —— 请重新运行本脚本，"
              "或检查磁盘空间/权限。")
        return 3

    print("\n[rebuild] Index rebuilt with BAAI/bge-small-zh-v1.5 (512-dim), cosine space.")
    if total_found and not total_ok:
        print("[rebuild][ERROR] 没有任何文件成功入库，索引可能是空的！")
        return 2
    print("[rebuild] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
