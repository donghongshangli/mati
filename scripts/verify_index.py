#!/usr/bin/env python3
"""
索引健康校验（可重复运行、自愈）。

为什么需要它
──────────
实测发现：**刚重建完的索引可能没有完整落盘**。重建进程退出时，个别集合的
HNSW 段文件仍是半成品（例如 `neb_chemistry_grade_10` 的 `data_level0.bin`
只有 226K，而同规模的集合约 1800K，且缺少 `index_metadata.pickle`）。
这种状态下该集合的向量查询会抛

    Error creating hnsw segment reader: Nothing found on disk
    Error executing plan: Internal error: Error finding id

**关键点：这不是索引损坏，而是"未完成"**。Chroma 在后续进程打开该集合时
会检测到并重建它，之后再打开就完全正常 —— 也就是说索引会**自愈**，
但只有被人访问过才会自愈。

如果不做这一步，后果是**静默零召回**：整个集合查不到东西，上层只能看到
「没检索到相关内容」，完全看不出是索引问题。实测这会让评测的 Recall@10
在 65%~94% 之间随机波动（同一份代码、同一个索引，只是每次运行的运气不同）。

本脚本做的事
──────────
1. 遍历所有集合，逐一对向量查询做一次探针（带 `doc_type != parent` 过滤，
   与真实检索路径一致）；
2. 失败的集合等一小会儿再试（给 Chroma 留出重建时间）；
3. 多轮之后仍失败的集合会被明确报出来，退出码非 0。

用法
────
    python scripts/verify_index.py
    python scripts/verify_index.py --rounds 4 --db mati_data/chroma_db
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("verify_index")


def probe(client, dim: int = 512):
    """
    对每个集合做一次向量查询探针。

    Returns:
        (ok_names, failed)   failed = [(集合名, 原因), ...]
    """
    # 探针向量必须是**非零**的：余弦空间下零向量未定义（0/0），
    # 用它自检会把「查询失败」和「向量非法」混在一起。
    vec = [0.05] * dim
    ok, failed = [], []
    for col in sorted(client.list_collections(), key=lambda c: c.name):
        if col.count() == 0:
            failed.append((col.name, "集合为空"))
            continue
        try:
            res = col.query(
                query_embeddings=[vec], n_results=1,
                where={"doc_type": {"$ne": "parent"}},
            )
            ids = (res.get("ids") or [[]])[0]
            if not ids:
                failed.append((col.name, "查询返回空结果"))
            else:
                ok.append(col.name)
        except Exception as e:  # noqa: BLE001
            failed.append((col.name, str(e)[:110]))
    return ok, failed


def repair_collection(client, name: str) -> bool:
    """
    重建单个集合的向量索引：读出全部记录 → 删集合 → 原样重建 → 回写。

    为什么需要它
    ───────────
    实测 `neb_chemistry_grade_10`（878 条向量）的 `data_level0.bin` 只有 226K，
    而同等规模的集合约 2300K —— 也就是说它的 HNSW 索引**在磁盘上基本是空的**
    （226K 只是空索引的基线结构）。这种集合每次查询都要在内存里重新构建索引，
    构建失败时整个集合就查不到东西（间歇性零召回）。

    Chroma 不会自动持久化这种重建结果，所以只能主动重写记录来强制建一次干净的索引。
    """
    try:
        col = client.get_collection(name)
        meta = getattr(col, "metadata", None) or {}
        n = col.count()
        got = col.get(include=["embeddings", "documents", "metadatas"])
        ids = got.get("ids") or []
        embs = got.get("embeddings")
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        if not ids or embs is None:
            print(f"         [修复] {name}: 读不到记录（{len(ids)} 条），跳过")
            return False
        print(f"         [修复] {name}: 读出 {len(ids)} 条，正在重写以重建索引…")
        client.delete_collection(name)
        new_col = client.create_collection(name=name, metadata=dict(meta))
        B = 100
        for i in range(0, len(ids), B):
            j = min(i + B, len(ids))
            new_col.add(ids=ids[i:j],
                        embeddings=[list(e) for e in embs[i:j]],
                        documents=docs[i:j],
                        metadatas=metas[i:j] or None)
        print(f"         [修复] {name}: 已回写 {len(ids)} 条")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"         [修复] {name}: 失败 —— {str(e)[:110]}")
        return False


def verify(db_path: str, rounds: int = 3, delay: float = 1.0, dim: int = 512,
           repair: bool = False) -> bool:
    import chromadb

    for r in range(1, max(1, rounds) + 1):
        # 每轮都**重新打开客户端**：自愈发生在"打开集合"时，
        # 复用同一个客户端不会触发重建。
        client = chromadb.PersistentClient(path=db_path)
        ok, failed = probe(client, dim)
        print(f"[verify] 第 {r}/{rounds} 轮：{len(ok)} 个集合正常，"
              f"{len(failed)} 个异常")
        for name, reason in failed:
            print(f"         [异常] {name}: {reason}")
        if not failed:
            print("[verify] 索引健康：全部集合向量检索可用 ✓")
            return True
        # 多轮仍失败 -> 主动重写记录强制重建索引（比"再试一次"更根本）
        if repair and r == max(1, rounds) - 1:
            print("[verify] 仍有异常，开始修复（重写记录以重建索引）…")
            for name, _ in list(failed):
                repair_collection(client, name)
        del client
        if r < rounds:
            time.sleep(delay)

    print(f"[verify][ERROR] {len(failed)} 个集合在多轮校验后仍不可用："
          f"{[n for n, _ in failed]}")
    print("[verify] 若持续失败，请重新摄取这些集合对应的源文件"
          "（重建索引后再跑一次本脚本）。")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Mati 索引健康校验（自愈重试）")
    ap.add_argument("--db", default="mati_data/chroma_db")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=512, help="嵌入维度（bge-small-zh-v1.5 = 512）")
    ap.add_argument("--repair", action="store_true",
                    help="多轮仍失败的集合，主动重写其记录以强制重建向量索引")
    args = ap.parse_args()
    return 0 if verify(args.db, args.rounds, args.delay, args.dim, args.repair) else 1


if __name__ == "__main__":
    sys.exit(main())
