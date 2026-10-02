#!/usr/bin/env python3
"""
检索层自检脚本（P0 验收用）。

检查项：
  1. 集合发现：库中集合的 subject / grade / 向量空间 / 块数
  2. 学科路由：每个 GUI 学科键实际选中的集合（原实现除 Science 外全部返回空）
  3. 召回验证：样例问题的候选池规模、证据分与书目溯源
  4. 零证据拒答：库外问题是否被正确拒答
  5. 缓存统计

用法（需完整 ML 依赖环境）：
    python scripts/check_retrieval.py
    python scripts/check_retrieval.py --db mati_data/chroma_db
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GUI_SUBJECTS = ["Science", "Math", "Chinese", "English Grammar",
                "Computer Science", "Social Studies"]

PROBES = [
    ("Science", "10", "什么是向心力？"),
    ("Science", "10", "平抛运动的规律是什么？"),
    ("Science", "11", "静电场中场强的定义式是什么？"),
    ("Math", "10", "函数的奇偶性怎么判断？"),
    ("Math", "11", "导数的几何意义是什么？"),
    ("Chinese", "10", "《乡土中国》的核心观点是什么？"),
    ("Computer Science", "10", "什么是算法？"),
    ("Computer Science", "11", "数据结构中的栈和队列有什么区别？"),
]

OUT_OF_SCOPE = ("请介绍一下量子色动力学中的渐近自由现象，以及"
                "中世纪欧洲庄园经济的赋税制度。")


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mati_data/chroma_db")
    ap.add_argument("--no-llm", action="store_true", help="不加载大模型，仅验证检索")
    args = ap.parse_args()

    from system.rag.rag_retrieval_engine import RAGRetrievalEngine

    engine = RAGRetrievalEngine(
        chroma_db_path=args.db,
        load_llm=not args.no_llm,
    )

    # ---------------------------------------------------------- 1. 集合发现
    section("1. 集合发现")
    colls = engine._discover_collections()
    if not colls:
        print("  [FAIL] 未发现任何可用集合")
        return 1
    print(f"  {'集合':<34}{'学科':<18}{'年级':<6}")
    print("  " + "-" * 60)
    for c in colls:
        print(f"  {c['name']:<34}{c['subject']:<18}{c['grade']:<6}")
    print(f"  共 {len(colls)} 个集合，"
          f"学科：{sorted({c['subject'] for c in colls})}")

    # 向量空间核对
    try:
        # 复用引擎已持有的客户端：**同一进程内不要对同一路径建第二个
        # PersistentClient**，否则新建的那个会间歇性查询失败（静默零召回，
        # 实测 100% 复现）。见 RAGRetrievalEngine.__init__ 的说明。
        client = engine.chroma_client
        spaces = {c.name: (c.metadata or {}).get("hnsw:space")
                  for c in client.list_collections()}
        bad = {k: v for k, v in spaces.items() if v != "cosine"}
        if bad:
            print(f"  [WARN] 非 cosine 空间的集合：{bad}"
                  "（1 - distance 将不等于余弦相似度）")
        else:
            print("  [OK] 全部集合均为 cosine 空间")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] 向量空间核对跳过：{e}")

    # ---------------------------------------------------------- 2. 学科路由
    section("2. 学科路由（原实现除 Science 外全部返回空集合）")
    route_fail = []
    for subj in GUI_SUBJECTS:
        names = engine._get_relevant_collections(subj, "10")
        flag = "OK  " if names else "EMPTY"
        if not names:
            route_fail.append(subj)
        print(f"  [{flag}] {subj:<18} -> {names}")
    if route_fail:
        print(f"  [WARN] 以下学科未路由到集合：{route_fail}")

    # -------------------------------------------------- 2b. 细分学科收窄
    section("2b. 细分学科收窄（科学 = 物理 + 化学 + 生物，可只查单科）")
    for disc in ("物理", "化学", "生物", "physics", "chemistry"):
        names = engine._get_relevant_collections("Science", "10", disc)
        print(f"  Science + discipline={disc:<10} -> {names}")
    fam = engine._get_relevant_collections("Science", "10")
    print(f"  Science（不指定细分）        -> {fam}")

    # ---------------------------------------------------------- 3. 召回验证
    section("3. 召回验证（候选池 / 证据分 / 书目溯源）")
    for subj, grade, q in PROBES:
        res = engine.query(query_text=q, subject=subj, grade=grade)
        stats = res.get("context_stats", {})
        srcs = res.get("sources", [])
        books = []
        for m in srcs:
            label = m.get("book_title") or m.get("source") or "?"
            if label not in books:
                books.append(label)
        print(f"\n  Q[{subj}/{grade}] {q}")
        print(f"    集合     : {res.get('collections_used')}")
        print(f"    候选/入选: {stats.get('candidates')} -> {stats.get('selected')}"
              f"   token {stats.get('tokens_used')}/{stats.get('tokens_budget')}")
        print(f"    最高证据分: {res.get('best_evidence_score')}   "
              f"置信度: {res.get('confidence')}   类型: {res.get('type')}")
        for b in books[:3]:
            print(f"    依据     : {b}")
        assert res.get("answer"), "答案为空"

    # ---------------------------------------------------------- 4. 零证据拒答
    section("4. 零证据拒答（库外问题）")
    res = engine.query(query_text=OUT_OF_SCOPE, subject="Science", grade="10")
    print(f"  类型     : {res.get('type')}")
    print(f"  证据条数 : {len(res.get('context_used') or [])}")
    print(f"  置信度   : {res.get('confidence')}")
    print(f"  答案     : {str(res.get('answer'))[:110]}")

    # ---------------------------------------------------------- 5. 缓存
    section("5. 缓存统计")
    print(f"  {engine.cache.stats()}")

    section("自检完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
