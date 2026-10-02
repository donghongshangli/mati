#!/usr/bin/env python3
"""
Mati RAG 评测脚本（分阶段）。

为什么需要它
──────────
P0 之前项目**没有任何评测基线**，所有参数改动都只能靠"感觉变好了"判断。
本脚本把链路的 6 个阶段拆开，每个阶段单独可测，这样：

  · 改一个权重（如来源权重）不需要重跑全链路，几秒内拿到指标对比；
  · 失败可归因到具体阶段（是没路由到？还是召回漏了？还是被截断了？）；
  · 支持消融开关（--dense-only / --no-rewrite），量化每个组件的真实贡献。

指标体系（方案 §5.1）
────────────────────
检索层  Recall@5 / Recall@10、MRR@10、路由准确率、零召回率
证据层  证据保留率（送入模型的 token / 精排池 token）
生成层  拒答正确率、误拒率（负样本/正样本）
系统层  置信度区分度（正负样本置信度均值差）

用法
────
    # 纯检索评测（不加载 1GB 模型，秒级）
    python scripts/eval_rag.py

    # 消融：关掉混合检索（等价 P0 的纯稠密行为）
    python scripts/eval_rag.py --dense-only

    # 关掉查询改写
    python scripts/eval_rag.py --no-rewrite

    # 三条基线一次跑完并对比
    python scripts/eval_rag.py --ablation

    # 输出逐题明细（失败归因用）
    python scripts/eval_rag.py --json eval_detail.json --verbose
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_rag")

DEFAULT_EVAL_SET = "tests/eval/rag_eval_set.jsonl"


# ─────────────────────────────────────────────────────────── 数据集与映射
def load_eval_set(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
    return items


def build_book_to_collection(client, allowed) -> dict:
    """
    扫描索引，构建 book_id -> 集合名 映射。

    用于判断「路由准确率」：gold 教材所在集合是否被选中。
    只取每集合前 2000 条元数据，库规模当前的量级完全够用。
    """
    mapping = {}
    for col in client.list_collections():
        if not allowed(col.name):
            continue
        got = col.get(include=["metadatas"], limit=2000)
        for m in got.get("metadatas") or []:
            bid = (m or {}).get("book_id")
            if bid:
                mapping.setdefault(bid, set()).add(col.name)
    return mapping


# ─────────────────────────────────────────────────────────── 单题评测
def eval_one(engine, item, book2coll) -> dict:
    """跑一题，返回该题的阶段指标。"""
    t0 = time.time()
    res = engine.query(
        query_text=item["q"],
        subject=item.get("subject", "science"),
        grade=item.get("grade"),
        discipline=item.get("discipline"),
        already_normalized=False,
        capture_stages=True,
    )
    elapsed = time.time() - t0
    stages = res.get("stage_debug") or {}
    gold = set(item.get("gold_books") or [])
    is_negative = item.get("type") == "negative"

    pool = stages.get("candidate_pool") or []
    reranked = stages.get("reranked") or []
    evidence = stages.get("evidence") or {}
    routed = set(stages.get("routed_collections") or [])

    def first_gold_rank(seq, key="book_id", limit=None):
        seq = seq[:limit] if limit else seq
        for i, c in enumerate(seq, start=1):
            if c.get("book_id") in gold:
                return i
        return None

    # 路由准确率：gold 教材所在的任一集合是否被选中（负样本不适用）
    route_hit = None
    if gold and not is_negative:
        gold_colls = set()
        for b in gold:
            gold_colls |= book2coll.get(b, set())
        route_hit = bool(gold_colls & routed) if gold_colls else None

    rec5 = first_gold_rank(pool, limit=5) is not None if gold else None
    rec10 = first_gold_rank(pool, limit=10) is not None if gold else None
    rank10 = first_gold_rank(pool, limit=10) if gold else None
    rec_ev = bool(gold & set(evidence.get("book_ids") or [])) if gold else None

    n_ev = evidence.get("n", 0)
    zero_recall = (n_ev == 0)
    pool_tokens = evidence.get("pool_tokens", 0) or 0
    used_tokens = evidence.get("tokens_used", 0) or 0
    retention = (used_tokens / pool_tokens) if pool_tokens > 0 else None

    data_fail = (res.get("type") == "no_evidence") or zero_recall

    # ── P2 指标（方案 §3.4 步骤 3.4 / §5.1）
    # 引用：只统计**模型真的写了标记**的那些题。没写标记不计入分母 ——
    # 1.5B 模型不遵循编号指令是常态，把它算成失败会得到一个没有意义的数字。
    cc = res.get("citation_check") or {}
    n_cit = cc.get("n_cited", 0) or 0
    n_oob = cc.get("n_out_of_range", 0) or 0
    cite_acc = (cc.get("n_valid", 0) / n_cit) if n_cit else None
    # 治理过滤：证据里本不该出现的噪声块（目录/封面/已下架）有多少漏进来了
    noise_in_ev = evidence.get("noise_in_evidence", 0) or 0
    archived_in_ev = evidence.get("archived_in_evidence", 0) or 0

    return {
        "id": item["id"],
        "q": item["q"],
        "type": item.get("type", "concept"),
        "register": item.get("register", "textbook"),
        "cross_grade": bool(item.get("cross_grade")),
        "is_negative": is_negative,
        "gold_books": sorted(gold),
        "routed": sorted(routed),
        "route_hit": route_hit,
        "recall@5": rec5,
        "recall@10": rec10,
        "mrr_rank@10": rank10,
        "recall@evidence": rec_ev,
        "pool_n": len(pool),
        "rerank_n": len(reranked),
        "evidence_n": n_ev,
        "zero_recall": zero_recall,
        "retention": retention,
        "tokens_used": used_tokens,
        "tokens_budget": evidence.get("tokens_budget", 0),
        "pool_tokens": pool_tokens,
        "refused": bool(data_fail),
        "confidence": res.get("confidence", 0.0),
        "best_evidence_score": res.get("best_evidence_score", 0.0),
        "query_coverage": evidence.get("best_query_coverage"),
        "best_dense": evidence.get("best_dense"),
        "refuse_threshold": evidence.get("refuse_threshold"),
        # P2
        "n_citations": n_cit,
        "n_cite_invalid": n_oob,
        "citation_accuracy": cite_acc,
        "n_citations_available": len(res.get("citations") or []),
        "noise_in_evidence": noise_in_ev,
        "archived_in_evidence": archived_in_ev,
        "meta_completeness": evidence.get("meta_completeness"),
        "variants": (stages.get("rewrite") or {}).get("variants", []),
        "bm25_n": (stages.get("retrieve") or {}).get("bm25_n", 0),
        "dense_n": (stages.get("retrieve") or {}).get("dense_n", 0),
        "parents": stages.get("parents", {}),
        "elapsed_s": round(elapsed, 3),
        "error": res.get("error", ""),
    }


# ─────────────────────────────────────────────────────────── 汇总
def summarize(rows) -> dict:
    pos = [r for r in rows if not r["is_negative"]]
    neg = [r for r in rows if r["is_negative"]]

    def rate(seq, key, predicate=lambda v: v is True):
        vals = [r[key] for r in seq if r.get(key) is not None]
        if not vals:
            return None
        return sum(1 for v in vals if predicate(v)) / len(vals)

    def mean(seq, key):
        vals = [r[key] for r in seq if r.get(key) is not None]
        return statistics.fmean(vals) if vals else None

    ranks = [r["mrr_rank@10"] for r in pos if r.get("mrr_rank@10")]
    mrr = statistics.fmean([1.0 / k for k in ranks]) if ranks else 0.0

    pos_refused = sum(1 for r in pos if r["refused"])
    neg_refused = sum(1 for r in neg if r["refused"])
    conf_pos = [r["confidence"] for r in pos if r.get("confidence") is not None]
    conf_neg = [r["confidence"] for r in neg if r.get("confidence") is not None]

    # 预算利用率：入选证据 token / 证据预算。
    # 与「证据保留率」是两个不同的量，必须分开看：
    #   保留率 = 入选 token / **精排池** token —— 精排池大小由 rerank_k 决定（固定 10 条），
    #            而预算只有 2000 token，父块约 1000 token，因此保留率**结构性**上限约 20%。
    #            方案 §5.1 写的「保留率 ≥ 0.70」在 rerank_k=10 的设定下不可达：
    #            要达标必须把精排池裁到与预算同量级（rerank_k≈2），代价是失去选择余地。
    #   利用率 = 入选 token / **预算** —— 衡量"给模型的证据窗口有没有被用满"，
    #            这才是真正影响答案质量的量。
    util = [
        (r["tokens_used"] / r["tokens_budget"])
        for r in pos
        if r.get("tokens_used") and r.get("tokens_budget")
    ]

    return {
        "n_pos": len(pos),
        "n_neg": len(neg),
        "n_error": sum(1 for r in rows if r.get("error")),
        "n_valid": sum(1 for r in rows if not r.get("error")),
        "route_acc": rate(pos, "route_hit"),
        "recall@5": rate(pos, "recall@5"),
        "recall@10": rate(pos, "recall@10"),
        "mrr@10": mrr,
        "recall@evidence": rate(pos, "recall@evidence"),
        "zero_recall_rate": rate(pos, "zero_recall", lambda v: v is True),
        "retention": mean(pos, "retention"),
        "budget_utilization": statistics.fmean(util) if util else None,
        "refuse_correct_rate": (neg_refused / len(neg)) if neg else None,
        "false_refuse_rate": (pos_refused / len(pos)) if pos else None,
        "confidence_pos": statistics.fmean(conf_pos) if conf_pos else None,
        "confidence_neg": statistics.fmean(conf_neg) if conf_neg else None,
        "confidence_gap": (statistics.fmean(conf_pos) - statistics.fmean(conf_neg))
                          if conf_pos and conf_neg else None,
        "best_evidence_pos": mean(pos, "best_evidence_score"),
        "best_evidence_neg": mean(neg, "best_evidence_score"),
        # 证据分区分度：不依赖生成模型也能测的「正负样本可分性」。
        # 与 confidence_gap 的区别是它不需要加载 1GB 模型，适合快速回归。
        "evidence_gap": (mean(pos, "best_evidence_score") or 0.0)
                        - (mean(neg, "best_evidence_score") or 0.0),
        # 按语体拆分：口语化提问才是「同义词改写」的用武之地，
        # 只报总体会把这个收益稀释掉（题面用标准术语时改写本就无事可做）。
        "recall@10_textbook": rate([r for r in pos if r.get("register") != "colloquial"],
                                   "recall@10"),
        "recall@10_colloquial": rate([r for r in pos if r.get("register") == "colloquial"],
                                     "recall@10"),
        # 跨年级题：提问年级与 gold 教材所属年级不一致（如高二学生问必修三内容）。
        # 这一项专门衡量「年级软过滤」是否真的解决了跨年级内容的假拒答。
        "recall@10_cross_grade": rate([r for r in pos if r.get("cross_grade")], "recall@10"),
        # ── P2 指标（方案 §3.4 步骤 3.3 / 3.4）
        # 引用准确率：分母只算**模型真的写了标记**的题。
        # 「模型没写标记」和「模型写了但编号错的」必须分开报 ——
        # 前者是 1.5B 模型的指令遵循问题，后者才是可信度问题。
        "citation_acc": mean(rows, "citation_accuracy"),
        "cite_coverage": (sum(1 for r in rows if r.get("n_citations", 0) > 0) / len(rows)
                          if rows else None),
        "cite_invalid_total": sum(r.get("n_cite_invalid", 0) or 0 for r in rows),
        # 治理过滤漏网率：证据里混进了多少本该被挡掉的目录/封面/下架块。
        # 正常应恒为 0；>0 说明 where 下推或 Python 兜底有漏洞。
        "noise_leak": sum(r.get("noise_in_evidence", 0) or 0 for r in rows),
        "archived_leak": sum(r.get("archived_in_evidence", 0) or 0 for r in rows),
        # 方案步骤 3.1 验收线：元数据完整率 ≥ 95%
        "meta_completeness": mean(rows, "meta_completeness"),
        "avg_latency_s": mean(rows, "elapsed_s"),
    }


def coverage_sweep(rows) -> list:
    """
    查询覆盖率阈值的标定扫描。

    拒答正确率（负样本被挡）与误拒率（正样本被误挡）是一对矛盾，
    阈值必须在**评测集上反推**而不是拍脑袋 —— 这正是本脚本存在的意义。
    返回 [(阈值, 拒答正确率, 误拒率, 正样本平均覆盖, 负样本平均覆盖), ...]
    """
    pos = [r for r in rows if not r["is_negative"] and r.get("query_coverage") is not None]
    neg = [r for r in rows if r["is_negative"] and r.get("query_coverage") is not None]
    out = []
    for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        ok = sum(1 for r in neg if (r.get("query_coverage") or 0) < th) / len(neg) if neg else 0
        fp = sum(1 for r in pos if (r.get("query_coverage") or 0) < th) / len(pos) if pos else 0
        out.append((th, ok, fp))
    return out


def fmt(v, digits=3, pct=False):
    if v is None:
        return "  n/a"
    if pct:
        return f"{v*100:5.1f}%"
    return f"{v:.{digits}f}"


def print_report(tag: str, s: dict, rows) -> None:
    print(f"\n{'='*72}\n【{tag}】\n{'='*72}")
    print(f"  样本：正 {s['n_pos']} / 负 {s['n_neg']}")
    if s.get("n_error"):
        print(f"  ⚠️  {s['n_error']} 题评测异常（无检索结果），"
              f"下方指标仅基于 {s['n_valid']} 题 —— 不可用于结论")
    print(f"  路由准确率      : {fmt(s['route_acc'], pct=True)}")
    print(f"  Recall@5        : {fmt(s['recall@5'], pct=True)}")
    print(f"  Recall@10       : {fmt(s['recall@10'], pct=True)}")
    print(f"  MRR@10          : {fmt(s['mrr@10'])}")
    print(f"  Recall@证据     : {fmt(s['recall@evidence'], pct=True)}")
    print(f"  零召回率        : {fmt(s['zero_recall_rate'], pct=True)}")
    print(f"  证据保留率      : {fmt(s['retention'], pct=True)}"
          f"（入选/精排池；受 rerank_k 限制，见脚本内说明）")
    print(f"  证据预算利用率  : {fmt(s['budget_utilization'], pct=True)}"
          f"（入选/预算；衡量证据窗口是否用满）")
    print(f"  拒答正确率(负)  : {fmt(s['refuse_correct_rate'], pct=True)}")
    print(f"  误拒率(正)      : {fmt(s['false_refuse_rate'], pct=True)}")
    print(f"  置信度 正/负    : {fmt(s['confidence_pos'])} / {fmt(s['confidence_neg'])}"
          f"   (区分度 {fmt(s['confidence_gap'])})")
    print(f"  证据分 正/负    : {fmt(s['best_evidence_pos'])} / {fmt(s['best_evidence_neg'])}"
          f"   (区分度 {fmt(s['evidence_gap'])})")
    print(f"  Recall@10 分语体: 书面 {fmt(s['recall@10_textbook'], pct=True)}  "
          f"口语 {fmt(s['recall@10_colloquial'], pct=True)}")
    print(f"  Recall@10 跨年级: {fmt(s['recall@10_cross_grade'], pct=True)}")

    # ── P2：知识治理与可信生成
    leak = (s.get("noise_leak", 0) or 0) + (s.get("archived_leak", 0) or 0)
    print(f"\n  ── P2 知识治理与可信生成 ──")
    print(f"  引用标注率      : {fmt(s.get('cite_coverage'), pct=True)}"
          f"（模型主动写了 [n] 的题占比）")
    print(f"  引用准确率      : {fmt(s.get('citation_acc'), pct=True)}"
          f"（仅统计写了标记的题；越界 {s.get('cite_invalid_total', 0)} 处）")
    if s.get("cite_coverage") is not None and s["cite_coverage"] < 0.5:
        print(f"     ⚠️ 标注率偏低：1.5B 模型不遵循编号指令属预期，"
              f"编号本身由检索层生成，不会因此产生假来源")
    print(f"  治理过滤漏网    : {leak} 块"
          f"（目录/封面 {s.get('noise_leak', 0)}，已下架 {s.get('archived_leak', 0)}）"
          + ("  ✓" if leak == 0 else "  ← 应恒为 0，说明过滤有漏洞"))
    mc = s.get("meta_completeness")
    if mc is not None:
        print(f"  元数据完整率    : {fmt(mc, pct=True)}"
              f"（方案验收线 ≥95%{'' if mc >= 0.95 else '  ← 未达标'}）")
    print(f"  平均单题耗时    : {fmt(s['avg_latency_s'], digits=2)} s")

    sw = coverage_sweep(rows)
    if sw:
        print(f"\n  ── 查询覆盖率阈值标定（拒答正确率 vs 误拒率）──")
        print(f"     {'阈值':>6}{'拒答正确率':>12}{'误拒率':>10}")
        for th, ok, fp in sw:
            mark = "  ←" if fp <= 0.10 and ok >= 0.60 else ""
            print(f"     {th:>6.2f}{ok*100:>11.1f}%{fp*100:>9.1f}%{mark}")

    failed = [r for r in rows if not r["is_negative"] and r.get("recall@10") is False]
    if failed:
        print(f"\n  ⚠️ Recall@10 未命中的 {len(failed)} 题（前 12 条）：")
        for r in failed[:12]:
            print(f"    [{r['id']}] {r['q'][:34]:<36} gold={r['gold_books']} "
                  f"pool={r['pool_n']} ev={r['evidence_n']} 路由={len(r['routed'])}")


# ─────────────────────────────────────────────────────────── 主流程
def run_once(args, items, book2coll, tag: str, client, **engine_kwargs):
    from system.rag.rag_retrieval_engine import RAGRetrievalEngine

    engine = RAGRetrievalEngine(
        chroma_db_path=args.db,
        load_llm=False,               # 评测检索层不需要 1GB 生成模型
        # 复用外部客户端：**同一进程内不要对同一路径建第二个 PersistentClient**，
        # 否则第二个客户端会有 1–2 个集合间歇性查询失败（静默零召回，
        # 实测 100% 复现），评测结果会随机波动，完全不可比。
        chroma_client=client,
        **engine_kwargs,
    )
    # 评测流量不是生产流量：关掉「检索失败落盘」，否则每跑一次评测就会往
    # mati_data/retrieval_failures.jsonl 里灌几百条合成失败样本，
    # 污染那张本应用于「挖掘真实失败模式、反哺同义词表」的语料
    # （实测一次评测即写入 383 条，全部是评测集原题）。
    try:
        engine.rewriter.failure_log = None
    except AttributeError:
        pass
    rows = []
    for i, item in enumerate(items, 1):
        try:
            rows.append(eval_one(engine, item, book2coll))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"题目 {item['id']} 评测异常：{e}")
            # 必须**保留该题原有的正负样本属性**。
            # 曾在这里硬编码 is_negative=False —— 结果报错的负样本被算成正样本，
            # 样本构成被悄悄改掉：78 题里 35 题报错时，「正 71 / 负 7」
            # 与「Recall@10 100%」都是在残缺样本上算出来的假数字。
            rows.append({
                "id": item["id"], "q": item["q"],
                "type": item.get("type", "concept"),
                "register": item.get("register", "textbook"),
                "cross_grade": bool(item.get("cross_grade")),
                "is_negative": item.get("type") == "negative",
                "error": str(e), "elapsed_s": 0.0, "refused": False,
            })
        if args.verbose and i % 10 == 0:
            print(f"  ... {i}/{len(items)}", flush=True)
    s = summarize(rows)
    print_report(tag, s, rows)
    return s, rows


def run_combo(args, items, book2coll, tag, client, repeats=1, **engine_kwargs):
    """
    跑一组配置，可重复多次并取指标均值。

    为什么要支持多次重复（重要）
    ──────────────────────────
    本机 Chroma 1.4.0 的 HNSW 索引惰性落盘，个别集合的持久化索引可能是半成品，
    导致**同一份代码、同一个索引，单次运行之间会有 ±3~5 个百分点的波动**
    （表现为某个集合整体零召回，且每次"中招"的集合可能不同）。
    拿单次数字下结论会把这种噪声当成"改好了"或"改坏了"。
    因此默认重复 2 次取均值，并额外报告**逐题翻转数**作为稳定性指标：
    翻转数为 0 才说明结论稳定可信。

    Returns:
        (mean_summary, first_run_rows, stability_info)
    """
    runs = []
    for i in range(max(1, repeats)):
        label = tag if repeats == 1 else f"{tag} · 第{i + 1}次"
        s, rows = run_once(args, items, book2coll, label, client, **engine_kwargs)
        runs.append((s, rows))

    numeric = {k for k, v in runs[0][0].items() if isinstance(v, (int, float))}
    mean = dict(runs[0][0])
    for k in numeric:
        vals = [s[k] for s, _ in runs if isinstance(s.get(k), (int, float))]
        if vals:
            mean[k] = statistics.fmean(vals)

    # 稳定性：以首个 run 为基准，统计后续 run 里 recall@10 / 证据条数翻转过多少题
    base = {r["id"]: (r.get("recall@10"), r.get("evidence_n")) for r in runs[0][1]}
    flips = 0
    for _, rows in runs[1:]:
        cur = {r["id"]: (r.get("recall@10"), r.get("evidence_n")) for r in rows}
        flips += sum(1 for i in base if base[i] != cur.get(i))
    stability = {"repeats": repeats, "item_flips": flips,
                 "flips_per_run": (flips / max(1, repeats - 1)) if repeats > 1 else 0}
    return mean, runs[0][1], stability


def main():
    ap = argparse.ArgumentParser(description="Mati RAG 分阶段评测")
    ap.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    ap.add_argument("--db", default="mati_data/chroma_db")
    ap.add_argument("--json", default=None, help="逐题明细输出路径")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dense-only", action="store_true",
                    help="关闭混合检索（等价 P0 纯稠密行为，用于 A/B）")
    ap.add_argument("--no-rewrite", action="store_true", help="关闭 Query 改写")
    ap.add_argument("--hard-grade", action="store_true",
                    help="启用硬过滤路由（默认 soft：同家族全部年级参与召回）")
    ap.add_argument("--ablation", action="store_true",
                    help="依次跑 A 纯稠密 / B 混合 / C 混合+改写 / D +软年级 并对比")
    ap.add_argument("--repeat", type=int, default=2,
                    help="每组配置重复运行次数（默认 2，取均值）。"
                         "本机 Chroma 的 HNSW 惰性落盘会让单次结果有 ±3~5pt 波动，"
                         "重复取值 + 逐题翻转数才能得到可信结论。")
    args = ap.parse_args()

    items = load_eval_set(args.eval_set)
    print(f"评测集：{args.eval_set}（{len(items)} 题，"
          f"正 {sum(1 for i in items if i.get('type') != 'negative')} / "
          f"负 {sum(1 for i in items if i.get('type') == 'negative')}）")

    import chromadb
    from system.rag.retrieval_config import RetrievalConfig
    cfg = RetrievalConfig.load()
    client = chromadb.PersistentClient(path=args.db)
    book2coll = build_book_to_collection(client, cfg.is_collection_allowed)
    print(f"索引：{len(client.list_collections())} 个集合，"
          f"{len(book2coll)} 本教材已建立教材→集合映射")

    results = {}
    if args.ablation:
        # 四组递进消融：每一组只在上一步基础上多打开一个组件，
        # 这样每个组件的边际贡献都能单独读出来。
        combos = [
            ("A 纯稠密+硬年级", dict(hybrid_enabled=False, rewrite_enabled=False,
                                     grade_mode="hard")),
            ("B +BM25混合", dict(hybrid_enabled=True, rewrite_enabled=False,
                                 grade_mode="hard")),
            ("C +Query改写", dict(hybrid_enabled=True, rewrite_enabled=True,
                                  grade_mode="hard")),
            ("D +软年级路由", dict(hybrid_enabled=True, rewrite_enabled=True,
                                   grade_mode="soft")),
        ]
    else:
        combos = [("评测结果", dict(hybrid_enabled=not args.dense_only,
                                  rewrite_enabled=not args.no_rewrite,
                                  grade_mode="hard" if args.hard_grade else "soft"))]

    for tag, kw in combos:
        s, rows, stability = run_combo(args, items, book2coll, tag, client,
                                       repeats=args.repeat, **kw)
        results[tag] = {"summary": s, "rows": rows, "stability": stability}
        print(f"  [稳定性] {tag}: 重复 {stability['repeats']} 次，"
              f"每次约 {stability['flips_per_run']:.1f} 题结果翻转"
              + ("（0 表示完全可复现）" if stability["repeats"] > 1 else
                 "（未重复；加 --repeat 2 可获得稳定性估计）"))

    if args.ablation:
        print(f"\n{'='*72}\n【消融对比】（每组重复 {args.repeat} 次取均值）\n{'='*72}")
        keys = ["route_acc", "recall@5", "recall@10", "mrr@10", "recall@evidence",
                "zero_recall_rate", "retention", "budget_utilization",
                "refuse_correct_rate",
                "false_refuse_rate", "evidence_gap",
                "recall@10_textbook", "recall@10_colloquial",
                "recall@10_cross_grade"]
        heads = list(results.keys())
        print(f"{'指标':<20}" + "".join(f"{h[:18]:>20}" for h in heads))
        for k in keys:
            pct = k not in ("mrr@10", "evidence_gap")
            line = f"{k:<20}"
            for h in heads:
                line += f"{fmt(results[h]['summary'][k], pct=pct):>20}"
            print(line)

    if args.json:
        out = {tag: {"summary": r["summary"], "rows": r["rows"]}
               for tag, r in results.items()}
        Path(args.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n逐题明细已写入：{args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
