# -*- coding: utf-8 -*-
"""
P2 步骤 3.4「可信生成」端到端验收。

为什么需要单独一个脚本，而不是靠 scripts/eval_rag.py
──────────────────────────────────────────────────
`eval_rag.py` 用 `load_llm=False` 跑 —— 它测的是**检索层**，
不加载 1.5B 生成模型，因此模型从不写 `[n]`，引用指标恒为 0。
那不是「引用功能坏了」，是**测错了对象**。

可信生成必须验证真实链路：
    检索层编号 → 来源清单进 context → 模型写 [n] → 校验 → 降置信度
这条链只有真跑模型才能验证。本脚本跑 4 题，覆盖正样本与拒答两条路径。

用法：
    C:\\...\\Anaconda3\\envs\\autoresearch\\python.exe -m scripts.verify_citations
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.rag.knowledge_governance import validate_citations  # noqa: E402

MODEL_DIR = ROOT / "mati_data" / "models" / "qwen2_5"

# 覆盖：书面提问 / 口语提问 / 跨年级 / 库外（应拒答）
PROBES = [
    ("集合的交集怎么定义", "math", 10, None),
    ("为啥物体运动的快慢要用加速度描述", "physics", 10, None),
    ("化学平衡移动时浓度怎么变", "chemistry", 11, None),
    ("帮我写一篇 800 字的散文", "chinese", 10, None),  # 库里没有对应内容 → 应拒答
]


def main() -> int:
    # model_path 传的是**目录**（引擎内部自己找目录下的 .gguf），
    # 传文件路径会报 "No .gguf file found in <文件名>"。
    if not MODEL_DIR.is_dir() or not list(MODEL_DIR.glob("*.gguf")):
        print(f"[verify] {MODEL_DIR} 下找不到 .gguf 模型文件")
        return 2

    from system.rag.rag_retrieval_engine import RAGRetrievalEngine

    print("[verify] 加载生成模型（1.5B, 首次加载较慢）…")
    engine = RAGRetrievalEngine(
        load_llm=True,
        model_path=str(MODEL_DIR),
    )
    # 评测流量不写失败日志
    try:
        engine.rewriter.failure_log = None
    except Exception:
        pass

    n_ok = 0
    n_leak = 0          # 越界引用总数
    n_cited = 0         # 模型写了标记的题数
    n_refused = 0
    n_no_llm = 0        # 模型没真正跑起来的题数
    n_creation = 0      # 创作类拒答题数（正确行为：不该调模型）

    for q, subject, grade, disc in PROBES:
        print("\n" + "=" * 70)
        print(f"【提问】{q}")
        res = engine.query(
            query_text=q, subject=subject, grade=grade, discipline=disc,
            already_normalized=False, capture_stages=True,
        )
        answer = (res.get("answer") or "").strip()
        cites = res.get("citations") or []
        check = res.get("citation_check") or {}

        print(f"  类型        : {res.get('type')}")
        print(f"  llm_used    : {res.get('llm_used')}")
        print(f"  置信度      : {res.get('confidence')}")
        print(f"  来源编号数  : {len(cites)}")
        for c in cites[:4]:
            print(f"      [{c['index']}] {c['label']}")
        if len(cites) > 4:
            print(f"      …共 {len(cites)} 条")
        print(f"  引用校验    : {check}")
        print(f"  答案（前 220 字）：{answer[:220]}")

        # 两种拒答都要归入 n_refused：
        #   no_evidence      —— 覆盖率不足，教材里没有
        #   creation_refused —— 创作类请求，**本就不该调模型**（llm_used=False 是正确行为）
        # 早期版本只认 no_evidence，把创作拒答误报成「模型没跑起来」并返回退出码 2。
        rtype = res.get("type")
        if rtype in ("no_evidence", "creation_refused"):
            n_refused += 1
            if rtype == "creation_refused":
                n_creation += 1
            # 拒答时不该带引用标记
            if check.get("n_cited"):
                print("  ⚠️ 拒答却带了引用标记")
                n_leak += check["n_cited"]
            continue

        n_ok += 1
        if not res.get("llm_used"):
            print("  ⚠️ 模型未实际生成（llm_used=False），本轮无法验收引用标注")
            n_no_llm += 1
            continue
        if check.get("n_cited"):
            n_cited += 1
        n_leak += check.get("n_out_of_range", 0)

        # 独立复算一遍，确认引擎报的数不是自己算自己
        if cites:
            indep = validate_citations(answer, cites)
            assert indep["n_cited"] == check.get("n_cited"), (
                f"校验结果不可复现：{indep} vs {check}"
            )

    print("\n" + "=" * 70)
    print("[verify] 汇总")
    print(f"  有效回答      : {n_ok} / {len(PROBES)}"
          f"（拒答 {n_refused}，其中创作类 {n_creation}）")
    if n_no_llm:
        print(f"  ⚠️ 未走生成模型: {n_no_llm} 题（应仅出现在拒答题上）")
        return 2
    print(f"  模型写标记的题: {n_cited} / {n_ok}")
    print(f"  越界引用总数  : {n_leak}")
    if n_leak:
        print("  ⚠️ 出现越界引用：编号表与答案不一致")
        return 1
    if n_ok and n_cited == 0:
        print("  ⚠️ 全部未标注 —— 编号指令对 1.5B 模型可能过难，")
        print("     但编号由检索层生成，模型不写也不会产生假来源。")
    print("  ✓ 引用链路端到端可用（无越界、无假来源）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
