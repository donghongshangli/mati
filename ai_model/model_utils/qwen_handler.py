#!/usr/bin/env python3
"""
Qwen2.5 handler for Mati Learning System (Chinese-adapted)
CPU-first, single-call implementation.

Loads **Qwen2.5-1.5B-Instruct (GGUF)** via llama-cpp-python and builds prompts
using the ChatML template expected by Qwen2.5.
"""

import re
import logging
from typing import Iterator, Tuple
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class QwenHandler:
    """Lightweight single-phase handler for i3 processors (Qwen2.5-1.5B)."""

    # Qwen2.5 uses ChatML: stop at the end-of-turn token.
    STOP_SEQUENCES = ["<|im_end|>", "<|endoftext|>", "</s>"]

    def __init__(self, model_path: str, require_citations: bool = True):
        self.model_path = model_path
        self.llm = None
        self.require_citations = require_citations
        # Chinese educational tutor system prompt.
        # Follows common open-source RAG + Qwen2.5 best practice: explicit
        # role, task, constraints, output format, and grounding/fallback rules.
        # 方案C：更口语化、可承接追问、可适当举例。
        #
        # 关于第 8 条「引用标记」——**编号与来源由检索层生成**（见
        # system/rag/knowledge_governance.build_citations），模型只写 [n]。
        # 这样设计的原因（方案 §6 风险表）：1.5B 模型的引用跟随能力不可靠，
        # 但它**无法编造**不存在的来源 —— 写 [7] 而实际只有 3 条证据时，
        # 越界会被生成后的确定性校验抓出来并降低置信度。
        # 因此这条要求是「鼓励」而非「依赖」：模型不写标记不影响正确性。
        self.system_prompt = (
            "你是 Mati，一名专业、耐心、严谨的中文教师，服务于中学教育场景。\n"
            "回答要求：\n"
            "1. 使用简体中文回答，语言亲切、自然，像老师面对面讲解。\n"
            "2. 优先依据提供的「参考材料」作答，并使用其中的专业术语；"
            "若材料不足以回答，先明确说明“材料中未找到相关信息”，再基于常识简要作答。\n"
            "3. 先给出直接结论或定义，再分点展开解释；可适当举例，帮助理解。\n"
            "4. 回答长度适中（一般 150–250 字）；若用户追问（如“为什么”“再解释”“继续”“还有吗”），"
            "请结合「对话历史」与「参考材料」继续深入讲解。\n"
            "5. 不要输出练习题、不要复述问题、不要罗列无关内容。\n"
            "6. 数学式子一律用纯文本 + Unicode 符号书写：变量直接写字母（x、y、f（x）），"
            "乘号写 ×、根号写 √、不等号写 ≤ ≥ ≠、增量写 Δ，分式写成 (Δy)/(Δx)。"
            "严禁使用 LaTeX 标记（如 \\[ \\]、$、\\frac、\\lim、\\sqrt、\\text）——"
            "界面是纯文本答案框，学生只会看到一堆反斜杠。\n"
            "7. 直接沿用参考材料中的数学符号与字母写法，不要另改造型或其他记号。\n"
            "8. 参考材料末尾附有「来源清单」，形如 [1] 书名 · 章节。"
            "请在关键结论后面加上对应的来源编号（如 牛顿第一定律适用于一切物体[1]），"
            "最多标 2–3 处即可；不要在每句话后都标，也不要编造清单里没有的编号。\n"
            "9. 不要自己写「依据：」「来源：」这类汇总行 —— 系统会自动附上，"
            "你自己写反而可能重复或写错。"
        )
    
    def load_model(self):
        if self.llm is not None:
            return
        
        logger.info("Loading Qwen2.5-1.5B-Instruct (CPU-optimized)...")
        self.llm = Llama(
            model_path=self.model_path,
            # ── 关于启动时那行 llama.cpp 提示 ──────────────────────────────
            #   llama_context: n_ctx_per_seq (4096) < n_ctx_train (32768)
            #                  -- the full capacity of the model will not be utilized
            #
            # 它是 **info 级提示，不是错误**，也不影响推理质量：
            # "未用满模型训练时的上下文容量" ≠ "上下文不够用"。
            # 只有当 prompt + 输出**超过** n_ctx 被截断时才会掉质量，
            # 而我们是**远小于**训练长度，模型处在完全正常的区间内。
            #
            # 为什么选 4096（Qwen2.5-1.5B 实测元数据）：
            #   n_layer=28, n_head=12, n_head_kv=2, n_embd=1536 → head_dim=128
            #   KV cache = 2(K,V) × 28层 × 2头 × 128维 = 14336 值/token
            #   本项目 f16_kv=False（KV 存 fp32，比 f16 稳但占两倍）→ 2 字节/值
            #     → 约 28 KB/token（若改 f16_kv=True 则减半，约 14 KB/token）
            #
            #     n_ctx     KV cache（fp32 / f16）
            #      2048      57 MB   / 29 MB
            #      4096     114 MB   / 57 MB   ← 当前取值
            #      8192     229 MB   / 114 MB
            #     32768     917 MB   / 458 MB  ← 拉满的代价，低配 CPU 不划算
            #
            # 4096 的依据（不是拍脑袋）：system prompt 约 300 + 证据上限 2000
            # + 3 轮历史 + 输出 768。原先 2048 会在 3 轮历史（约 660 token）时
            # 把证据预算挤到只剩 280 —— 等于没有证据。提到 4096 后不再被侵蚀。
            #
            # 想调大：改这里的 n_ctx，并同步
            # `mati_data/config/retrieval.yaml → context_budget.n_ctx`
            # （检索引擎按 n_ctx 反推证据预算，两处必须一致，否则会溢出）。
            n_ctx=4096,
            n_batch=96,
            n_threads=4,      
            use_mmap=True,
            use_mlock=False,
            f16_kv=False,
            verbose=False
        )
        logger.info("Model loaded!")
    
    def warm_up(self):
        """
        Warming up the model with a dummy inference.
        """
        if not self.llm:
            self.load_model()
        
        logger.info("Warming up model...")
        try:
            # Quick dummy inference to initialize the engine
            dummy_prompt = "Instruct: What is 2+2?\nOutput: Answer:"
            self.llm(
                dummy_prompt,
                max_tokens=10,
                temperature=0.1,
                stream=False
            )
            logger.info("Model warmed up!")
        except Exception as e:
            logger.warning(f"Warm-up failed (non-critical): {e}")
    
    def _build_prompt(self, question: str, context: str = "", history=None) -> str:
        context = (context or "").strip()

        # 对话历史（可追问支持）：最近最多 3 轮
        history_text = ""
        if history:
            turns = []
            for q, a in history[-3:]:
                q = (q or "").strip()
                a = (a or "").strip()
                if q and a:
                    turns.append(f"学生：{q}\n老师：{a}")
            if turns:
                history_text = "【对话历史】\n" + "\n".join(turns)

        # 组装用户消息：历史 + 参考材料 + 问题
        parts = ["请用中文回答用户的问题，语言亲切、清晰、准确。"]
        if history_text:
            parts.append(history_text)
        if context:
            # 上下文预算由 system/rag/context_budget.py 统一按 token 管理，
            # 此处不再做二次截断（原实现按 850 字符截断，会把检索层辛苦
            # 召回的 1600 字符证据静默砍掉一半以上）。
            #
            # 安全上限必须也是**按 token** 判定的，不能沿用字符数：
            # 中英混排 / 公式（ASCII）的字符-token 比差异很大 ——
            # 2000 token 的中文证据约 2000 字符，但 2000 token 的公式文本
            # 可达 6000+ 字符。用固定字符上限会在含公式的教材上静默截断，
            # 正是原 E1/E4「字符预算 ≠ token 预算」的同一类错误。
            SAFETY_CAP_TOKENS = 4000   # 远高于证据预算（2000），仅防外部传入超长文本
            try:
                from system.rag.text_utils import estimate_tokens as _est_tokens
                est = _est_tokens(context)
            except Exception:  # noqa: BLE001 - 兜底：按中文最保守的 1 字符 ≈ 1 token
                est = len(context)
            if est > SAFETY_CAP_TOKENS:
                keep = max(1, int(len(context) * SAFETY_CAP_TOKENS / est))
                logger.warning(
                    f"参考材料超过安全上限（约 {est} > {SAFETY_CAP_TOKENS} token），"
                    f"已按比例截断到 {keep} 字符；"
                    "正常情况下应由 context_budget 按 token 预算组装。"
                )
                context = context[:keep]
            parts.append("请优先依据下面的「参考材料」作答，并使用其中的专业术语；若材料中未包含答案，请先说明“材料中未找到相关信息”，再简要作答。")
            parts.append(f"参考材料：\n{context}")
        parts.append(f"问题：{question}")
        user_msg = "\n\n".join(parts)

        return (
            "<|im_start|>system\n"
            f"{self.system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_msg}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    
    def _clean_answer(self, answer: str) -> str:
        if not answer:
            return ""
        
        answer = answer.strip()
        
  
        answer = re.sub(r'^(Q:|A:)\s*', '', answer, flags=re.I)
        answer = re.sub(r'\n(Q:|A:)\s*', '\n', answer, flags=re.I)
        

        off_markers = ["Exercise:", "Practice:", "Try this:", "Use Case:", "Real-world",
                       "练习：", "习题：", "试试：", "用例：", "现实世界"]
        for marker in off_markers:
            if marker in answer:
                answer = answer.split(marker)[0].strip()
        

        answer = re.sub(r'\s+', ' ', answer)
        

        if answer and not answer[-1] in ".!?":
            answer += '.'
        
        return answer
    
    @staticmethod
    def _units(text: str) -> set:
        """
        Split text into comparison units that work for both Chinese and English:
        each CJK character counts as one unit, each Latin word/token as another.
        """
        return set(
            re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", text.lower())
        )

    def _calculate_confidence(self, answer: str, question: str) -> float:
        if not answer:
            return 0.3
        # Length heuristic: >=5 tokens for English words, or >=15 CJK chars.
        length = len(self._units(answer))
        if length < 5:
            return 0.3

        q_units = self._units(question)
        a_units = self._units(answer)
        relevance = len(q_units & a_units) / max(1, len(q_units))

        return min(1.0, 0.5 + relevance * 0.5)
    
    def get_answer_stream(self, question: str, context: str = "", history=None) -> Iterator[str]:
        if not self.llm:
            self.load_model()
        
        if not question or len(question.strip()) < 3:
            yield "请输入有效的问题。"
            return
        
        prompt = self._build_prompt(question, context, history=history)
        
        try:

            for chunk in self.llm(
                prompt,
                max_tokens=768,  # Detailed, complete answers
                # Qwen2.5 official recommendation: temperature 0.7 / top_p 0.8 /
                # repetition_penalty 1.05. For educational answers we stay a bit
                # more conservative (lower temperature, higher top_p).
                temperature=0.6,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=self.STOP_SEQUENCES,
                stream=True
            ):
                if chunk and "choices" in chunk:
                    if len(chunk["choices"]) > 0:
                        text = chunk["choices"][0].get("text", "")
                        if text:
                            yield text
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield "生成答案时出错，请重试。"
    
    def get_answer(self, question: str, context: str = "", history=None) -> Tuple[str, float]:
        if not self.llm:
            self.load_model()
        
        if not question or len(question.strip()) < 3:
            return "请输入有效的问题。", 0.1
        
        prompt = self._build_prompt(question, context, history=history)
        
        try:
            response = self.llm(
                prompt,
                max_tokens=768,
                # See note in get_answer_stream: conservative values for education.
                temperature=0.6,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=self.STOP_SEQUENCES,
                stream=False
            )
            
            raw = response["choices"][0]["text"].strip()
            answer_clean = self._clean_answer(raw)
            confidence = self._calculate_confidence(answer_clean, question)
            
            return answer_clean, confidence
        
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return "生成答案时出错，请重试。", 0.1
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generating a raw response from a custom prompt."""
        if not self.llm:
            self.load_model()
            
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.3, # Lower temperature for grading/logic
                top_p=0.9,
                repeat_penalty=1.1,
                stop=self.STOP_SEQUENCES + ["Student Answer:", "Question:"], # Extra stops for grading
                stream=False
            )
            return response["choices"][0]["text"].strip()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""

    def cleanup(self):
        if self.llm:
            del self.llm
            self.llm = None
            logger.info("Model cleaned up")