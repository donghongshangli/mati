"""
User Edge Case Handler for Mati Learning System

Handles common user interaction edge cases to improve user experience
and save inference computation.
"""

import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class UserEdgeCaseHandler:
    """
    Detects and handles special cases in user queries
    before they reach the heavy RAG/LLM pipeline.
    """
    
    def __init__(self):
        self.greetings = {
            "hi", "hello", "namaste", "namaskar", "hey",
            "good morning", "good afternoon", "good evening"
        }
        
        self.apologies = {
            "sorry", "i'm sorry", "my bad", "apologies"
        }
        
        self.gratitude = {
            "thank you", "thanks", "thx", "dhanyabad"
        }

    def check_edge_cases(self, query: str) -> Optional[str]:
        query_lower = query.strip().lower()
        
        # 1. Empty Query
        if not query_lower:
            return "请输入你的问题，我很乐意帮助你学习！"
            
        # 2. Short Vague Queries
        if len(query_lower) < 4:
            return "能再说详细一点吗？你的问题太短了，我还不太理解。"
            
        # 3. Greetings
        if query_lower in self.greetings:
            return "你好！我是 Mati，你的学习助手。今天想学习什么呢？"
            
        for g in self.greetings:
            if query_lower.startswith(g + " "):
                pass

        # 4. Gratitude
        if query_lower in self.gratitude:
            return "不客气！继续加油学习！"
            
        # 5. Apologies
        if query_lower in self.apologies:
            return "没关系！我们继续学习吧。"
            
        # 6. "I don't understand"
        if "don't understand" in query_lower or "confused" in query_lower:
            return None

        return None
        
    def is_math_query(self, query: str) -> bool:
        math_symbols = ['+', '=', '/', '*', '√', '^', 'solve', 'calculate', 'equation']
        count = sum(1 for s in math_symbols if s in query.lower())
        return count >= 1