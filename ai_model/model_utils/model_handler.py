#!/usr/bin/env python3
"""
Model Handler for Mati Learning System
CPU-first with Simple Handler interface
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Iterator, Optional

from .qwen_handler import QwenHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleHandler:
    """Lightweight single-phase interface for RAG queries."""
    
    
    def __init__(self, phi_handler):
        self.phi_handler = phi_handler
    
    def get_answer(self, query_text: str, context_text: str, history=None) -> Tuple[str, float]:
        """Single-step answer generation."""
        try:
            return self.phi_handler.get_answer(query_text, context_text, history=history)
        except Exception as e:
            logger.error(f"SimpleHandler error: {e}")
            return "生成答案时出错，请重试。", 0.0


class ModelHandler:
    """Model handler with SimpleHandler interface for i3 optimization (Qwen2.5-1.5B)."""
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join("mati_data", "models", "qwen2_5")
        
        from pathlib import Path
        model_dir = Path(model_path)
        gguf_files = list(model_dir.glob("*.gguf"))
        
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf file found in {model_path}")
        
        model_file = str(gguf_files[0])
        logger.info(f"Using model: {model_file}")
        
        self.model_path = model_path
        self.handler = QwenHandler(model_file)
        
        try:
            logger.info("Loading Qwen2.5-1.5B...")
            self.handler.load_model()
            self.handler.warm_up()
            logger.info("Model ready!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.simple_handler = SimpleHandler(self.handler)
    
    def get_answer(self, question: str, context: str = "", answer_length: str = "medium", history=None) -> Tuple[str, float]:
        try:
            return self.handler.get_answer(question, context, history=history)
        except Exception as e:
            logger.error(f"Error: {e}")
            return "我在理解您的问题时遇到了困难，请重试。", 0.1
    
    def get_answer_stream(self, question: str, context: str = "", answer_length: str = "medium", history=None) -> Iterator[str]:
        try:
            yield from self.handler.get_answer_stream(question, context, history=history)
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield "我在理解您的问题时遇到了困难，请重试。"
            
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Video passthrough for raw prompt generation."""
        try:
             return self.handler.generate_response(prompt, max_tokens)
        except Exception as e:
             logger.error(f"Gen Error: {e}")
             return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Qwen2.5-1.5B-Instruct",
            "version": "2.5",
            "format": "GGUF",
            "backend": "llama-cpp-python",
            "optimized_for": "i3_cpu",
            "context_size": "2048 tokens",
            "prompt_format": "ChatML"
        }
    
    def cleanup(self):
        try:
            self.handler.cleanup()
            logger.info("Model cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")