#!/usr/bin/env python3
"""
Semantic Embedding Generator for Mati Learning System

Generates high-quality semantic embeddings using Sentence Transformers.
Optimized for local execution on i3 processors.

Features:
- Uses `BAAI/bge-small-zh-v1.5` (Chinese-optimized, ~95MB, 512-dim)
- Batch processing for efficiency
- Caching of embeddings
- GPU support (if available, falls back to CPU)
- Chinese + English multilingual support (512D)
"""

import os
import json
import logging
import torch
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ---------------------------------------------------------------------------
# 离线优先环境准备（必须在 import huggingface_hub / transformers 之前执行！）
# huggingface_hub 在 import 时把 HF_HUB_OFFLINE / HF_ENDPOINT 缓存为常量，
# 之后设置无效。直连 huggingface.co 在部分网络会被重置（ConnectionResetError
# 10054），因此：本地缓存存在则强制离线加载；缺失则走 hf-mirror 国内镜像。
# ---------------------------------------------------------------------------
_HF_MODEL = "BAAI/bge-small-zh-v1.5"
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / (
    "models--" + _HF_MODEL.replace("/", "--"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if _HF_CACHE.exists():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("Missing dependency: sentence-transformers")
    logger.error("Install with: pip install sentence-transformers")
    raise

# Official BGE retrieval instruction for Chinese (v1.5 models).
# Only queries get the instruction; passages/documents are embedded WITHOUT it.
# Source: https://github.com/FlagOpen/FlagEmbedding (BAAI/bge-small-zh-v1.5)
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def prepare_hf_environment(model_name: str) -> None:
    """
    离线优先的模型加载环境准备。

    - 本地缓存存在：设置 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE，
      强制使用本地缓存，避免每次启动联网检查被网络环境重置
      （ConnectionResetError 10054）而报错。
    - 本地缓存缺失：默认使用 hf-mirror 国内镜像下载（直连 huggingface.co
      在此网络环境不可用）。
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + model_name.replace("/", "--"))
    if cache_dir.exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


class EmbeddingGenerator:
    """
    Generates semantic embeddings using Sentence BERT.
    Default model: BAAI/bge-small-zh-v1.5 (512 dimensions, ~95MB, Chinese-optimized)
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = None,
        batch_size: int = 16, # Reduced from 32 for i3 stability
        cache_dir: str = None
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        
        torch.set_num_threads(2)
        self.model_name = model_name
        self.batch_size = batch_size
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Embedding Generator...")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Device: {self.device}")
        
        prepare_hf_environment(model_name)
        try:
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=cache_dir
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"   Dimensions: {self.embedding_dim}")

    def generate_query_embeddings(
        self,
        texts: Union[str, List[str]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Embed retrieval QUERIES with the official BGE query instruction
        ("为这个句子生成表示以用于检索相关文章：").

        BGE v1.5 models are trained so that queries carry the instruction while
        passages do not; using it measurably improves retrieval ranking.
        """
        if isinstance(texts, str):
            return self.generate_embeddings(BGE_QUERY_INSTRUCTION + texts)
        return self.generate_embeddings([BGE_QUERY_INSTRUCTION + t for t in texts])

    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
            
        try:
            valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
            valid_texts = [texts[i] for i in valid_indices]
            
            if not valid_texts:
                return np.array([]) if not is_single else np.zeros(self.embedding_dim)

            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            if len(valid_texts) < len(texts):
                full_embeddings = np.zeros((len(texts), self.embedding_dim))
                for i, valid_idx in enumerate(valid_indices):
                    full_embeddings[valid_idx] = embeddings[i]
                result = full_embeddings
            else:
                result = embeddings
                
            return result[0] if is_single else result
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise

    def process_chunk_file(
        self,
        input_file: str,
        output_file: str = None
    ) -> Dict:
        if output_file is None:
            output_file = input_file
            
        logger.info(f"Processing chunks from: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                chunks = data
            elif isinstance(data, dict) and 'chunks' in data:
                chunks = data['chunks']
            else:
                chunks = []
                for k, v in data.items():
                    if isinstance(v, list) and len(v) > 0 and 'text' in v[0]:
                        chunks = v
                        break
            
            if not chunks:
                logger.warning("No chunks found in file")
                return {'processed': 0, 'error': "No chunks found"}
                
            texts = [c.get('text', '') for c in chunks]
            
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.generate_embeddings(texts, show_progress=True)
            
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk['embedding'] = embeddings[i].tolist()
            
            output_data = {'chunks': chunks} if isinstance(data, list) else data

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False)
            
            logger.info(f"Saved updated chunks to {output_file}")
            
            return {
                'processed': len(chunks),
                'model': self.model_name,
                'dimension': self.embedding_dim
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {'processed': 0, 'error': str(e)}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate semantic embeddings for text chunks")
    parser.add_argument("input", help="Input JSON file or directory containing chunks")
    parser.add_argument("--output", help="Output path (optional)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5", help="SentenceTransformer model name")
    
    args = parser.parse_args()
    
    generator = EmbeddingGenerator(
        model_name=args.model,
        batch_size=args.batch_size
    )
    
    if os.path.isfile(args.input):
        generator.process_chunk_file(args.input, args.output)
    elif os.path.isdir(args.input):
        input_path = Path(args.input)
        files = list(input_path.glob("*chunks*.json"))
        logger.info(f"📚 Found {len(files)} chunk files to process")
        
        for f in files:
            generator.process_chunk_file(str(f))

if __name__ == "__main__":
    main()