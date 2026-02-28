"""Memes Manager 模块

提供表情包管理功能，包括：
- 表情数据结构定义（index.py）
- 词嵌入管理（embedding_wrapper.py）
- 顶层管理接口（manager.py）
"""

from .embedding_wrapper import EmbeddingsData, MemesData, MemesEmbeddingManager
from .index import (
    EmbeddingConfig,
    EmbeddingSearchResult,
    Emotion,
    EmotionEntry,
    FuzzySearchResult,
    Meme,
    MemesStats,
    MemesTable,
)
from .manager import MemesManager, MemesManagerConfig

__all__ = [
    # index.py
    "Meme",
    "Emotion",
    "EmotionEntry",
    "FuzzySearchResult",
    "EmbeddingSearchResult",
    "EmbeddingConfig",
    "MemesStats",
    "MemesTable",
    # embedding_wrapper.py
    "EmbeddingsData",
    "MemesData",
    "MemesEmbeddingManager",
    # manager.py
    "MemesManagerConfig",
    "MemesManager",
]
