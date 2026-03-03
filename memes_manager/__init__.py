"""Memes Manager 模块

提供表情包管理功能，包括：
- 表情数据结构定义
- 词嵌入管理（embedding_wrapper.py）
- 顶层管理接口
- 错误类型定义
"""

from .embedding_wrapper import EmbeddingsData, MemesData, MemesEmbeddingManager
from .errors import (
    MemesConfigError,
    MemesEmbeddingDimensionMismatchError,
    MemesEmbeddingDisabledError,
    MemesEmbeddingError,
    MemesEmotionNotFoundError,
    MemesError,
    MemesFileError,
    MemesFileNotFoundError,
    MemesLLMError,
    MemesMemeNotFoundError,
    MemesNotFoundError,
    MemesNotInitializedError,
    MemesParseError,
    MemesProviderError,
    MemesProviderNotFoundError,
)
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
    # errors.py
    "MemesError",
    "MemesNotInitializedError",
    "MemesFileError",
    "MemesFileNotFoundError",
    "MemesEmbeddingError",
    "MemesEmbeddingDisabledError",
    "MemesEmbeddingDimensionMismatchError",
    "MemesProviderError",
    "MemesProviderNotFoundError",
    "MemesNotFoundError",
    "MemesMemeNotFoundError",
    "MemesEmotionNotFoundError",
    "MemesConfigError",
    "MemesParseError",
    "MemesLLMError",
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
