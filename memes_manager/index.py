"""表情包数据结构和索引

定义表情包的核心数据结构，包括 Meme、Emotion、MemesTable 等。
"""

from dataclasses import dataclass, field
from pathlib import Path

from fuzzywuzzy import fuzz

from astrbot.api import logger

from .errors import (
    MemesEmbeddingDimensionMismatchError,
    MemesEmbeddingDisabledError,
    MemesMemeNotFoundError,
)


@dataclass
class Meme:
    internal_path: Path
    description: str
    description_embedding: list[float] | None = None


# 用于表示表情包情感的一个中文词语
Emotion = str


@dataclass
class EmotionEntry:
    """情感条目，包含对应的词嵌入向量"""

    embedding: list[float] | None = None


@dataclass
class FuzzySearchResult:
    """模糊匹配搜索结果"""

    meme: Meme
    emotion: str
    score: int  # 范围 [0, 100]，值越大表示越匹配


@dataclass
class EmbeddingSearchResult:
    """词嵌入搜索结果"""

    meme: Meme
    emotion: str
    emotion_similarity: float  # 范围 [-1, 1]，值越大表示越相似
    description_similarity: float


@dataclass
class MemesStats:
    """表情包统计信息"""

    total_memes: int  # 总表情数量
    total_emotions: int  # 总情感数量
    embedding_enabled: bool  # 是否启用了词嵌入
    emotions_with_embedding: int = 0  # 已计算嵌入的情感数量
    memes_with_description_embedding: int = 0  # 已计算描述嵌入的表情数量
    embedding_provider_id: str | None = None  # Embedding Provider ID
    embedding_dim: int | None = None  # 词嵌入维度


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度

    Args:
        a: 向量 a
        b: 向量 b

    Returns:
        余弦相似度，范围 [-1, 1]，值越大表示越相似

    Precondition:
        len(a) > 0 and len(b) > 0
    """
    if len(a) == 0:
        raise RuntimeError("向量 a 不能为空")
    if len(b) == 0:
        raise RuntimeError("向量 b 不能为空")

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


@dataclass
class EmbeddingConfig:
    provider_id: str
    dim: int


@dataclass
class MemesTable:
    """通过 Emotion 索引的表情列表，支持词嵌入查询

    该类仅负责管理词嵌入数据和执行查询，不涉及模型调用。
    模型调用应由外部模块（如 embedding_wrapper.py）处理。

    支持两种嵌入向量：
    1. Emotion 嵌入：情感词的语义向量
    2. Description 嵌入：表情描述的语义向量

    Contract:
        - 所有 set_* 方法要求 embedding_config 已设置（如果涉及嵌入操作）
        - 所有 search_* 方法要求 embedding_config 已设置（如果使用嵌入搜索）
    """

    embedding_config: EmbeddingConfig | None
    by_emotion: dict[Emotion, list[Meme]] = field(default_factory=dict)
    emotion_entries: dict[Emotion, EmotionEntry] = field(default_factory=dict)

    def __init__(self, embedding_config: EmbeddingConfig | None):
        self.by_emotion = {}
        self.emotion_entries = {}
        self.embedding_config = embedding_config

    def add(self, emotion: Emotion, meme: Meme):
        """添加表情到指定情感类别"""
        if emotion not in self.by_emotion:
            self.by_emotion[emotion] = []
        self.by_emotion[emotion].append(meme)

        # 确保 emotion_entries 中有对应的条目
        if emotion not in self.emotion_entries:
            self.emotion_entries[emotion] = EmotionEntry()

    def remove(
        self, path: Path, remove_emotion_embedding: bool = True
    ) -> Emotion | None:
        """删除指定路径的表情

        Args:
            path: 表情文件路径
            remove_emotion_embedding: 是否在情感下无表情时移除情感嵌入

        Returns:
            被删除表情的情感，如果表情不存在则返回 None
        """
        for emotion, memes in self.by_emotion.items():
            for i, meme in enumerate(memes):
                if meme.internal_path == path:
                    memes.pop(i)
                    # 如果该情感下没有表情了，删除该情感
                    if not memes:
                        del self.by_emotion[emotion]
                        if remove_emotion_embedding and emotion in self.emotion_entries:
                            del self.emotion_entries[emotion]
                    return emotion
        return None

    def contains(self, path: Path) -> bool:
        for emotion, memes in self.by_emotion.items():
            for meme in memes:
                if meme.internal_path == path:
                    return True
        return False

    def clean_emotion_embedding(self, emotion: Emotion):
        """清理某个情感下的词嵌入

        如果该情感下没有表情，则移除该情感的嵌入数据。
        """
        if self.by_emotion.get(emotion) is None:
            if emotion in self.emotion_entries:
                del self.emotion_entries[emotion]
        elif len(self.by_emotion[emotion]) == 0:
            del self.by_emotion[emotion]
            if emotion in self.emotion_entries:
                del self.emotion_entries[emotion]

    def set_emotion_embedding(self, emotion: str, embedding: list[float]):
        """设置某个 Emotion 的词嵌入向量

        Args:
            emotion: 情感词
            embedding: 词嵌入向量

        Raises:
            MemesEmbeddingDisabledError: 词嵌入未启用
            MemesEmbeddingDimensionMismatchError: 向量维度与配置不匹配
        """
        if self.embedding_config is None:
            raise MemesEmbeddingDisabledError()

        if len(embedding) != self.embedding_config.dim:
            raise MemesEmbeddingDimensionMismatchError(
                expected_dim=self.embedding_config.dim,
                actual_dim=len(embedding),
            )

        # 确保 emotion 存在于 emotion_entries 中
        if emotion not in self.emotion_entries:
            self.emotion_entries[emotion] = EmotionEntry()

        self.emotion_entries[emotion].embedding = embedding

    def get_emotion_embedding(self, emotion: str) -> list[float] | None:
        """获取某个 Emotion 的词嵌入向量

        Returns:
            词嵌入向量，如果不存在则返回 None
        """
        entry = self.emotion_entries.get(emotion)
        return entry.embedding if entry else None

    def get_emotions_without_embedding(self) -> list[str]:
        """获取所有尚未计算词嵌入的 Emotion"""
        return [
            emotion
            for emotion, entry in self.emotion_entries.items()
            if entry.embedding is None
        ]

    def get_all_emotions(self) -> list[str]:
        """获取所有 Emotion"""
        return list(self.emotion_entries.keys())

    def set_description_embedding(self, path: Path, embedding: list[float]):
        """设置某个 Meme 描述的词嵌入向量

        Args:
            path: Meme 的 internal_path
            embedding: 词嵌入向量

        Raises:
            MemesEmbeddingDisabledError: 词嵌入未启用
            MemesEmbeddingDimensionMismatchError: 向量维度与配置不匹配
            MemesMemeNotFoundError: 找不到对应路径的 Meme
        """
        if self.embedding_config is None:
            raise MemesEmbeddingDisabledError()

        # 验证向量维度
        if len(embedding) != self.embedding_config.dim:
            raise MemesEmbeddingDimensionMismatchError(
                expected_dim=self.embedding_config.dim,
                actual_dim=len(embedding),
            )

        # 查找对应的 Meme 并设置嵌入
        for memes in self.by_emotion.values():
            for meme in memes:
                if meme.internal_path == path:
                    meme.description_embedding = embedding
                    return

        raise MemesMemeNotFoundError(path)

    def get_description_embedding(self, path: Path) -> list[float] | None:
        """获取某个 Meme 描述的词嵌入向量

        Returns:
            词嵌入向量，如果不存在则返回 None
        """
        for memes in self.by_emotion.values():
            for meme in memes:
                if meme.internal_path == path:
                    return meme.description_embedding
        return None

    def get_memes_without_description_embedding(self) -> list[Meme]:
        """获取所有尚未计算描述嵌入的 Meme"""
        result = []
        for memes in self.by_emotion.values():
            for meme in memes:
                if meme.description_embedding is None:
                    result.append(meme)
        return result

    def get_all_memes(self) -> list[Meme]:
        """获取所有 Meme"""
        result = []
        for memes in self.by_emotion.values():
            result.extend(memes)
        return result

    def search_by_embedding(
        self,
        queries: list[list[float]],
        max_candidates: int,
    ) -> list[EmbeddingSearchResult]:
        """基于词嵌入向量的相似度查询表情

        同时搜索 Emotion 嵌入和 Description 嵌入，返回最佳匹配结果。

        Args:
            queries: 查询词的嵌入向量列表
            max_candidates: 最多返回的候选项数量

        Returns:
            EmbeddingSearchResult 列表，按相似度降序排序。

        Raises:
            MemesEmbeddingDisabledError: 词嵌入未启用
            MemesEmbeddingDimensionMismatchError: 查询向量维度与配置不匹配

        Precondition:
            len(queries) > 0
            max_candidates > 0
        """
        if len(queries) == 0:
            raise RuntimeError("queries 不能为空")
        if max_candidates <= 0:
            raise RuntimeError("max_candidates 必须大于 0")

        if self.embedding_config is None:
            raise MemesEmbeddingDisabledError()

        if not all(len(q) == self.embedding_config.dim for q in queries):
            actual_dims = [len(q) for q in queries]
            raise MemesEmbeddingDimensionMismatchError(
                expected_dim=self.embedding_config.dim,
                actual_dim=actual_dims[0]
                if len(set(actual_dims)) == 1
                else actual_dims,  # type: ignore
            )

        logger.debug(
            f"词嵌入搜索: 维度={self.embedding_config.dim}, 最大候选数={max_candidates}"
        )

        # key: internal_path, value: (Meme, Emotion, similarity)
        emotion_candidates: dict[Emotion, float] = {}

        # 1. 基于 Emotion 嵌入计算相似度
        for emotion, entry in self.emotion_entries.items():
            if entry.embedding is None:
                continue

            similarity = max(_cosine_similarity(q, entry.embedding) for q in queries)
            emotion_candidates[emotion] = similarity

        # 获取 Top k
        sorted_emotions = sorted(
            emotion_candidates.items(), key=lambda x: x[1], reverse=True
        )
        top_emotions = sorted_emotions[:max_candidates]

        candidates: dict[Path, tuple[Meme, Emotion, float]] = {}

        # 2. 基于 Description 嵌入计算相似度
        for emotion, emotion_similarity in top_emotions:
            for meme in self.by_emotion.get(emotion, []):
                if meme.description_embedding is None:
                    continue

                similarity = max(
                    _cosine_similarity(q, meme.description_embedding) for q in queries
                )
                path = meme.internal_path
                candidates[path] = (meme, emotion, similarity)

        # 按相似度降序排序并返回前 max_candidates 个
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda x: x[2] * emotion_candidates[x[1]],
            reverse=True,
        )
        results = [
            EmbeddingSearchResult(
                meme=meme,
                emotion=emotion,
                emotion_similarity=emotion_candidates[emotion],
                description_similarity=similarity,
            )
            for meme, emotion, similarity in sorted_candidates[:max_candidates]
        ]
        logger.debug(f"词嵌入搜索完成: 返回 {len(results)} 个结果")
        return results

    def search_keyword(
        self, keywords: list[str], max_candidates: int
    ) -> list[FuzzySearchResult]:
        """在所有表情的情感和描述中使用模糊匹配查询候选项

        Args:
            keywords: 搜索关键词
            max_candidates: 最多返回的候选项数量

        Returns:
            FuzzySearchResult 列表，按分数降序排序。

        Precondition:
            len(keywords) > 0
            max_candidates > 0
        """
        if len(keywords) == 0:
            raise RuntimeError("keywords 不能为空")
        if max_candidates <= 0:
            raise RuntimeError("max_candidates 必须大于 0")

        logger.debug(f"模糊匹配搜索: 关键词='{keywords}', 最大候选数={max_candidates}")

        # 用于存储所有候选项及其分数，key 为 internal_path 用于去重
        candidates: dict[Path, tuple[Meme, Emotion, int]] = {}

        for emotion, memes in self.by_emotion.items():
            for meme in memes:
                # 计算情感和描述的匹配分数
                emotion_score = max(fuzz.partial_ratio(kw, emotion) for kw in keywords)
                description_score = max(
                    fuzz.partial_ratio(kw, meme.description) for kw in keywords
                )
                max_score = max(emotion_score, description_score)

                # 如果该表情已经被记录，保留更高的分数
                if meme.internal_path in candidates:
                    _, _, existing_score = candidates[meme.internal_path]
                    if max_score > existing_score:
                        candidates[meme.internal_path] = (meme, emotion, max_score)
                else:
                    candidates[meme.internal_path] = (meme, emotion, max_score)

        # 按分数降序排序并返回前 max_candidates 个
        sorted_candidates = sorted(
            candidates.values(), key=lambda x: x[2], reverse=True
        )
        results = [
            FuzzySearchResult(meme=meme, emotion=emotion, score=score)
            for meme, emotion, score in sorted_candidates[:max_candidates]
        ]
        logger.debug(f"模糊匹配搜索完成: 返回 {len(results)} 个结果")
        return results
