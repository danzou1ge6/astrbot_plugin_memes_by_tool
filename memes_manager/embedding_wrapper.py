"""表情包词嵌入管理器

负责调用词嵌入模型并管理词嵌入数据的持久化。

存储方案：
- memes.json: 人类可读的表情信息（路径、情感、描述），不压缩
- embeddings.json.gz: 词嵌入向量数据，gzip 压缩

两份文件独立索引，允许修改 memes.json 而不破坏 embeddings.json.gz。
"""

import gzip
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from astrbot.api import logger
from astrbot.core.provider.provider import EmbeddingProvider
from astrbot.core.star.context import Context

from .index import (
    EmbeddingConfig,
    EmbeddingSearchResult,
    FuzzySearchResult,
    Meme,
    MemesStats,
    MemesTable,
)


@dataclass
class EmbeddingsData:
    """词嵌入数据结构"""

    config: EmbeddingConfig
    emotions: dict[str, list[float]] = field(default_factory=dict)
    # path (str) -> embedding
    descriptions: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "config": {"dim": self.config.dim, "provider_id": self.config.provider_id},
            "emotions": self.emotions,
            "descriptions": self.descriptions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingsData":
        """从字典创建"""
        logger.debug(
            f"加载词嵌入：情感词包括{list(data.get('emotions', {}).keys())}，描述包括{list(data.get('descriptions', {}).keys())}"
        )

        return cls(
            config=EmbeddingConfig(
                dim=data["config"]["dim"], provider_id=data["config"]["provider_id"]
            ),
            emotions=data.get("emotions", {}),
            descriptions=data.get("descriptions", {}),
        )


@dataclass
class MemesData:
    """表情数据结构（人类可读）"""

    memes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {"memes": self.memes}

    @classmethod
    def from_dict(cls, data: dict) -> "MemesData":
        """从字典创建"""
        return cls(memes=data.get("memes", []))


class MemesEmbeddingManager:
    """表情包词嵌入管理器

    负责管理 MemesTable 的词嵌入数据，包括：
    - 从文件加载/保存数据
    - 调用 EmbeddingProvider 计算词嵌入
    - 同步 memes.json 和 embeddings.json.gz
    """

    def __init__(
        self,
        memes_table: MemesTable,
        context: Context,
        data_dir: Path,
    ):
        """初始化管理器

        Args:
            memes_table: 表情表实例
            context: AstrBot 上下文，用于获取 EmbeddingProvider
            data_dir: 数据存储目录
            embedding_provider_id: 指定使用的 EmbeddingProvider ID，
                                  如果为 None 则使用第一个可用的 provider
        """
        self.memes_table = memes_table
        self.context = context
        self.data_dir = data_dir
        self._embedding_provider: EmbeddingProvider | None = None

        # 确保数据目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 文件路径
        self._memes_file = self.data_dir / "memes.json"
        self._embeddings_file = self.data_dir / "embeddings.json.gz"

    def get_embedding_provider(self) -> EmbeddingProvider | None:
        """获取 EmbeddingProvider 实例

        Returns:
            EmbeddingProvider 实例，如果没有可用的则返回 None

        Raises:
            KeyError: 找不到设置的provider
        """
        if self._embedding_provider is not None:
            return self._embedding_provider

        providers = self.context.get_all_embedding_providers()
        if not providers:
            return None

        if self.memes_table.embedding_config is not None:
            for provider in providers:
                if provider.meta().id == self.memes_table.embedding_config.provider_id:
                    self._embedding_provider = provider
                    return provider
            raise KeyError(
                f"没有找到provider {self.memes_table.embedding_config.provider_id}"
            )
        else:
            self._embedding_provider = None

        return self._embedding_provider

    def load_memes(self) -> bool:
        """从 memes.json 加载表情数据

        Returns:
            是否成功加载
        """
        if not self._memes_file.exists():
            logger.info(f"表情数据文件不存在: {self._memes_file}")
            return False

        try:
            with open(self._memes_file, encoding="utf-8") as f:
                data = json.load(f)

            memes_data = MemesData.from_dict(data)

            for item in memes_data.memes:
                path = Path(item["path"])
                emotion = item["emotion"]
                description = item.get("description", "")

                meme = Meme(
                    internal_path=path,
                    description=description,
                )
                self.memes_table.add(emotion, meme)

            logger.info(f"成功加载 {len(memes_data.memes)} 个表情数据")
            return True
        except Exception as e:
            logger.error(f"加载表情数据失败: {e}")
            return False

    def save_memes(self) -> bool:
        """保存表情数据到 memes.json

        Returns:
            是否成功保存
        """
        try:
            memes_data = MemesData()
            for emotion, memes in self.memes_table.by_emotion.items():
                for meme in memes:
                    memes_data.memes.append(
                        {
                            "path": str(meme.internal_path),
                            "emotion": emotion,
                            "description": meme.description,
                        }
                    )

            with open(self._memes_file, "w", encoding="utf-8") as f:
                json.dump(memes_data.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(
                f"成功保存 {len(memes_data.memes)} 个表情数据到 {self._memes_file}"
            )
            return True
        except Exception as e:
            logger.error(f"保存表情数据失败: {e}")
            return False

    def load_embeddings(self) -> bool:
        """从 embeddings.json.gz 加载词嵌入数据

        Returns:
            是否成功加载
        """
        if self.memes_table.embedding_config is None:
            logger.info("未启用词嵌入，跳过加载词嵌入数据")
            return False

        if not self._embeddings_file.exists():
            logger.debug(f"词嵌入数据文件不存在: {self._embeddings_file}")
            return False

        try:
            with gzip.open(self._embeddings_file, "rt", encoding="utf-8") as f:
                data = json.load(f)

            embeddings_data = EmbeddingsData.from_dict(data)

            if self.memes_table.embedding_config != embeddings_data.config:
                logger.warning(
                    f"配置的词嵌入为{self.memes_table.embedding_config} ，而词嵌入数据使用{embeddings_data.config}，无法加载"
                )
                return False

            # 加载 emotion 嵌入
            for emotion, embedding in embeddings_data.emotions.items():
                self.memes_table.set_emotion_embedding(emotion, embedding)

            # 加载 description 嵌入
            for path_str, embedding in embeddings_data.descriptions.items():
                try:
                    path = Path(path_str)
                    self.memes_table.set_description_embedding(path, embedding)
                except KeyError:
                    # 该路径的 Meme 不存在（可能已被删除），跳过
                    pass

            logger.info(
                f"成功加载词嵌入数据: {len(embeddings_data.emotions)} 个情感, "
                f"{len(embeddings_data.descriptions)} 个描述, "
                f"维度={embeddings_data.config.dim}"
            )
            return True
        except Exception as e:
            logger.error(f"加载词嵌入数据失败: {e}")
            return False

    def save_embeddings(self) -> bool:
        """保存词嵌入数据到 embeddings.json.gz

        Returns:
            是否成功保存
        """
        if self.memes_table.embedding_config is None:
            logger.info("未启用词嵌入，跳过加载词嵌入数据")
            return False

        try:
            embeddings_data = EmbeddingsData(self.memes_table.embedding_config)

            # 保存 emotion 嵌入
            for emotion in self.memes_table.get_all_emotions():
                embedding = self.memes_table.get_emotion_embedding(emotion)
                if embedding:
                    embeddings_data.emotions[emotion] = embedding

            # 保存 description 嵌入
            for meme in self.memes_table.get_all_memes():
                if meme.description_embedding:
                    embeddings_data.descriptions[str(meme.internal_path)] = (
                        meme.description_embedding
                    )

            with gzip.open(self._embeddings_file, "wt", encoding="utf-8") as f:
                json.dump(embeddings_data.to_dict(), f)

            logger.info(
                f"成功保存词嵌入数据: {len(embeddings_data.emotions)} 个情感, "
                f"{len(embeddings_data.descriptions)} 个描述"
            )
            return True
        except Exception as e:
            logger.error(f"保存词嵌入数据失败: {e}")
            return False

    async def compute_emotion_embedding(self, emotion: str) -> list[float] | None:
        """计算单个 Emotion 的词嵌入

        Args:
            emotion: 情感词

        Returns:
            词嵌入向量，如果失败返回 None
        """
        provider = self.get_embedding_provider()
        if not provider:
            logger.warning("无可用的 EmbeddingProvider，无法计算情感嵌入")
            return None

        try:
            embedding = await provider.get_embedding(emotion)
            logger.debug(f"成功计算情感嵌入: {emotion}")
            return embedding
        except Exception as e:
            logger.error(f"计算情感嵌入失败 '{emotion}': {e}")
            return None

    async def compute_description_embedding(
        self, description: str
    ) -> list[float] | None:
        """计算描述文本的词嵌入

        Args:
            description: 描述文本

        Returns:
            词嵌入向量，如果失败返回 None
        """
        provider = self.get_embedding_provider()
        if not provider:
            logger.warning("无可用的 EmbeddingProvider，无法计算描述嵌入")
            return None

        try:
            embedding = await provider.get_embedding(description)
            logger.debug(f"成功计算描述嵌入: {description[:30]}...")
            return embedding
        except Exception as e:
            logger.error(f"计算描述嵌入失败 '{description[:30]}...': {e}")
            return None

    async def compute_embeddings_batch(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> tuple[int, int]:
        """批量计算所有缺失的词嵌入（更高效）

        使用 provider.get_embeddings_batch() 进行批量计算。

        Args:
            progress_callback: 进度回调函数，参数为 (current, total, message)

        Returns:
            (成功数量, 失败数量)
        """
        provider = self.get_embedding_provider()
        if not provider:
            logger.warning("无可用的 EmbeddingProvider，无法计算词嵌入")
            return (0, 0)

        success_count = 0
        fail_count = 0

        # 1. 批量计算 Emotion 嵌入
        emotions_without_embedding = self.memes_table.get_emotions_without_embedding()
        if emotions_without_embedding:
            logger.info(f"开始计算 {len(emotions_without_embedding)} 个情感嵌入...")
            if progress_callback:
                progress_callback(
                    0, len(emotions_without_embedding), "批量计算情感嵌入..."
                )

            try:
                embeddings = await provider.get_embeddings(emotions_without_embedding)
                for emotion, embedding in zip(emotions_without_embedding, embeddings):
                    self.memes_table.set_emotion_embedding(emotion, embedding)
                    success_count += 1
                logger.info(f"批量计算情感嵌入完成: {success_count} 个成功")
            except Exception as e:
                logger.warning(f"批量计算情感嵌入失败，改为逐个计算: {e}")
                # 批量失败，逐个尝试
                for emotion in emotions_without_embedding:
                    embedding = await self.compute_emotion_embedding(emotion)
                    if embedding:
                        self.memes_table.set_emotion_embedding(emotion, embedding)
                        success_count += 1
                    else:
                        fail_count += 1

        # 2. 批量计算 Description 嵌入
        memes_without_embedding = (
            self.memes_table.get_memes_without_description_embedding()
        )
        if memes_without_embedding:
            logger.info(f"开始计算 {len(memes_without_embedding)} 个描述嵌入...")
            descriptions = [m.description for m in memes_without_embedding]

            if progress_callback:
                progress_callback(0, len(descriptions), "批量计算描述嵌入...")

            try:
                embeddings = await provider.get_embeddings(descriptions)
                for meme, embedding in zip(memes_without_embedding, embeddings):
                    self.memes_table.set_description_embedding(
                        meme.internal_path, embedding
                    )
                    success_count += 1
                logger.info(f"批量计算描述嵌入完成: {success_count} 个成功")
            except Exception as e:
                logger.warning(f"批量计算描述嵌入失败，改为逐个计算: {e}")
                # 批量失败，逐个尝试
                for meme in memes_without_embedding:
                    embedding = await self.compute_description_embedding(
                        meme.description
                    )
                    if embedding:
                        self.memes_table.set_description_embedding(
                            meme.internal_path, embedding
                        )
                        success_count += 1
                    else:
                        fail_count += 1

        logger.info(f"词嵌入计算完成: 成功={success_count}, 失败={fail_count}")
        return (success_count, fail_count)

    async def search(
        self,
        emotion_query: str,
        description_query: str,
        max_candidates: int = 10,
        use_embedding: bool = True,
        fallback_to_fuzzy: bool = True,
    ) -> list[EmbeddingSearchResult] | list[FuzzySearchResult]:
        """统一查询接口

        优先使用词嵌入查询，失败时可选回退到模糊匹配。

        Args:
            query: 查询文本
            max_candidates: 最多返回的候选项数量
            use_embedding: 是否使用词嵌入查询
            fallback_to_fuzzy: 当词嵌入查询失败时是否回退到模糊匹配

        Returns:
            搜索结果列表，类型为 EmbeddingSearchResult 或 FuzzySearchResult，
            按分数降序排序。
        """
        if use_embedding:
            provider = self.get_embedding_provider()
            if provider:
                try:
                    emotion_query_embedding = await provider.get_embedding(
                        emotion_query
                    )
                    description_query_embedding = await provider.get_embedding(
                        description_query
                    )
                    results = self.memes_table.search_by_embedding(
                        emotion_query_embedding,
                        description_query_embedding,
                        max_candidates,
                    )
                    logger.info(
                        f"词嵌入搜索完成: 查询='{emotion_query}' '{description_query}', 结果数={len(results)}"
                    )
                    return results
                except Exception as e:
                    logger.warning(f"词嵌入搜索失败，尝试回退到模糊匹配: {e}")

        if fallback_to_fuzzy or not use_embedding:
            results = self.memes_table.search_keyword(
                emotion_query, description_query, max_candidates
            )
            logger.info(
                f"词嵌入搜索完成: 查询='{emotion_query}' '{description_query}', 结果数={len(results)}"
            )
            return results

        return []

    def clear_orphan_embeddings(self) -> int:
        """清理孤儿向量（memes.json 中已不存在的 Meme 的嵌入）

        Returns:
            清理的嵌入数量
        """
        if self.memes_table.embedding_config is None:
            logger.warning("未启用词嵌入模型，跳过清理孤儿向量")
            return 0

        logger.debug("开始清理孤儿向量...")

        # 获取当前所有 Meme 的路径
        current_paths = set()
        for memes in self.memes_table.by_emotion.values():
            for meme in memes:
                current_paths.add(str(meme.internal_path))

        # 从 embeddings 数据中清理
        cleaned = 0
        embeddings_data = EmbeddingsData(self.memes_table.embedding_config)

        # 重新加载嵌入数据
        if self._embeddings_file.exists():
            try:
                with gzip.open(self._embeddings_file, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                embeddings_data = EmbeddingsData.from_dict(data)
            except Exception as e:
                logger.warning(f"加载嵌入数据失败，无法清理孤儿向量: {e}")

        # 清理 description 嵌入中的孤儿
        orphan_paths = [
            path for path in embeddings_data.descriptions if path not in current_paths
        ]
        for path in orphan_paths:
            del embeddings_data.descriptions[path]
            cleaned += 1

        # 清理 emotion 嵌入中的孤儿
        current_emotions = {
            emo
            for emo in self.memes_table.get_all_emotions()
            if len(self.memes_table.by_emotion.get(emo, [])) != 0
        }
        orphan_emotions = [
            emotion
            for emotion in embeddings_data.emotions
            if emotion not in current_emotions
        ]
        for emotion in orphan_emotions:
            del embeddings_data.emotions[emotion]
            del self.memes_table.by_emotion[emotion]
            cleaned += 1

        # 保存清理后的数据
        if cleaned > 0:
            try:
                with gzip.open(self._embeddings_file, "wt", encoding="utf-8") as f:
                    json.dump(embeddings_data.to_dict(), f)
                logger.info(f"成功清理 {cleaned} 个孤儿向量")
            except Exception as e:
                logger.error(f"保存清理后的嵌入数据失败: {e}")

        if cleaned == 0:
            logger.debug("未发现孤儿向量")

        return cleaned

    def get_stats(self) -> MemesStats:
        """获取统计信息

        Returns:
            统计信息对象
        """
        total_memes = len(self.memes_table.get_all_memes())
        total_emotions = len(self.memes_table.get_all_emotions())
        emotions_with_embedding = total_emotions - len(
            self.memes_table.get_emotions_without_embedding()
        )
        memes_with_description_embedding = total_memes - len(
            self.memes_table.get_memes_without_description_embedding()
        )

        embedding_enabled = self.memes_table.embedding_config is not None
        embedding_provider_id = None
        embedding_dim = None

        if self.memes_table.embedding_config is not None:
            embedding_provider_id = self.memes_table.embedding_config.provider_id
            embedding_dim = self.memes_table.embedding_config.dim

        return MemesStats(
            total_memes=total_memes,
            total_emotions=total_emotions,
            embedding_enabled=embedding_enabled,
            emotions_with_embedding=emotions_with_embedding,
            memes_with_description_embedding=memes_with_description_embedding,
            embedding_provider_id=embedding_provider_id,
            embedding_dim=embedding_dim,
        )

    async def add_meme(
        self,
        emotion: str,
        path: Path,
        description: str,
        compute_embedding: bool = True,
        save: bool = True,
    ) -> bool:
        """添加一个新的表情

        Args:
            emotion: 情感类别
            path: 表情文件路径（相对路径）
            description: 表情描述
            compute_embedding: 是否立即计算词嵌入
            save: 是否立即保存到文件

        Returns:
            是否成功添加
        """
        try:
            meme = Meme(
                internal_path=path,
                description=description,
            )
            self.memes_table.add(emotion, meme)
            logger.info(f"添加表情: {path} -> {emotion}")

            if compute_embedding and self.memes_table.embedding_config is not None:
                # 计算情感嵌入（如果还没有）
                if self.memes_table.get_emotion_embedding(emotion) is None:
                    emotion_embedding = await self.compute_emotion_embedding(emotion)
                    if emotion_embedding:
                        self.memes_table.set_emotion_embedding(
                            emotion, emotion_embedding
                        )

                # 计算描述嵌入
                desc_embedding = await self.compute_description_embedding(description)
                if desc_embedding:
                    self.memes_table.set_description_embedding(path, desc_embedding)

            if save:
                self.save_memes()
                if self.memes_table.embedding_config is not None:
                    self.save_embeddings()

            return True
        except Exception as e:
            logger.error(f"添加表情失败: {e}")
            return False

    def remove_meme(self, path: Path, save: bool = True) -> bool:
        """删除一个表情

        Args:
            path: 表情文件路径
            save: 是否立即保存到文件

        Returns:
            是否成功删除（如果表情不存在则返回 False）
        """
        try:
            removed = self.memes_table.remove(path)
            if removed:
                logger.info(f"删除表情: {path}")
                if save:
                    self.save_memes()
                    if self.memes_table.embedding_config is not None:
                        self.save_embeddings()
                return True
            else:
                logger.warning(f"表情不存在: {path}")
                return False
        except Exception as e:
            logger.error(f"删除表情失败: {e}")
            return False

    async def initialize(self) -> bool:
        """初始化：加载数据并计算缺失的嵌入

        Returns:
            是否成功初始化
        """
        logger.info("开始初始化 MemesEmbeddingManager...")

        # 1. 加载 memes.json
        logger.debug("加载 memes.json")
        self.load_memes()

        # 2. 加载 embeddings.json.gz
        logger.debug("加载 embeddings.json.gz")
        self.load_embeddings()

        # 3. 检查是否有可用的 embedding provider
        logger.debug("检查 EmbeddingProvider")
        provider = self.get_embedding_provider()
        if provider is None:
            # 没有 provider，只能使用已有的嵌入数据
            logger.info("无可用的 EmbeddingProvider，跳过词嵌入计算")
            return True

        # 5. 计算缺失的嵌入
        await self.compute_embeddings_batch()

        # 6. 保存嵌入数据
        self.save_embeddings()

        return True
