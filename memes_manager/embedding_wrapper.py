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

from .atomic_write import atomic_write_gzip_json, atomic_write_json
from .errors import (
    MemesEmbeddingDisabledError,
    MemesEmbeddingError,
    MemesFileError,
    MemesMemeNotFoundError,
    MemesParseError,
    MemesProviderNotFoundError,
)
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

    Contract:
        - 所有涉及文件操作的方法可能抛出 MemesFileError
        - 所有涉及嵌入计算的方法可能抛出 MemesEmbeddingError
        - 当 embedding_config 为 None 时，嵌入相关方法抛出 MemesEmbeddingDisabledError
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
            MemesProviderNotFoundError: 配置的 provider ID 找不到
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
            raise MemesProviderNotFoundError(
                provider_id=self.memes_table.embedding_config.provider_id,
                provider_type="EmbeddingProvider",
            )
        else:
            self._embedding_provider = None

        return self._embedding_provider

    def load_memes(self) -> None:
        """从 memes.json 加载表情数据

        Raises:
            MemesFileError: 文件读取失败
            MemesParseError: JSON 解析失败
        """
        if not self._memes_file.exists():
            logger.info(f"表情数据文件不存在: {self._memes_file}")
            return

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
        except json.JSONDecodeError as e:
            raise MemesParseError(f"解析 {self._memes_file} 失败", e)
        except OSError:
            raise MemesFileError(f"读取 {self._memes_file} 失败", self._memes_file)
        except Exception as e:
            raise MemesParseError(f"加载表情数据失败: {e}")

    def save_memes(self) -> None:
        """保存表情数据到 memes.json

        使用原子写入确保数据完整性。

        Raises:
            MemesFileError: 文件写入失败
        """
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

        # 原子写入
        atomic_write_json(self._memes_file, memes_data.to_dict())

        logger.info(f"成功保存 {len(memes_data.memes)} 个表情数据到 {self._memes_file}")

    def load_embeddings(self) -> bool:
        """从 embeddings.json.gz 加载词嵌入数据

        Returns:
            是否成功加载（如果未启用词嵌入或文件不存在则返回 False）

        Raises:
            MemesFileError: 文件读取失败
            MemesParseError: JSON 解析失败
            MemesEmbeddingError: 配置不匹配
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
                    f"配置的词嵌入为{self.memes_table.embedding_config}，而词嵌入数据使用{embeddings_data.config}，无法加载"
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
                except MemesMemeNotFoundError:
                    # 该路径的 Meme 不存在（可能已被删除），跳过
                    logger.debug(f"跳过已删除表情的嵌入: {path_str}")
                    pass

            logger.info(
                f"成功加载词嵌入数据: {len(embeddings_data.emotions)} 个情感, "
                f"{len(embeddings_data.descriptions)} 个描述, "
                f"维度={embeddings_data.config.dim}"
            )
            return True
        except json.JSONDecodeError as e:
            raise MemesParseError(f"解析 {self._embeddings_file} 失败", e)
        except OSError:
            raise MemesFileError(
                f"读取 {self._embeddings_file} 失败", self._embeddings_file
            )
        except MemesEmbeddingError:
            raise
        except Exception as e:
            raise MemesEmbeddingError("加载词嵌入数据失败", e)

    def save_embeddings(self) -> None:
        """保存词嵌入数据到 embeddings.json.gz

        使用原子写入确保数据完整性。

        Raises:
            MemesFileError: 文件写入失败

        Note:
            如果词嵌入未启用（embedding_config 为 None），此方法会静默跳过，
            不会抛出异常。调用者应在需要时先检查 embedding_config 是否为 None。
        """
        if self.memes_table.embedding_config is None:
            logger.debug("未启用词嵌入，跳过保存词嵌入数据")
            return

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

        # 原子写入
        atomic_write_gzip_json(self._embeddings_file, embeddings_data.to_dict())

        logger.info(
            f"成功保存词嵌入数据: {len(embeddings_data.emotions)} 个情感, "
            f"{len(embeddings_data.descriptions)} 个描述"
        )

    async def compute_emotion_embedding(self, emotion: str) -> list[float]:
        """计算单个 Emotion 的词嵌入

        Args:
            emotion: 情感词

        Returns:
            词嵌入向量

        Raises:
            MemesEmbeddingDisabledError: 没有可用的 EmbeddingProvider
            MemesEmbeddingError: 计算失败
        """
        provider = self.get_embedding_provider()
        if not provider:
            raise MemesEmbeddingDisabledError()

        try:
            embedding = await provider.get_embedding(emotion)
            logger.debug(f"成功计算情感嵌入: {emotion}")
            return embedding
        except Exception as e:
            raise MemesEmbeddingError(f"计算情感嵌入失败 '{emotion}'", e)

    async def compute_description_embedding(self, description: str) -> list[float]:
        """计算描述文本的词嵌入

        Args:
            description: 描述文本

        Returns:
            词嵌入向量

        Raises:
            MemesEmbeddingDisabledError: 没有可用的 EmbeddingProvider
            MemesEmbeddingError: 计算失败
        """
        provider = self.get_embedding_provider()
        if not provider:
            raise MemesEmbeddingDisabledError()

        try:
            embedding = await provider.get_embedding(description)
            logger.debug(f"成功计算描述嵌入: {description[:30]}...")
            return embedding
        except Exception as e:
            raise MemesEmbeddingError(f"计算描述嵌入失败 '{description[:30]}...'", e)

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

        Note:
            此方法会尽可能完成计算，单个失败不会中断整体流程。
            失败会在日志中记录警告。
        """
        provider = self.get_embedding_provider()
        if not provider:
            logger.info("无可用的 EmbeddingProvider，跳过词嵌入计算")
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
                    try:
                        embedding = await self.compute_emotion_embedding(emotion)
                        self.memes_table.set_emotion_embedding(emotion, embedding)
                        success_count += 1
                    except MemesEmbeddingError as embed_err:
                        logger.warning(f"计算情感嵌入失败 '{emotion}': {embed_err}")
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
                    try:
                        embedding = await self.compute_description_embedding(
                            meme.description
                        )
                        self.memes_table.set_description_embedding(
                            meme.internal_path, embedding
                        )
                        success_count += 1
                    except MemesEmbeddingError as embed_err:
                        logger.warning(
                            f"计算描述嵌入失败 '{meme.description[:30]}...': {embed_err}"
                        )
                        fail_count += 1

        logger.info(f"词嵌入计算完成: 成功={success_count}, 失败={fail_count}")
        return (success_count, fail_count)

    async def search(
        self,
        queries: list[str],
        max_candidates: int = 10,
        use_embedding: bool = True,
        fallback_to_fuzzy: bool = True,
    ) -> list[EmbeddingSearchResult] | list[FuzzySearchResult]:
        """统一查询接口

        优先使用词嵌入查询，失败时可选回退到模糊匹配。

        Args:
            queries: 查询文本列表
            max_candidates: 最多返回的候选项数量
            use_embedding: 是否使用词嵌入查询
            fallback_to_fuzzy: 当词嵌入查询失败时是否回退到模糊匹配

        Returns:
            搜索结果列表，类型为 EmbeddingSearchResult 或 FuzzySearchResult，
            按分数降序排序。

        Precondition:
            len(queries) > 0
            max_candidates > 0
        """
        if len(queries) == 0:
            raise RuntimeError("queries 不能为空")
        if max_candidates <= 0:
            raise RuntimeError("max_candidates 必须大于 0")

        if use_embedding:
            provider = self.get_embedding_provider()
            if provider:
                try:
                    embeddings = await provider.get_embeddings(queries)
                    results = self.memes_table.search_by_embedding(
                        embeddings,
                        max_candidates,
                    )
                    logger.info(
                        f"词嵌入搜索完成: 查询='{queries}', 结果数={len(results)}"
                    )
                    return results
                except Exception as e:
                    logger.warning(f"词嵌入搜索失败，尝试回退到模糊匹配: {e}")

        if fallback_to_fuzzy or not use_embedding:
            results = self.memes_table.search_keyword(queries, max_candidates)
            logger.info(f"模糊匹配搜索完成: 查询='{queries}', 结果数={len(results)}")
            return results

        return []

    def clear_orphan_embeddings(self) -> int:
        """清理孤儿向量（memes.json 中已不存在的 Meme 的嵌入）

        Returns:
            清理的嵌入数量

        Raises:
            MemesFileError: 保存清理后的数据失败
        """
        if self.memes_table.embedding_config is None:
            logger.info("未启用词嵌入模型，跳过清理孤儿向量")
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
                return 0

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
            self._save_embeddings_data(embeddings_data)
            logger.info(f"成功清理 {cleaned} 个孤儿向量")

        if cleaned == 0:
            logger.debug("未发现孤儿向量")

        return cleaned

    def _save_embeddings_data(self, embeddings_data: EmbeddingsData) -> None:
        """保存嵌入数据到文件（原子写入）

        Args:
            embeddings_data: 要保存的嵌入数据

        Raises:
            MemesFileError: 保存失败
        """
        atomic_write_gzip_json(self._embeddings_file, embeddings_data.to_dict())

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
    ) -> None:
        """添加一个新的表情

        Args:
            emotion: 情感类别
            path: 表情文件路径（相对路径）
            description: 表情描述
            compute_embedding: 是否立即计算词嵌入
            save: 是否立即保存到文件

        Raises:
            MemesFileError: 保存文件失败
            MemesEmbeddingError: 计算嵌入失败（仅当 compute_embedding=True 时）
        """
        meme = Meme(
            internal_path=path,
            description=description,
        )
        self.memes_table.add(emotion, meme)
        logger.info(f"添加表情: {path} -> {emotion}")

        if compute_embedding and self.memes_table.embedding_config is not None:
            # 计算情感嵌入（如果还没有）
            if self.memes_table.get_emotion_embedding(emotion) is None:
                try:
                    emotion_embedding = await self.compute_emotion_embedding(emotion)
                    self.memes_table.set_emotion_embedding(emotion, emotion_embedding)
                except MemesEmbeddingError as e:
                    logger.warning(f"计算情感嵌入失败，跳过: {e}")

            # 计算描述嵌入
            try:
                desc_embedding = await self.compute_description_embedding(description)
                self.memes_table.set_description_embedding(path, desc_embedding)
            except MemesEmbeddingError as e:
                logger.warning(f"计算描述嵌入失败，跳过: {e}")

        if save:
            self.save_memes()
            if self.memes_table.embedding_config is not None:
                self.save_embeddings()

    def remove_meme(self, path: Path, save: bool = True) -> None:
        """删除一个表情

        Args:
            path: 表情文件路径
            save: 是否立即保存到文件

        Raises:
            MemesMemeNotFoundError: 表情不存在
            MemesFileError: 保存文件失败
        """
        removed = self.memes_table.remove(path)
        if removed is None:
            raise MemesMemeNotFoundError(path)

        logger.info(f"删除表情: {path}")
        if save:
            self.save_memes()
            if self.memes_table.embedding_config is not None:
                self.save_embeddings()

    def contains(self, path: Path) -> bool:
        return self.memes_table.contains(path)

    async def update_meme(
        self,
        emotion: str,
        old_path: Path,
        new_path: Path,
        description: str,
        compute_embedding: bool = True,
        save: bool = True,
    ) -> None:
        """更新一个已有的表情

        Args:
            emotion: 情感类别
            path: 表情文件路径（相对路径）
            description: 表情描述
            compute_embedding: 是否立即计算词嵌入
            save: 是否立即保存到文件

        Raises:
            MemesMemeNotFoundError: 表情不存在
            MemesFileError: 保存文件失败
            MemesEmbeddingError: 计算嵌入失败（仅当 compute_embedding=True 时）
        """
        removed_emotion = self.memes_table.remove(
            old_path, remove_emotion_embedding=False
        )
        if removed_emotion is None:
            raise MemesMemeNotFoundError(old_path)

        logger.info(f"更新表情：删除 {old_path} 的旧记录")

        await self.add_meme(emotion, new_path, description, compute_embedding, save)

        self.memes_table.clean_emotion_embedding(removed_emotion)

    async def initialize(self) -> None:
        """初始化：加载数据并计算缺失的嵌入

        Raises:
            MemesFileError: 加载表情数据文件失败
            MemesParseError: 解析表情数据失败

        Note:
            词嵌入相关的错误（如加载嵌入数据失败、计算嵌入失败）会被记录为警告，
            不会中断初始化流程，因为表情管理器可以在没有词嵌入的情况下工作。
        """
        logger.info("开始初始化 MemesEmbeddingManager...")

        # 1. 加载 memes.json（必须成功）
        logger.debug("加载 memes.json")
        self.load_memes()

        # 2. 加载 embeddings.json.gz（可选，失败不影响核心功能）
        logger.debug("加载 embeddings.json.gz")
        try:
            self.load_embeddings()
        except (MemesFileError, MemesParseError, MemesEmbeddingError) as e:
            logger.warning(f"加载词嵌入数据失败，将使用模糊匹配搜索: {e}")

        # 3. 检查是否有可用的 embedding provider
        logger.debug("检查 EmbeddingProvider")
        try:
            provider = self.get_embedding_provider()
        except MemesProviderNotFoundError as e:
            logger.warning(f"EmbeddingProvider 配置错误: {e}")
            provider = None

        if provider is None:
            # 没有 provider，只能使用已有的嵌入数据
            logger.info("无可用的 EmbeddingProvider，跳过词嵌入计算")
            return

        # 4. 计算缺失的嵌入
        try:
            await self.compute_embeddings_batch()
        except Exception as e:
            logger.warning(f"批量计算词嵌入失败: {e}")

        # 5. 保存嵌入数据
        try:
            self.save_embeddings()
        except (MemesFileError, MemesEmbeddingError) as e:
            logger.warning(f"保存词嵌入数据失败: {e}")

        logger.info("MemesEmbeddingManager 初始化完成")
