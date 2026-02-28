"""MemesManager 顶层接口

提供表情包管理的统一接口，包括：
- 表情文件管理（复制、删除、列表）
- LLM 自动生成情感和描述
- 词嵌入搜索功能
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from astrbot.api import logger
from astrbot.core.provider.provider import Provider
from astrbot.core.star.context import Context

from .embedding_wrapper import MemesEmbeddingManager
from .index import (
    EmbeddingConfig,
    EmbeddingSearchResult,
    Emotion,
    FuzzySearchResult,
    Meme,
    MemesStats,
)


@dataclass
class MemesManagerConfig:
    """MemesManager 配置

    Attributes:
        data_dir: 数据目录，表情文件和配置将存放在此目录下
        chat_provider_id: 用于 LLM 生成的 provider ID，用于自动生成情感和描述
        embedding_provider_id: 用于词嵌入的 provider ID，用于语义搜索
    """

    data_dir: Path
    max_candidates: int
    chat_provider_id: str | None = None
    embedding_provider_id: str | None = None


class MemesManager:
    """顶层表情管理器

    包装 MemesEmbeddingManager，提供以下功能：
    - 表情文件管理：将文件复制到统一目录、删除文件等
    - LLM 生成：调用 LLM 自动生成表情的情感标签和描述
    - 搜索功能：基于词嵌入或模糊匹配搜索表情
    """

    def __init__(
        self,
        context: Context,
        config: MemesManagerConfig,
    ):
        """初始化 MemesManager

        Args:
            context: AstrBot 上下文，用于获取 Provider
            config: 配置对象
        """
        self.context = context
        self.config = config
        self._initialized = False

        # 确保数据目录存在
        config.data_dir.mkdir(parents=True, exist_ok=True)

        # 表情文件统一存放在 data_dir/memes 下
        self.memes_dir = config.data_dir / "memes"
        self.memes_dir.mkdir(parents=True, exist_ok=True)

        # 创建 EmbeddingConfig
        embedding_config = None
        if config.embedding_provider_id:
            providers = context.get_all_embedding_providers()
            for provider in providers:
                if provider.meta().id == config.embedding_provider_id:
                    embedding_config = EmbeddingConfig(
                        provider_id=config.embedding_provider_id,
                        dim=provider.get_dim(),
                    )
                    break

        # 创建 MemesEmbeddingManager
        from .index import MemesTable

        if embedding_config is not None:
            logger.info(
                f"词嵌入已启用，使用模型{embedding_config.provider_id}，维数{embedding_config.dim}"
            )

        memes_table = MemesTable(embedding_config)
        self.embedding_manager = MemesEmbeddingManager(
            memes_table=memes_table,
            context=context,
            data_dir=config.data_dir,
        )

    async def initialize(self) -> bool:
        """初始化管理器，加载现有数据

        Returns:
            是否成功初始化
        """
        if self._initialized:
            logger.warning("MemesManager 已经初始化过了")
            return True

        logger.info("初始化 MemesManager...")
        logger.info(f"配置{self.config}")

        result = await self.embedding_manager.initialize()
        if result:
            self._initialized = True
            logger.info("MemesManager 初始化成功")
        else:
            logger.error("MemesManager 初始化失败")
        return result

    async def add_meme_from_file(
        self,
        file_path: Path,
        emotion: str | None = None,
        memo: str | None = None,
        description: str | None = None,
        auto_generate: bool = True,
        copy_file: bool = True,
    ) -> Path | None:
        """从文件添加表情

        Args:
            file_path: 表情文件路径
            emotion: 情感标签，如果为 None 且 auto_generate=True，则自动生成
            description: 描述，如果为 None 且 auto_generate=True，则自动生成
            auto_generate: 是否自动生成情感和描述（使用 LLM）
            copy_file: 是否将文件复制到统一目录，False 则使用原路径

        Returns:
            添加后的相对路径，如果成功呢添加
        """
        if not self._initialized:
            raise RuntimeError("未初始化")

        try:
            # 检查文件是否存在
            if not file_path.exists():
                logger.error(f"表情文件不存在: {file_path}")
                return None

            # 自动生成情感和描述
            if auto_generate and (emotion is None or description is None):
                (
                    gen_emotion,
                    gen_secondary,
                    gen_description,
                ) = await self.generate_emotion_and_description(file_path)
                if emotion is None:
                    emotion = gen_emotion
                if description is None:
                    description = gen_description
                if memo is None:
                    memo = gen_secondary

            if emotion is None or description is None or memo is None:
                raise RuntimeError("没有设置情感和描述，并且调用LLM生成失败")

            # 确定最终存储路径
            if copy_file:
                dest_path = await self._copy_to_memes_dir(file_path, emotion, memo)
                if dest_path is None:
                    return None
            else:
                dest_path = file_path

            # 添加到 embedding manager
            if await self.embedding_manager.add_meme(
                emotion=emotion,
                path=dest_path,
                description=description,
                compute_embedding=True,
                save=True,
            ):
                return dest_path
            else:
                return None
        except Exception as e:
            logger.error(f"添加表情失败: {e}")
            return None

    async def _copy_to_memes_dir(
        self, file_path: Path, emotion: str, memo: str
    ) -> Path | None:
        """将文件复制到 memes 目录

        Args:
            file_path: 源文件路径
            emotion: 情感类别
            memo: 助记名称
        Returns:
            目标文件路径，相对于self.memes_dir，失败返回 None
        """
        try:
            # 生成目标路径，避免重名
            dest_path = self.memes_dir / emotion / f"{memo}{file_path.suffix}"
            if dest_path.exists():
                # 如果文件已存在，添加序号
                counter = 1
                while dest_path.exists():
                    dest_path = self.memes_dir / f"{memo}_{counter}{file_path.suffix}"
                    counter += 1

            # 确保目标目录存在
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # 复制文件
            shutil.copy2(file_path, dest_path)
            logger.info(f"复制表情文件: {file_path} -> {dest_path}")
            # 返回相对于 data_dir 的相对路径
            relative_path = dest_path.relative_to(self.memes_dir)
            return relative_path
        except Exception as e:
            logger.error(f"复制文件失败: {e}")
            return None

    async def generate_emotion_and_description(
        self,
        image_path: Path,
    ) -> tuple[str | None, str | None, str | None]:
        """调用 LLM 分析图片生成情感和描述

        Args:
            image_path: 图片路径

        Returns:
            (emotion, memo, description) 元组，如果失败则返回 (None, None, None)
        """

        logger.debug(f"为图片{image_path}调用LLM生成描述")

        if not self.config.chat_provider_id:
            logger.warning("未配置 chat_provider_id，无法调用 LLM 生成")
            return (None, None, None)

        try:
            # 获取 chat provider
            provider = self.context.get_provider_by_id(self.config.chat_provider_id)
            if not provider or not isinstance(provider, Provider):
                logger.error(f"找不到 Chat Provider: {self.config.chat_provider_id}")
                return (None, None, None)

            # 构建 prompt
            prompt = """请分析这张表情包图片，并返回以下 JSON 格式的结果：
{
    "keyword": "用一个中文词语描述这个图片的关键词，关键词表达表情的核心情感或者动作，如高兴、震惊、伤心、恼怒、后悔、害怕，或者敬礼、祈祷、奔跑等。优先使用情感词。",
    "secondary": "用一个中文词语描述这个图片的次级关键词，必须要和keyword不同。如果keyword是情感词，则此处不要再使用情感词。",
    "description": "用一句简短的中文描述这个表情的内容或含义"
}
如果下列某词语适合作为这张图片的关键词，则选择它：
""" + ",".join(self.get_all_emotions())

            # 构造图片 URL（本地文件使用 file:/// 协议）
            image_url = f"file:///{image_path.absolute()}"

            # 调用 LLM
            response = await self.context.llm_generate(
                chat_provider_id=self.config.chat_provider_id,
                prompt=prompt,
                image_urls=[image_url],
            )

            # 解析响应

            result_text = response.completion_text.strip()
            # 尝试提取 JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            emotion = result.get("keyword")
            secondary = result.get("secondary")
            description = result.get("description")

            logger.info(
                f"LLM 生成结果: emotion={emotion}, secondary={secondary}, description={description}"
            )
            return (emotion, secondary, description)

        except json.JSONDecodeError as e:
            logger.error(f"解析 LLM 响应失败: {e}")
            return (None, None, None)
        except Exception as e:
            logger.error(f"调用 LLM 生成失败: {e}")
            return (None, None, None)

    async def search(
        self, emotion_query: str, description_query: str
    ) -> list[EmbeddingSearchResult] | list[FuzzySearchResult]:
        """搜索表情

        优先使用词嵌入搜索，如果没有词嵌入则回退到模糊匹配。

        Args:
            query: 查询文本
            max_candidates: 最多返回的候选项数量

        Returns:
            搜索结果列表
        """
        if not self._initialized:
            raise RuntimeError("未初始化")

        return await self.embedding_manager.search(
            emotion_query, description_query, self.config.max_candidates
        )

    def list_memes(self) -> list[tuple[Emotion, Meme]]:
        """列出所有表情

        Returns:
            表情列表，每个元素是一个 (Emotion, Meme) 元组
        """
        if not self._initialized:
            raise RuntimeError("MemesManager 未初始化，请先调用 initialize()")

        result = []
        for emotion, meme_list in self.embedding_manager.memes_table.by_emotion.items():
            for meme in meme_list:
                result.append((emotion, meme))
        return result

    def get_stats(self) -> MemesStats:
        """获取统计信息

        Returns:
            统计信息对象，包含总数、情感数量、嵌入状态等
        """
        if not self._initialized:
            raise RuntimeError("MemesManager 未初始化，请先调用 initialize()")

        return self.embedding_manager.get_stats()

    def remove_meme(self, path: Path, delete_file: bool = False) -> bool:
        """删除表情

        Args:
            path: 表情文件路径
            delete_file: 是否同时删除文件

        Returns:
            是否成功删除
        """
        if not self._initialized:
            raise RuntimeError("未初始化")

        try:
            # 从 embedding manager 删除
            result = self.embedding_manager.remove_meme(path, save=True)

            # 如果需要删除文件且删除成功
            if result and delete_file:
                if path.exists():
                    path.unlink()
                    logger.info(f"删除表情文件: {path}")

            return result
        except Exception as e:
            logger.error(f"删除表情失败: {e}")
            return False

    def get_meme_by_path(self, path: Path) -> tuple[Emotion, Meme, Path] | None:
        """根据路径获取表情信息

        Args:
            path: 表情相对路径

        Returns:
            (Emotion, Meme, 表情文件绝对路径) 元组，如果不存在则返回 None
        """
        if not self._initialized:
            raise RuntimeError("MemesManager 未初始化，请先调用 initialize()")

        for emotion, meme_list in self.embedding_manager.memes_table.by_emotion.items():
            for meme in meme_list:
                if meme.internal_path == path:
                    file_path = self.memes_dir / path
                    if not file_path.exists():
                        raise RuntimeError(f"文件{file_path}不存在")
                    return (emotion, meme, file_path)
        return None

    def get_memes_by_emotion(self, emotion: str) -> list[Meme]:
        """根据情感获取表情列表

        Args:
            emotion: 情感标签

        Returns:
            Meme 对象列表
        """
        if not self._initialized:
            raise RuntimeError("MemesManager 未初始化，请先调用 initialize()")

        return self.embedding_manager.memes_table.by_emotion.get(emotion, [])

    def get_all_emotions(self) -> list[str]:
        """获取所有情感标签

        Returns:
            情感标签列表
        """
        if not self._initialized:
            raise RuntimeError("MemesManager 未初始化，请先调用 initialize()")

        return self.embedding_manager.memes_table.get_all_emotions()
