from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import BaseMessageComponent, Image, Plain
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .memes_manager import (
    EmbeddingSearchResult,
    FuzzySearchResult,
    MemesManager,
    MemesManagerConfig,
)


@register(
    "astrbot_plugin_memes_by_tool",
    "danzou1ge6",
    "使用工具调用为大模型提供表情包能力",
    "0.0.1",
)
class MyPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.memes_manager: MemesManager | None = None
        embedding_provider_id = config.get("embedding_provider")
        if embedding_provider_id == "":
            embedding_provider_id = None

        self.config = MemesManagerConfig(
            data_dir=Path(get_astrbot_data_path()) / "plugin_data" / self.name,
            chat_provider_id=config.get("chat_provider"),
            embedding_provider_id=embedding_provider_id,
            max_candidates=config["max_candidates"],
        )

    async def initialize(self):
        """初始化插件，创建 MemesManager 实例"""

        self.memes_manager = MemesManager(self.context, self.config)  # type: ignore
        await self.memes_manager.initialize()
        logger.info("表情包管理器初始化完成")

    @filter.command_group("表情工具")
    def meme_tool(self):
        """表情包管理工具"""
        pass

    @meme_tool.command("情感")
    async def list_emotions(self, event: AstrMessageEvent):
        """列出所有情感标签

        用法: /表情工具 情感
        """
        if not self.memes_manager:
            yield event.plain_result("表情管理器未初始化")
            return

        try:
            emotions = self.memes_manager.get_all_emotions()
            if not emotions:
                yield event.plain_result("当前没有任何情感标签")
                return

            result = "所有情感标签：\n"
            for i, emotion in enumerate(emotions, 1):
                memes = self.memes_manager.get_memes_by_emotion(emotion)
                result += f"{i}. {emotion} ({len(memes)} 个表情)\n"

            yield event.plain_result(result.strip())
        except Exception as e:
            logger.error(f"列出情感失败: {e}")
            yield event.plain_result(f"操作失败: {e}")

    @meme_tool.command("列出")
    async def list_memes(self, event: AstrMessageEvent, emotion: str = ""):
        """列出指定情感下的所有表情

        用法: /表情工具 列出 <情感>
        示例: /表情工具 列出 开心
        """
        if not self.memes_manager:
            yield event.plain_result("表情管理器未初始化")
            return

        if not emotion:
            yield event.plain_result("请指定情感标签，例如: /表情工具 列出 开心")
            return

        try:
            memes = self.memes_manager.get_memes_by_emotion(emotion)
            if not memes:
                yield event.plain_result(f"情感 '{emotion}' 下没有任何表情")
                return

            result = f"情感 '{emotion}' 下的表情：\n"
            for i, meme in enumerate(memes, 1):
                result += f"{i}. {meme.internal_path}\n   描述: {meme.description}\n"

            yield event.plain_result(result.strip())
        except Exception as e:
            logger.error(f"列出表情失败: {e}")
            yield event.plain_result(f"操作失败: {e}")

    @meme_tool.command("搜索")
    async def search_meme(
        self,
        event: AstrMessageEvent,
        emotion_query: str = "",
        description_query: str = "",
    ):
        """搜索表情

        用法: /表情工具 搜索 <情感关键词> <描述关键词>
        示例: /表情工具 搜索 开心 大笑
        """
        if not self.memes_manager:
            yield event.plain_result("表情管理器未初始化")
            return

        if not emotion_query or not description_query:
            yield event.plain_result("请指定搜索关键词，例如: /表情工具 搜索 开心")
            return

        try:
            results = await self.memes_manager.search(emotion_query, description_query)
            result_text = self._format_search_results(results)
            yield event.plain_result(result_text)
        except Exception as e:
            logger.error(f"搜索表情失败: {e}")
            yield event.plain_result(f"操作失败: {e}")

    @meme_tool.command("添加")
    async def add_meme(self, event: AstrMessageEvent):
        """添加消息中的所有图片到表情库

        用法: /表情工具 添加 (需要回复包含图片的消息或发送带图片的消息)
        """
        if not self.memes_manager:
            yield event.plain_result("表情管理器未初始化")
            return

        try:
            # 获取消息中的所有图片
            messages = event.get_messages()
            images = []

            for msg in messages:
                if isinstance(msg, Image):
                    images.append(msg)

            if not images:
                yield event.plain_result("未找到图片，请发送带图片的消息")
                return

            yield event.plain_result(f"正在处理 {len(images)} 张图片，请稍候...")

            # 下载并添加每张图片
            success_count = 0
            failed_count = 0

            added_paths = []

            for i, image in enumerate(images, 1):
                try:
                    # 使用 convert_to_file_path 方法获取图片路径
                    image_path = Path(await image.convert_to_file_path())

                    # 添加表情（自动生成情感和描述）
                    success = await self.memes_manager.add_meme_from_file(
                        file_path=image_path,
                        auto_generate=True,  # 自动生成情感和描述
                        copy_file=True,  # 复制到表情库目录
                    )

                    if success is not None:
                        added_paths.append(success)
                        success_count += 1
                        logger.info(f"成功添加图片 {i}/{len(images)}: {image_path}")
                    else:
                        failed_count += 1
                        logger.warning(f"添加图片失败 {i}/{len(images)}: {image_path}")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"处理图片 {i}/{len(images)} 时出错: {e}")

            result_msg = (
                f"处理完成！\n成功添加: {success_count} 张\n失败: {failed_count} 张\n\n"
            )

            for i, path in enumerate(added_paths, 1):
                r = self.memes_manager.get_meme_by_path(path)
                assert r is not None
                emotion, meme, _ = r
                result_msg += f"{i}. [{emotion}] {meme.internal_path}\n"
                result_msg += f"   描述: {meme.description}\n"

            yield event.plain_result(result_msg)
        except Exception as e:
            logger.error(f"添加表情失败: {e}")
            yield event.plain_result(f"操作失败: {e}")

    @meme_tool.command("手动添加")
    async def manual_add_meme(
        self,
        event: AstrMessageEvent,
        emotion: str = "",
        memo: str = "",
        description: str = "",
    ):
        """手动添加表情

        用法: /表情工具 手动添加 <情感> <助记词> <描述>
        示例: /表情工具 手动添加 开心 大笑 一个小人咧开嘴，举起手，开心地大笑
        注意: 需要回复包含图片的消息
        """
        if not self.memes_manager:
            yield event.plain_result("表情管理器未初始化")
            return

        if not emotion or not description:
            yield event.plain_result(
                "用法: /表情工具 手动添加 <情感> <助记词> <描述>"
                "示例: /表情工具 手动添加 开心 大笑 一个小人咧开嘴，举起手，开心地大笑"
            )
            return

        try:
            # 获取消息中的图片
            messages = event.get_messages()
            images = []

            for msg in messages:
                if isinstance(msg, Image):
                    images.append(msg)

            if not images:
                yield event.plain_result("未找到图片，请发送带图片的消息")
                return

            if len(images) > 1:
                yield event.plain_result("请仅发送一张图片")

            yield event.plain_result("正在处理图片...")

            # 下载并添加每张图片
            success_count = 0
            failed_count = 0

            for i, image in enumerate(images, 1):
                try:
                    # 使用 convert_to_file_path 方法获取图片路径
                    image_path = Path(await image.convert_to_file_path())

                    # 添加表情（使用指定的情感和描述）
                    success = await self.memes_manager.add_meme_from_file(
                        file_path=image_path,
                        emotion=emotion,
                        memo=memo,
                        description=description,
                        auto_generate=False,  # 使用指定的情感和描述
                        copy_file=True,  # 复制到表情库目录
                    )

                    if success:
                        success_count += 1
                        logger.info(f"成功添加图片 {i}/{len(images)}: {image_path}")
                    else:
                        failed_count += 1
                        logger.warning(f"添加图片失败 {i}/{len(images)}: {image_path}")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"处理图片 {i}/{len(images)} 时出错: {e}")

            yield event.plain_result(
                f"处理完成！\n成功添加: {success_count} 张\n失败: {failed_count} 张\n"
                f"情感: {emotion}\n描述: {description}"
            )
        except Exception as e:
            logger.error(f"手动添加表情失败: {e}")
            yield event.plain_result(f"操作失败: {e}")

    @meme_tool.command("删除")
    async def delete_meme(self, event: AstrMessageEvent, path: str = ""):
        """删除指定路径的表情

        用法: /表情工具 删除 <路径>
        示例: /表情工具 列出 开心 (先查看表情路径)
              /表情工具 删除 data/plugins/.../meme.jpg
        """
        if not self.memes_manager:
            yield event.plain_result("表情管理器未初始化")
            return

        if not path:
            yield event.plain_result(
                "请指定要删除的表情路径\n"
                "用法: /表情工具 删除 <路径>\n"
                "提示: 使用 '/表情工具 列出 <情感>' 查看表情路径"
            )
            return

        try:
            meme_path = Path(path)
            success = self.memes_manager.remove_meme(meme_path, delete_file=True)

            if success:
                yield event.plain_result(f"成功删除表情: {path}")
            else:
                yield event.plain_result(f"删除失败: 未找到路径为 '{path}' 的表情")
        except Exception as e:
            logger.error(f"删除表情失败: {e}")
            yield event.plain_result(f"操作失败: {e}")

    @meme_tool.command("发送")
    async def send_meme(self, event: AstrMessageEvent, path: str = ""):
        """发送指定路径的表情图片

        用法: /表情工具 发送 <路径>
        示例: /表情工具 列出 开心 (先查看表情路径)
              /表情工具 发送 data/plugins/.../meme.jpg
        """
        if not path:
            yield event.plain_result(
                "请指定要发送的表情路径\n"
                "用法: /表情工具 发送 <路径>\n"
                "提示: 使用 '/表情工具 列出 <情感>' 查看表情路径"
            )
            return

        try:
            assert self.memes_manager is not None
            info = self.memes_manager.get_meme_by_path(Path(path))
            if info is None:
                yield event.plain_result(f"表情{path}不存在")
                return

            emotion, meme, file_path = info

            # 构建消息链
            chain = [
                Plain(f"\n情感: {emotion}\n描述: {meme.description}"),
                Image.fromFileSystem(str(file_path)),
            ]

            yield event.chain_result(chain)

        except Exception as e:
            logger.error(f"发送表情失败: {e}")
            yield event.plain_result(f"操作失败: {e}")

    @meme_tool.command("清理词嵌入")
    async def clean_embeddings(self, event: AstrMessageEvent):
        """清理孤儿词嵌入向量

        用法: /表情工具 清理词嵌入
        """
        if not self.memes_manager:
            yield event.plain_result("表情管理器未初始化")
            return

        try:
            # 获取清理前的统计
            stats_before = self.memes_manager.get_stats()

            # 执行清理
            cleaned_count = (
                self.memes_manager.embedding_manager.clear_orphan_embeddings()
            )

            # 获取清理后的统计
            stats_after = self.memes_manager.get_stats()

            yield event.plain_result(
                f"清理完成！\n"
                f"清理前: {stats_before.emotions_with_embedding} 个情感嵌入, "
                f"{stats_before.memes_with_description_embedding} 个描述嵌入\n"
                f"清理后: {stats_after.emotions_with_embedding} 个情感嵌入, "
                f"{stats_after.memes_with_description_embedding} 个描述嵌入\n"
                f"共清理了 {cleaned_count} 个孤儿向量"
            )
        except Exception as e:
            logger.error(f"清理词嵌入失败: {e}")
            yield event.plain_result(f"操作失败: {e}")

    @meme_tool.command("统计")
    async def show_stats(self, event: AstrMessageEvent):
        """显示表情库统计信息

        用法: /表情工具 统计
        """
        if not self.memes_manager:
            yield event.plain_result("表情管理器未初始化")
            return

        try:
            stats = self.memes_manager.get_stats()

            result = "表情库统计信息：\n"
            result += f"- 总表情数: {stats.total_memes}\n"
            result += f"- 总情感数: {stats.total_emotions}\n"
            result += (
                f"- 词嵌入状态: {'已启用' if stats.embedding_enabled else '未启用'}\n"
            )

            if stats.embedding_enabled:
                result += f"- 已计算情感嵌入: {stats.emotions_with_embedding}/{stats.total_emotions}\n"
                result += f"- 已计算描述嵌入: {stats.memes_with_description_embedding}/{stats.total_memes}\n"
                result += f"- Embedding Provider: {stats.embedding_provider_id}\n"
                result += f"- 嵌入维度: {stats.embedding_dim}\n"

            yield event.plain_result(result.strip())
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            yield event.plain_result(f"操作失败: {e}")

    def _format_search_results(
        self, results: list[EmbeddingSearchResult] | list[FuzzySearchResult]
    ) -> str:
        """格式化搜索结果为文本

        Args:
            results: 搜索结果列表

        Returns:
            格式化后的文本
        """
        if not results:
            return "未找到相关表情"

        result_text = "搜索结果：\n\n"
        for i, search_result in enumerate(results, 1):
            meme = search_result.meme
            emotion = search_result.emotion

            # 根据结果类型显示相似度/分数
            if isinstance(search_result, EmbeddingSearchResult):
                result_text += f"{i}. [{emotion}] {meme.internal_path}\n"
                result_text += f"   描述: {meme.description}\n"
                result_text += (
                    f"   情感余弦距离: {search_result.emotion_similarity:.3f}\n"
                )
                result_text += (
                    f"   描述余弦距离: {search_result.description_similarity:.3f}\n\n"
                )
            elif isinstance(search_result, FuzzySearchResult):
                result_text += f"{i}. [{emotion}] {meme.internal_path}\n"
                result_text += f"   描述: {meme.description}\n"
                result_text += f"   匹配度: {search_result.score}\n\n"

        return result_text.strip()

    @filter.llm_tool(name="memes_list_emotions")
    async def memes_list_emotions(self, event: AstrMessageEvent):
        """列出所有可用的表情情感标签。

        使用场景：当需要了解当前有哪些情感标签可供选择时使用此工具。

        Returns:
            string: 所有情感标签的列表，格式为：
                "情感标签列表：
                1. 开心 (5 个表情)
                2. 悲伤 (3 个表情)
                ..."
        """
        if not self.memes_manager:
            raise RuntimeError("self.memes_manager未初始化")

        try:
            emotions = self.memes_manager.get_all_emotions()
            if not emotions:
                yield "当前没有任何情感标签"

            result = "情感标签列表：\n"
            for i, emotion in enumerate(emotions, 1):
                memes = self.memes_manager.get_memes_by_emotion(emotion)
                result += f"{i}. {emotion} ({len(memes)} 个表情)\n"

            yield result.strip()
        except Exception as e:
            logger.error(f"列出情感失败: {e}")
            yield f"操作失败: {e}"

    @filter.llm_tool(name="memes_search")
    async def memes_search(
        self,
        event: AstrMessageEvent,
        emotion_query: str,
        description_query: str,
    ):
        """根据情感和描述关键词搜索候选表情。

        使用场景：当需要查找符合特定情感和描述的表情时使用此工具。
        可以通过情感关键词（如"开心"、"悲伤"）和描述关键词（如"大笑"、"哭泣"）来搜索表情。

        Args:
            emotion_query(string): 情感关键词，用于匹配表情的情感标签，例如"开心"、"悲伤"、"愤怒"等
            description_query(string): 描述关键词，用于匹配表情的描述文本，例如"大笑"、"哭泣"、"思考"等

        Returns:
            string: 搜索结果列表，格式为：
                "搜索结果：

                1. [情感] 表情路径
                   描述: 表情描述
                   情感余弦距离: 0.xxx
                   描述余弦距离: 0.xxx

                2. ..."
        """
        if not self.memes_manager:
            raise RuntimeError("self.memes_manager未初始化")

        if not emotion_query or not description_query:
            yield "请提供情感关键词和描述关键词"

        try:
            results = await self.memes_manager.search(emotion_query, description_query)
            result_text = self._format_search_results(results)
            yield result_text
        except Exception as e:
            logger.error(f"搜索表情失败: {e}")
            yield f"操作失败: {e}"

    @filter.llm_tool(name="memes_send")
    async def memes_send(self, event: AstrMessageEvent, path: str):
        """发送指定路径的表情图片。

        使用场景：当需要向用户展示某个表情时使用此工具。
        通常在搜索到表情后，选择合适的表情发送给用户。

        Args:
            path(string): 表情的路径，从搜索结果中获取，例如 "开心/happy.jpg"

        Returns:
            MessageEventResult: 发送的表情图片，包含情感标签和描述信息
        """
        if not path:
            yield "请提供表情路径"

        try:
            assert self.memes_manager is not None
            info = self.memes_manager.get_meme_by_path(Path(path))
            if info is None:
                yield event.plain_result(f"表情 {path} 不存在")
                return

            _, _, file_path = info

            # 构建消息链
            chain: list[BaseMessageComponent] = [
                Image.fromFileSystem(str(file_path)),
            ]

            yield event.chain_result(chain)
            yield "发送成功"

        except Exception as e:
            logger.error(f"发送表情失败: {e}")
            yield f"操作失败: {e}"

    async def terminate(self):
        """插件销毁时调用"""
        logger.info("表情包管理器插件已卸载")
