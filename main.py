"""表情包管理器插件

提供表情包管理功能，包括添加、删除、搜索表情等。
作为应用层，负责捕获底层异常并记录 ERROR 日志。
"""

from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.message.message_event_result import MessageChain

from .memes_manager import (
    EmbeddingSearchResult,
    FuzzySearchResult,
    MemesError,
    MemesManager,
    MemesManagerConfig,
    MemesMemeNotFoundError,
)


class MyPlugin(Star):
    """表情包管理器插件

    作为应用层，负责：
    - 初始化 MemesManager
    - 处理用户命令
    - 捕获底层异常并记录 ERROR 日志
    - 向用户返回友好的错误信息
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.memes_manager: MemesManager | None = None
        embedding_provider_id = config.get("embedding_provider")
        if embedding_provider_id == "":
            embedding_provider_id = None

        self.context = context

        self.config = MemesManagerConfig(
            data_dir=StarTools.get_data_dir(),
            chat_provider_id=config.get("chat_provider"),
            embedding_provider_id=embedding_provider_id,
            max_candidates=config["max_candidates"],
        )

    async def initialize(self):
        """初始化插件，创建 MemesManager 实例"""
        self.memes_manager = MemesManager(self.context, self.config)
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
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        emotions = self.memes_manager.get_all_emotions()
        if not emotions:
            yield event.plain_result("当前没有任何情感标签")
            return

        result = "所有情感标签：\n"
        for i, emotion in enumerate(emotions, 1):
            memes = self.memes_manager.get_memes_by_emotion(emotion)
            result += f"{i}. {emotion} ({len(memes)} 个表情)\n"

        yield event.plain_result(result.strip())

    @meme_tool.command("列出")
    async def list_memes(self, event: AstrMessageEvent, emotion: str = ""):
        """列出指定情感下的所有表情

        用法: /表情工具 列出 <情感>
        示例: /表情工具 列出 开心
        """
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        if not emotion:
            yield event.plain_result("请指定情感标签，例如: /表情工具 列出 开心")
            return

        memes = self.memes_manager.get_memes_by_emotion(emotion)
        if not memes:
            yield event.plain_result(f"情感 '{emotion}' 下没有任何表情")
            return

        result = f"情感 '{emotion}' 下的表情：\n"
        for i, meme in enumerate(memes, 1):
            result += f"{i}. {meme.internal_path}\n   描述: {meme.description}\n"

        yield event.plain_result(result.strip())

    @meme_tool.command("搜索")
    async def search_meme(self, event: AstrMessageEvent, query: str):
        """搜索表情

        用法: /表情工具 搜索 <逗号分开的关键词列表>
        示例: /表情工具 搜索 开心，大笑
        """
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        if not query:
            yield event.plain_result("请指定搜索关键词，例如: /表情工具 搜索 开心")
            return

        if "，" in query:
            queries = query.split("，")
        else:
            queries = query.split(",")

        results = await self.memes_manager.search(queries)
        result_text = self._format_search_results(results, show_similarity=True)
        yield event.plain_result(result_text)

    @meme_tool.command("添加")
    async def add_meme(self, event: AstrMessageEvent):
        """添加消息中的所有图片到表情库

        用法: /表情工具 添加 (需要回复包含图片的消息或发送带图片的消息)
        """
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

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
        errors = []

        for i, image in enumerate(images, 1):
            try:
                # 使用 convert_to_file_path 方法获取图片路径
                image_path = Path(await image.convert_to_file_path())

                # 添加表情（自动生成情感和描述）
                path = await self.memes_manager.add_meme_from_file(
                    file_path=image_path,
                    auto_generate=True,
                    copy_file=True,
                )
                added_paths.append(path)
                success_count += 1
                logger.info(f"成功添加图片 {i}/{len(images)}: {image_path}")

            except MemesError as e:
                failed_count += 1
                errors.append(f"图片 {i}: {e}")
                logger.error(f"添加图片失败 {i}/{len(images)}: {e}")
            except Exception as e:
                failed_count += 1
                errors.append(f"图片 {i}: 未知错误")
                logger.error(f"处理图片 {i}/{len(images)} 时发生未知错误: {e}")

        result_msg = (
            f"处理完成！\n成功添加: {success_count} 张\n失败: {failed_count} 张\n\n"
        )

        for i, path in enumerate(added_paths, 1):
            try:
                r = self.memes_manager.get_meme_by_path(path)
                emotion, meme, _ = r
                result_msg += f"{i}. [{emotion}] {meme.internal_path}\n"
                result_msg += f"   描述: {meme.description}\n"
            except MemesError:
                result_msg += f"{i}. {path}\n"

        if errors:
            result_msg += "\n失败详情:\n"
            for err in errors[:5]:  # 最多显示5个错误
                result_msg += f"- {err}\n"
            if len(errors) > 5:
                result_msg += f"... 还有 {len(errors) - 5} 个错误\n"

        yield event.plain_result(result_msg)

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
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        if not emotion or not description or not memo:
            yield event.plain_result(
                "用法: /表情工具 手动添加 <情感> <助记词> <描述>\n"
                "示例: /表情工具 手动添加 开心 大笑 一个小人咧开嘴，举起手，开心地大笑"
            )
            return

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
            return

        yield event.plain_result("正在处理图片...")

        # 下载并添加图片
        image = images[0]
        image_path = Path(await image.convert_to_file_path())

        path = await self.memes_manager.add_meme_from_file(
            file_path=image_path,
            emotion=emotion,
            memo=memo,
            description=description,
            auto_generate=False,
            copy_file=True,
        )

        logger.info(f"成功添加图片: {image_path}")
        yield event.plain_result(
            f"添加成功！\n路径: {path}\n情感: {emotion}\n描述: {description}"
        )

    @meme_tool.command("删除")
    async def delete_meme(self, event: AstrMessageEvent, path: str = ""):
        """删除指定路径的表情

        用法: /表情工具 删除 <路径>
        示例: /表情工具 列出 开心 (先查看表情路径)
              /表情工具 删除 开心/happy.jpg
        """
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        if not path:
            yield event.plain_result(
                "请指定要删除的表情路径\n"
                "用法: /表情工具 删除 <路径>\n"
                "提示: 使用 '/表情工具 列出 <情感>' 查看表情路径"
            )
            return

        try:
            meme_path = Path(path)
            self.memes_manager.remove_meme(meme_path, delete_file=True)
            yield event.plain_result(f"成功删除表情: {path}")
        except MemesMemeNotFoundError:
            yield event.plain_result(f"删除失败: 未找到路径为 '{path}' 的表情")

    @meme_tool.command("更新")
    async def manual_update_meme(
        self,
        event: AstrMessageEvent,
        old_path: str = "",
        emotion: str = "",
        memo: str = "",
        description: str = "",
    ):
        """手动更新表情

        用法: /表情工具 更新 <路径> <情感> <助记词> <描述>
        示例: /表情工具 更新 伤心/大哭.jpg 开心 大笑 一个小人咧开嘴，举起手，开心地大笑
        """
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        if not old_path or not memo or not emotion or not description:
            yield event.plain_result(
                "用法: /表情工具 更新 <路径> <情感> <助记词> <描述>\n"
                "示例: /表情工具 更新 伤心/大哭.jpg 开心 大笑 一个小人咧开嘴，举起手，开心地大笑"
            )
            return

        try:
            await self.memes_manager.update_meme(
                Path(old_path), emotion, memo, description
            )
            yield event.plain_result("更新成功")
        except MemesMemeNotFoundError as e:
            logger.error(f"更新表情失败: {e}")
            yield event.plain_result("更新失败: 表情不存在")

    @meme_tool.command("发送")
    async def send_meme(self, event: AstrMessageEvent, path: str = ""):
        """发送指定路径的表情图片

        用法: /表情工具 发送 <路径>
        示例: /表情工具 列出 开心 (先查看表情路径)
              /表情工具 发送 开心/happy.jpg
        """
        if not path:
            yield event.plain_result(
                "请指定要发送的表情路径\n"
                "用法: /表情工具 发送 <路径>\n"
                "提示: 使用 '/表情工具 列出 <情感>' 查看表情路径"
            )
            return

        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        try:
            info = self.memes_manager.get_meme_by_path(Path(path))
            emotion, meme, file_path = info

            # 构建消息链
            chain = [
                Plain(f"\n情感: {emotion}\n描述: {meme.description}"),
                Image.fromFileSystem(str(file_path)),
            ]

            yield event.chain_result(chain)

        except MemesMemeNotFoundError:
            yield event.plain_result(f"表情 '{path}' 不存在")

    @meme_tool.command("清理词嵌入")
    async def clean_embeddings(self, event: AstrMessageEvent):
        """清理孤儿词嵌入向量

        用法: /表情工具 清理词嵌入
        """
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        # 获取清理前的统计
        stats_before = self.memes_manager.get_stats()

        # 执行清理
        cleaned_count = self.memes_manager.embedding_manager.clear_orphan_embeddings()

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

    @meme_tool.command("统计")
    async def show_stats(self, event: AstrMessageEvent):
        """显示表情库统计信息

        用法: /表情工具 统计
        """
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        stats = self.memes_manager.get_stats()

        result = "表情库统计信息：\n"
        result += f"- 总表情数: {stats.total_memes}\n"
        result += f"- 总情感数: {stats.total_emotions}\n"
        result += f"- 词嵌入状态: {'已启用' if stats.embedding_enabled else '未启用'}\n"

        if stats.embedding_enabled:
            result += f"- 已计算情感嵌入: {stats.emotions_with_embedding}/{stats.total_emotions}\n"
            result += f"- 已计算描述嵌入: {stats.memes_with_description_embedding}/{stats.total_memes}\n"
            result += f"- Embedding Provider: {stats.embedding_provider_id}\n"
            result += f"- 嵌入维度: {stats.embedding_dim}\n"

        yield event.plain_result(result.strip())

    def _format_search_results(
        self,
        results: list[EmbeddingSearchResult] | list[FuzzySearchResult],
        show_similarity: bool = False,
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
                if show_similarity:
                    result_text += (
                        f"   情感余弦距离: {search_result.emotion_similarity:.3f}\n"
                    )
                    result_text += (
                        f"   描述余弦距离: {search_result.description_similarity:.3f}\n"
                    )
            elif isinstance(search_result, FuzzySearchResult):
                result_text += f"{i}. [{emotion}] {meme.internal_path}\n"
                result_text += f"   描述: {meme.description}\n"
                if show_similarity:
                    result_text += f"   匹配度: {search_result.score}\n"

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
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        emotions = self.memes_manager.get_all_emotions()
        if not emotions:
            yield "当前没有任何情感标签"
            return

        result = "情感标签列表：\n"
        for i, emotion in enumerate(emotions, 1):
            memes = self.memes_manager.get_memes_by_emotion(emotion)
            result += f"{i}. {emotion} ({len(memes)} 个表情)\n"

        yield result.strip()

    @filter.llm_tool(name="memes_search")
    async def memes_search(self, event: AstrMessageEvent, keywords: str):
        """根据关键词列表搜索候选表情。

        使用场景：当需要查找符合特定情感和描述的表情时使用此工具。
        可以通过多个情感关键词（如"开心"、"悲伤"）和描述关键词（如"大笑"、"哭泣"）来搜索表情。

        Args:
            keywords(string): 关键词列表，用中文或英文逗号分开

        Returns:
            string: 搜索结果列表，格式为：
                "搜索结果：

                1. [情感] 表情路径
                   描述: 表情描述

                2. ..."
        """
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        if not keywords:
            yield "请提供情感关键词和描述关键词"
            return

        if "，" in keywords:
            kwds = keywords.split("，")
        else:
            kwds = keywords.split(",")

        results = await self.memes_manager.search(kwds)
        result_text = self._format_search_results(results)
        yield result_text

    @filter.llm_tool(name="memes_add")
    async def memes_add(
        self,
        event: AstrMessageEvent,
        file_path: str,
        emotion: str,
        memo: str,
        description: str,
    ):
        """添加新的图片文件到表情库

        使用场景：当看到合适的表情图片后将其添加到表情库。
        要为表情提供情感词和描述。

        Args:
            file_path(string): 图片文件在文件系统中的路径
            emotion(string): 用一个中文词语描述这个图片的关键词，关键词表达表情的核心情感或者动作，如高兴、震惊、伤心、恼怒、后悔、害怕、不耐烦，或者敬礼、祈祷、奔跑等。优先使用情感词。
            memo(string): 用一个中文词语或短语描述这个图片的次级关键词，必须要和emotion不同。如果emotion是情感词，则此处不要再使用情感词。
            description(string): 用一句简短的中文描述这个表情的内容或含义

        Returns:
            新表情的内部路径
        """
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        path = await self.memes_manager.add_meme_from_file(
            file_path=Path(file_path),
            emotion=emotion,
            memo=memo,
            description=description,
            auto_generate=False,
            copy_file=True,
        )

        yield str(path)

    @filter.llm_tool(name="memes_update")
    async def memes_update(
        self,
        event: AstrMessageEvent,
        path: str,
        emotion: str,
        memo: str,
        description: str,
    ):
        """修改已有表情图片的情感词和描述

        使用场景：当用户指出表情使用不恰当，并告知正确用法。
        根据用户提供的正确用法决定新的情感词和描述。

        Args:
            path(string): 表情的内部路径，从搜索结果中获取，也用于发送表情，如“开心/大笑.jpg"
            emotion(string): 用一个中文词语描述这个图片的关键词，关键词表达表情的核心情感或者动作，如高兴、震惊、伤心、恼怒、后悔、害怕、不耐烦，或者敬礼、祈祷、奔跑等。优先使用情感词。
            memo(string): 用一个中文词语或短语描述这个图片的次级关键词，必须要和emotion不同。如果emotion是情感词，则此处不要再使用情感词。
            description(string): 用一句简短的中文描述这个表情的内容或含义

        Returns:
            是否成功更新
        """
        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        try:
            await self.memes_manager.update_meme(Path(path), emotion, memo, description)
            yield "更新成功"

        except MemesMemeNotFoundError as e:
            logger.error(f"更新表情失败: {e}")
            yield "更新失败: 表情不存在"

    @filter.llm_tool(name="memes_send")
    async def memes_send(self, event: AstrMessageEvent, path: str):
        """发送指定路径的表情图片。

        使用场景：当需要向用户展示某个表情时使用此工具。
        通常在搜索到表情后，选择合适的表情发送给用户。

        Args:
            path(string): 表情的内部路径，从搜索结果中获取，例如 "开心/大笑.jpg"

        Returns:
            是否成功发送
        """
        if not path:
            yield "请提供表情路径"
            return

        if self.memes_manager is None:
            raise RuntimeError("MemesManager 未初始化")

        try:
            _, _, file_path = self.memes_manager.get_meme_by_path(Path(path))

            await self.context.send_message(
                event.unified_msg_origin, MessageChain().file_image(str(file_path))
            )
            yield "发送成功"

        except MemesMemeNotFoundError:
            yield f"表情 '{path}' 不存在，请重新搜索"

    async def terminate(self):
        """插件销毁时调用"""
        logger.info("表情包管理器插件已卸载")
