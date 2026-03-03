"""MemesManager 层级错误类型定义

按照错误处理规则：
- Expected Errors：由外部输入或环境因素导致的错误，应该向上传播
- Logic Errors：由内部 bug 导致的错误，应该快速失败

所有的 Expected Errors 都继承自 MemesError 基类。
"""

from pathlib import Path


class MemesError(Exception):
    """MemesManager 层级错误基类

    所有可能由外部因素导致的预期错误都应该继承此类。
    """

    pass


class MemesNotInitializedError(MemesError):
    """未初始化错误

    当 MemesManager 或相关组件未调用 initialize() 就使用时抛出。
    这是一个 Expected Error，因为初始化顺序由调用者控制。
    """

    def __init__(self, message: str = "MemesManager 未初始化，请先调用 initialize()"):
        super().__init__(message)


class MemesFileError(MemesError):
    """文件操作相关错误

    当文件不存在、无法读取、无法写入等情况下抛出。
    """

    def __init__(self, message: str, path: Path | None = None):
        self.path = path
        if path:
            message = f"{message}: {path}"
        super().__init__(message)


class MemesFileNotFoundError(MemesFileError):
    """文件不存在错误

    当请求的文件路径不存在时抛出。
    """

    def __init__(self, path: Path):
        super().__init__("文件不存在", path)


class MemesEmbeddingError(MemesError):
    """词嵌入相关错误

    当词嵌入计算、加载、保存等操作失败时抛出。
    """

    def __init__(self, message: str, cause: Exception | None = None):
        self.cause = cause
        if cause:
            message = f"{message}: {cause}"
        super().__init__(message)


class MemesEmbeddingDisabledError(MemesEmbeddingError):
    """词嵌入未启用错误

    当尝试使用词嵌入功能但未配置 EmbeddingProvider 时抛出。
    """

    def __init__(self, message: str = "词嵌入功能未启用，请配置 EmbeddingProvider"):
        super().__init__(message)


class MemesEmbeddingDimensionMismatchError(MemesEmbeddingError):
    """词嵌入维度不匹配错误

    当向量维度与配置的维度不一致时抛出。
    """

    def __init__(self, expected_dim: int, actual_dim: int):
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        super().__init__(f"向量维度不匹配：期望 {expected_dim}，实际 {actual_dim}")


class MemesProviderError(MemesError):
    """Provider 相关错误

    当 Provider 不可用或调用失败时抛出。
    """

    def __init__(
        self,
        message: str,
        provider_id: str | None = None,
        cause: Exception | None = None,
    ):
        self.provider_id = provider_id
        self.cause = cause
        if provider_id:
            message = f"{message} (provider: {provider_id})"
        if cause:
            message = f"{message}: {cause}"
        super().__init__(message)


class MemesProviderNotFoundError(MemesProviderError):
    """Provider 未找到错误

    当请求的 Provider 不存在时抛出。
    """

    def __init__(self, provider_id: str, provider_type: str = "Provider"):
        super().__init__(f"未找到{provider_type}", provider_id)


class MemesNotFoundError(MemesError):
    """资源未找到错误

    当请求的表情或情感不存在时抛出。
    """

    def __init__(self, resource_type: str, identifier: str | Path):
        self.resource_type = resource_type
        self.identifier = identifier
        super().__init__(f"{resource_type}不存在: {identifier}")


class MemesMemeNotFoundError(MemesNotFoundError):
    """表情未找到错误

    当请求的表情不存在时抛出。
    """

    def __init__(self, path: Path | str):
        super().__init__("表情", path)


class MemesEmotionNotFoundError(MemesNotFoundError):
    """情感未找到错误

    当请求的情感标签不存在时抛出。
    """

    def __init__(self, emotion: str):
        super().__init__("情感", emotion)


class MemesConfigError(MemesError):
    """配置相关错误

    当配置无效或缺失时抛出。
    """

    def __init__(self, message: str, config_key: str | None = None):
        self.config_key = config_key
        if config_key:
            message = f"{message} (配置项: {config_key})"
        super().__init__(message)


class MemesParseError(MemesError):
    """解析相关错误

    当解析数据（如 JSON）失败时抛出。
    """

    def __init__(self, message: str, cause: Exception | None = None):
        self.cause = cause
        if cause:
            message = f"{message}: {cause}"
        super().__init__(message)


class MemesLLMError(MemesError):
    """LLM 调用相关错误

    当调用 LLM 生成失败时抛出。
    """

    def __init__(self, message: str, cause: Exception | None = None):
        self.cause = cause
        if cause:
            message = f"{message}: {cause}"
        super().__init__(message)
