"""数据源适配器抽象基类。

每个数据源实现 discover -> parse -> validate 三阶段接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from classic_chinese_llm.data.schemas import SourceDocument


class BaseSource(ABC):
    """数据源适配器抽象基类。

    子类只需实现三个抽象方法:
        discover()  — 发现可处理的原始文件
        parse()     — 解析文件为 SourceDocument 列表
        validate()  — 校验单篇文档是否合格

    生命周期: discover → [parse → validate] × N
    """

    name: str = ""  # 来源唯一标识，如 "daizhige"
    display_name: str = ""  # 人类可读名称，如 "殆知阁"

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir) / self.name
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def discover(self, raw_dir: Path) -> list[Path]:
        """阶段 1: 发现可处理的文件。

        Args:
            raw_dir: 原始数据根目录（raw_dir/{self.name}/ 下存放本来源的文件）

        Returns:
            待处理文件的绝对路径列表（按处理顺序排序）。
            空目录返回空列表（不报错）。
        """
        ...

    @abstractmethod
    def parse(self, file_path: Path) -> list[SourceDocument]:
        """阶段 2: 解析文件，产出 SourceDocument 列表。

        Args:
            file_path: discover() 返回的单个文件路径。

        Returns:
            SourceDocument 列表。解析失败时返回空列表并记录 warning。
        """
        ...

    def validate(self, doc: SourceDocument) -> bool:
        """阶段 3: 校验单篇文档是否合格。

        默认实现: 正文长度 >= 10 字符。
        子类可重写以添加来源特定的校验规则。
        """
        return len(doc.text.strip()) >= 10

    def post_process(self, docs: list[SourceDocument]) -> list[SourceDocument]:
        """可选的采集后处理（去重、合并等）。默认不做任何操作。"""
        return docs

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"
