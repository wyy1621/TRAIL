"""无监督训练方法基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseUnsuperviseMethod(ABC):
    """无监督训练方法基类。

    子类必须实现：
    - validate_config
    - build_optimizer
    - build_scheduler
    - train_epoch
    - val_epoch
    - compute_anomaly_score
    """

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> None:
        """校验当前方法所需配置。"""

    @abstractmethod
    def build_optimizer(self, model, config: dict[str, Any]):
        """返回当前方法自定义优化器；返回 None 表示使用壳层默认实现。"""

    @abstractmethod
    def build_scheduler(self, optimizer, config: dict[str, Any], steps_per_epoch: int):
        """返回当前方法自定义 scheduler；返回 None 表示使用壳层默认实现。"""

    @abstractmethod
    def train_epoch(
        self,
        model,
        train_loader,
        optimizer,
        scheduler,
        device,
        config: dict[str, Any],
        epoch: int,
        global_step: int,
        model_save_root: str,
    ):
        """执行一个完整训练 epoch。"""

    @abstractmethod
    def val_epoch(
        self,
        model,
        val_loader,
        device,
        config: dict[str, Any],
        epoch: int,
    ):
        """执行一个完整验证 epoch。"""

    @abstractmethod
    def compute_anomaly_score(self, input_x, model_output, config: dict[str, Any]):
        """定义该方法最终 anomaly score 的计算方式。"""
