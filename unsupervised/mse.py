"""
通用 MSE 无监督训练方法。

当前方法含义：
- 把模型当作重建器使用；
- 训练目标是重建输入；
- 异常依据来自重建误差；
- 最终 anomaly score 会从 `(B, L, D)` 规约成 `(B, L)`。
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from common import datetime_format
from common.performance import exclude_if_active
from .base import BaseUnsuperviseMethod
from .training import (
    get_filled_point_weight,
    observed_filled_element_loss_max,
    get_score_cfg,
    get_training_cfg,
    reduce_observed_weighted_loss,
    save_checkpoint,
    summarize_observed_time_scores,
    unpack_input_and_observed_mask,
)


log = logging.getLogger(__name__)


class MSEBase(BaseUnsuperviseMethod):
    """MSE 重建训练方法实现。"""

    def validate_config(self, config: dict[str, Any]) -> None:
        """校验 MSE 方法所需关键配置。"""
        training_cfg = get_training_cfg(config)
        required_keys = ['batch_size', 'learning_rate', 'max_epochs', 'save_model_per_epoch']
        for key in required_keys:
            if key not in training_cfg:
                raise KeyError(f'MSE 方法缺少 training.{key} 配置')

    def build_optimizer(self, model: torch.nn.Module, config: dict[str, Any]):
        return None

    def build_scheduler(self, optimizer, config: dict[str, Any], steps_per_epoch: int):
        return None

    def train_epoch(
        self,
        model: torch.nn.Module,
        train_loader,
        optimizer,
        scheduler,
        device: torch.device,
        config: dict[str, Any],
        epoch: int,
        global_step: int,
        model_save_root: str,
    ) -> tuple[dict[str, float], int]:
        """执行一个完整的 MSE 训练 epoch。"""
        training_cfg = get_training_cfg(config)
        dataset_name = config['Dataset']
        filled_point_weight = get_filled_point_weight(config)
        save_steps = np.linspace(1, len(train_loader), int(training_cfg['save_model_per_epoch']) + 2, dtype=int).tolist()[1:-1]

        model.train()
        epoch_total_loss = 0.0
        epoch_score_mean = 0.0

        for i, batch in enumerate(train_loader):
            input_x, feature_mask = unpack_input_and_observed_mask(batch, dataset_name)
            input_x = input_x.to(device)
            feature_mask = feature_mask.to(device)
            optimizer.zero_grad()

            recon_x = model(input_x, None, None, None)
            if isinstance(recon_x, tuple):
                raise TypeError('MSE 方法要求模型输出单个重建张量，当前输出为 tuple')
            recon_element_loss = F.mse_loss(recon_x, input_x, reduction='none')
            recon_loss = reduce_observed_weighted_loss(
                recon_element_loss,
                feature_mask,
                filled_point_weight,
                mode='mean',
            )
            recon_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_total_loss += recon_loss.item()
            step_scores = self.compute_anomaly_score(input_x, recon_x, config)
            time_mask = feature_mask.any(dim=-1)
            step_score_mean = summarize_observed_time_scores(step_scores, time_mask)['mean'].item()
            epoch_score_mean += step_score_mean

            if i < 1000 or i % 10 == 0 or i == len(train_loader) - 1:
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None and hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                with exclude_if_active("training.reconstructor"):
                    log.info(
                        'Epoch: %d, Step %d/%d, Loss: %.4f, LR: %.6e',
                        epoch,
                        i,
                        len(train_loader),
                        recon_loss.item(),
                        current_lr,
                    )

            if i in save_steps:
                with exclude_if_active("training.reconstructor"):
                    checkpoint_name = f'{model.__class__.__name__}_epoch_{epoch}_globalstep_{global_step}_{datetime_format(fmt="%Y%m%d-%H%M%S")}.pth'
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        path=f'{model_save_root}/{checkpoint_name}',
                    )

            global_step += 1

        metrics = {
            'loss': epoch_total_loss / len(train_loader),
            'score_mean': epoch_score_mean / len(train_loader),
        }
        return metrics, global_step

    def val_epoch(
        self,
        model: torch.nn.Module,
        val_loader,
        device: torch.device,
        config: dict[str, Any],
        epoch: int,
    ) -> dict[str, float]:
        """执行一个完整的 MSE 验证 epoch。"""
        dataset_name = config['Dataset']
        filled_point_weight = get_filled_point_weight(config)

        model.eval()
        val_total_loss = 0.0
        observed_max_mse_all: list[float] = []
        filled_max_mse_all: list[float] = []
        score_mean_all: list[float] = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                input_x, feature_mask = unpack_input_and_observed_mask(batch, dataset_name)
                input_x = input_x.to(device)
                feature_mask = feature_mask.to(device)
                recon_x = model(input_x, None, None, None)
                if isinstance(recon_x, tuple):
                    raise TypeError('MSE 方法要求模型输出单个重建张量，当前输出为 tuple')
                recon_element_loss = F.mse_loss(recon_x, input_x, reduction='none')
                recon_loss = reduce_observed_weighted_loss(
                    recon_element_loss,
                    feature_mask,
                    filled_point_weight,
                    mode='mean',
                )
                val_total_loss += recon_loss.item()

                observed_mse_max, filled_mse_max = observed_filled_element_loss_max(recon_element_loss, feature_mask)
                observed_mse_max_value = observed_mse_max.item()
                filled_mse_max_value = filled_mse_max.item()
                observed_max_mse_all.append(observed_mse_max_value)
                filled_max_mse_all.append(filled_mse_max_value)
                time_mask = feature_mask.any(dim=-1)
                score_mean = summarize_observed_time_scores(self.compute_anomaly_score(input_x, recon_x, config), time_mask)['mean'].item()
                score_mean_all.append(score_mean)

                log.info(
                    'Epoch: %d, Validation Step %d/%d, Loss: %.4f, Observed Max Per-Sample MSE: %.4f, Filled Max Per-Sample MSE: %.4f',
                    epoch,
                    i,
                    len(val_loader),
                    recon_loss.item(),
                    observed_mse_max_value,
                    filled_mse_max_value,
                )

        metrics = {
            'loss': val_total_loss / len(val_loader),
            'observed_max_mse': float(np.mean(observed_max_mse_all)) if observed_max_mse_all else float('nan'),
            'filled_max_mse': float(np.mean(filled_max_mse_all)) if filled_max_mse_all else float('nan'),
            'score_mean': float(np.mean(score_mean_all)) if score_mean_all else 0.0,
        }
        return metrics

    def compute_anomaly_score(self, input_x: torch.Tensor, model_output: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
        """把逐元素 MSE 误差 `(B, L, D)` 规约为 `(B, L)` 的时间点分数。"""
        score_cfg = get_score_cfg(config)
        reduction = score_cfg.get('feature_reduction', 'mean')
        mse = F.mse_loss(model_output, input_x, reduction='none')
        if reduction == 'mean':
            return mse.mean(dim=-1)
        if reduction == 'max':
            return mse.max(dim=-1).values
        raise ValueError(f'Unsupported MSE score reduction: {reduction}')

    @staticmethod
    def _unpack_batch(batch, dataset_name: str) -> torch.Tensor:
        if dataset_name == 'CF':
            _, _, input_x, _ = batch
            return input_x
        if dataset_name == 'RADBench':
            _, _, _, input_x, _ = batch
            return input_x
        raise ValueError(f'Unsupported dataset: {dataset_name}')


METHOD = MSEBase()
