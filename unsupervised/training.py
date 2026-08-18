"""无监督训练公共工具。"""

import copy
import datetime
import logging
import os
import shutil
from typing import Any
from zoneinfo import ZoneInfo

import dotenv
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import LambdaLR

from common import GPUAutoSelector, beijing
from data_pipeline.lmdb_store import should_persist_workers, BgpLMDBDataset, RADBenchLMDBDataset, DataLoader


logging.Formatter.converter = beijing
log = logging.getLogger(__name__)


_UNSUP_RESERVED_KEYS = {
    'model_name',
    'model_args',
    'training',
    'score',
    'method_name',
    'checkpoint',
}


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_unsupervise_section(config: dict[str, Any]) -> dict[str, Any]:
    if 'UnsuperviseModel' not in config:
        raise KeyError('配置中缺少 UnsuperviseModel')
    return config['UnsuperviseModel']


def resolve_model_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """把当前启用模型段解析成 build_unsupervised_model() 所需参数。"""
    unsup_cfg = get_unsupervise_section(config)
    model_name = unsup_cfg['model_name']

    if 'model_args' in unsup_cfg:
        model_cfg = {'model_name': model_name}
        model_cfg.update(copy.deepcopy(unsup_cfg['model_args']))
    else:
        model_cfg = {
            key: copy.deepcopy(value)
            for key, value in unsup_cfg.items()
            if key not in _UNSUP_RESERVED_KEYS
        }
        model_cfg['model_name'] = model_name

    if 'win_size' not in model_cfg:
        raise KeyError('当前模型配置缺少 win_size')
    if 'step' not in model_cfg:
        model_cfg['step'] = model_cfg['win_size']

    return model_cfg


def get_training_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return get_unsupervise_section(config).get('training', {})


def get_score_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return get_unsupervise_section(config).get('score', {})


def get_method_name(config: dict[str, Any]) -> str:
    unsup_cfg = get_unsupervise_section(config)
    explicit_method = unsup_cfg.get('method_name')
    if explicit_method:
        return explicit_method
    if unsup_cfg['model_name'] == 'AnomalyTransformer':
        return 'anomaly_transformer'
    if unsup_cfg['model_name'] == 'OmniAnomaly':
        return 'omni_anomaly'
    if unsup_cfg['model_name'] == 'CATCH':
        return 'catch'
    return 'mse'


def prepare_run_directory(
    config: dict[str, Any],
    model: torch.nn.Module,
    config_path: str,
    method_name: str,
) -> str:
    """Create a timestamped checkpoint directory and snapshot training inputs."""
    date_part = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    checkpoint_root = config.get('checkpoint_root', 'Checkpoints/Unsupervise')

    model_save_root = os.path.join(
        checkpoint_root,
        f'{model.__class__.__name__}_{method_name}_{date_part}',
    )
    os.makedirs(model_save_root, exist_ok=True)

    unsupervise_root = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(unsupervise_root, '..'))
    train_script_path = os.path.join(unsupervise_root, 'train_impl.py')
    method_script_path = os.path.join(unsupervise_root, f'{method_name}.py')
    env_path = os.path.join(repo_root, '.env')

    shutil.copyfile(config_path, os.path.join(model_save_root, os.path.basename(config_path)))
    shutil.copyfile(train_script_path, os.path.join(model_save_root, os.path.basename(train_script_path)))
    shutil.copyfile(method_script_path, os.path.join(model_save_root, os.path.basename(method_script_path)))
    if os.path.isfile(env_path):
        shutil.copyfile(env_path, os.path.join(model_save_root, os.path.basename(env_path)))
    return model_save_root


def build_datasets(config: dict[str, Any]) -> tuple[Any, Any]:
    """构建 no-sparse train/val dataset，并返回时间点级 observed mask。

    Args:
        config: no-sparse 无监督训练配置。

    Returns:
        `(train_dataset, val_dataset)`。RADBench batch 末尾额外包含 `(B, L)` observed mask。
    """
    dataset_name = config['Dataset']
    model_cfg = resolve_model_cfg(config)
    train_lmdb_path = config['train_lmdb_path']
    val_lmdb_path = config['val_lmdb_path']
    win_size = model_cfg['win_size']
    step = model_cfg.get('step', win_size)

    if dataset_name != 'RADBench':
        raise ValueError(f'pipeline_no_sparse 仅支持 RADBench no-sparse LMDB，当前为 {dataset_name}')

    train_dataset = RADBenchLMDBDataset(
        lmdb_file_path=train_lmdb_path,
        win_size=win_size,
        step=step,
        return_mask=True,
    )
    val_dataset = RADBenchLMDBDataset(
        lmdb_file_path=val_lmdb_path,
        win_size=win_size,
        step=step,
        return_mask=True,
    )
    return train_dataset, val_dataset


def unpack_input_and_observed_mask(batch, dataset_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """从 no-sparse batch 中取出输入序列和 feature-level nonzero mask。

    Args:
        batch: `RADBenchLMDBDataset(return_mask=True)` 的 DataLoader batch，字段数必须为 6。
        dataset_name: 数据集名称；该 no-sparse 入口只支持 `RADBench`。

    Returns:
        `input_x` 形状为 `(B, L, D)`，`feature_mask` 形状为 `(B, L, D)`。
    """
    if dataset_name != 'RADBench':
        raise ValueError(f'pipeline_no_sparse 仅支持 RADBench no-sparse LMDB，当前为 {dataset_name}')
    if len(batch) != 6:
        raise ValueError(f'RADBench no-sparse batch 应包含 6 个字段，当前为 {len(batch)}')
    _, _, _, input_x, _, feature_mask = batch
    if feature_mask.ndim != 3:
        raise ValueError(f'feature_mask 应为 (B, L, D)，当前 shape={tuple(feature_mask.shape)}')
    if tuple(feature_mask.shape) != tuple(input_x.shape):
        raise ValueError(f'feature_mask shape={tuple(feature_mask.shape)} 与 input_x shape={tuple(input_x.shape)} 不一致')
    return input_x, feature_mask


def get_filled_point_weight(config: dict[str, Any]) -> float:
    """读取 feature_mask 为 0 的位置在 reconstruction loss 中的保留权重。"""
    weight = float(get_training_cfg(config).get('filled_point_weight', 0.001))
    if weight < 0:
        raise ValueError(f'training.filled_point_weight 必须非负，当前为 {weight}')
    return weight


def reduce_observed_weighted_loss(
    element_loss: torch.Tensor,
    feature_mask: torch.Tensor,
    filled_point_weight: float,
    *,
    mode: str = 'mean',
) -> torch.Tensor:
    """按 feature-level mask 和 filled 权重规约逐元素 loss。

    Args:
        element_loss: 逐元素 loss，形状通常为 `(B, L, D)`。
        feature_mask: feature-level mask `(B, L, D)`；1 表示该特征值真实非零，0 表示该位置使用 filled 权重。
        filled_point_weight: `feature_mask == 0` 位置的 loss 权重；`feature_mask > 0` 位置权重固定为 1.0。
        mode: `mean` 表示加权逐元素均值；`sample_sum_mean` 表示先对每个样本加权求和，再取 batch 均值。

    Returns:
        标量 loss。
    """
    if filled_point_weight < 0:
        raise ValueError(f'filled_point_weight 必须非负，当前为 {filled_point_weight}')
    if feature_mask.ndim != 3:
        raise ValueError(f'feature_mask 应为 (B, L, D)，当前 shape={tuple(feature_mask.shape)}')
    if tuple(feature_mask.shape) != tuple(element_loss.shape):
        raise ValueError(
            f'feature_mask shape={tuple(feature_mask.shape)} 与 element_loss shape={tuple(element_loss.shape)} 不一致'
        )

    feature_mask = feature_mask.to(device=element_loss.device, dtype=element_loss.dtype)
    loss_weight = torch.where(
        feature_mask > 0,
        torch.ones_like(feature_mask),
        torch.full_like(feature_mask, float(filled_point_weight)),
    )
    weight_sum = loss_weight.sum()
    if weight_sum.item() <= 0:
        raise ValueError('feature_mask 当前 batch 没有有效 feature，且 filled_point_weight 为 0，无法计算 reconstruction loss')
    weighted_loss = element_loss * loss_weight

    if mode == 'mean':
        return weighted_loss.sum() / weight_sum
    if mode == 'sample_sum_mean':
        return weighted_loss.flatten(start_dim=1).sum(dim=1).mean()
    raise ValueError(f'Unsupported loss reduction mode: {mode}')


def observed_filled_element_loss_max(
    element_loss: torch.Tensor,
    feature_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """分别计算有效特征和未学习特征位置的逐元素 loss 最大值。

    Args:
        element_loss: 逐元素 loss，形状通常为 `(B, L, D)`。
        feature_mask: feature-level mask `(B, L, D)`；1 表示该特征值真实非零，0 表示该位置不参与重建学习。

    Returns:
        `(observed_max, filled_max)`。`observed_max` 是有效特征位置的逐元素 loss 最大值；
        `filled_max` 是未学习特征位置的逐元素 loss 最大值。若当前 batch 没有对应类别，返回 `nan`。
    """
    if feature_mask.ndim != 3:
        raise ValueError(f'feature_mask 应为 (B, L, D)，当前 shape={tuple(feature_mask.shape)}')
    if tuple(feature_mask.shape) != tuple(element_loss.shape):
        raise ValueError(
            f'feature_mask shape={tuple(feature_mask.shape)} 与 element_loss shape={tuple(element_loss.shape)} 不一致'
        )

    observed_element_mask = feature_mask.to(device=element_loss.device).bool()
    filled_element_mask = ~observed_element_mask

    observed_max = element_loss[observed_element_mask].max() if torch.any(observed_element_mask) else element_loss.new_tensor(float('nan'))
    filled_max = element_loss[filled_element_mask].max() if torch.any(filled_element_mask) else element_loss.new_tensor(float('nan'))
    return observed_max, filled_max


def summarize_observed_time_scores(
    time_score: torch.Tensor,
    time_mask: torch.Tensor,
    *,
    quantile: float | None = None,
) -> dict[str, torch.Tensor]:
    """只在真实观测时间点上统计 `(B, L)` 时间点 score。

    Args:
        time_score: 时间点级 score，形状为 `(B, L)`；若只输出最后一点，可为 `(B, 1)`。
        time_mask: 时间点级 mask `(B, L)`；由 feature-level mask 在特征维做 `any` 得到。
        quantile: 可选分位数，取值范围 `[0, 1]`。

    Returns:
        包含 `min`、`max`、`mean` 的统计字典；若传入 `quantile`，额外包含 `quantile`。
        当前 batch 没有真实观测 score 时，对应统计值为 `nan`。
    """
    if time_score.ndim != 2:
        raise ValueError(f'time_score 应为 (B, L)，当前 shape={tuple(time_score.shape)}')
    if time_mask.ndim != 2:
        raise ValueError(f'time_mask 应为 (B, L)，当前 shape={tuple(time_mask.shape)}')
    if time_score.shape[0] != time_mask.shape[0]:
        raise ValueError(
            f'time_score batch={time_score.shape[0]} 与 time_mask batch={time_mask.shape[0]} 不一致'
        )
    if time_score.shape[1] == time_mask.shape[1]:
        score_mask = time_mask
    elif time_score.shape[1] == 1:
        score_mask = time_mask[:, -1:]
    else:
        raise ValueError(
            f'time_score shape={tuple(time_score.shape)} 与 time_mask shape={tuple(time_mask.shape)} 无法对齐'
        )
    if quantile is not None and not 0.0 <= float(quantile) <= 1.0:
        raise ValueError(f'quantile 必须位于 [0, 1]，当前为 {quantile}')

    score_mask = score_mask.to(device=time_score.device).bool()
    observed_scores = time_score[score_mask]
    nan_value = time_score.new_tensor(float('nan'))
    if observed_scores.numel() == 0:
        summary = {'min': nan_value, 'max': nan_value, 'mean': nan_value}
    else:
        summary = {
            'min': observed_scores.min(),
            'max': observed_scores.max(),
            'mean': observed_scores.mean(),
        }
    if quantile is not None:
        summary['quantile'] = torch.quantile(observed_scores.reshape(-1), float(quantile)) if observed_scores.numel() > 0 else nan_value
    return summary


def build_dataloaders(config: dict[str, Any], train_dataset: Any, val_dataset: Any) -> tuple[DataLoader, DataLoader]:
    training_cfg = get_training_cfg(config)
    batch_size = int(training_cfg['batch_size'])
    num_workers = int(training_cfg.get('num_workers', 8))
    pin_memory = bool(training_cfg.get('pin_memory', False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=should_persist_workers(num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=should_persist_workers(num_workers),
    )
    return train_loader, val_loader


def build_default_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> optim.Optimizer:
    """壳层默认优化器，保持旧 MSE 训练脚本的 AdamW 语义。"""
    training_cfg = get_training_cfg(config)
    return optim.AdamW(
        model.parameters(),
        lr=float(training_cfg['learning_rate']),
        weight_decay=float(training_cfg.get('weight_decay', 0.01)),
    )


def build_default_scheduler(
    optimizer: optim.Optimizer,
    config: dict[str, Any],
    steps_per_epoch: int,
) -> LambdaLR:
    """壳层默认 scheduler，保持旧脚本的 inverse square root warmup。"""
    training_cfg = get_training_cfg(config)
    use_warmup = bool(training_cfg.get('use_warmup', True))

    if use_warmup:
        warmup_steps = int(training_cfg.get('warmup_epochs', 1)) * steps_per_epoch
        warmup_steps = max(warmup_steps, 1)

        def inverse_sqrt_schedule(step: int) -> float:
            current_step = max(int(step), 1)
            if current_step < warmup_steps:
                return current_step / warmup_steps
            return (warmup_steps / current_step) ** 0.5

        return LambdaLR(optimizer, lr_lambda=inverse_sqrt_schedule)

    return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer | None,
    scheduler: Any,
    epoch: int,
    global_step: int,
    path: str,
    extra_state: dict[str, Any] | None = None,
) -> str:
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, 'state_dict'):
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if extra_state:
        checkpoint['extra_state'] = extra_state

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    log.info('Checkpoint saved to %s', path)
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: Any = None,
    device: torch.device | None = None,
) -> tuple[int, int, dict[str, Any]] | None:
    """恢复 checkpoint，epoch 语义是“已完成轮数”，恢复后从下一轮继续。"""
    if not os.path.exists(path):
        log.error('Checkpoint %s does not exist', path)
        return None

    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and hasattr(scheduler, 'load_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = int(checkpoint.get('epoch', 0))
    global_step = int(checkpoint.get('global_step', 0))
    extra_state = checkpoint.get('extra_state', {})
    log.info('Loaded checkpoint from %s (epoch %d, step %d)', path, epoch, global_step)
    return epoch, global_step, extra_state


def get_resume_checkpoint_path(config: dict[str, Any]) -> str:
    return str(get_training_cfg(config).get('resume_checkpoint', '')).strip()


def load_env() -> None:
    dotenv.load_dotenv()
    GPUAutoSelector().select()
