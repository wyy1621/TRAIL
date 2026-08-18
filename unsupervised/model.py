"""TimesNet-only extraction of the source model factory."""

from __future__ import annotations

import torch
import torch.nn as nn

from .timesnet import TimesNet


def get_unsupervised_reconstruction(
    model_name: str,
    model_output: torch.Tensor | tuple,
) -> torch.Tensor:
    """Extract the reconstruction tensor using the source TimesNet semantics."""
    if not isinstance(model_output, torch.Tensor):
        raise TypeError(f"{model_name} 推理时应返回重建张量")
    return model_output


def build_unsupervised_model(
    config: dict | None = None,
    *,
    model_cfg: dict | None = None,
) -> nn.Module:
    """Build the source TimesNet branch and reject models outside this repository."""
    if model_cfg is None:
        if config is None:
            raise ValueError("config 和 model_cfg 不能同时为 None")
        model_cfg = config["UnsuperviseModel"]

    model_name = model_cfg["model_name"]
    if model_name == "TimesNet":
        return TimesNet(
            seq_len=model_cfg["win_size"],
            pred_len=0,
            d_model=model_cfg["d_model"],
            d_ff=model_cfg["d_ff"],
            task_name="anomaly_detection",
            enc_in=model_cfg["feature_dim"],
            c_out=model_cfg["feature_dim"],
            dropout=model_cfg["dropout"],
        )
    raise ValueError(f"Unsupported unsupervised model: {model_name}")
