"""RADBench no-sparse 无监督训练壳层。"""

import importlib
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.performance import measure_if_active

from .model import build_unsupervised_model
from .training import (
    build_dataloaders,
    build_datasets,
    build_default_optimizer,
    build_default_scheduler,
    get_device,
    get_method_name,
    get_resume_checkpoint_path,
    get_training_cfg,
    load_checkpoint,
    load_config,
    load_env,
    prepare_run_directory,
    resolve_model_cfg,
    save_checkpoint,
    set_random_seed,
    setup_logging,
)


log = logging.getLogger(__name__)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')


def import_method_module(method_name: str):
    """Import one training method within the unsupervised package.

    Args:
        method_name: Module name configured as the unsupervised method.

    Returns:
        Imported method module with its package context preserved.
    """
    return importlib.import_module(f'.{method_name}', package=__package__)


def main(config_path: str = CONFIG_PATH) -> str:
    """Train through the configured final epoch and return its checkpoint path."""
    load_env()
    setup_logging()
    config = load_config(config_path)
    set_random_seed(int(config['seed']))
    device = get_device()

    method_name = get_method_name(config)
    method_module = import_method_module(method_name)
    method = method_module.METHOD
    method.validate_config(config)

    model_cfg = resolve_model_cfg(config)
    model = build_unsupervised_model(model_cfg=model_cfg).to(device)

    train_dataset, val_dataset = build_datasets(config)
    train_loader, val_loader = build_dataloaders(config, train_dataset, val_dataset)

    method_optimizer = method.build_optimizer(model, config)
    optimizer = method_optimizer if method_optimizer is not None else build_default_optimizer(model, config)

    method_scheduler = method.build_scheduler(optimizer, config, len(train_loader))
    scheduler = method_scheduler if method_scheduler is not None else build_default_scheduler(optimizer, config, len(train_loader))

    model_save_root = prepare_run_directory(
        config=config,
        model=model,
        config_path=config_path,
        method_name=method_name,
    )

    global_step = 1
    start_epoch = 1
    resume_checkpoint = get_resume_checkpoint_path(config)
    if resume_checkpoint:
        if method_name == 'catch':
            resumed = load_checkpoint(
                path=resume_checkpoint,
                model=model,
                optimizer=optimizer['main'],
                scheduler=scheduler['main'],
                device=device,
            )
            if resumed is not None:
                _, _, extra_state = resumed
                if 'mask_optimizer_state_dict' not in extra_state:
                    raise KeyError('CATCH checkpoint 缺少 extra_state.mask_optimizer_state_dict')
                if 'mask_scheduler_state_dict' not in extra_state:
                    raise KeyError('CATCH checkpoint 缺少 extra_state.mask_scheduler_state_dict')
                optimizer['mask'].load_state_dict(extra_state['mask_optimizer_state_dict'])
                scheduler['mask'].load_state_dict(extra_state['mask_scheduler_state_dict'])
        else:
            resumed = load_checkpoint(
                path=resume_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
        if resumed is not None:
            loaded_epoch, loaded_global_step, _ = resumed
            start_epoch = loaded_epoch + 1
            global_step = loaded_global_step + 1
            log.info('Resuming training from epoch %d, step %d', start_epoch, global_step)

    max_epochs = int(get_training_cfg(config)['max_epochs'])
    log.info('Starting unsupervised training with method: %s', method_name)

    final_checkpoint_path: str | None = None
    for epoch in range(start_epoch, max_epochs + 1):
        with measure_if_active(
            "training.reconstructor", cuda=device.type == "cuda"
        ):
            train_metrics, global_step = method.train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                config=config,
                epoch=epoch,
                global_step=global_step,
                model_save_root=model_save_root,
            )
        log.info('Epoch %d train metrics: %s', epoch, train_metrics)

        val_metrics = method.val_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            config=config,
            epoch=epoch,
        )
        log.info('Epoch %d validation metrics: %s', epoch, val_metrics)

        checkpoint_name = f'{os.path.basename(model_save_root)}.pth'
        final_checkpoint_path = os.path.join(model_save_root, checkpoint_name)
        if method_name == 'catch':
            save_checkpoint(
                model=model,
                optimizer=optimizer['main'],
                scheduler=scheduler['main'],
                epoch=epoch,
                global_step=global_step,
                path=final_checkpoint_path,
                extra_state={
                    'mask_optimizer_state_dict': optimizer['mask'].state_dict(),
                    'mask_scheduler_state_dict': scheduler['mask'].state_dict(),
                },
            )
        else:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                path=final_checkpoint_path,
            )

    if final_checkpoint_path is None:
        raise RuntimeError(
            f'No training epoch executed: start_epoch={start_epoch}, max_epochs={max_epochs}'
        )
    return final_checkpoint_path


if __name__ == '__main__':
    main()
