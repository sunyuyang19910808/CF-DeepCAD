from __future__ import annotations

import os
from collections import OrderedDict

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from trainer.base import TrainClock
from trainer.scheduler import GradualWarmupScheduler
from utils import cycle

from constraint_fused_deepcad_step.application.device import resolve_device
from constraint_fused_deepcad_step.application.train_use_case import build_train_use_case
from constraint_fused_deepcad_step.config.config_step import ConfigStep
from constraint_fused_deepcad_step.infrastructure.checkpoint_repository_fs import ModelCheckpointRepositoryFs
from constraint_fused_deepcad_step.infrastructure.dataset_step import get_step_dataloader
from constraint_fused_deepcad_step.infrastructure.experiment_tracker import ExperimentTracker


def save_ckpt(use_case, optimizer, scheduler, clock: TrainClock, cfg, name: str) -> str:
    repo = ModelCheckpointRepositoryFs(cfg.model_dir)
    path = repo.save(
        {
            "clock": clock.make_checkpoint(),
            "model_state_dict": use_case.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        "{}.pth".format(name),
    )
    return path


def load_ckpt(use_case, optimizer, scheduler, clock: TrainClock, cfg) -> None:
    ckpt_name = cfg.ckpt if cfg.ckpt == "latest" else "ckpt_epoch{}".format(cfg.ckpt)
    repo = ModelCheckpointRepositoryFs(cfg.model_dir)
    checkpoint = repo.load("{}.pth".format(ckpt_name), map_location=resolve_device(cfg))
    use_case.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    clock.restore_checkpoint(checkpoint["clock"])


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def main() -> None:
    cfg = ConfigStep("train")
    device = resolve_device(cfg)
    use_case = build_train_use_case(cfg, device)
    params = list(use_case.model.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = GradualWarmupScheduler(optimizer, 1.0, cfg.warmup_step)
    clock = TrainClock()

    if cfg.cont:
        try:
            load_ckpt(use_case, optimizer, scheduler, clock, cfg)
        except FileNotFoundError:
            pass

    train_loader = get_step_dataloader("train", cfg, shuffle=True)
    val_loader = cycle(get_step_dataloader("validation", cfg, shuffle=False))

    train_tb = SummaryWriter(os.path.join(cfg.log_dir, "train.events"))
    val_tb = SummaryWriter(os.path.join(cfg.log_dir, "val.events"))
    tracker = ExperimentTracker(cfg.artifact_dir, cfg.exp_name, cfg.data_root)
    tracker.write_manifest(
        {
            key: getattr(cfg, key)
            for key in dir(cfg)
            if not key.startswith("_") and isinstance(getattr(cfg, key), (int, float, str, bool, dict))
        },
        dataset_split="train",
    )
    stop_requested = False
    best_ckpt_path = ""
    for epoch in range(clock.epoch, cfg.nr_epochs + 1):
        cfg._current_epoch = epoch
        use_case.train()
        pbar = tqdm(train_loader)
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            out = use_case.execute(batch, epoch=epoch)
            out["loss"].backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            metrics = OrderedDict(
                loss=_scalar(out["loss"]),
                loss_cmd=_scalar(out["loss_cmd"]),
                loss_cmd_raw=_scalar(out["loss_cmd_raw"]),
                loss_args=_scalar(out["loss_args"]),
                loss_args_raw=_scalar(out["loss_args_raw"]),
                loss_geom=_scalar(out["loss_geom"]),
                geom_weighted=_scalar(out["geom_weighted"]),
                geom_ratio=_scalar(out["geom_effective_ratio"]),
                geom_target_ratio=_scalar(out["geom_target_ratio"]),
                gamma=_scalar(out["gamma_geom"]),
                geom_h=_scalar(out["geom_h"]),
                geom_v=_scalar(out["geom_v"]),
                geom_parallel=_scalar(out["geom_parallel"]),
                geom_perpendicular=_scalar(out["geom_perpendicular"]),
                positive_count_h=_scalar(out["positive_count_h"]),
                positive_count_v=_scalar(out["positive_count_v"]),
                positive_count_parallel=_scalar(out["positive_count_parallel"]),
                positive_count_perpendicular=_scalar(out["positive_count_perpendicular"]),
            )
            pbar.set_description("EPOCH[{}][{}]".format(epoch, batch_idx))
            pbar.set_postfix(metrics)

            if clock.step % cfg.log_frequency == 0:
                for key, value in metrics.items():
                    train_tb.add_scalar(key, value, clock.step)
                row = {"epoch": epoch, "step": clock.step, **metrics}
                tracker.append_train_metrics(row)

            if clock.step % cfg.val_frequency == 0:
                use_case.eval()
                with torch.no_grad():
                    val_out = use_case.execute(next(val_loader), epoch=epoch)
                val_metrics = OrderedDict(
                    loss=_scalar(val_out["loss"]),
                    loss_cmd=_scalar(val_out["loss_cmd"]),
                    loss_args=_scalar(val_out["loss_args"]),
                    loss_geom=_scalar(val_out["loss_geom"]),
                    geom_weighted=_scalar(val_out["geom_weighted"]),
                    geom_ratio=_scalar(val_out["geom_effective_ratio"]),
                    geom_target_ratio=_scalar(val_out["geom_target_ratio"]),
                    gamma=_scalar(val_out["gamma_geom"]),
                )
                for key, value in val_metrics.items():
                    val_tb.add_scalar(key, value, clock.step)
                use_case.train()

            clock.tick()
            if cfg.max_steps and clock.step >= cfg.max_steps:
                stop_requested = True
                break

        latest_path = save_ckpt(use_case, optimizer, scheduler, clock, cfg, "latest")
        best_ckpt_path = latest_path
        if epoch % cfg.save_frequency == 0:
            save_ckpt(use_case, optimizer, scheduler, clock, cfg, "ckpt_epoch{}".format(epoch))
        clock.tock()
        if stop_requested:
            break

    tracker.write_best_checkpoint(best_ckpt_path)
    train_tb.close()
    val_tb.close()


if __name__ == "__main__":
    main()
