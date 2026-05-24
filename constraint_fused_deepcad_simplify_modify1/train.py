from __future__ import annotations

import csv
import os
from collections import OrderedDict

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from trainer.base import TrainClock
from trainer.scheduler import GradualWarmupScheduler
from utils import cycle, ensure_dir

from constraint_fused_deepcad_simplify_modify1.application.evaluate_axis_constraints import resolve_device
from constraint_fused_deepcad_simplify_modify1.application.train_use_case import build_train_use_case
from constraint_fused_deepcad_simplify_modify1.config.config_constraint_fused_simplify_modify1 import (
    ConfigConstraintFusedSimplifyModify1,
)
from constraint_fused_deepcad_simplify_modify1.infrastructure.dataset_simplify_modify1 import (
    get_simplify_modify1_dataloader,
)


def save_ckpt(use_case, optimizer, scheduler, clock: TrainClock, cfg, name: str) -> None:
    ensure_dir(cfg.model_dir)
    save_path = os.path.join(cfg.model_dir, "{}.pth".format(name))
    torch.save(
        {
            "clock": clock.make_checkpoint(),
            "model_state_dict": use_case.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        save_path,
    )


def load_ckpt(use_case, optimizer, scheduler, clock: TrainClock, cfg) -> None:
    ckpt_name = cfg.ckpt if cfg.ckpt == "latest" else "ckpt_epoch{}".format(cfg.ckpt)
    load_path = os.path.join(cfg.model_dir, "{}.pth".format(ckpt_name))
    checkpoint = torch.load(load_path, map_location=resolve_device(cfg))
    use_case.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    clock.restore_checkpoint(checkpoint["clock"])


def append_csv_row(csv_path: str, row: dict) -> None:
    ensure_dir(os.path.dirname(csv_path))
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    cfg = ConfigConstraintFusedSimplifyModify1("train")
    device = resolve_device(cfg)
    use_case = build_train_use_case(cfg, device)
    params = []
    for module in use_case.modules():
        params.extend(list(module.parameters()))
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = GradualWarmupScheduler(optimizer, 1.0, cfg.warmup_step)
    clock = TrainClock()

    if cfg.cont:
        load_ckpt(use_case, optimizer, scheduler, clock, cfg)

    train_loader = get_simplify_modify1_dataloader("train", cfg, shuffle=True)
    val_loader = cycle(get_simplify_modify1_dataloader("validation", cfg, shuffle=False))

    train_tb = SummaryWriter(os.path.join(cfg.log_dir, "train.events"))
    val_tb = SummaryWriter(os.path.join(cfg.log_dir, "val.events"))
    metrics_csv = os.path.join(cfg.artifact_dir, "train_metrics.csv")

    stop_requested = False
    for epoch in range(clock.epoch, cfg.nr_epochs + 1):
        use_case.train()
        pbar = tqdm(train_loader)
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            out = use_case.execute(batch)
            out["loss"].backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            metrics = OrderedDict(
                loss=float(out["loss"].item()),
                loss_cmd=float(out["loss_cmd"].item()),
                loss_cmd_only=float(out["loss_cmd_only"].item()),
                loss_args=float(out["loss_args"].item()),
                axis_loss=float(out["axis_loss"].item()),
                pred_loss=float(out["pred_loss"].item()),
                geom_loss=float(out["geom_loss"].item()),
            )
            pbar.set_description("EPOCH[{}][{}]".format(epoch, batch_idx))
            pbar.set_postfix(metrics)

            if clock.step % cfg.log_frequency == 0:
                for key, value in metrics.items():
                    train_tb.add_scalar(key, value, clock.step)
                append_csv_row(metrics_csv, {"epoch": epoch, "step": clock.step, **metrics})

            if clock.step % cfg.val_frequency == 0:
                use_case.eval()
                with torch.no_grad():
                    val_out = use_case.execute(next(val_loader))
                for key in ("loss", "loss_cmd", "axis_loss", "pred_loss", "geom_loss"):
                    val_tb.add_scalar(key, float(val_out[key].item()), clock.step)
                use_case.train()

            clock.tick()
            if cfg.max_steps and clock.step >= cfg.max_steps:
                stop_requested = True
                break

        save_ckpt(use_case, optimizer, scheduler, clock, cfg, "latest")
        if epoch % cfg.save_frequency == 0:
            save_ckpt(use_case, optimizer, scheduler, clock, cfg, "ckpt_epoch{}".format(epoch))
        clock.tock()
        if stop_requested:
            break

    train_tb.close()
    val_tb.close()


if __name__ == "__main__":
    main()
