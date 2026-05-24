from __future__ import annotations

import csv
import os
import sys
from collections import OrderedDict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from trainer.base import TrainClock
from trainer.scheduler import GradualWarmupScheduler
from utils import cycle, ensure_dir

from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.generate_use_case import resolve_device
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.loss_schedule import (
    resolve_aux_weights,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.train_use_case import build_train_use_case
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.config.config_constraint_fused_high_modify import (
    ConfigConstraintFusedHighModify,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.infrastructure.checkpoint_repository_fs import (
    ModelCheckpointRepositoryFs,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.infrastructure.dataset_high_modify import (
    get_high_modify_dataloader,
)
from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.infrastructure.experiment_tracker import ExperimentTracker


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


def _resolve_checkpoint_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def load_init_weights(use_case, cfg, device) -> None:
    init_path = (getattr(cfg, "init_weights_from", "") or "").strip()
    if not init_path:
        return
    if cfg.cont:
        raise ValueError("--init_weights_from cannot be used together with --continue.")
    resolved = _resolve_checkpoint_path(init_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError("init_weights_from not found: {}".format(resolved))
    checkpoint = torch.load(resolved, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    use_case.load_state_dict(state_dict)
    print("Loaded init weights (model only) from: {}".format(resolved))


def append_csv_row(csv_path: str, row: dict) -> None:
    ensure_dir(os.path.dirname(csv_path))
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    cfg = ConfigConstraintFusedHighModify("train")
    device = resolve_device(cfg)
    use_case = build_train_use_case(cfg, device)
    params = []
    for module in use_case.modules():
        params.extend(list(module.parameters()))
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = GradualWarmupScheduler(optimizer, 1.0, cfg.warmup_step)
    clock = TrainClock()

    if cfg.cont:
        try:
            load_ckpt(use_case, optimizer, scheduler, clock, cfg)
            if clock.minibatch > 0:
                clock.tock()
        except FileNotFoundError:
            pass
    else:
        load_init_weights(use_case, cfg, device)

    train_loader = get_high_modify_dataloader("train", cfg, shuffle=True)
    val_loader = cycle(get_high_modify_dataloader("validation", cfg, shuffle=False))

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
    metrics_csv = os.path.join(cfg.artifact_dir, "train_metrics.csv")

    stop_requested = False
    best_ckpt_path = ""
    for epoch in range(clock.epoch, cfg.nr_epochs + 1):
        use_case.train()
        pbar = tqdm(train_loader)
        aux_weights = resolve_aux_weights(cfg, epoch)
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            out = use_case.execute(batch, aux_weights=aux_weights)
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
                pred_loss=float(out["pred_loss"].item()),
                recon_loss=float(out["recon_loss"].item()),
                unary_recon_loss=float(out["unary_recon_loss"].item()),
                pair_recon_loss=float(out["pair_recon_loss"].item()),
                geom_loss=float(out["geom_loss"].item()),
                geom_horizontal=float(out["geom_horizontal"].item()),
                geom_vertical=float(out["geom_vertical"].item()),
                geom_parallel=float(out["geom_parallel"].item()),
                geom_perpendicular=float(out["geom_perpendicular"].item()),
                aux_alpha=float(out["aux_alpha"]),
                aux_beta=float(out["aux_beta"]),
                aux_gamma=float(out["aux_gamma"]),
            )
            if getattr(cfg, "use_hard_geom_bce", False) or getattr(cfg, "use_corrected_line_start", False):
                metrics.update(
                    geom_h_loss=float(out["geom_h_loss"].item()),
                    geom_v_loss=float(out["geom_v_loss"].item()),
                    geom_para_loss=float(out["geom_para_loss"].item()),
                    geom_perp_loss=float(out["geom_perp_loss"].item()),
                    aux_gamma_h=float(out["aux_gamma_h"]),
                    aux_gamma_v=float(out["aux_gamma_v"]),
                    aux_gamma_para=float(out["aux_gamma_para"]),
                    aux_gamma_perp=float(out["aux_gamma_perp"]),
                )
            pbar.set_description("EPOCH[{}][{}]".format(epoch, batch_idx))
            pbar.set_postfix(metrics)

            if clock.step % cfg.log_frequency == 0:
                for key, value in metrics.items():
                    train_tb.add_scalar(key, value, clock.step)
                row = {"epoch": epoch, "step": clock.step, **metrics}
                append_csv_row(metrics_csv, row)

            if clock.step % cfg.val_frequency == 0:
                use_case.eval()
                with torch.no_grad():
                    val_out = use_case.execute(next(val_loader), aux_weights=aux_weights)
                for key in ("loss", "loss_cmd", "pred_loss", "recon_loss", "geom_loss", "aux_alpha", "aux_beta", "aux_gamma"):
                    value = val_out[key]
                    val_tb.add_scalar(key, float(value.item() if hasattr(value, "item") else value), clock.step)
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
    tracker.write_qualitative_cases([])
    train_tb.close()
    val_tb.close()


if __name__ == "__main__":
    main()
