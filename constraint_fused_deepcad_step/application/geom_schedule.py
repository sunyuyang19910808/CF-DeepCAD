from __future__ import annotations

from dataclasses import dataclass


def _warmup_progress(cfg, epoch: int) -> float:
    """Linear ramp in [0, 1] over geom_warmup_start_epoch .. geom_warmup_end_epoch."""
    start = int(getattr(cfg, "geom_warmup_start_epoch", 1))
    end = int(getattr(cfg, "geom_warmup_end_epoch", 1))
    if end <= start:
        return 1.0 if epoch >= start else 0.0
    if epoch < start:
        return 0.0
    if epoch >= end:
        return 1.0
    return float(epoch - start) / float(end - start)


def _geom_enabled(cfg) -> bool:
    if not getattr(cfg, "enable_geom_loss", False):
        return False
    if getattr(cfg, "geom_log_only", False):
        return False
    return True


def resolve_gamma_geom(cfg, epoch: int) -> float:
    """Resolve fixed gamma_geom with optional warmup schedule (legacy mode)."""
    if not _geom_enabled(cfg):
        return 0.0

    gamma = float(getattr(cfg, "gamma_geom", 0.0))
    if float(getattr(cfg, "geom_target_ratio", 0.0)) > 0.0:
        return 0.0

    start = int(getattr(cfg, "geom_warmup_start_epoch", 1))
    end = int(getattr(cfg, "geom_warmup_end_epoch", 1))
    if end <= start:
        return gamma if epoch >= start else 0.0
    if epoch < start:
        return 0.0
    if epoch >= end:
        return gamma
    progress = float(epoch - start) / float(end - start)
    return gamma * progress


def resolve_target_geom_ratio(cfg, epoch: int) -> float:
    """Effective target geom_effective_ratio for adaptive gamma (with warmup)."""
    if not _geom_enabled(cfg):
        return 0.0
    target = float(getattr(cfg, "geom_target_ratio", 0.0))
    if target <= 0.0:
        return 0.0
    warmup_start = int(getattr(cfg, "geom_warmup_start_epoch", 1))
    progress = _warmup_progress(cfg, epoch)
    if epoch < warmup_start:
        return 0.0
    ratio_start = float(getattr(cfg, "geom_target_ratio_start", 0.0))
    if ratio_start > 0.0:
        return ratio_start + (target - ratio_start) * progress
    return target * progress


@dataclass
class GeomLossEma:
    """EMA smoother for main-task and geom losses used in adaptive gamma."""

    decay: float
    main: float | None = None
    geom: float | None = None

    def update(self, main: float, geom: float) -> tuple[float, float]:
        if self.decay <= 0.0:
            return main, geom
        if self.main is None:
            self.main = main
            self.geom = geom
        else:
            keep = self.decay
            blend = 1.0 - keep
            self.main = keep * self.main + blend * main
            self.geom = keep * self.geom + blend * geom
        return self.main, self.geom


def resolve_adaptive_gamma_geom(
    cfg,
    epoch: int,
    main_task_loss,
    loss_geom,
    ema: GeomLossEma | None = None,
) -> tuple[float, float]:
    """Derive gamma_geom from target geom_ratio: gamma = target * main / geom.

    Returns (gamma_geom, geom_target_ratio_effective).
    """
    target_ratio = resolve_target_geom_ratio(cfg, epoch)
    if target_ratio <= 0.0:
        return 0.0, 0.0

    main = float(main_task_loss.detach().item())
    geom = float(loss_geom.detach().item())
    if ema is not None:
        main, geom = ema.update(main, geom)

    gamma_max = float(getattr(cfg, "gamma_geom", 0.1))
    gamma_min = float(getattr(cfg, "geom_gamma_min", 1e-6))
    gamma = target_ratio * main / max(geom, 1e-8)
    gamma = max(gamma_min, min(gamma, gamma_max))
    return gamma, target_ratio
