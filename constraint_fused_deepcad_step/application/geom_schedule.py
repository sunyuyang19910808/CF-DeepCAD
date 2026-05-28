from __future__ import annotations


def resolve_gamma_geom(cfg, epoch: int) -> float:
    """Resolve effective gamma_geom with optional warmup schedule."""
    if not getattr(cfg, "enable_geom_loss", False):
        return 0.0
    if getattr(cfg, "geom_log_only", False):
        return 0.0

    gamma = float(getattr(cfg, "gamma_geom", 0.0))
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
