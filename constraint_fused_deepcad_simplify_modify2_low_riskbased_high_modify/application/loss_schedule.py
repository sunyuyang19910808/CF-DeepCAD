from __future__ import annotations


def resolve_aux_weights(cfg, epoch: int) -> tuple[float, float, float]:
    """Return the effective auxiliary loss weights for the current epoch."""
    alpha = float(getattr(cfg, "alpha", 0.0))
    beta = float(getattr(cfg, "beta", 0.0))
    gamma = float(getattr(cfg, "gamma", 0.0))
    if getattr(cfg, "aux_schedule", "constant") != "warmup":
        return alpha, beta, gamma

    start = int(getattr(cfg, "aux_warmup_start_epoch", 10))
    end = int(getattr(cfg, "aux_warmup_end_epoch", 30))
    if epoch <= start:
        ratio = 0.0
    elif epoch >= end:
        ratio = 1.0
    else:
        ratio = float(epoch - start) / float(max(end - start, 1))
    return alpha * ratio, beta * ratio, gamma * ratio
