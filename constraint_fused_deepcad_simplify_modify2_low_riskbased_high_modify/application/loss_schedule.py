from __future__ import annotations


def resolve_aux_weights(cfg, epoch: int) -> tuple[float, float, float, float, float, float]:
    """Return ``(alpha, beta, gamma_h, gamma_v, gamma_para, gamma_perp)`` for ``epoch``.

    Backward compatibility:

    * If the user only sets the legacy ``--gamma X``, the four per-component gammas all default
      to ``X`` (matching A1/A2/A2b/A2c behaviour where ``gamma * geom_loss`` collapses to the
      same value as ``gamma * (L_h + L_v + L_para + L_perp)`` because the four soft components
      share the combined denominator).
    * If the user explicitly sets ``--gamma_horizontal`` / ``--gamma_vertical`` /
      ``--gamma_parallel`` / ``--gamma_perpendicular`` (A2d path), those overrides take effect.
    * ``aux_schedule=warmup`` scales every weight linearly from epoch ``aux_warmup_start_epoch``
      to ``aux_warmup_end_epoch``.
    """

    alpha = float(getattr(cfg, "alpha", 0.0))
    beta = float(getattr(cfg, "beta", 0.0))
    legacy_gamma = float(getattr(cfg, "gamma", 0.0))

    def _pick(name: str) -> float:
        value = getattr(cfg, name, None)
        if value is None:
            return legacy_gamma
        return float(value)

    gamma_h = _pick("gamma_horizontal")
    gamma_v = _pick("gamma_vertical")
    gamma_para = _pick("gamma_parallel")
    gamma_perp = _pick("gamma_perpendicular")

    if getattr(cfg, "aux_schedule", "constant") != "warmup":
        return alpha, beta, gamma_h, gamma_v, gamma_para, gamma_perp

    start = int(getattr(cfg, "aux_warmup_start_epoch", 10))
    end = int(getattr(cfg, "aux_warmup_end_epoch", 30))
    if epoch <= start:
        ratio = 0.0
    elif epoch >= end:
        ratio = 1.0
    else:
        ratio = float(epoch - start) / float(max(end - start, 1))

    return (
        alpha * ratio,
        beta * ratio,
        gamma_h * ratio,
        gamma_v * ratio,
        gamma_para * ratio,
        gamma_perp * ratio,
    )
