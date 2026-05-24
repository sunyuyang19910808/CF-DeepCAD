import torch

from constraint_fused_deepcad.domain.entities import ConstraintAwareLatent
from constraint_fused_deepcad.generation.decoder_adapter import ConstraintAwareDecoderAdapter
from model.model_utils import _make_batch_first


class GenerateFromLatentUseCase:
    def __init__(self, decoder_adapter: ConstraintAwareDecoderAdapter):
        self.decoder_adapter = decoder_adapter

    def execute(self, z: torch.Tensor) -> dict:
        if z.dim() == 2:
            z = z.unsqueeze(0)
        latent = ConstraintAwareLatent(z)
        cmd_logits_sf, args_logits_sf, _ = self.decoder_adapter(latent.tensor, constraint_memory=None, constraint_mask=None)
        cmd_logits, args_logits = _make_batch_first(cmd_logits_sf, args_logits_sf)
        return {"command_logits": cmd_logits, "args_logits": args_logits}
