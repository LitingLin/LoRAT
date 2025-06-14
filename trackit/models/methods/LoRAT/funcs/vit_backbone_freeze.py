from itertools import chain
import torch.nn as nn


def freeze_vit_backbone_(model: nn.Module, freeze_out_norm: bool = True) -> nn.Module:
    """
    Freeze all backbone weights of a ViT‑style model *in‑place*.

    Notes
    -----
    * The routine assumes the model exposes attributes
      `patch_embed`, `blocks`, `pos_embed`, and `norm`.
    * Call *before* optimizer creation.
    """
    to_freeze = chain(
        model.patch_embed.parameters(),
        model.blocks.parameters(),
        (model.pos_embed,), model.norm.parameters() if freeze_out_norm else ()
    )
    for p in to_freeze:
        p.requires_grad = False
    return model
