#!/usr/bin/env python
"""
SwinTransformerSysEntropyAttention.py

This file imports the original network from swin_unet.py and extends it by adding
final-layer entropy-based attention gating. The new class, SwinTransformerSysEntropyAttention,
inherits from the original SwinTransformerSys and only modifies the final forward pass.

Modifications:
  1. An auxiliary head (self.aux_head) is used to produce logits from the final decoder features.
  2. A per-pixel entropy map is computed from these logits.
  3. The gating factor is computed as exp(-β · entropy) and then normalized so that it sums to 1
     over the spatial dimensions.
  4. The final decoder features are reweighted by this normalized gating before final upsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the original network.
from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

# ------------------------------------------------------------------------------
# Helper: compute_entropy
# ------------------------------------------------------------------------------
def compute_entropy(logits):
    """
    Compute per-pixel entropy from logits.

    Args:
        logits: Tensor of shape (B, num_classes, H, W).

    Returns:
        Tensor of shape (B, 1, H, W) containing the per-pixel entropy.
    """
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1, keepdim=True)
    return entropy

# ------------------------------------------------------------------------------
# New Class: SwinTransformerSysEntropyAttention
# ------------------------------------------------------------------------------
class SwinTransformerSysEntropyAttention(SwinTransformerSys):
    """
    Extends the original SwinTransformerSys by adding final-layer entropy-based attention gating.
    In the final decoder stage, the feature map is reshaped to 2D, an auxiliary head produces logits,
    from which a per-pixel entropy map is computed. A gating factor is then calculated as
      gating = exp(-1.0 * entropy)
and normalized so that the sum over the spatial dimensions is 1. This normalized gating is used to
reweight the final decoder features before the final upsampling.
    """
    def __init__(self, *args, **kwargs):
        # Ensure that final_entropy is enabled.
        kwargs['final_entropy'] = True
        super().__init__(*args, **kwargs)
        self.beta = 1.0  # Beta value for gating factor

    def forward(self, x):
        # Run encoder and decoder as in the original network.
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        B, L, C = x.shape
        H, W = self.patches_resolution  # Final patch grid resolution

        # Reshape final decoder features from (B, L, C) to (B, C, H, W)
        x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Use the auxiliary head (already defined if final_entropy is enabled) to get logits.
        logits = self.aux_head(x_2d)  # Shape: (B, num_classes, H, W)

        # Compute per-pixel entropy from the logits.
        ent_map = compute_entropy(logits)  # Shape: (B, 1, H, W)

        # Compute gating factor (using beta=1.0) and normalize it.
        gating = torch.exp(-self.beta * ent_map)
        # Normalize gating so that the sum over spatial dimensions is 1 for each sample.
        gating = gating / (gating.sum(dim=(2,3), keepdim=True) + 1e-9)

        # Reweight the final decoder features with the normalized gating factor.
        x_2d = x_2d * gating

        # Restore shape to (B, L, C)
        x = x_2d.permute(0, 2, 3, 1).view(B, L, C)

        # Final upsampling as in the original network.
        x = self.up_x4(x)
        return x

# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    model = SwinTransformerSysEntropyAttention(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=2,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        depths_decoder=[1, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        final_upsample="expand_first"
    )
    inp = torch.randn(1, 3, 224, 224)
    out = model(inp)
    print("Output shape:", out.shape)
