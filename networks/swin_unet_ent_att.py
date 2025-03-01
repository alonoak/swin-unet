"""
swin_unet_final_entropy.py

Demonstration of a Swin-UNet architecture in Python, focusing on entropy-based
attention gating only in the FINAL decoder stage. This code is simplified for clarity
and may omit advanced features (e.g., shifted windows, relative position biases).

You can adapt or expand it for real-world usage.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# 1. Patch Embedding & Merging
###############################################################################

class PatchEmbed(nn.Module):
    """
    Splits the input image into patches via a stride=patch_size convolution,
    then normalizes channel embeddings.
    """

    def __init__(self, in_chans=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        b_, c_, h_, w_ = x.shape

        # Flatten for LayerNorm
        x_t = x.permute(0, 2, 3, 1).contiguous().view(b_, h_ * w_, c_)
        x_t = self.norm(x_t)

        # Reshape back
        x_t = x_t.view(b_, h_, w_, c_).permute(0, 3, 1, 2).contiguous()
        return x_t


class PatchMerging(nn.Module):
    """
    Downsamples the spatial resolution by 2x, merging each 2x2 block of patches
    and linearly projecting from 4*C -> 2*C channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        newH, newW = H // 2, W // 2

        # group 2x2
        x = x.view(B, C, newH, 2, newW, 2)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, newH, newW, 2, 2, C)
        x = x.view(B, newH * newW, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)  # -> (B, newH*newW, 2*C)

        x = x.view(B, newH, newW, -1).permute(0, 3, 1, 2).contiguous()
        return x


###############################################################################
# 2. Window-Based Attention & Swin Block (no shift)
###############################################################################

class WindowAttention(nn.Module):
    """
    Simple local multi-head self-attention for each window.
    Entropy gating is optionally integrated if 'entropy_map' is provided.
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, entropy_map=None, beta=1.0):
        """
        x: (B, N, C) => flattened tokens for a local window
        entropy_map: (B, 1, N) or None => gating factor. shape must match.
        """
        B, N, C = x.shape

        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)  # (B, heads, N, N)

        if entropy_map is not None:
            # shape => (B, 1, N)
            gating = torch.exp(-beta * entropy_map).unsqueeze(1)  # => (B,1,1,N)
            attn = attn * gating
            attn_sum = attn.sum(dim=-1, keepdim=True) + 1e-9
            attn = attn / attn_sum

        out = torch.matmul(attn, V)  # => (B, heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        return out


class SwinBlock(nn.Module):
    """
    Minimal Swin block without shifting, optionally supports entropy-based gating in WindowAttention.
    """

    def __init__(self, dim, window_size=7, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, entropy_map=None, beta=1.0):
        """
        x: (B, C, H, W)
        entropy_map: (B, 1, H, W) or None
        """
        B, C, H, W = x.shape

        # 1) LN
        x_t = x.permute(0, 2, 3, 1).contiguous()  # => (B, H, W, C)
        x_t = self.norm1(x_t)

        # 2) Partition into windows
        ws = self.window_size
        # shape => (B, H//ws, ws, W//ws, ws, C)
        x_t = x_t.view(B, H // ws, ws, W // ws, ws, C)
        x_t = x_t.permute(0, 1, 3, 2, 4, 5).contiguous()
        b_2 = x_t.shape[0]
        N = ws * ws

        x_t = x_t.view(b_2, N, C)  # => (B_local, N, C)

        # if we have an entropy map, partition similarly
        ent_t = None
        if entropy_map is not None:
            # => (B, 1, H, W)
            e_t = entropy_map.permute(0, 2, 3, 1).contiguous()  # => (B,H,W,1)
            e_t = e_t.view(B, H // ws, ws, W // ws, ws, 1)
            e_t = e_t.permute(0, 1, 3, 2, 4, 5).contiguous()
            e_t = e_t.view(b_2, N, 1)
            ent_t = e_t

        # 3) Window attention with gating
        attn_out = self.attn(x_t, ent_t, beta=beta)  # => (b_2, N, C)

        # 4) Reverse window partition
        attn_out = attn_out.view(B, H // ws, W // ws, ws, ws, C)
        attn_out = attn_out.permute(0, 1, 3, 2, 4, 5).contiguous()
        attn_out = attn_out.view(B, H, W, C)

        # Residual
        x_res = x.permute(0, 2, 3, 1) + attn_out
        x_res = x_res.permute(0, 3, 1, 2).contiguous()

        # 5) MLP
        x_t2 = x_res.permute(0, 2, 3, 1)
        x_t2 = self.norm2(x_t2)
        B_, H_, W_, C_ = x_t2.shape
        x_t2 = x_t2.view(B_, H_ * W_, C_)
        x_t2 = self.mlp(x_t2)
        x_t2 = x_t2.view(B_, H_, W_, C_)

        x_res2 = x_res.permute(0, 2, 3, 1) + x_t2
        x_out = x_res2.permute(0, 3, 1, 2).contiguous()
        return x_out


###############################################################################
# 3. Encoder & Decoder Stages
###############################################################################

class SwinEncoderStage(nn.Module):
    """
    One encoder stage: multiple SwinBlocks + optional patch merging.
    """

    def __init__(self, dim, depth, window_size=7, num_heads=4, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, window_size, num_heads) for _ in range(depth)
        ])
        self.downsample = downsample
        if downsample:
            self.merger = PatchMerging(dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (skip_feature, downsampled_feature)
        """
        for blk in self.blocks:
            x = blk(x)

        if self.downsample:
            x_down = self.merger(x)
            return x, x_down
        else:
            return x, None


class SwinDecoderStage(nn.Module):
    """
    A typical decoder stage: upsample -> concat skip -> unify -> pass through Swin blocks.
    (No entropy gating here.)
    """

    def __init__(self, in_dim, skip_dim, out_dim, depth=2, window_size=7, num_heads=4):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_dim, skip_dim, kernel_size=2, stride=2)
        self.unify = nn.Conv2d(skip_dim + skip_dim, out_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            SwinBlock(out_dim, window_size, num_heads) for _ in range(depth)
        ])

    def forward(self, x, skip):
        # x: lower-res from previous decoder step
        # skip: same resolution from encoder
        x_up = self.upsample(x)  # => (B, skip_dim, ?, ?)
        x_cat = torch.cat([x_up, skip], dim=1)  # => (B, skip_dim*2, ?, ?)
        x_cat = self.unify(x_cat)

        for blk in self.blocks:
            x_cat = blk(x_cat)  # no entropy gating
        return x_cat


###############################################################################
# 4. Final Decoder Stage WITH Entropy Gating
###############################################################################

def compute_entropy(logits):
    probs = F.softmax(logits, dim=1)
    ent = -torch.sum(probs * torch.log(probs + 1e-9), dim=1, keepdim=True)
    return ent


class FinalDecoderStageWithEntropy(nn.Module):
    """
    Replaces a normal decoder stage with an entropy-based gating step in the final Swin block.
    1) Upsample
    2) Concat skip => unify channels
    3) Produce auxiliary logits => compute entropy
    4) Reweight final Swin block with entropy_map
    """

    def __init__(self, in_dim, skip_dim, out_dim,
                 num_classes=2, window_size=7, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_dim, skip_dim, kernel_size=2, stride=2)
        self.unify = nn.Conv2d(skip_dim + skip_dim, out_dim, kernel_size=1)

        # Final SwinBlock
        self.swin_block = SwinBlock(out_dim, window_size, num_heads, mlp_ratio)

        # Aux head for entropy
        self.aux_head = nn.Conv2d(out_dim, num_classes, kernel_size=1)

    def forward(self, x, skip):
        x_up = self.upsample(x)
        x_cat = torch.cat([x_up, skip], dim=1)
        x_cat = self.unify(x_cat)  # => (B, out_dim, H, W)

        # produce logits => entropy
        logits = self.aux_head(x_cat)  # => (B, num_classes, H, W)
        ent_map = compute_entropy(logits)  # => (B, 1, H, W)

        # pass x_cat + ent_map into the final attention block
        x_out = self.swin_block(x_cat, entropy_map=ent_map)
        return x_out


###############################################################################
# 5. SwinUNet with Final Entropy Gating in the Last Decoder Stage
###############################################################################

class SwinUNetFinalEntropy(nn.Module):
    """
    Swin-UNet architecture:
    - PatchEmbed => multiple encoder stages
    - symmetrical decoder stages => final stage includes entropy gating
    """

    def __init__(self,
                 in_chans=3,
                 num_classes=2,
                 base_dim=64,
                 depths=[2, 2, 2, 2],
                 num_heads=[2, 4, 4, 8],
                 window_size=7):
        super().__init__()

        # 1) Patch embedding
        self.patch_embed = PatchEmbed(in_chans, base_dim, patch_size=4)

        # 2) Encoder stages
        self.encoder_stages = nn.ModuleList()
        current_dim = base_dim
        dims = []

        for i, d in enumerate(depths):
            stage = SwinEncoderStage(
                dim=current_dim,
                depth=d,
                window_size=window_size,
                num_heads=num_heads[i],
                downsample=(i < len(depths) - 1)
            )
            self.encoder_stages.append(stage)
            dims.append(current_dim)

            if i < len(depths) - 1:
                current_dim *= 2

        # after last stage, current_dim is final
        self.bottleneck_dim = current_dim

        # 3) Decoder stages (except final)
        self.decoder_stages = nn.ModuleList()
        # we have len(depths)-1 decode steps
        for i in range(len(depths) - 1, 1, -1):
            in_dim = dims[i]
            skip_dim = dims[i - 1]
            out_dim = dims[i - 1]
            dec = SwinDecoderStage(in_dim, skip_dim, out_dim,
                                   depth=2, window_size=window_size,
                                   num_heads=num_heads[i - 1])
            self.decoder_stages.append(dec)

        # 4) Final decoder stage WITH entropy gating
        # If we had 4 stages, i=1 => dims[1]= ...
        in_dim = dims[1]
        skip_dim = dims[0]
        out_dim = dims[0]
        self.final_entropy_stage = FinalDecoderStageWithEntropy(
            in_dim, skip_dim, out_dim,
            num_classes=num_classes,
            window_size=window_size,
            num_heads=num_heads[0],
            mlp_ratio=4.0
        )

        # 5) final seg head
        self.seg_head = nn.Conv2d(out_dim, num_classes, kernel_size=1)

    def forward(self, x):
        """
        x: (B, in_chans, H, W)
        """
        # 1) patch embed
        x0 = self.patch_embed(x)  # => (B, base_dim, H/4, W/4)

        # 2) encoder
        skips = []
        xi = x0
        for enc in self.encoder_stages:
            skip_feat, down_feat = enc(xi)
            skips.append(skip_feat)
            xi = down_feat if down_feat is not None else skip_feat

        # The last skip is the bottleneck
        # if there are 4 stages, we have skip indices: 0,1,2,3
        # skip[3] => bottleneck
        bottleneck = skips[-1]

        # 3) decode
        # We do len(depths)-2 normal stages, then the final stage
        dec_intermediate = bottleneck
        # If we had 4 total stages => we decode 2 times here
        # skip indices => skip[-2], skip[-3], ...
        for i, dec in enumerate(self.decoder_stages):
            skip_feat = skips[-2 - i]
            dec_intermediate = dec(dec_intermediate, skip_feat)

        # final stage uses skip index = 0
        skip_top = skips[0]
        x_final = self.final_entropy_stage(dec_intermediate, skip_top)

        # final seg
        out = self.seg_head(x_final)
        # optionally upsample to original size
        return out


###############################################################################
# 6. Example usage / test
###############################################################################

if __name__ == "__main__":
    model = SwinUNetFinalEntropy(
        in_chans=3,
        num_classes=2,
        base_dim=64,
        depths=[2, 2, 2, 2],
        num_heads=[2, 4, 4, 8],
        window_size=7
    )

    # Suppose input is 256x256
    inp = torch.randn(1, 3, 256, 256)
    out = model(inp)
    print("Output shape:", out.shape)
    # e.g. => (1, 2, 64, 64)
    # If you want the final segmentation in 256x256, you can do:
    # out_upsampled = F.interpolate(out, scale_factor=4, mode='bilinear')
    # print("Upsampled shape:", out_upsampled.shape)
