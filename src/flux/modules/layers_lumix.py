import math
import torch
from torch import nn
from einops import rearrange
from torch import Tensor

from ..math import attention


# === TensorLoRA: low-rank cross-modality adapter ===
class TensorLoRA(nn.Module):
    """
    Tensor LoRA adapter used by QBA.

    The contraction order is:
    x(j) x G2(i) -> G1(i,j,q) -> G3(i)

    This is memory efficient when d_in is much larger than sequence length,
    rank, and number of modalities.
    """
    def __init__(self, num_modalities, in_features, out_features,
                 rank=8, alpha=None, lora_weight=1.0,
                 use_gradient_checkpointing=False,
                 device=None, dtype=None):
        super().__init__()
        self.M = num_modalities
        self.d_in = in_features
        self.d_out = out_features
        self.R = rank
        self.alpha = (alpha if alpha is not None else 1.0) / math.sqrt(rank)
        self.lora_weight = lora_weight
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.G1 = nn.Parameter(torch.randn(self.M, self.M, self.R, device=device, dtype=dtype) * 0.02)
        self.G2 = nn.Parameter(torch.randn(self.M, self.d_in, self.R, self.R, device=device, dtype=dtype) / math.sqrt(self.d_in * self.R))
        self.G3 = nn.Parameter(torch.zeros(self.M, self.d_out, self.R, device=device, dtype=dtype))

    def _forward_impl(self, x):
        """
        x: [M, L, d_in]
        return: [M, L, d_out]
        """
        orig_dtype = x.dtype
        x = x.to(self.G1.dtype)

        # ---------- Step 1 ----------
        B = torch.einsum('jlu,iurq->ijlrq', x, self.G2)

        # ---------- Step 2 ----------
        H = torch.einsum('ijlrq,ijq->ijlr', B, self.G1)

        # ---------- Step 3 ----------
        Y_all = torch.einsum('ijlr,ior->ijlo', H, self.G3)

        # ---------- Step 4 ----------
        Y = Y_all.sum(dim=1)  # [i,L,d_out]

        # ---------- Step 5 ----------
        return (Y * self.alpha * self.lora_weight).to(orig_dtype)

    def forward(self, x):
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        else:
            return self._forward_impl(x)



class LumiXSingleStreamProcessor(nn.Module):
    """
    LumiX processor for SingleStreamBlock.
    
    Features:
    - Query Broadcast Attention from color modality (index=2) for stability
    - Each output modality processes inputs from all modalities via Tensor LoRA
    - Efficient Tensor LoRA decomposition reduces parameter count
    - Zero Python loops, fully vectorized operations
    """
    
    def __init__(self, dim: int, rank: int = 16, network_alpha=None, lora_weight: float = 1.0, 
                 num_modalities: int = 5, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.lora_weight = lora_weight
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.num_modalities = num_modalities
        
        # Modality order: [diffuse_reflectance, diffuse_illumination, color, depth, normal]
        self.modality_names = ['diffuse_reflectance', 'diffuse_illumination', 'color', 'depth', 'normal']
        
        # Tensor LoRA layers. Keep attribute names for checkpoint compatibility.
        self.kv_tt_lora = TensorLoRA(
            num_modalities, dim, dim * 2, 
            rank=rank, alpha=network_alpha, lora_weight=lora_weight,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        self.proj_tt_lora = TensorLoRA(
            num_modalities, dim, dim,
            rank=rank, alpha=network_alpha, lora_weight=lora_weight,
            use_gradient_checkpointing=use_gradient_checkpointing
        )

    def _forward_impl(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        """Forward pass with QBA and Tensor LoRA."""
        B = x.shape[0]  # M modalities
        L = x.shape[1]
        H = attn.num_heads
        D_head = self.dim // H
        
        # ==== Step 1: Modulation ====
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift  # [M, L, D]

        # ==== Step 2: Original QKV ====
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=H)
        q, k = attn.norm(q, k, v)

        # Query Broadcast Attention: broadcast the color query to every modality.
        q_shared = q[2:3].expand(B, -1, -1, -1)

        # ==== Step 3: Tensor LoRA K/V adaptation ====
        kv_delta = self.kv_tt_lora(x_mod)  # [M, L, 2D]
        kv_delta_reshaped = kv_delta.view(B, L, 2, H, D_head).permute(0, 2, 3, 1, 4)
        k_delta, v_delta = kv_delta_reshaped.unbind(1)

        k_modified = k + k_delta
        v_modified = v + v_delta

        # ==== Step 4: Attention ====
        attn_out = attention(q_shared, k_modified, v_modified, pe=pe)

        # ==== Step 5: Output projection ====
        mlp_input = torch.cat((attn_out, attn.mlp_act(mlp)), dim=-1)
        output = attn.linear2(mlp_input)

        # ==== Step 6: Tensor LoRA projection adaptation ====
        proj_delta = self.proj_tt_lora(output)  # [M, L, D]
        output_modified = output + proj_delta

        # ==== Step 7: Residual connection ====
        return x + mod.gate * output_modified

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, attn, x, vec, pe, use_reentrant=False
            )
        else:
            return self._forward_impl(attn, x, vec, pe)


class LumiXDoubleStreamProcessor(nn.Module):
    """
    LumiX processor for DoubleStreamBlock.
    
    Features:
    - Query Broadcast Attention from color modality for image branch
    - Text branch passes through unchanged  
    - Each output modality processes inputs from all modalities via Tensor LoRA
    - Efficient parameter usage through Tensor LoRA decomposition
    """
    
    def __init__(self, dim: int, rank: int = 16, network_alpha=None, lora_weight: float = 1.0,
                 num_modalities: int = 5, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.lora_weight = lora_weight
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.num_modalities = num_modalities
        
        # Modality order: [diffuse_reflectance, diffuse_illumination, color, depth, normal]
        self.modality_names = ['diffuse_reflectance', 'diffuse_illumination', 'color', 'depth', 'normal']
        
        # Tensor LoRA layers for image branch. Keep attribute names for checkpoint compatibility.
        self.kv_tt_lora = TensorLoRA(
            num_modalities, dim, dim * 2,
            rank=rank, alpha=network_alpha, lora_weight=lora_weight,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        self.proj_tt_lora = TensorLoRA(
            num_modalities, dim, dim,
            rank=rank, alpha=network_alpha, lora_weight=lora_weight,
            use_gradient_checkpointing=use_gradient_checkpointing
        )

    def _forward_impl(self, attn, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor):
        """Forward pass for DoubleStream block with QBA and Tensor LoRA."""
        B_img = img.shape[0]
        L_img = img.shape[1]
        H = attn.num_heads
        D_head = self.dim // H

        # ==== Modulation ====
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # ==== Image branch with Tensor LoRA ====
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=H)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # Apply Tensor LoRA K/V adaptation.
        kv_delta = self.kv_tt_lora(img_modulated)
        kv_delta = kv_delta.view(B_img, L_img, 2, H, D_head).permute(0, 2, 3, 1, 4)
        k_delta, v_delta = kv_delta.unbind(1)
        img_k_modified = img_k + k_delta
        img_v_modified = img_v + v_delta

        # Query Broadcast Attention: broadcast the color query to every modality.
        img_q_shared = img_q[2:3].expand(B_img, -1, -1, -1)

        # ==== Text branch (unchanged) ====
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=H)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # ==== Joint attention ====
        q = torch.cat((txt_q, img_q_shared), dim=2)
        k = torch.cat((txt_k, img_k_modified), dim=2)
        v = torch.cat((txt_v, img_v_modified), dim=2)

        attn_out = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn_out[:, : txt.shape[1]], attn_out[:, txt.shape[1] :]

        # ==== Image projection with Tensor LoRA ====
        img_attn_proj = attn.img_attn.proj(img_attn)
        proj_delta = self.proj_tt_lora(img_attn_proj)
        img_attn_proj_modified = img_attn_proj + proj_delta

        # ==== Text projection (unchanged) ====
        txt_attn_proj = attn.txt_attn.proj(txt_attn)

        # ==== Final residual blocks ====
        img = img + img_mod1.gate * img_attn_proj_modified
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * txt_attn_proj
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        return img, txt

    def forward(self, attn, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, **attention_kwargs):
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, attn, img, txt, vec, pe, use_reentrant=False
            )
        else:
            return self._forward_impl(attn, img, txt, vec, pe)
