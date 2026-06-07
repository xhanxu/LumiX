import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from ..math import attention, rope
import torch.nn.functional as F

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

class FLuxSelfAttnProcessor:
    def __call__(self, attn, x, pe, **attention_kwargs):
        print('2' * 30)

        qkv = attn.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x)
        return x

class LoraFluxAttnProcessor(nn.Module):

    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight


    def __call__(self, attn, x, pe, **attention_kwargs):
        qkv = attn.qkv(x) + self.qkv_lora(x) * self.lora_weight
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x) + self.proj_lora(x) * self.lora_weight
        print('1' * 30)
        print(x.norm(), (self.proj_lora(x) * self.lora_weight).norm(), 'norm')
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
    def forward():
        pass


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class DoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class DoubleStreamBlockImageOnlyLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        # self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        # txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        # txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class MultiModalDoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.albedo_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        self.albedo_qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # txt stream
        self.albedo_proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)    # txt stream
        
        # material LoRA (batch=1)
        self.material_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.material_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        self.material_qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # txt stream
        self.material_proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)    # txt stream
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.normal_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        self.normal_qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # txt stream
        self.normal_proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)    # txt stream
        
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        batch_size = img.shape[0]
        
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        
        img_qkv_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_qkv_lora1(img_modulated[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_qkv_lora1(img_modulated[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_qkv_lora1(img_modulated[i:i+1]) * self.lora_weight
            img_qkv_deltas.append(delta)
        
        img_qkv_deltas = torch.cat(img_qkv_deltas, dim=0)
        img_qkv = img_qkv + img_qkv_deltas
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        
        txt_qkv_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_qkv_lora2(txt_modulated[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_qkv_lora2(txt_modulated[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_qkv_lora2(txt_modulated[i:i+1]) * self.lora_weight
            txt_qkv_deltas.append(delta)
        
        txt_qkv_deltas = torch.cat(txt_qkv_deltas, dim=0)
        txt_qkv = txt_qkv + txt_qkv_deltas
        
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img blocks with modality-specific projection LoRA (proj_lora1)
        img_proj_output = attn.img_attn.proj(img_attn)
        
        img_proj_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_proj_lora1(img_attn[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_proj_lora1(img_attn[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_proj_lora1(img_attn[i:i+1]) * self.lora_weight
            img_proj_deltas.append(delta)
        
        img_proj_deltas = torch.cat(img_proj_deltas, dim=0)
        img_proj_output = img_proj_output + img_proj_deltas
        
        img = img + img_mod1.gate * img_proj_output
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt blocks with modality-specific projection LoRA (proj_lora2)
        txt_proj_output = attn.txt_attn.proj(txt_attn)
        
        txt_proj_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_proj_lora2(txt_attn[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_proj_lora2(txt_attn[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_proj_lora2(txt_attn[i:i+1]) * self.lora_weight
            txt_proj_deltas.append(delta)
        
        txt_proj_deltas = torch.cat(txt_proj_deltas, dim=0)
        txt_proj_output = txt_proj_output + txt_proj_deltas
        
        txt = txt + txt_mod1.gate * txt_proj_output
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        del img_qkv_deltas, txt_qkv_deltas, img_proj_deltas, txt_proj_deltas, delta
        
        return img, txt

class MultiModalDoubleStreamBlockImageOnlyLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1, use_pooling: bool = False, pooling_type: str = "mean"):
        super().__init__()
        
        # Pooling configuration
        self.use_pooling = use_pooling
        self.pooling_type = pooling_type  # "mean", "max", "attention"
        
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.albedo_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # material LoRA (batch=1)
        self.material_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.material_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.normal_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # Pooling-specific components (if enabled)
        if self.use_pooling:
            if self.pooling_type == "attention":
                self.pooling_attention = nn.Linear(dim, 1, bias=False)
        
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        batch_size = img.shape[0]
        
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        
        if self.use_pooling:
            if self.pooling_type == "mean":
                base_img_qkv = torch.mean(img_qkv, dim=0, keepdim=True)  # [1, L, D]
            elif self.pooling_type == "max":
                base_img_qkv, _ = torch.max(img_qkv, dim=0, keepdim=True)  # [1, L, D]
            elif self.pooling_type == "attention":
                attn_weights = self.pooling_attention(img_modulated)  # [B, L, 1]
                attn_weights = torch.softmax(attn_weights, dim=0)
                
                base_img_qkv = torch.sum(img_qkv * attn_weights, dim=0, keepdim=True)  # [1, L, D]
            
            base_img_qkv = base_img_qkv.expand(batch_size, -1, -1)  # [B, L, D]
        
        img_qkv_deltas = []
        if self.use_pooling:
            for i in range(batch_size):
                if i == 0:  # albedo
                    delta = self.albedo_qkv_lora1(base_img_qkv[i:i+1]) * self.lora_weight
                elif i == 1:  # material
                    delta = self.material_qkv_lora1(base_img_qkv[i:i+1]) * self.lora_weight
                else:  # normal (i == 2)
                    delta = self.normal_qkv_lora1(base_img_qkv[i:i+1]) * self.lora_weight
                img_qkv_deltas.append(delta)
        else:
            for i in range(batch_size):
                if i == 0:  # albedo
                    delta = self.albedo_qkv_lora1(img_modulated[i:i+1]) * self.lora_weight
                elif i == 1:  # material
                    delta = self.material_qkv_lora1(img_modulated[i:i+1]) * self.lora_weight
                else:  # normal (i == 2)
                    delta = self.normal_qkv_lora1(img_modulated[i:i+1]) * self.lora_weight
                img_qkv_deltas.append(delta)
        
        img_qkv_deltas = torch.cat(img_qkv_deltas, dim=0)
        if self.use_pooling:
            img_qkv = base_img_qkv + img_qkv_deltas
        else:
            img_qkv = img_qkv + img_qkv_deltas
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img blocks with modality-specific projection LoRA (proj_lora1)
        img_proj_output = attn.img_attn.proj(img_attn)
        
        img_proj_deltas = []
        
        if self.use_pooling:
            if self.pooling_type == "mean":
                base_img_proj_output = torch.mean(img_proj_output, dim=0, keepdim=True)  # [1, L, D]
            elif self.pooling_type == "max":
                base_img_proj_output, _ = torch.max(img_proj_output, dim=0, keepdim=True)  # [1, L, D]
            elif self.pooling_type == "attention":
                attn_weights = self.pooling_attention(img_modulated)  # [B, L, 1] 
                attn_weights = torch.softmax(attn_weights, dim=0)
                base_img_proj_output = torch.sum(img_proj_output * attn_weights, dim=0, keepdim=True)  # [1, L, D]
            
            base_img_proj_output = base_img_proj_output.expand(batch_size, -1, -1)  # [B, L, D]
            
            for i in range(batch_size):
                if i == 0:  # albedo
                    delta = self.albedo_proj_lora1(base_img_proj_output[i:i+1]) * self.lora_weight
                elif i == 1:  # material
                    delta = self.material_proj_lora1(base_img_proj_output[i:i+1]) * self.lora_weight
                else:  # normal (i == 2)
                    delta = self.normal_proj_lora1(base_img_proj_output[i:i+1]) * self.lora_weight
                img_proj_deltas.append(delta)
            
            img_proj_deltas = torch.cat(img_proj_deltas, dim=0)
            img_proj_output = base_img_proj_output + img_proj_deltas
        else:
            for i in range(batch_size):
                if i == 0:  # albedo
                    delta = self.albedo_proj_lora1(img_attn[i:i+1]) * self.lora_weight
                elif i == 1:  # material
                    delta = self.material_proj_lora1(img_attn[i:i+1]) * self.lora_weight
                else:  # normal (i == 2)
                    delta = self.normal_proj_lora1(img_attn[i:i+1]) * self.lora_weight
                img_proj_deltas.append(delta)
            
            img_proj_deltas = torch.cat(img_proj_deltas, dim=0)
            img_proj_output = img_proj_output + img_proj_deltas
        
        img = img + img_mod1.gate * img_proj_output
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        del img_qkv_deltas, img_proj_deltas, delta
        
        return img, txt

class MultiModalDoubleStreamBlockImageOnlySharedKVLoraProcessor(nn.Module):
    """
    Multi-modal DoubleStreamBlock processor with SharedKV mechanism, Image-Only LoRA.
    Each modality uses its own Q but shares K,V computed from all modalities.
    Based on MultiModalDoubleStreamBlockImageOnlyLoraProcessor but with SharedKV mechanism.
    """
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.albedo_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # material LoRA (batch=1)
        self.material_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.material_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.normal_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        batch_size = img.shape[0]
        
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        
        img_qkv_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_qkv_lora1(img_modulated[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_qkv_lora1(img_modulated[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_qkv_lora1(img_modulated[i:i+1]) * self.lora_weight
            img_qkv_deltas.append(delta)
        
        img_qkv_deltas = torch.cat(img_qkv_deltas, dim=0)
        img_qkv = img_qkv + img_qkv_deltas
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        
        # txt: [B, H, L_txt, D], img: [B, H, L_img, D] -> combined: [B, H, L_txt+L_img, D]
        combined_k = torch.cat((txt_k, img_k), dim=2)  # [B, H, L_txt+L_img, D]
        combined_v = torch.cat((txt_v, img_v), dim=2)  # [B, H, L_txt+L_img, D]
        
        shared_k = rearrange(combined_k, "B H L D -> 1 H (B L) D")  # [1, H, B*(L_txt+L_img), D]
        shared_v = rearrange(combined_v, "B H L D -> 1 H (B L) D")  # [1, H, B*(L_txt+L_img), D]
        
        txt_attn_results = []
        img_attn_results = []
        
        for i in range(batch_size):
            modal_txt_q = txt_q[i:i+1]  # [1, H, L_txt, D]
            modal_img_q = img_q[i:i+1]  # [1, H, L_img, D]
            
            modal_q = torch.cat((modal_txt_q, modal_img_q), dim=2)  # [1, H, L_txt+L_img, D]
            
            modal_attn = attention(modal_q, shared_k, shared_v, pe=pe[i:i+1])  # [1, L_txt+L_img, H*D]
            
            L_txt = txt.shape[1]
            modal_txt_attn = modal_attn[:, :L_txt]  # [1, L_txt, H*D]
            modal_img_attn = modal_attn[:, L_txt:]  # [1, L_img, H*D]
            
            txt_attn_results.append(modal_txt_attn)
            img_attn_results.append(modal_img_attn)
        
        txt_attn = torch.cat(txt_attn_results, dim=0)  # [B, L_txt, H*D]
        img_attn = torch.cat(img_attn_results, dim=0)  # [B, L_img, H*D]

        # calculate the img blocks with modality-specific projection LoRA (proj_lora1)
        img_proj_output = attn.img_attn.proj(img_attn)
        
        img_proj_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_proj_lora1(img_attn[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_proj_lora1(img_attn[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_proj_lora1(img_attn[i:i+1]) * self.lora_weight
            img_proj_deltas.append(delta)
        
        img_proj_deltas = torch.cat(img_proj_deltas, dim=0)
        img_proj_output = img_proj_output + img_proj_deltas
        
        img = img + img_mod1.gate * img_proj_output
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        del (img_qkv_deltas, img_proj_deltas, delta, combined_k, combined_v, 
             shared_k, shared_v, txt_attn_results, img_attn_results, modal_attn)
        
        return img, txt

class MultiModalDoubleStreamBlockImageOnlyAlbedoAlignedKVLoraProcessor(nn.Module):
    """
    Multi-modal DoubleStreamBlock processor with albedo-aligned K,V mechanism, Image-Only LoRA.
    - albedo: uses its own Q, K, V
    - material/normal: use their own Q, but K,V are concat of [albedo_K,V + own_K,V]
    Based on MultiModalDoubleStreamBlockImageOnlyLoraProcessor but with albedo-aligned KV mechanism.
    """
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1, 
                 use_alignment_lora: bool = False, align_rank: int = 4):
        super().__init__()
        
        self.use_alignment_lora = use_alignment_lora
        
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.albedo_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # material LoRA (batch=1)
        self.material_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.material_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.normal_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        if use_alignment_lora:
            self.material_align_qkv_lora1 = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.material_align_proj_lora1 = LoRALinearLayer(dim, dim, align_rank, network_alpha)
            self.normal_align_qkv_lora1 = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.normal_align_proj_lora1 = LoRALinearLayer(dim, dim, align_rank, network_alpha)
        
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        batch_size = img.shape[0]
        assert batch_size == 3, f"Expected batch_size=3 for albedo/material/normal, got {batch_size}"
        
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        
        albedo_img_mod, material_img_mod, normal_img_mod = img_modulated[0:1], img_modulated[1:2], img_modulated[2:3]
        
        albedo_qkv_delta = self.albedo_qkv_lora1(albedo_img_mod) * self.lora_weight
        material_qkv_delta = self.material_qkv_lora1(material_img_mod) * self.lora_weight
        normal_qkv_delta = self.normal_qkv_lora1(normal_img_mod) * self.lora_weight
        
        if self.use_alignment_lora:
            material_align_delta = self.material_align_qkv_lora1(material_img_mod) * self.lora_weight
            normal_align_delta = self.normal_align_qkv_lora1(normal_img_mod) * self.lora_weight
            material_qkv_delta = material_qkv_delta + material_align_delta
            normal_qkv_delta = normal_qkv_delta + normal_align_delta
        
        img_qkv_deltas = torch.cat([albedo_qkv_delta, material_qkv_delta, normal_qkv_delta], dim=0)
        img_qkv = img_qkv + img_qkv_deltas
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        
        # txt: [B, H, L_txt, D], img: [B, H, L_img, D] -> combined: [B, H, L_txt+L_img, D]
        combined_k = torch.cat((txt_k, img_k), dim=2)  # [B, H, L_txt+L_img, D]
        combined_v = torch.cat((txt_v, img_v), dim=2)  # [B, H, L_txt+L_img, D]
        
        albedo_k = combined_k[0:1]
        albedo_v = combined_v[0:1]
        
        L_txt = txt.shape[1]
        
        material_k = torch.cat([albedo_k, combined_k[1:2]], dim=2)  # [1, H, 2*(L_txt+L_img), D]
        material_v = torch.cat([albedo_v, combined_v[1:2]], dim=2)  # [1, H, 2*(L_txt+L_img), D]
        normal_k = torch.cat([albedo_k, combined_k[2:3]], dim=2)    # [1, H, 2*(L_txt+L_img), D]
        normal_v = torch.cat([albedo_v, combined_v[2:3]], dim=2)    # [1, H, 2*(L_txt+L_img), D]
        
        albedo_q = torch.cat((txt_q[0:1], img_q[0:1]), dim=2)
        material_q = torch.cat((txt_q[1:2], img_q[1:2]), dim=2)
        normal_q = torch.cat((txt_q[2:3], img_q[2:3]), dim=2)
        
        albedo_attn = attention(albedo_q, combined_k[0:1], combined_v[0:1], pe=pe[0:1])
        material_attn = attention(material_q, material_k, material_v, pe=pe[1:2])
        normal_attn = attention(normal_q, normal_k, normal_v, pe=pe[2:3])
        
        combined_attn = torch.cat([albedo_attn, material_attn, normal_attn], dim=0)
        txt_attn = combined_attn[:, :L_txt]
        img_attn = combined_attn[:, L_txt:]

        # calculate the img blocks with modality-specific projection LoRA (proj_lora1)
        img_proj_output = attn.img_attn.proj(img_attn)
        
        albedo_img_attn, material_img_attn, normal_img_attn = img_attn[0:1], img_attn[1:2], img_attn[2:3]
        
        albedo_proj_delta = self.albedo_proj_lora1(albedo_img_attn) * self.lora_weight
        material_proj_delta = self.material_proj_lora1(material_img_attn) * self.lora_weight
        normal_proj_delta = self.normal_proj_lora1(normal_img_attn) * self.lora_weight
        
        if self.use_alignment_lora:
            material_align_proj_delta = self.material_align_proj_lora1(material_img_attn) * self.lora_weight
            normal_align_proj_delta = self.normal_align_proj_lora1(normal_img_attn) * self.lora_weight
            material_proj_delta = material_proj_delta + material_align_proj_delta
            normal_proj_delta = normal_proj_delta + normal_align_proj_delta
        
        img_proj_deltas = torch.cat([albedo_proj_delta, material_proj_delta, normal_proj_delta], dim=0)
        img_proj_output = img_proj_output + img_proj_deltas
        
        img = img + img_mod1.gate * img_proj_output
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        return img, txt

class MultiModalDoubleStreamBlockImageOnlyNormalAlignedKVLoraProcessor(nn.Module):
    """
    Multi-modal DoubleStreamBlock processor with normal-aligned K,V mechanism, Image-Only LoRA.
    - normal: uses its own Q, K, V (geometry drives everything)
    - albedo/material: use their own Q, but K,V are concat of [normal_K,V + own_K,V]
    Based on MultiModalDoubleStreamBlockImageOnlyLoraProcessor but with normal-aligned KV mechanism.
    """
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1, 
                 use_alignment_lora: bool = False, align_rank: int = 4):
        super().__init__()
        
        self.use_alignment_lora = use_alignment_lora
        
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.albedo_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # material LoRA (batch=1)
        self.material_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.material_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.normal_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        if use_alignment_lora:
            self.albedo_align_qkv_lora1 = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.albedo_align_proj_lora1 = LoRALinearLayer(dim, dim, align_rank, network_alpha)
            self.material_align_qkv_lora1 = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.material_align_proj_lora1 = LoRALinearLayer(dim, dim, align_rank, network_alpha)
        
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        batch_size = img.shape[0]
        assert batch_size == 3, f"Expected batch_size=3 for albedo/material/normal, got {batch_size}"
        
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        
        albedo_img_mod, material_img_mod, normal_img_mod = img_modulated[0:1], img_modulated[1:2], img_modulated[2:3]
        
        albedo_qkv_delta = self.albedo_qkv_lora1(albedo_img_mod) * self.lora_weight
        material_qkv_delta = self.material_qkv_lora1(material_img_mod) * self.lora_weight
        normal_qkv_delta = self.normal_qkv_lora1(normal_img_mod) * self.lora_weight
        
        if self.use_alignment_lora:
            material_align_delta = self.material_align_qkv_lora1(material_img_mod) * self.lora_weight
            normal_align_delta = self.normal_align_qkv_lora1(normal_img_mod) * self.lora_weight
            material_qkv_delta = material_qkv_delta + material_align_delta
            normal_qkv_delta = normal_qkv_delta + normal_align_delta
        
        img_qkv_deltas = torch.cat([albedo_qkv_delta, material_qkv_delta, normal_qkv_delta], dim=0)
        img_qkv = img_qkv + img_qkv_deltas
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        
        # txt: [B, H, L_txt, D], img: [B, H, L_img, D] -> combined: [B, H, L_txt+L_img, D]
        combined_k = torch.cat((txt_k, img_k), dim=2)  # [B, H, L_txt+L_img, D]
        combined_v = torch.cat((txt_v, img_v), dim=2)  # [B, H, L_txt+L_img, D]
        
        normal_k = combined_k[2:3]
        normal_v = combined_v[2:3]

        L_txt = txt.shape[1]
        
        albedo_k = torch.cat([normal_k, combined_k[0:1]], dim=2)    # [1, H, 2*(L_txt+L_img), D]
        albedo_v = torch.cat([normal_v, combined_v[0:1]], dim=2)    # [1, H, 2*(L_txt+L_img), D]
        material_k = torch.cat([normal_k, combined_k[1:2]], dim=2)  # [1, H, 2*(L_txt+L_img), D]
        material_v = torch.cat([normal_v, combined_v[1:2]], dim=2)  # [1, H, 2*(L_txt+L_img), D]
        
        
        albedo_q = torch.cat((txt_q[0:1], img_q[0:1]), dim=2)
        material_q = torch.cat((txt_q[1:2], img_q[1:2]), dim=2)
        normal_q = torch.cat((txt_q[2:3], img_q[2:3]), dim=2)
        
        albedo_attn = attention(albedo_q, albedo_k, albedo_v, pe=pe[0:1])
        material_attn = attention(material_q, material_k, material_v, pe=pe[1:2])
        normal_attn = attention(normal_q, normal_k, normal_v, pe=pe[2:3])
        
        combined_attn = torch.cat([albedo_attn, material_attn, normal_attn], dim=0)
        txt_attn = combined_attn[:, :L_txt]
        img_attn = combined_attn[:, L_txt:]

        # calculate the img blocks with modality-specific projection LoRA (proj_lora1)
        img_proj_output = attn.img_attn.proj(img_attn)
        
        albedo_img_attn, material_img_attn, normal_img_attn = img_attn[0:1], img_attn[1:2], img_attn[2:3]
        
        albedo_proj_delta = self.albedo_proj_lora1(albedo_img_attn) * self.lora_weight
        material_proj_delta = self.material_proj_lora1(material_img_attn) * self.lora_weight
        normal_proj_delta = self.normal_proj_lora1(normal_img_attn) * self.lora_weight
        
        if self.use_alignment_lora:
            albedo_align_proj_delta = self.albedo_align_proj_lora1(albedo_img_attn) * self.lora_weight
            material_align_proj_delta = self.material_align_proj_lora1(material_img_attn) * self.lora_weight
            albedo_proj_delta = albedo_proj_delta + albedo_align_proj_delta
            material_proj_delta = material_proj_delta + material_align_proj_delta
        
        img_proj_deltas = torch.cat([albedo_proj_delta, material_proj_delta, normal_proj_delta], dim=0)
        img_proj_output = img_proj_output + img_proj_deltas
        
        img = img + img_mod1.gate * img_proj_output
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        return img, txt

class MultiModalDoubleStreamBlockImageOnlyCausalSharedKVLoraProcessor(nn.Module):
    """
    Multi-modal DoubleStreamBlock processor with albedo-aligned K,V mechanism, Image-Only LoRA.
    - albedo: uses its own Q, K, V
    - material/normal: use their own Q, but K,V are concat of [albedo_K,V + own_K,V]
    Based on MultiModalDoubleStreamBlockImageOnlyLoraProcessor but with albedo-aligned KV mechanism.
    """
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1, 
                 use_alignment_lora: bool = False, align_rank: int = 4):
        super().__init__()
        
        self.use_alignment_lora = use_alignment_lora
        
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.albedo_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # material LoRA (batch=1)
        self.material_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.material_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.normal_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        if use_alignment_lora:
            self.material_align_qkv_lora1 = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.material_align_proj_lora1 = LoRALinearLayer(dim, dim, align_rank, network_alpha)
            self.normal_align_qkv_lora1 = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.normal_align_proj_lora1 = LoRALinearLayer(dim, dim, align_rank, network_alpha)
        
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        batch_size = img.shape[0]
        assert batch_size == 3, f"Expected batch_size=3 for albedo/material/normal, got {batch_size}"
        
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        
        albedo_img_mod, material_img_mod, normal_img_mod = img_modulated[0:1], img_modulated[1:2], img_modulated[2:3]
        
        albedo_qkv_delta = self.albedo_qkv_lora1(albedo_img_mod) * self.lora_weight
        material_qkv_delta = self.material_qkv_lora1(material_img_mod) * self.lora_weight
        normal_qkv_delta = self.normal_qkv_lora1(normal_img_mod) * self.lora_weight
        
        if self.use_alignment_lora:
            material_align_delta = self.material_align_qkv_lora1(material_img_mod) * self.lora_weight
            normal_align_delta = self.normal_align_qkv_lora1(normal_img_mod) * self.lora_weight
            material_qkv_delta = material_qkv_delta + material_align_delta
            normal_qkv_delta = normal_qkv_delta + normal_align_delta
        
        img_qkv_deltas = torch.cat([albedo_qkv_delta, material_qkv_delta, normal_qkv_delta], dim=0)
        img_qkv = img_qkv + img_qkv_deltas
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        
        # txt: [B, H, L_txt, D], img: [B, H, L_img, D] -> combined: [B, H, L_txt+L_img, D]
        combined_k = torch.cat((txt_k, img_k), dim=2)  # [B, H, L_txt+L_img, D]
        combined_v = torch.cat((txt_v, img_v), dim=2)  # [B, H, L_txt+L_img, D]
        
        albedo_k = combined_k[0:1]
        albedo_v = combined_v[0:1]
        
        L_txt = txt.shape[1]
        
        normal_k = torch.cat([albedo_k, combined_k[2:3]], dim=2)    # [1, H, 2*(L_txt+L_img), D]
        normal_v = torch.cat([albedo_v, combined_v[2:3]], dim=2)    # [1, H, 2*(L_txt+L_img), D]

        material_k = torch.cat([normal_k, combined_k[1:2]], dim=2)  # [1, H, 3*(L_txt+L_img), D]
        material_v = torch.cat([normal_v, combined_v[1:2]], dim=2)  # [1, H, 3*(L_txt+L_img), D]

        
        albedo_q = torch.cat((txt_q[0:1], img_q[0:1]), dim=2)
        material_q = torch.cat((txt_q[1:2], img_q[1:2]), dim=2)
        normal_q = torch.cat((txt_q[2:3], img_q[2:3]), dim=2)
        
        albedo_attn = attention(albedo_q, combined_k[0:1], combined_v[0:1], pe=pe[0:1])
        material_attn = attention(material_q, material_k, material_v, pe=pe[1:2])
        normal_attn = attention(normal_q, normal_k, normal_v, pe=pe[2:3])
        
        combined_attn = torch.cat([albedo_attn, material_attn, normal_attn], dim=0)
        txt_attn = combined_attn[:, :L_txt]
        img_attn = combined_attn[:, L_txt:]

        # calculate the img blocks with modality-specific projection LoRA (proj_lora1)
        img_proj_output = attn.img_attn.proj(img_attn)
        
        albedo_img_attn, material_img_attn, normal_img_attn = img_attn[0:1], img_attn[1:2], img_attn[2:3]
        
        albedo_proj_delta = self.albedo_proj_lora1(albedo_img_attn) * self.lora_weight
        material_proj_delta = self.material_proj_lora1(material_img_attn) * self.lora_weight
        normal_proj_delta = self.normal_proj_lora1(normal_img_attn) * self.lora_weight
        
        if self.use_alignment_lora:
            material_align_proj_delta = self.material_align_proj_lora1(material_img_attn) * self.lora_weight
            normal_align_proj_delta = self.normal_align_proj_lora1(normal_img_attn) * self.lora_weight
            material_proj_delta = material_proj_delta + material_align_proj_delta
            normal_proj_delta = normal_proj_delta + normal_align_proj_delta
        
        img_proj_deltas = torch.cat([albedo_proj_delta, material_proj_delta, normal_proj_delta], dim=0)
        img_proj_output = img_proj_output + img_proj_deltas
        
        img = img + img_mod1.gate * img_proj_output
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        return img, txt

class MultiModalDoubleStreamBlockImageOnlyCausalSharedKV_N_A_M_LoraProcessor(nn.Module):
    """
    Multi-modal DoubleStreamBlock processor with albedo-aligned K,V mechanism, Image-Only LoRA.
    - normal -> albedo -> material (geometry drives everything)
    Based on MultiModalDoubleStreamBlockImageOnlyLoraProcessor but with albedo-aligned KV mechanism.
    """
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1, 
                 use_alignment_lora: bool = False, align_rank: int = 4):
        super().__init__()
        
        self.use_alignment_lora = use_alignment_lora
        
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.albedo_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # material LoRA (batch=1)
        self.material_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.material_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.normal_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        
        if use_alignment_lora:
            self.albedo_align_qkv_lora1 = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.albedo_align_proj_lora1 = LoRALinearLayer(dim, dim, align_rank, network_alpha)
            self.material_align_qkv_lora1 = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.material_align_proj_lora1 = LoRALinearLayer(dim, dim, align_rank, network_alpha)
            
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        batch_size = img.shape[0]
        assert batch_size == 3, f"Expected batch_size=3 for albedo/material/normal, got {batch_size}"
        
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        
        albedo_img_mod, material_img_mod, normal_img_mod = img_modulated[0:1], img_modulated[1:2], img_modulated[2:3]
        
        albedo_qkv_delta = self.albedo_qkv_lora1(albedo_img_mod) * self.lora_weight
        material_qkv_delta = self.material_qkv_lora1(material_img_mod) * self.lora_weight
        normal_qkv_delta = self.normal_qkv_lora1(normal_img_mod) * self.lora_weight
        
        if self.use_alignment_lora:
            albedo_align_delta = self.albedo_align_qkv_lora1(albedo_img_mod) * self.lora_weight
            material_align_delta = self.material_align_qkv_lora1(material_img_mod) * self.lora_weight
            
            albedo_qkv_delta = albedo_qkv_delta + albedo_align_delta
            material_qkv_delta = material_qkv_delta + material_align_delta
            

        img_qkv_deltas = torch.cat([albedo_qkv_delta, material_qkv_delta, normal_qkv_delta], dim=0)
        img_qkv = img_qkv + img_qkv_deltas
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        
        # txt: [B, H, L_txt, D], img: [B, H, L_img, D] -> combined: [B, H, L_txt+L_img, D]
        combined_k = torch.cat((txt_k, img_k), dim=2)  # [B, H, L_txt+L_img, D]
        combined_v = torch.cat((txt_v, img_v), dim=2)  # [B, H, L_txt+L_img, D]
        
        normal_k = combined_k[2:3]
        normal_v = combined_v[2:3]

        L_txt = txt.shape[1]
        
        albedo_k = torch.cat([normal_k, combined_k[0:1]], dim=2)    # [1, H, 2*(L_txt+L_img), D]
        albedo_v = torch.cat([normal_v, combined_v[0:1]], dim=2)    # [1, H, 2*(L_txt+L_img), D]

        material_k = torch.cat([albedo_k, combined_k[1:2]], dim=2)  # [1, H, 3*(L_txt+L_img), D]
        material_v = torch.cat([albedo_v, combined_v[1:2]], dim=2)  # [1, H, 3*(L_txt+L_img), D]

        
        albedo_q = torch.cat((txt_q[0:1], img_q[0:1]), dim=2)
        material_q = torch.cat((txt_q[1:2], img_q[1:2]), dim=2)
        normal_q = torch.cat((txt_q[2:3], img_q[2:3]), dim=2)
        
        albedo_attn = attention(albedo_q, albedo_k, albedo_v, pe=pe[0:1])
        material_attn = attention(material_q, material_k, material_v, pe=pe[1:2])
        normal_attn = attention(normal_q, normal_k, normal_v, pe=pe[2:3])
        
        combined_attn = torch.cat([albedo_attn, material_attn, normal_attn], dim=0)
        txt_attn = combined_attn[:, :L_txt]
        img_attn = combined_attn[:, L_txt:]

        # calculate the img blocks with modality-specific projection LoRA (proj_lora1)
        img_proj_output = attn.img_attn.proj(img_attn)
        
        albedo_img_attn, material_img_attn, normal_img_attn = img_attn[0:1], img_attn[1:2], img_attn[2:3]
        
        albedo_proj_delta = self.albedo_proj_lora1(albedo_img_attn) * self.lora_weight
        material_proj_delta = self.material_proj_lora1(material_img_attn) * self.lora_weight
        normal_proj_delta = self.normal_proj_lora1(normal_img_attn) * self.lora_weight
        
        if self.use_alignment_lora:
            material_align_proj_delta = self.material_align_proj_lora1(material_img_attn) * self.lora_weight
            albedo_align_proj_delta = self.albedo_proj_lora1_align_proj_lora1(normal_img_attn) * self.lora_weight
            material_proj_delta = material_proj_delta + material_align_proj_delta
            albedo_proj_delta = albedo_proj_delta + albedo_align_proj_delta

        img_proj_deltas = torch.cat([albedo_proj_delta, material_proj_delta, normal_proj_delta], dim=0)
        img_proj_output = img_proj_output + img_proj_deltas
        
        img = img + img_mod1.gate * img_proj_output
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        return img, txt

class IPDoubleStreamBlockProcessor(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

    def __call__(self, attn, img, txt, vec, pe, image_proj, ip_scale=1.0, **attention_kwargs):

        # Prepare image for attention
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)


        # IP-adapter processing
        ip_query = img_q  # latent sample query
        ip_key = self.ip_adapter_double_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_double_stream_v_proj(image_proj)

        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value,
            dropout_p=0.0,
            is_causal=False
        )
        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)

        img = img + ip_scale * ip_attention

        return img, txt


class MultiModalSharedKVDoubleStreamBlockLoraProcessor(nn.Module):
    """
    Multi-modal LoRA processor with shared K,V mechanism for DoubleStreamBlock.
    Each modality uses its own Q but shares K,V computed from all modalities.
    """
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.albedo_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        self.albedo_qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # txt stream
        self.albedo_proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)    # txt stream
        
        # material LoRA (batch=1)
        self.material_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.material_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        self.material_qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # txt stream
        self.material_proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)    # txt stream
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # img stream
        self.normal_proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)    # img stream
        self.normal_qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)  # txt stream
        self.normal_proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)    # txt stream
        
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        batch_size = img.shape[0]
        
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        
        img_qkv_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_qkv_lora1(img_modulated[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_qkv_lora1(img_modulated[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_qkv_lora1(img_modulated[i:i+1]) * self.lora_weight
            img_qkv_deltas.append(delta)
        
        img_qkv_deltas = torch.cat(img_qkv_deltas, dim=0)
        img_qkv = img_qkv + img_qkv_deltas
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        
        txt_qkv_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_qkv_lora2(txt_modulated[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_qkv_lora2(txt_modulated[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_qkv_lora2(txt_modulated[i:i+1]) * self.lora_weight
            txt_qkv_deltas.append(delta)
        
        txt_qkv_deltas = torch.cat(txt_qkv_deltas, dim=0)
        txt_qkv = txt_qkv + txt_qkv_deltas
        
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        
        # txt: [B, H, L_txt, D], img: [B, H, L_img, D] -> combined: [B, H, L_txt+L_img, D]
        combined_k = torch.cat((txt_k, img_k), dim=2)  # [B, H, L_txt+L_img, D]
        combined_v = torch.cat((txt_v, img_v), dim=2)  # [B, H, L_txt+L_img, D]
        
        shared_k = rearrange(combined_k, "B H L D -> 1 H (B L) D")  # [1, H, B*(L_txt+L_img), D]
        shared_v = rearrange(combined_v, "B H L D -> 1 H (B L) D")  # [1, H, B*(L_txt+L_img), D]
        
        txt_attn_results = []
        img_attn_results = []
        
        for i in range(batch_size):
            modal_txt_q = txt_q[i:i+1]  # [1, H, L_txt, D]
            modal_img_q = img_q[i:i+1]  # [1, H, L_img, D]
            
            modal_q = torch.cat((modal_txt_q, modal_img_q), dim=2)  # [1, H, L_txt+L_img, D]
            
            modal_attn = attention(modal_q, shared_k, shared_v, pe=pe[i:i+1])  # [1, L_txt+L_img, H*D]
            
            L_txt = txt.shape[1]
            modal_txt_attn = modal_attn[:, :L_txt]  # [1, L_txt, H*D]
            modal_img_attn = modal_attn[:, L_txt:]  # [1, L_img, H*D]
            
            txt_attn_results.append(modal_txt_attn)
            img_attn_results.append(modal_img_attn)
        
        txt_attn = torch.cat(txt_attn_results, dim=0)  # [B, L_txt, H*D]
        img_attn = torch.cat(img_attn_results, dim=0)  # [B, L_img, H*D]

        # calculate the img blocks with modality-specific projection LoRA (proj_lora1)
        img_proj_output = attn.img_attn.proj(img_attn)
        
        img_proj_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_proj_lora1(img_attn[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_proj_lora1(img_attn[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_proj_lora1(img_attn[i:i+1]) * self.lora_weight
            img_proj_deltas.append(delta)
        
        img_proj_deltas = torch.cat(img_proj_deltas, dim=0)
        img_proj_output = img_proj_output + img_proj_deltas
        
        img = img + img_mod1.gate * img_proj_output
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt blocks with modality-specific projection LoRA (proj_lora2)
        txt_proj_output = attn.txt_attn.proj(txt_attn)
        
        txt_proj_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_proj_lora2(txt_attn[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_proj_lora2(txt_attn[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_proj_lora2(txt_attn[i:i+1]) * self.lora_weight
            txt_proj_deltas.append(delta)
        
        txt_proj_deltas = torch.cat(txt_proj_deltas, dim=0)
        txt_proj_output = txt_proj_output + txt_proj_deltas
        
        txt = txt + txt_mod1.gate * txt_proj_output
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        
        del (img_qkv_deltas, txt_qkv_deltas, img_proj_deltas, txt_proj_deltas, 
             txt_attn_results, img_attn_results, modal_attn, delta)
        
        return img, txt

class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        processor = DoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor = None,
        ip_scale: float =1.0,
    ) -> tuple[Tensor, Tensor]:
        if image_proj is None:
            return self.processor(self, img, txt, vec, pe)
        else:
            return self.processor(self, img, txt, vec, pe, image_proj, ip_scale)

class IPSingleStreamBlockProcessor(nn.Module):
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPSingleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)

    def __call__(
        self,
        attn: nn.Module,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0
    ) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # IP-adapter processing
        ip_query = q
        ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_single_stream_v_proj(image_proj)

        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)


        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value
        )
        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")

        attn_out = attn_1 + ip_scale * ip_attention

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_out, attn.mlp_act(mlp)), 2))
        out = x + mod.gate * output

        return out


class SingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = output + self.proj_lora(torch.cat((attn_1, attn.mlp_act(mlp)), 2)) * self.lora_weight
        output = x + mod.gate * output
        return output
    
class SimpleSingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = output + self.proj_lora(output) * self.lora_weight
        output = x + mod.gate * output
        return output

class SimpleMultiModalSingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1, use_pooling: bool = False, pooling_type: str = "mean"):
        super().__init__()
        
        # Pooling configuration
        self.use_pooling = use_pooling
        self.pooling_type = pooling_type  # "mean", "max", "attention"
        
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.albedo_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # material LoRA (batch=1)  
        self.material_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.material_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.normal_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # Pooling-specific components (if enabled)
        if self.use_pooling:
            if self.pooling_type == "attention":
                self.pooling_attention = nn.Linear(dim, 1, bias=False)
        
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        batch_size = x.shape[0] # 3 1536 3072
        
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
        if self.use_pooling:
            if self.pooling_type == "mean":
                base_qkv = torch.mean(qkv, dim=0, keepdim=True)  # [1, L, D]
            elif self.pooling_type == "max":
                base_qkv, _ = torch.max(qkv, dim=0, keepdim=True)  # [1, L, D]
            elif self.pooling_type == "attention":
                attn_weights = self.pooling_attention(x_mod)  # [B, L, 1]
                attn_weights = torch.softmax(attn_weights, dim=0)
                
                base_qkv = torch.sum(qkv * attn_weights, dim=0, keepdim=True)  # [1, L, D]
            
            base_qkv = base_qkv.expand(batch_size, -1, -1)  # [B, L, D]
        
        qkv_deltas = []
        if self.use_pooling:
            for i in range(batch_size):
                if i == 0:  # albedo
                    delta = self.albedo_qkv_lora(base_qkv[i:i+1]) * self.lora_weight
                elif i == 1:  # material
                    delta = self.material_qkv_lora(base_qkv[i:i+1]) * self.lora_weight
                else:  # normal (i == 2)
                    delta = self.normal_qkv_lora(base_qkv[i:i+1]) * self.lora_weight
                qkv_deltas.append(delta)
        else:
            for i in range(batch_size):
                if i == 0:  # albedo
                    delta = self.albedo_qkv_lora(x_mod[i:i+1]) * self.lora_weight
                elif i == 1:  # material
                    delta = self.material_qkv_lora(x_mod[i:i+1]) * self.lora_weight
                else:  # normal (i == 2)
                    delta = self.normal_qkv_lora(x_mod[i:i+1]) * self.lora_weight
                qkv_deltas.append(delta)
        
        qkv_deltas = torch.cat(qkv_deltas, dim=0)
        if self.use_pooling:
            qkv = base_qkv + qkv_deltas
        else:
            qkv = qkv + qkv_deltas

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        mlp_input = torch.cat((attn_1, attn.mlp_act(mlp)), 2)
        output = attn.linear2(mlp_input)
        
        proj_deltas = []
        
        if self.use_pooling:
            if self.pooling_type == "mean":
                base_output = torch.mean(output, dim=0, keepdim=True)  # [1, L, D]
            elif self.pooling_type == "max":
                base_output, _ = torch.max(output, dim=0, keepdim=True)  # [1, L, D]
            elif self.pooling_type == "attention":
                attn_weights = self.pooling_attention(x_mod)  # [B, L, 1] 
                attn_weights = torch.softmax(attn_weights, dim=0)
                base_output = torch.sum(output * attn_weights, dim=0, keepdim=True)  # [1, L, D]
            
            base_output = base_output.expand(batch_size, -1, -1)  # [B, L, D]
            
            for i in range(batch_size):
                if i == 0:  # albedo
                    delta = self.albedo_proj_lora(base_output[i:i+1]) * self.lora_weight
                elif i == 1:  # material
                    delta = self.material_proj_lora(base_output[i:i+1]) * self.lora_weight
                else:  # normal (i == 2)
                    delta = self.normal_proj_lora(base_output[i:i+1]) * self.lora_weight
                proj_deltas.append(delta)
            
            proj_deltas = torch.cat(proj_deltas, dim=0)
            output = base_output + proj_deltas
        else:
            for i in range(batch_size):
                if i == 0:  # albedo
                    delta = self.albedo_proj_lora(output[i:i+1]) * self.lora_weight
                elif i == 1:  # material
                    delta = self.material_proj_lora(output[i:i+1]) * self.lora_weight
                else:  # normal (i == 2)
                    delta = self.normal_proj_lora(output[i:i+1]) * self.lora_weight
                proj_deltas.append(delta)
            
            proj_deltas = torch.cat(proj_deltas, dim=0)
            output = output + proj_deltas
        
        output = x + mod.gate * output
        del proj_deltas, qkv_deltas, delta
        return output



class SimpleMultiModalSharedKVSingleStreamBlockLoraProcessor(nn.Module):
    """
    Simple Multi-modal LoRA processor with shared K,V mechanism for SingleStreamBlock.
    Each modality uses its own Q but shares K,V computed from all modalities.
    Based on SimpleMultiModalSingleStreamBlockLoraProcessor but with SharedKV mechanism.
    """
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.albedo_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # material LoRA (batch=1)  
        self.material_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.material_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.normal_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        batch_size = x.shape[0]
        
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
        qkv_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_qkv_lora(x_mod[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_qkv_lora(x_mod[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_qkv_lora(x_mod[i:i+1]) * self.lora_weight
            qkv_deltas.append(delta)
        
        qkv_deltas = torch.cat(qkv_deltas, dim=0)
        qkv = qkv + qkv_deltas

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)
        
        # q: [B, H, L, D], k: [B, H, L, D], v: [B, H, L, D]
        
        # shared_k: [1, H, L*B, D], shared_v: [1, H, L*B, D] 
        shared_k = rearrange(k, "B H L D -> 1 H (B L) D")  # [1, H, 3*L, D]
        shared_v = rearrange(v, "B H L D -> 1 H (B L) D")  # [1, H, 3*L, D]
        
        attn_results = []
        for i in range(batch_size):
            modal_q = q[i:i+1]  # [1, H, L, D]
            
            modal_attn = attention(modal_q, shared_k, shared_v, pe=pe[i:i+1])  # [1, L, H*D]
            attn_results.append(modal_attn)
        
        attn_1 = torch.cat(attn_results, dim=0)  # [B, L, H*D]

        # compute activation in mlp stream, cat again and run second linear layer
        mlp_input = torch.cat((attn_1, attn.mlp_act(mlp)), 2)
        output = attn.linear2(mlp_input)
        
        proj_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_proj_lora(output[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_proj_lora(output[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_proj_lora(output[i:i+1]) * self.lora_weight
            proj_deltas.append(delta)
        
        proj_deltas = torch.cat(proj_deltas, dim=0)
        output = output + proj_deltas
        output = x + mod.gate * output
        
        del proj_deltas, qkv_deltas, delta, attn_results, modal_attn
        
        return output


class SimpleMultiModalAlbedoAlignedKVSingleStreamBlockLoraProcessor(nn.Module):
    """
    Simple Multi-modal LoRA processor with albedo-aligned K,V mechanism for SingleStreamBlock.
    - albedo: uses its own Q, K, V
    - material/normal: use their own Q, but K,V are concat of [albedo_K,V + own_K,V]
    Based on SimpleMultiModalSingleStreamBlockLoraProcessor but with albedo-aligned KV mechanism.
    
    Additional alignment LoRA option:
    - use_alignment_lora: adds dedicated LoRA layers for material/normal alignment
    - align_rank: controls the capacity of alignment LoRA layers
    """
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1, 
                 use_alignment_lora: bool = False, align_rank: int = 2):
        super().__init__()
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.albedo_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # material LoRA (batch=1)  
        self.material_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.material_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.normal_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        self.use_alignment_lora = use_alignment_lora
        if use_alignment_lora:
            self.material_align_qkv_lora = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.material_align_proj_lora = LoRALinearLayer(dim, dim, align_rank, network_alpha)
            
            self.normal_align_qkv_lora = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.normal_align_proj_lora = LoRALinearLayer(dim, dim, align_rank, network_alpha)
        
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        batch_size = x.shape[0]
        assert batch_size == 3, f"Expected batch_size=3 for albedo/material/normal, got {batch_size}"
        
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
        albedo_x, material_x, normal_x = x_mod[0:1], x_mod[1:2], x_mod[2:3]
        
        albedo_qkv_delta = self.albedo_qkv_lora(albedo_x) * self.lora_weight
        material_qkv_delta = self.material_qkv_lora(material_x) * self.lora_weight
        normal_qkv_delta = self.normal_qkv_lora(normal_x) * self.lora_weight
        
        if self.use_alignment_lora:
            material_align_delta = self.material_align_qkv_lora(material_x) * self.lora_weight
            normal_align_delta = self.normal_align_qkv_lora(normal_x) * self.lora_weight
            material_qkv_delta = material_qkv_delta + material_align_delta
            normal_qkv_delta = normal_qkv_delta + normal_align_delta
        
        qkv_deltas = torch.cat([albedo_qkv_delta, material_qkv_delta, normal_qkv_delta], dim=0)
        qkv = qkv + qkv_deltas

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)
        
        albedo_k, albedo_v = k[0:1], v[0:1]  # [1, H, L, D]
        
        material_k = torch.cat([albedo_k, k[1:2]], dim=2)  # [1, H, 2*L, D]
        material_v = torch.cat([albedo_v, v[1:2]], dim=2)  # [1, H, 2*L, D]
        normal_k = torch.cat([albedo_k, k[2:3]], dim=2)    # [1, H, 2*L, D]
        normal_v = torch.cat([albedo_v, v[2:3]], dim=2)    # [1, H, 2*L, D]
        
        albedo_attn = attention(q[0:1], k[0:1], v[0:1], pe=pe[0:1])
        material_attn = attention(q[1:2], material_k, material_v, pe=pe[1:2])
        normal_attn = attention(q[2:3], normal_k, normal_v, pe=pe[2:3])
        
        attn_1 = torch.cat([albedo_attn, material_attn, normal_attn], dim=0)  # [3, L, H*D]

        # compute activation in mlp stream, cat again and run second linear layer
        mlp_input = torch.cat((attn_1, attn.mlp_act(mlp)), 2)
        output = attn.linear2(mlp_input)
        
        albedo_out, material_out, normal_out = output[0:1], output[1:2], output[2:3]
        
        albedo_proj_delta = self.albedo_proj_lora(albedo_out) * self.lora_weight
        material_proj_delta = self.material_proj_lora(material_out) * self.lora_weight
        normal_proj_delta = self.normal_proj_lora(normal_out) * self.lora_weight
        
        if self.use_alignment_lora:
            material_align_proj_delta = self.material_align_proj_lora(material_out) * self.lora_weight
            normal_align_proj_delta = self.normal_align_proj_lora(normal_out) * self.lora_weight
            material_proj_delta = material_proj_delta + material_align_proj_delta
            normal_proj_delta = normal_proj_delta + normal_align_proj_delta
        
        proj_deltas = torch.cat([albedo_proj_delta, material_proj_delta, normal_proj_delta], dim=0)
        output = output + proj_deltas
        output = x + mod.gate * output
        
        return output

class SimpleMultiModalNormalAlignedKVSingleStreamBlockLoraProcessor(nn.Module):
    """
    Simple Multi-modal LoRA processor with normal-aligned K,V mechanism for SingleStreamBlock.
    - normal: uses its own Q, K, V (geometry drives everything)
    - albedo/material: use their own Q, but K,V are concat of [normal_K,V + own_K,V]
    Based on SimpleMultiModalSingleStreamBlockLoraProcessor but with normal-aligned KV mechanism.
    
    Additional alignment LoRA option:
    - use_alignment_lora: adds dedicated LoRA layers for albedo/material alignment
    - align_rank: controls the capacity of alignment LoRA layers
    """
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1, 
                 use_alignment_lora: bool = False, align_rank: int = 2):
        super().__init__()
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.albedo_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # material LoRA (batch=1)  
        self.material_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.material_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.normal_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        self.use_alignment_lora = use_alignment_lora
        if use_alignment_lora:
            self.albedo_align_qkv_lora = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.albedo_align_proj_lora = LoRALinearLayer(dim, dim, align_rank, network_alpha)
            
            self.material_align_qkv_lora = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.material_align_proj_lora = LoRALinearLayer(dim, dim, align_rank, network_alpha)
        
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        batch_size = x.shape[0]
        assert batch_size == 3, f"Expected batch_size=3 for albedo/material/normal, got {batch_size}"
        
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
        albedo_x, material_x, normal_x = x_mod[0:1], x_mod[1:2], x_mod[2:3]
        
        albedo_qkv_delta = self.albedo_qkv_lora(albedo_x) * self.lora_weight
        material_qkv_delta = self.material_qkv_lora(material_x) * self.lora_weight
        normal_qkv_delta = self.normal_qkv_lora(normal_x) * self.lora_weight
        
        if self.use_alignment_lora:
            material_align_delta = self.material_align_qkv_lora(material_x) * self.lora_weight
            normal_align_delta = self.normal_align_qkv_lora(normal_x) * self.lora_weight
            material_qkv_delta = material_qkv_delta + material_align_delta
            normal_qkv_delta = normal_qkv_delta + normal_align_delta
        
        qkv_deltas = torch.cat([albedo_qkv_delta, material_qkv_delta, normal_qkv_delta], dim=0)
        qkv = qkv + qkv_deltas

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)
        
        normal_k, normal_v = k[2:3], v[2:3]  # [1, H, L, D]

        albedo_k = torch.cat([normal_k, k[0:1]], dim=2)  # [1, H, 2*L, D]
        albedo_v = torch.cat([normal_v, v[0:1]], dim=2)  # [1, H, 2*L, D]
        material_k = torch.cat([normal_k, k[2:3]], dim=2)    # [1, H, 2*L, D]
        material_v = torch.cat([normal_v, v[2:3]], dim=2)    # [1, H, 2*L, D]
        
        normal_attn = attention(q[2:3], normal_k, normal_v, pe=pe[2:3])
        albedo_attn = attention(q[0:1], albedo_k, albedo_v, pe=pe[0:1])
        material_attn = attention(q[1:2], material_k, material_v, pe=pe[1:2])
        
        
        attn_1 = torch.cat([albedo_attn, material_attn, normal_attn], dim=0)  # [3, L, H*D]

        # compute activation in mlp stream, cat again and run second linear layer
        mlp_input = torch.cat((attn_1, attn.mlp_act(mlp)), 2)
        output = attn.linear2(mlp_input)
        
        albedo_out, material_out, normal_out = output[0:1], output[1:2], output[2:3]
        
        albedo_proj_delta = self.albedo_proj_lora(albedo_out) * self.lora_weight
        material_proj_delta = self.material_proj_lora(material_out) * self.lora_weight
        normal_proj_delta = self.normal_proj_lora(normal_out) * self.lora_weight
        
        if self.use_alignment_lora:
            albedo_align_proj_delta = self.albedo_align_proj_lora(albedo_out) * self.lora_weight
            material_align_proj_delta = self.material_align_proj_lora(material_out) * self.lora_weight
            albedo_proj_delta = albedo_proj_delta + albedo_align_proj_delta
            material_proj_delta = material_proj_delta + material_align_proj_delta
        
        proj_deltas = torch.cat([albedo_proj_delta, material_proj_delta, normal_proj_delta], dim=0)
        output = output + proj_deltas
        output = x + mod.gate * output
        
        return output

class SimpleMultiModalCausalSharedKVSingleStreamBlockLoraProcessor(nn.Module):
    """
    Simple Multi-modal LoRA processor with causal K,V sharing mechanism for SingleStreamBlock.
    Causal sharing chain: albedo -> normal -> material
    - albedo: uses only its own Q, K, V
    - normal: uses its own Q, but K,V are concat of [albedo_K,V + own_K,V]  
    - material: uses its own Q, but K,V are concat of [albedo_K,V + normal_K,V + own_K,V]
    
    Additional alignment LoRA option:
    - use_alignment_lora: adds dedicated LoRA layers for normal/material alignment
    - align_rank: controls the capacity of alignment LoRA layers
    """
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1, 
                 use_alignment_lora: bool = False, align_rank: int = 2):
        super().__init__()
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.albedo_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # material LoRA (batch=1)  
        self.material_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.material_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.normal_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        self.use_alignment_lora = use_alignment_lora
        if use_alignment_lora:
            self.material_align_qkv_lora = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.material_align_proj_lora = LoRALinearLayer(dim, dim, align_rank, network_alpha)
            
            self.normal_align_qkv_lora = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.normal_align_proj_lora = LoRALinearLayer(dim, dim, align_rank, network_alpha)
        
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        batch_size = x.shape[0]
        assert batch_size == 3, f"Expected batch_size=3 for albedo/material/normal, got {batch_size}"
        
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
        albedo_x, material_x, normal_x = x_mod[0:1], x_mod[1:2], x_mod[2:3]
        
        albedo_qkv_delta = self.albedo_qkv_lora(albedo_x) * self.lora_weight
        material_qkv_delta = self.material_qkv_lora(material_x) * self.lora_weight
        normal_qkv_delta = self.normal_qkv_lora(normal_x) * self.lora_weight
        
        if self.use_alignment_lora:
            material_align_delta = self.material_align_qkv_lora(material_x) * self.lora_weight
            normal_align_delta = self.normal_align_qkv_lora(normal_x) * self.lora_weight
            material_qkv_delta = material_qkv_delta + material_align_delta
            normal_qkv_delta = normal_qkv_delta + normal_align_delta
        
        qkv_deltas = torch.cat([albedo_qkv_delta, material_qkv_delta, normal_qkv_delta], dim=0)
        qkv = qkv + qkv_deltas

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)
        
        albedo_k, albedo_v = k[0:1], v[0:1]  # [1, H, L, D]
        
        normal_k = torch.cat([albedo_k, k[2:3]], dim=2)    # [1, H, 2*L, D]
        normal_v = torch.cat([albedo_v, v[2:3]], dim=2)    # [1, H, 2*L, D]

        material_k = torch.cat([normal_k, k[1:2]], dim=2)  # [1, H, 2*L+L, D]
        material_v = torch.cat([normal_v, v[1:2]], dim=2)  # [1, H, 2*L+L, D]
        
        
        albedo_attn = attention(q[0:1], k[0:1], v[0:1], pe=pe[0:1])
        material_attn = attention(q[1:2], material_k, material_v, pe=pe[1:2])
        normal_attn = attention(q[2:3], normal_k, normal_v, pe=pe[2:3])
        
        attn_1 = torch.cat([albedo_attn, material_attn, normal_attn], dim=0)  # [3, L, H*D]

        # compute activation in mlp stream, cat again and run second linear layer
        mlp_input = torch.cat((attn_1, attn.mlp_act(mlp)), 2)
        output = attn.linear2(mlp_input)
        
        albedo_out, material_out, normal_out = output[0:1], output[1:2], output[2:3]
        
        albedo_proj_delta = self.albedo_proj_lora(albedo_out) * self.lora_weight
        material_proj_delta = self.material_proj_lora(material_out) * self.lora_weight
        normal_proj_delta = self.normal_proj_lora(normal_out) * self.lora_weight
        
        if self.use_alignment_lora:
            material_align_proj_delta = self.material_align_proj_lora(material_out) * self.lora_weight
            normal_align_proj_delta = self.normal_align_proj_lora(normal_out) * self.lora_weight
            material_proj_delta = material_proj_delta + material_align_proj_delta
            normal_proj_delta = normal_proj_delta + normal_align_proj_delta
        
        proj_deltas = torch.cat([albedo_proj_delta, material_proj_delta, normal_proj_delta], dim=0)
        output = output + proj_deltas
        output = x + mod.gate * output
        
        return output

class SimpleMultiModalCausalSharedKV_N_A_M_SingleStreamBlockLoraProcessor(nn.Module):
    """
    Simple Multi-modal LoRA processor with causal K,V sharing mechanism for SingleStreamBlock.
    Causal sharing chain: albedo -> normal -> material
    - albedo: uses only its own Q, K, V
    - normal: uses its own Q, but K,V are concat of [albedo_K,V + own_K,V]  
    - material: uses its own Q, but K,V are concat of [albedo_K,V + normal_K,V + own_K,V]
    
    Additional alignment LoRA option:
    - use_alignment_lora: adds dedicated LoRA layers for normal/material alignment
    - align_rank: controls the capacity of alignment LoRA layers
    """
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1, 
                 use_alignment_lora: bool = False, align_rank: int = 2):
        super().__init__()
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.albedo_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # material LoRA (batch=1)  
        self.material_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.material_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.normal_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        self.use_alignment_lora = use_alignment_lora
        if use_alignment_lora:
            self.albedo_align_qkv_lora = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.albedo_align_proj_lora = LoRALinearLayer(dim, dim, align_rank, network_alpha)

            self.material_align_qkv_lora = LoRALinearLayer(dim, dim * 3, align_rank, network_alpha)
            self.material_align_proj_lora = LoRALinearLayer(dim, dim, align_rank, network_alpha)
            
            
        
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        batch_size = x.shape[0]
        assert batch_size == 3, f"Expected batch_size=3 for albedo/material/normal, got {batch_size}"
        
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
        albedo_x, material_x, normal_x = x_mod[0:1], x_mod[1:2], x_mod[2:3]
        
        albedo_qkv_delta = self.albedo_qkv_lora(albedo_x) * self.lora_weight
        material_qkv_delta = self.material_qkv_lora(material_x) * self.lora_weight
        normal_qkv_delta = self.normal_qkv_lora(normal_x) * self.lora_weight
        
        if self.use_alignment_lora:
            albedo_align_delta = self.albedo_align_qkv_lora(albedo_x) * self.lora_weight
            material_align_delta = self.material_align_qkv_lora(material_x) * self.lora_weight
            
            albedo_qkv_delta = albedo_qkv_delta + albedo_align_delta
            material_qkv_delta = material_qkv_delta + material_align_delta
        
        qkv_deltas = torch.cat([albedo_qkv_delta, material_qkv_delta, normal_qkv_delta], dim=0)
        qkv = qkv + qkv_deltas

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        normal_k, normal_v = k[2:3], v[2:3]  # [1, H, L, D]
        
        albedo_k = torch.cat([normal_k, k[0:1]], dim=2)  # [1, H, 2*L+L, D]
        albedo_v = torch.cat([normal_v, v[0:1]], dim=2)  # [1, H, 2*L+L, D]

        material_k = torch.cat([albedo_k, k[1:2]], dim=2)    # [1, H, 2*L, D]
        material_v = torch.cat([albedo_v, v[1:2]], dim=2)    # [1, H, 2*L, D]

        albedo_attn = attention(q[0:1], albedo_k, albedo_v, pe=pe[0:1])
        material_attn = attention(q[1:2], material_k, material_v, pe=pe[1:2])
        normal_attn = attention(q[2:3], normal_k, normal_v, pe=pe[2:3])
        
        attn_1 = torch.cat([albedo_attn, material_attn, normal_attn], dim=0)  # [3, L, H*D]

        # compute activation in mlp stream, cat again and run second linear layer
        mlp_input = torch.cat((attn_1, attn.mlp_act(mlp)), 2)
        output = attn.linear2(mlp_input)
        
        albedo_out, material_out, normal_out = output[0:1], output[1:2], output[2:3]
        
        albedo_proj_delta = self.albedo_proj_lora(albedo_out) * self.lora_weight
        material_proj_delta = self.material_proj_lora(material_out) * self.lora_weight
        normal_proj_delta = self.normal_proj_lora(normal_out) * self.lora_weight
        
        if self.use_alignment_lora:
            albedo_align_proj_delta = self.albedo_align_proj_lora(albedo_out) * self.lora_weight
            material_align_proj_delta = self.material_align_proj_lora(material_out) * self.lora_weight

            albedo_proj_delta = albedo_proj_delta + albedo_align_proj_delta
            material_proj_delta = material_proj_delta + material_align_proj_delta
            
        proj_deltas = torch.cat([albedo_proj_delta, material_proj_delta, normal_proj_delta], dim=0)
        output = output + proj_deltas
        output = x + mod.gate * output
        
        return output

class MultiModalSingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.albedo_proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        
        # material LoRA (batch=1)  
        self.material_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.material_proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.normal_proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        batch_size = x.shape[0] # 3 1536 3072
        
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
        qkv_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_qkv_lora(x_mod[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_qkv_lora(x_mod[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_qkv_lora(x_mod[i:i+1]) * self.lora_weight
            qkv_deltas.append(delta)
        
        qkv_deltas = torch.cat(qkv_deltas, dim=0)
        qkv = qkv + qkv_deltas

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        mlp_input = torch.cat((attn_1, attn.mlp_act(mlp)), 2)
        output = attn.linear2(mlp_input)
        
        proj_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_proj_lora(mlp_input[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_proj_lora(mlp_input[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_proj_lora(mlp_input[i:i+1]) * self.lora_weight
            proj_deltas.append(delta)
        
        proj_deltas = torch.cat(proj_deltas, dim=0)
        output = output + proj_deltas
        output = x + mod.gate * output
        del proj_deltas, qkv_deltas, delta
        return output


class MultiModalSharedKVSingleStreamBlockLoraProcessor(nn.Module):
    """
    Multi-modal LoRA processor with shared K,V mechanism for SingleStreamBlock.
    Each modality uses its own Q but shares K,V computed from all modalities.
    """
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        # albedo LoRA (batch=0)
        self.albedo_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.albedo_proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        
        # material LoRA (batch=1)  
        self.material_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.material_proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        
        # normal LoRA (batch=2)
        self.normal_qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.normal_proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        batch_size = x.shape[0]  # 3 1536 3072
        
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
        qkv_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_qkv_lora(x_mod[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_qkv_lora(x_mod[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_qkv_lora(x_mod[i:i+1]) * self.lora_weight
            qkv_deltas.append(delta)
        
        qkv_deltas = torch.cat(qkv_deltas, dim=0)
        qkv = qkv + qkv_deltas

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)
        
        # q: [B, H, L, D], k: [B, H, L, D], v: [B, H, L, D]
        
        # shared_k: [1, H, L*B, D], shared_v: [1, H, L*B, D] 
        shared_k = rearrange(k, "B H L D -> 1 H (B L) D")  # [1, H, 3*L, D]
        shared_v = rearrange(v, "B H L D -> 1 H (B L) D")  # [1, H, 3*L, D]
        
        attn_results = []
        for i in range(batch_size):
            modal_attn = attention(q[i:i+1], shared_k, shared_v, pe=pe[i:i+1])  # [1, L, H*D]
            attn_results.append(modal_attn)
        
        attn_results = torch.cat(attn_results, dim=0)  # [B, L, H*D]

        # compute activation in mlp stream, cat again and run second linear layer
        mlp_input = torch.cat((attn_results, attn.mlp_act(mlp)), 2)
        output = attn.linear2(mlp_input)
        
        proj_deltas = []
        for i in range(batch_size):
            if i == 0:  # albedo
                delta = self.albedo_proj_lora(mlp_input[i:i+1]) * self.lora_weight
            elif i == 1:  # material
                delta = self.material_proj_lora(mlp_input[i:i+1]) * self.lora_weight
            else:  # normal (i == 2)
                delta = self.normal_proj_lora(mlp_input[i:i+1]) * self.lora_weight
            proj_deltas.append(delta)
        
        proj_deltas = torch.cat(proj_deltas, dim=0)
        output = output + proj_deltas
        output = x + mod.gate * output
        
        del proj_deltas, qkv_deltas, delta, attn_results, modal_attn
        return output


class SingleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output
        return output

class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)


    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0
    ) -> Tensor:
        if image_proj is None:
            return self.processor(self, x, vec, pe)
        else:
            return self.processor(self, x, vec, pe, image_proj, ip_scale)


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x

class ImageProjModel(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

