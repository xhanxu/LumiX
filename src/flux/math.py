import torch
from einops import rearrange
from torch import Tensor


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    
    q_seq_len = xq.shape[-2]
    k_seq_len = xk.shape[-2]
    
    if q_seq_len != k_seq_len:
        if k_seq_len % q_seq_len == 0:
            repeat_factor = k_seq_len // q_seq_len
            freqs_cis_expanded = freqs_cis.repeat(1, 1, repeat_factor, 1, 1, 1)
            
            xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
            xk_out = freqs_cis_expanded[..., 0] * xk_[..., 0] + freqs_cis_expanded[..., 1] * xk_[..., 1]
        elif q_seq_len % k_seq_len == 0:
            freqs_cis_truncated = freqs_cis[:, :, :k_seq_len, :, :, :]
            
            xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
            xk_out = freqs_cis_truncated[..., 0] * xk_[..., 0] + freqs_cis_truncated[..., 1] * xk_[..., 1]
        else:
            if k_seq_len < q_seq_len:
                freqs_cis_k = freqs_cis[:, :, :k_seq_len, :, :, :]
            else:
                repeat_times = (k_seq_len + q_seq_len - 1) // q_seq_len
                freqs_cis_repeated = freqs_cis.repeat(1, 1, repeat_times, 1, 1, 1)
                freqs_cis_k = freqs_cis_repeated[:, :, :k_seq_len, :, :, :]
            
            xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
            xk_out = freqs_cis_k[..., 0] * xk_[..., 0] + freqs_cis_k[..., 1] * xk_[..., 1]
    else:
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
