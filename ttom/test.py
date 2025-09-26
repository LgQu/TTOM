# test_manual_attn_firstn.py
import math
import torch
import torch.nn.functional as F
from einops import rearrange

torch.backends.cuda.matmul.allow_tf32 = True  # Improve FP32 performance (Ampere+)

@torch.no_grad()
def manual_attn_weights_firstn(q_inst: torch.Tensor,
                               k_img: torch.Tensor,
                               num_heads: int,
                               n_tokens: int | None = None,
                               chunk_k: int | None = None,
                               upcast: bool = True) -> torch.Tensor:
    """
    Manually calculate cross-attention softmax weights from q_inst, k_img (no v needed).
    Compatible with your shape: q_inst=[B,L_q,H*d], k_img=[B,L_k,H*d]
    Returns:
        attn_map: [B, H, n, L_k]  (n = n_tokens or L_q)
    Parameters:
        num_heads: your head count H
        n_tokens : only take first n query tokens; None means use all L_q
        chunk_k  : if L_k is large, can chunk by columns (e.g. 1024/2048); None means no chunking
        upcast   : upcast to fp32 during computation for stability, then fall back to input dtype
    """
    assert q_inst.shape[-1] == k_img.shape[-1], "q/k last dimension mismatch"
    B, L_q, D = q_inst.shape
    _, L_k, Dk = k_img.shape
    assert D % num_heads == 0, "D must be divisible by num_heads"
    d_head = D // num_heads
    n = L_q if n_tokens is None else min(n_tokens, L_q)

    # Select first n queries
    q = q_inst[:, :n, :]            # [B, n, H*d]
    k = k_img                       # [B, L_k, H*d]

    # [B,H,L, d]
    q_h = rearrange(q, "b L (h d) -> b h L d", h=num_heads, d=d_head).contiguous()
    k_h = rearrange(k, "b L (h d) -> b h L d", h=num_heads, d=d_head).contiguous()

    # Upcast precision (more stable)
    comp_dtype = torch.float32 if upcast else q_h.dtype
    qh = q_h.to(comp_dtype)
    kh = k_h.to(comp_dtype)
    scale = 1.0 / math.sqrt(d_head)

    # No chunking: calculate scores→softmax at once
    if chunk_k is None or chunk_k >= L_k:
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale      # [B,H,n,L_k]
        attn = torch.softmax(scores, dim=-1)
        return attn.to(q_inst.dtype)

    # Chunking: two-pass log-sum-exp (stable/memory efficient)
    B_, H_, n_ = B, num_heads, n
    device = q_inst.device
    m = torch.full((B_, H_, n_), -float("inf"), dtype=comp_dtype, device=device)  # Row max
    z = torch.zeros((B_, H_, n_), dtype=comp_dtype, device=device)                # Row exp sum
    for start in range(0, L_k, chunk_k):
        end = min(start + chunk_k, L_k)
        ks  = kh[:, :, start:end, :]                              # [B,H,ck,d]
        sc  = torch.matmul(qh, ks.transpose(-2, -1)) * scale      # [B,H,n,ck]
        m_new = torch.maximum(m, sc.amax(dim=-1))                 # [B,H,n]
        z = torch.exp(m - m_new) * z + torch.exp(sc - m_new.unsqueeze(-1)).sum(dim=-1)
        m = m_new

    attn = torch.empty((B_, H_, n_, L_k), dtype=comp_dtype, device=device)
    for start in range(0, L_k, chunk_k):
        end = min(start + chunk_k, L_k)
        ks  = kh[:, :, start:end, :]
        sc  = torch.matmul(qh, ks.transpose(-2, -1)) * scale      # [B,H,n,ck]
        attn_chunk = torch.exp(sc - m.unsqueeze(-1)) / z.unsqueeze(-1)  # Normalize
        attn[:, :, :, start:end] = attn_chunk

    return attn.to(q_inst.dtype)

@torch.no_grad()
def run_one_case(B=2, H=4, L_q=17, L_k=1025, d_head=64, device="cuda" if torch.cuda.is_available() else "cpu",
                 dtype=torch.float32, n_tokens=8, chunk_k: int | None = 512):
    torch.manual_seed(0)
    D = H * d_head

    # Simulate your preprocessing: q_inst = self.norm_q(self.q(inst)); k_img = self.norm_k(self.k(x))
    q_inst = torch.randn(B, L_q, D, device=device, dtype=dtype)
    k_img  = torch.randn(B, L_k, D, device=device, dtype=dtype)

    # 1) Manual weights (full L_q, no chunking)
    A_full = manual_attn_weights_firstn(q_inst, k_img, num_heads=H, n_tokens=None, chunk_k=None, upcast=True)   # [B,H,L_q,L_k]
    # 2) Manual weights (first n_tokens, chunked)
    A_firstn_chunk = manual_attn_weights_firstn(q_inst, k_img, num_heads=H, n_tokens=n_tokens, chunk_k=chunk_k, upcast=True)  # [B,H,n,L_k]

    # Random v, and reshape to [B,H,L_k,d]
    v = torch.randn(B, L_k, D, device=device, dtype=dtype)
    v_h = rearrange(v, "b L (h d) -> b h L d", h=H, d=d_head).contiguous()

    # Multiply v with manual weights to get output
    x_manual_full   = torch.matmul(A_full.float(), v_h.float()).to(dtype)               # [B,H,L_q,d]
    x_manual_firstn = torch.matmul(A_firstn_chunk.float(), v_h.float()).to(dtype)       # [B,H,n,d]

    # Library function output (full L_q)
    q_h_full = rearrange(q_inst,           "b L (h d) -> b h L d", h=H, d=d_head)
    k_h      = rearrange(k_img,            "b L (h d) -> b h L d", h=H, d=d_head)
    x_lib_full = F.scaled_dot_product_attention(q_h_full, k_h, v_h)                     # [B,H,L_q,d]

    # Library function output (first n_tokens)
    q_h_firstn = q_h_full[:, :, :n_tokens, :]
    x_lib_firstn = F.scaled_dot_product_attention(q_h_firstn, k_h, v_h)                 # [B,H,n,d]

    # Error
    def report(name, a, b):
        max_abs = (a - b).abs().max().item()
        mean_abs = (a - b).abs().mean().item()
        print(f"{name:<20s} | shape={tuple(a.shape)} | max|Δ|={max_abs:.3e} | mean|Δ|={mean_abs:.3e}")
        return max_abs

    print(f"Device={device}, DTYPE={str(dtype).split('.')[-1]}, "
          f"B={B}, H={H}, L_q={L_q}, L_k={L_k}, d_head={d_head}, n={n_tokens}, chunk_k={chunk_k}")
    max1 = report("FULL (no-chunk)", x_manual_full,   x_lib_full)
    max2 = report("FIRST n (chunk)", x_manual_firstn, x_lib_firstn)

    # Reasonable tolerance (FP32 strict; FP16 relaxed)
    tol = 1e-6 if dtype == torch.float32 else 2e-3
    assert max1 < tol and max2 < tol, f"Error too large (tol={tol}), please relax tolerance in FP16 or force math kernel"

if __name__ == "__main__":
    # FP32 / CUDA (automatically use CPU if no CUDA)
    run_one_case(dtype=torch.float32)

    # Optional: FP16 / CUDA (tolerance relaxed to ~2e-3 in FP16)
    if torch.cuda.is_available():
        run_one_case(dtype=torch.float16)
