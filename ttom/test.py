# test_manual_attn_firstn.py
import math
import torch
import torch.nn.functional as F
from einops import rearrange

torch.backends.cuda.matmul.allow_tf32 = True  # 提升 FP32 性能（Ampere+）

@torch.no_grad()
def manual_attn_weights_firstn(q_inst: torch.Tensor,
                               k_img: torch.Tensor,
                               num_heads: int,
                               n_tokens: int | None = None,
                               chunk_k: int | None = None,
                               upcast: bool = True) -> torch.Tensor:
    """
    从 q_inst, k_img 手动计算 cross-attention 的 softmax 权重（不需要 v）。
    兼容你的shape：q_inst=[B,L_q,H*d], k_img=[B,L_k,H*d]
    返回:
        attn_map: [B, H, n, L_k]  （n = n_tokens 或 L_q）
    参数:
        num_heads: 你的 head 数 H
        n_tokens : 只取前 n 个 query token；None 表示用全部 L_q
        chunk_k  : 如果 L_k 很大，可按列分块（如 1024/2048）；None 表示不分块
        upcast   : 计算时上浮到 fp32 提升稳定性，然后回落到输入 dtype
    """
    assert q_inst.shape[-1] == k_img.shape[-1], "q/k 最后一维不一致"
    B, L_q, D = q_inst.shape
    _, L_k, Dk = k_img.shape
    assert D % num_heads == 0, "D 必须能被 num_heads 整除"
    d_head = D // num_heads
    n = L_q if n_tokens is None else min(n_tokens, L_q)

    # 选前 n 个 query
    q = q_inst[:, :n, :]            # [B, n, H*d]
    k = k_img                       # [B, L_k, H*d]

    # [B,H,L, d]
    q_h = rearrange(q, "b L (h d) -> b h L d", h=num_heads, d=d_head).contiguous()
    k_h = rearrange(k, "b L (h d) -> b h L d", h=num_heads, d=d_head).contiguous()

    # 上浮精度（更稳）
    comp_dtype = torch.float32 if upcast else q_h.dtype
    qh = q_h.to(comp_dtype)
    kh = k_h.to(comp_dtype)
    scale = 1.0 / math.sqrt(d_head)

    # 不分块：一次性算 scores→softmax
    if chunk_k is None or chunk_k >= L_k:
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale      # [B,H,n,L_k]
        attn = torch.softmax(scores, dim=-1)
        return attn.to(q_inst.dtype)

    # 分块：两遍 log-sum-exp（稳定/省显存）
    B_, H_, n_ = B, num_heads, n
    device = q_inst.device
    m = torch.full((B_, H_, n_), -float("inf"), dtype=comp_dtype, device=device)  # 行最大
    z = torch.zeros((B_, H_, n_), dtype=comp_dtype, device=device)                # 行 exp 和
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
        attn_chunk = torch.exp(sc - m.unsqueeze(-1)) / z.unsqueeze(-1)  # 归一化
        attn[:, :, :, start:end] = attn_chunk

    return attn.to(q_inst.dtype)

@torch.no_grad()
def run_one_case(B=2, H=4, L_q=17, L_k=1025, d_head=64, device="cuda" if torch.cuda.is_available() else "cpu",
                 dtype=torch.float32, n_tokens=8, chunk_k: int | None = 512):
    torch.manual_seed(0)
    D = H * d_head

    # 模拟你的前置：q_inst = self.norm_q(self.q(inst)); k_img = self.norm_k(self.k(x))
    q_inst = torch.randn(B, L_q, D, device=device, dtype=dtype)
    k_img  = torch.randn(B, L_k, D, device=device, dtype=dtype)

    # 1) 手动权重（全 L_q，不分块）
    A_full = manual_attn_weights_firstn(q_inst, k_img, num_heads=H, n_tokens=None, chunk_k=None, upcast=True)   # [B,H,L_q,L_k]
    # 2) 手动权重（前 n_tokens，分块）
    A_firstn_chunk = manual_attn_weights_firstn(q_inst, k_img, num_heads=H, n_tokens=n_tokens, chunk_k=chunk_k, upcast=True)  # [B,H,n,L_k]

    # 随机 v，并 reshape 到 [B,H,L_k,d]
    v = torch.randn(B, L_k, D, device=device, dtype=dtype)
    v_h = rearrange(v, "b L (h d) -> b h L d", h=H, d=d_head).contiguous()

    # 用手动权重乘 v 得到输出
    x_manual_full   = torch.matmul(A_full.float(), v_h.float()).to(dtype)               # [B,H,L_q,d]
    x_manual_firstn = torch.matmul(A_firstn_chunk.float(), v_h.float()).to(dtype)       # [B,H,n,d]

    # 库函数输出（全 L_q）
    q_h_full = rearrange(q_inst,           "b L (h d) -> b h L d", h=H, d=d_head)
    k_h      = rearrange(k_img,            "b L (h d) -> b h L d", h=H, d=d_head)
    x_lib_full = F.scaled_dot_product_attention(q_h_full, k_h, v_h)                     # [B,H,L_q,d]

    # 库函数输出（前 n_tokens）
    q_h_firstn = q_h_full[:, :, :n_tokens, :]
    x_lib_firstn = F.scaled_dot_product_attention(q_h_firstn, k_h, v_h)                 # [B,H,n,d]

    # 误差
    def report(name, a, b):
        max_abs = (a - b).abs().max().item()
        mean_abs = (a - b).abs().mean().item()
        print(f"{name:<20s} | shape={tuple(a.shape)} | max|Δ|={max_abs:.3e} | mean|Δ|={mean_abs:.3e}")
        return max_abs

    print(f"Device={device}, DTYPE={str(dtype).split('.')[-1]}, "
          f"B={B}, H={H}, L_q={L_q}, L_k={L_k}, d_head={d_head}, n={n_tokens}, chunk_k={chunk_k}")
    max1 = report("FULL (no-chunk)", x_manual_full,   x_lib_full)
    max2 = report("FIRST n (chunk)", x_manual_firstn, x_lib_firstn)

    # 合理容差（FP32 严格；FP16 放宽）
    tol = 1e-6 if dtype == torch.float32 else 2e-3
    assert max1 < tol and max2 < tol, f"误差过大（tol={tol}），请在 FP16 下放宽容差或强制 math kernel"

if __name__ == "__main__":
    # FP32 / CUDA（若无CUDA则自动用CPU）
    run_one_case(dtype=torch.float32)

    # 可选：FP16 / CUDA（FP16下容差放宽到 ~2e-3）
    if torch.cuda.is_available():
        run_one_case(dtype=torch.float16)
