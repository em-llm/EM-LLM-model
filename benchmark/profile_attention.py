import argparse
import time
import torch
from em_llm.attention.dot_product_attention.torch_impl import TorchMultiStageDotProductAttention

import inspect
from functools import wraps
from torch.cuda import nvtx

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        nvtx.range_push(f"attn.{func.__name__}")
        try:
            out = func(*args, **kwargs)
        finally:
            nvtx.range_pop()
        return out
    return wrapper

def sync():
    torch.cuda.synchronize()

def make_tensors(args):
    q = torch.randn(args.batch, args.num_heads, args.q_len, args.head_dim, device="cuda", dtype=args.dtype)
    k_local = torch.randn(args.batch, args.num_heads_kv, args.local_len, args.head_dim, device="cuda", dtype=args.dtype)
    v_local = torch.randn_like(k_local)
    k_global = torch.randn(args.batch, args.num_heads_kv, args.global_len, args.head_dim, device="cuda", dtype=args.dtype)
    v_global = torch.randn_like(k_global)
    return q, k_local, v_local, k_global, v_global

def run_current(q, k_local, v_local, k_global, v_global, n_local):
    attn = TorchMultiStageDotProductAttention(q.shape, q.dtype, q.device)
    attn.append(q, k_local, v_local, get_score=True, sliding_window=n_local)
    attn.append(q, k_local, v_local, end=True, get_score=False, sliding_window=None)
    return attn.get_result()[0]

def benchmark(fn, warmup=10, iters=50, torch_profiler=False):
    for _ in range(warmup):
        out = fn()
    sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    sync()

    if torch_profiler:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(10):
                out = fn()
        sync()

        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30,
        ))
        prof.export_chrome_trace(f"{fn.__name__}_trace.json")

    return start.elapsed_time(end) / iters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-heads-kv", type=int, default=8)
    parser.add_argument("--q-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--local-len", type=int, default=4096)
    parser.add_argument("--global-len", type=int, default=2688)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    args.dtype = getattr(torch, args.dtype)
    q, k_local, v_local, k_global, v_global = make_tensors(args)

    ms = benchmark(
        lambda: run_current(q, k_local, v_local, k_global, v_global, args.local_len),
        warmup=args.warmup,
        iters=args.iters,
        torch_profiler=args.torch_profiler,
    )
    print(f"current: {ms:.3f} ms")

if __name__ == "__main__":
    main()
