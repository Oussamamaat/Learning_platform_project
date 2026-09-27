"""S3Gen + watermark speedups, second half of the Chatterbox latency plan (after fast_t3.py).

Measured on the Akash RTX 3090 / Xeon E5-2697A v4 (s3gen_probe.py, 2026-09-27), sentence 01:
  - the 10-step CFM loop took ~0.75s whatever the mel length (406-600): launch-bound, like T3 was.
    Its estimator calls add_optional_chunk_mask 14x per pass, each with a `.item()` safety check:
    140 forced GPU->CPU syncs per sentence.
      eager 758 ms | sync-free 706 ms (bit-identical) | CUDA graph at exact length 349 ms
      (bit-identical) | CUDA graph padded to a length bucket 401 ms (mel max|diff| 1e-2 on a
      [-13, 4.6] range)
  - the PerTh watermark ran on the CPU (perth's default device): median 158 ms, up to 666 ms.
    On the GPU: median 15 ms, watermark still detected (1.0 on every bit), output diff 6e-5.

What this does:
  1. add_optional_chunk_mask without the `.item()` check. decoder.py only ever calls it with
     use_dynamic_chunk=False, static_chunk_size=0, where it returns the input mask unchanged; the
     skipped check can only fire for a batch row of length 0. Bit-identical.
  2. The whole 10-step CFM loop (solve_euler) as one CUDA graph per LENGTH BUCKET (mel length
     rounded up to a multiple of BUCKET, pad masked out). Exact-length graphs are bit-identical
     but each new length costs ~2s to capture, which would land on a live listener; buckets are
     all captured once at startup. Not bit-identical (1e-2 on the mel, far below the
     laptop-vs-server difference of the stock pipeline itself), so it goes through a blind listen.
  3. Watermark network on the GPU.

Use (after install_fast_t3, model on cuda):
    from fast_s3gen import install_fast_s3gen
    install_fast_s3gen(engine)
"""
import logging
import math
import time

import torch

import src.chatterbox_.models.s3gen.decoder as decoder_mod

logger = logging.getLogger(__name__)

BUCKET = 32          # mel frames (50/s): <= 0.64s of padded audio per bucket
MAX_TOKENS_PREWARM = 400  # speech tokens covered at startup (16s of audio); longer -> captured lazily


def chunk_mask_nosync(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size,
                      static_chunk_size, num_decoding_left_chunks, enable_full_context=True):
    """add_optional_chunk_mask for the only call shape decoder.py uses, minus its `.item()` check."""
    assert not use_dynamic_chunk and static_chunk_size == 0, "unexpected call shape"
    assert masks.dtype == torch.bool
    return masks


class GraphCFM:
    """Replaces CausalConditionalCFM.solve_euler (non-meanflow, batch 1) with bucketed graphs."""

    def __init__(self, cfm, bucket: int = BUCKET):
        self.cfm = cfm
        self.orig = cfm.solve_euler  # bound method of the class implementation
        self.bucket = bucket
        self.graphs = {}  # padded length -> (graph, static inputs, static output)
        self.pool = torch.cuda.graph_pool_handle()

    def _capture(self, Tp: int, dtype, device, n_steps: int):
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        st = dict(x=z(1, 80, Tp), t_span=z(n_steps + 1), mu=z(1, 80, Tp),
                  mask=torch.ones(1, 1, Tp, device=device, dtype=dtype), spks=z(1, 80),
                  cond=z(1, 80, Tp))
        fn = lambda: self.orig(st["x"], st["t_span"], st["mu"], st["mask"], st["spks"], st["cond"])
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = fn()
        self.graphs[Tp] = (g, st, out)

    def padded(self, T: int) -> int:
        return math.ceil(T / self.bucket) * self.bucket

    @torch.inference_mode()
    def __call__(self, x, t_span, mu, mask, spks, cond, meanflow=False):
        if meanflow or x.size(0) != 1 or spks is None or cond is None:
            return self.orig(x, t_span, mu, mask, spks, cond, meanflow=meanflow)
        T = x.shape[-1]
        Tp = self.padded(T)
        if Tp not in self.graphs:
            t0 = time.perf_counter()
            self._capture(Tp, x.dtype, x.device, t_span.numel() - 1)
            logger.warning(f"fast_s3gen: captured bucket {Tp} lazily ({time.perf_counter() - t0:.1f}s)")
        g, st, out = self.graphs[Tp]
        if st["t_span"].numel() != t_span.numel():
            return self.orig(x, t_span, mu, mask, spks, cond, meanflow=meanflow)
        for name, val in (("x", x), ("mu", mu), ("mask", mask), ("cond", cond)):
            buf = st[name]
            buf.zero_()                      # pad region: zero input, mask 0
            buf[..., :T].copy_(val)
        st["t_span"].copy_(t_span)
        st["spks"].copy_(spks)
        g.replay()
        return out[..., :T].clone()

    def prewarm(self, T_min: int, T_max: int, dtype, device, n_steps: int = 10):
        t0 = time.perf_counter()
        for Tp in range(self.padded(T_min), self.padded(T_max) + 1, self.bucket):
            if Tp not in self.graphs:
                self._capture(Tp, dtype, device, n_steps)
        torch.cuda.synchronize()
        return time.perf_counter() - t0


def install_fast_s3gen(engine, bucket: int = BUCKET, prewarm: bool = True):
    s3gen = engine.s3gen
    if next(s3gen.parameters()).device.type != "cuda":
        raise RuntimeError("fast_s3gen needs the model on cuda")
    decoder_mod.add_optional_chunk_mask = chunk_mask_nosync

    cfm = s3gen.flow.decoder
    gcfm = GraphCFM(cfm, bucket=bucket)
    cfm.solve_euler = gcfm  # instance attribute shadows the method; called as solve_euler(z, ...)

    if prewarm:
        # mel length = 2 x (prompt tokens + generated tokens); the prompt part is fixed per voice.
        prompt_mel = engine.conds.gen["prompt_feat"].shape[1]
        dtype = next(cfm.estimator.parameters()).dtype
        secs = gcfm.prewarm(prompt_mel + 2 * 8, prompt_mel + 2 * MAX_TOKENS_PREWARM, dtype,
                            torch.device("cuda"), n_steps=10)
        logger.info(f"fast_s3gen: {len(gcfm.graphs)} CFM buckets captured in {secs:.1f}s")

    engine.watermarker.perth_net.to("cuda")
    free, total = torch.cuda.mem_get_info()
    logger.info(f"fast_s3gen installed; GPU {(total - free) / 1e9:.2f} GB used")
    return gcfm
