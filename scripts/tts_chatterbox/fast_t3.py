"""CUDA-graph T3 decoder: the same model, the same math, the same hallucination guard, far fewer
CPU round-trips per token.

Why (plan: based-on-the-documentation-floating-puffin, "Chatterbox latency"): the B2 gate on an
RTX 3090 (outputs/bench_b2_rtx3090/) measured T3 at 21-25 tok/s with the GPU ~21% busy. Every
token runs an HF Llama forward -- 30 layers x a dozen small kernels, each launched from Python --
plus forced syncs, so the step is bound by the CPU launching work, not by the GPU doing it. Here
one decode step (embeddings, 30 layers over a preallocated KV cache, speech head) is recorded
once as a CUDA graph and replayed per token: one launch instead of hundreds.

What stays exactly as stock T3.inference (src/chatterbox_/models/t3/t3.py:238-423):
  - the prefill (conditioning + text + BOS) runs through the same HF path with eager attention,
    so the alignment guard's first chunk is computed the same way;
  - every per-token step after the forward pass: CFG combine, AlignmentStreamAnalyzer.step(),
    repetition penalty, temperature, min_p, top_p, multinomial, EOS check -- same code, same
    order, same RNG consumption;
  - fp32 weights, eager-attention formula (q.k^T / sqrt(d), fp32 softmax).
What changes: the decode forward pass itself, and where the guard's three attention rows come
from -- the graph writes (layer 9 head 2, layer 12 head 15, layer 13 head 11) into a small
buffer, instead of forward hooks copying every layer-sized attention tensor to the CPU.

Also fixes a stock leak: stock builds a new AlignmentStreamAnalyzer per call and never removes
its 3 forward hooks, so each call adds 3 more hooks that each copy attention to the CPU on every
forward. Variant A's T3 speed fell 21.4 -> 20.7 -> 20.0 tok/s across the B2 run's repeats; the
hook-free variant B stayed flat. Here the hooks are removed right after the prefill.

Use:
    from fast_t3 import install_fast_t3
    install_fast_t3(engine)          # after the adapter is merged and the model is on cuda
"""
import logging
import math
import types

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    MinPLogitsWarper, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper,
)

from src.chatterbox_.models.t3 import t3 as t3_mod
from src.chatterbox_.models.t3.inference.alignment_stream_analyzer import (
    LLAMA_ALIGNED_HEADS, AlignmentStreamAnalyzer,
)
from src.chatterbox_.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend

logger = logging.getLogger(__name__)

MAX_LEN = 1400  # prefill (~150 cond + <=250 text) + 1000 new tokens, the cap generate() passes
BUCKETS = (256, 384, 512, 640, 768, 1024)  # a typical sentence ends ~300-450 positions


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class GraphT3Decoder:
    """Static buffers + one captured decode step for a merged, fp32, cuda T3 (Llama backbone)."""

    def __init__(self, t3, max_len: int = MAX_LEN, batch: int = 2):
        tfmr = t3.tfmr
        cfg = tfmr.config
        assert cfg.num_key_value_heads == cfg.num_attention_heads, "GQA not handled"
        self.t3, self.tfmr, self.max_len, self.batch = t3, tfmr, max_len, batch
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        dev = t3.device
        dtype = next(t3.parameters()).dtype
        self.dtype = dtype

        shape = (self.n_layers, batch, self.n_heads, max_len, self.head_dim)
        self.k_cache = torch.zeros(shape, device=dev, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=dev, dtype=dtype)

        # RoPE table from the model's own rotary module (llama3 scaling included), every position.
        with torch.inference_mode():
            pos = torch.arange(max_len, device=dev)[None]
            cos, sin = tfmr.rotary_emb(torch.zeros(1, device=dev, dtype=dtype), pos)
        self.cos, self.sin = cos[0].clone(), sin[0].clone()  # (max_len, head_dim)
        self.arange = torch.arange(max_len, device=dev)
        self.neg = torch.finfo(dtype).min

        # Static inputs/outputs of the captured step.
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=dev)       # sampled speech token
        self.cache_pos = torch.zeros(1, dtype=torch.long, device=dev)    # where its K/V is written
        self.speech_pos = torch.zeros(1, 1, dtype=torch.long, device=dev)  # speech_pos_emb index
        self.logits = torch.zeros(batch, t3.hp.speech_tokens_dict_size, device=dev, dtype=dtype)
        self.guard = torch.zeros(len(LLAMA_ALIGNED_HEADS), max_len, device=dev, dtype=dtype)
        self.guard_layers = {layer: (i, head) for i, (layer, head) in enumerate(LLAMA_ALIGNED_HEADS)}

        # One graph per attention-length bucket: a step reads only the first L cache positions,
        # not all max_len (a full read is ~0.7 GB/step in fp32 -- as much as a third of the weights).
        self.buckets = sorted({b for b in BUCKETS if b < max_len} | {max_len})
        self.graphs = {}

    @torch.inference_mode()
    def _step(self, L: int):
        t3, B, H, D = self.t3, self.batch, self.n_heads, self.head_dim
        x = t3.speech_emb(self.tok) + t3.speech_pos_emb.emb(self.speech_pos)  # (1, 1, dim)
        x = torch.cat([x] * B)                                              # CFG: same embed twice
        cos = self.cos.index_select(0, self.cache_pos)[None, None]          # (1, 1, 1, D)
        sin = self.sin.index_select(0, self.cache_pos)[None, None]
        mask = torch.where(self.arange[:L] <= self.cache_pos, 0.0, self.neg).to(self.dtype)
        scale = 1.0 / math.sqrt(D)

        for li, layer in enumerate(self.tfmr.layers):
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            q = attn.q_proj(h).view(B, 1, H, D).transpose(1, 2)  # (B, H, 1, D)
            k = attn.k_proj(h).view(B, 1, H, D).transpose(1, 2)
            v = attn.v_proj(h).view(B, 1, H, D).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            self.k_cache[li].index_copy_(2, self.cache_pos, k)
            self.v_cache[li].index_copy_(2, self.cache_pos, v)
            w = torch.matmul(q, self.k_cache[li, :, :, :L].transpose(2, 3)) * scale + mask  # (B, H, 1, L)
            w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
            if li in self.guard_layers:
                gi, head = self.guard_layers[li]
                self.guard[gi, :L].copy_(w[0, head, 0])  # conditional batch row, as the stock hook reads
            o = torch.matmul(w, self.v_cache[li, :, :, :L]).transpose(1, 2).reshape(B, 1, H * D)
            x = x + attn.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        x = self.tfmr.norm(x)
        self.logits.copy_(t3.speech_head(x)[:, -1, :])

    def capture(self):
        pool = torch.cuda.graph_pool_handle()  # buckets never run concurrently: share memory
        for L in self.buckets:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):  # warm up allocator/kernels outside the graph (writes pos 0 only)
                    self._step(L)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self._step(L)
            self.graphs[L] = g
        torch.cuda.synchronize()

    def load_prefill(self, past) -> int:
        """Copy an HF cache (B, H, T, D per layer) into the static cache; return T. transformers
        4.46 returns the legacy tuple-of-(k, v) form when the call started without a cache."""
        if hasattr(past, "key_cache"):
            layers = list(zip(past.key_cache, past.value_cache))
        else:
            layers = [(k, v) for k, v in past]
        T = layers[0][0].shape[2]
        for li, (k, v) in enumerate(layers):
            self.k_cache[li, :, :, :T].copy_(k)
            self.v_cache[li, :, :, :T].copy_(v)
        return T

    def replay(self, token: torch.Tensor, cache_pos: int, speech_pos: int):
        self.tok.copy_(token.view(1, 1))
        self.cache_pos.fill_(cache_pos)
        self.speech_pos.fill_(speech_pos)
        L = next(b for b in self.buckets if cache_pos < b)
        self.graphs[L].replay()


@torch.inference_mode()
def fast_inference(self, *, t3_cond, text_tokens, initial_speech_tokens=None,
                   prepend_prompt_speech_tokens=None, num_return_sequences=1, max_new_tokens=None,
                   stop_on_eos=True, do_sample=True, temperature=0.8, top_p=0.95, min_p=0.05,
                   length_penalty=1.0, repetition_penalty=1.2, cfg_weight=0.5, _trace=None):
    """Drop-in for T3.inference (same signature). `_trace`, if a list, receives per-step
    (cfg-batch logits, guard rows) for the parity test."""
    dec: GraphT3Decoder = self._graph_decoder
    stock = self._stock_inference
    kwargs = dict(t3_cond=t3_cond, text_tokens=text_tokens, initial_speech_tokens=initial_speech_tokens,
                  prepend_prompt_speech_tokens=prepend_prompt_speech_tokens,
                  num_return_sequences=num_return_sequences, max_new_tokens=max_new_tokens,
                  stop_on_eos=stop_on_eos, do_sample=do_sample, temperature=temperature, top_p=top_p,
                  min_p=min_p, length_penalty=length_penalty,
                  repetition_penalty=repetition_penalty, cfg_weight=cfg_weight)

    # ---- identical to stock up to the prefill ----
    assert prepend_prompt_speech_tokens is None, "not implemented"
    t3_mod._ensure_BOT_EOT(text_tokens, self.hp)
    text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)
    if initial_speech_tokens is None:
        initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, len_cond = self.prepare_input_embeds(
        t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=initial_speech_tokens,
        cfg_weight=cfg_weight,
    )
    max_new_tokens = max_new_tokens or 1000  # generate() always passes 1000
    prefill_len = embeds.size(1) + 1  # + BOS
    if embeds.size(0) != dec.batch or prefill_len + max_new_tokens > dec.max_len:
        logger.warning(f"fast_t3: fallback to stock (batch {embeds.size(0)}, "
                       f"prefill {prefill_len} + {max_new_tokens} > {dec.max_len})")
        return stock(**kwargs)

    hooks_before = {li: set(l.self_attn._forward_hooks) for li, l in enumerate(self.tfmr.layers)}
    analyzer = AlignmentStreamAnalyzer(
        self.tfmr, None, text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
        alignment_layer_idx=9, eos_idx=self.hp.stop_speech_token,
    )
    patched_model = T3HuggingfaceBackend(
        config=self.cfg, llama=self.tfmr, speech_enc=self.speech_emb,
        speech_head=self.speech_head, alignment_stream_analyzer=analyzer,
    )
    try:
        device = embeds.device
        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token) + self.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = torch.cat([bos_embed, bos_embed])
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        generated_ids = bos_token.clone()
        predicted = []
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        output = patched_model(
            inputs_embeds=inputs_embeds, past_key_values=None, use_cache=True,
            output_attentions=True, output_hidden_states=True, return_dict=True,
        )
    finally:
        # The prefill filled analyzer.last_aligned_attns; nothing needs the hooks after that.
        for li, l in enumerate(self.tfmr.layers):
            for hid in set(l.self_attn._forward_hooks) - hooks_before[li]:
                del l.self_attn._forward_hooks[hid]

    T = dec.load_prefill(output.past_key_values)
    assert T == prefill_len
    logits_step = output.logits[:, -1, :]

    for i in range(max_new_tokens):
        if _trace is not None:
            _trace.append((logits_step.clone(),
                           [a.clone() for a in analyzer.last_aligned_attns]))
        # ---- per-token logic: same code and order as stock t3.py:364-401 ----
        cond = logits_step[0:1, :]
        uncond = logits_step[1:2, :]
        cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
        logits = cond + cfg * (cond - uncond)

        last_token = generated_ids[0, -1].item() if len(generated_ids[0]) > 0 else None
        logits = analyzer.step(logits, next_token=last_token)

        ids_for_proc = generated_ids[:1, ...]
        logits = repetition_penalty_processor(ids_for_proc, logits)
        if temperature != 1.0:
            logits = logits / temperature
        logits = min_p_warper(ids_for_proc, logits)
        logits = top_p_warper(ids_for_proc, logits)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        predicted.append(next_token)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if next_token.view(-1) == self.hp.stop_speech_token:
            break

        # ---- graph decode step instead of patched_model(...) ----
        pos = T + i
        dec.replay(next_token, cache_pos=pos, speech_pos=i + 1)
        rows = dec.guard[:, : pos + 1].cpu()  # the one host copy per token
        # Stock hook stores step_attention[0, head] with shape (1, Ti); only [:, i:j] is used.
        analyzer.last_aligned_attns = [rows[g: g + 1] for g in range(rows.size(0))]
        logits_step = dec.logits

    return torch.cat(predicted, dim=1)


def install_fast_t3(engine, max_len: int = MAX_LEN):
    """Swap engine.t3.inference for the graph version. Needs the merged fp32 model on cuda."""
    t3 = engine.t3
    if t3.device.type != "cuda":
        raise RuntimeError("fast_t3 needs the model on cuda")
    # Prefill needs attention maps for the guard; eager is what stock inference runs with.
    t3.tfmr.config._attn_implementation = "eager"
    t3._stock_inference = types.MethodType(type(t3).inference, t3)
    t3._graph_decoder = GraphT3Decoder(t3, max_len=max_len)
    t3._graph_decoder.capture()
    t3.inference = types.MethodType(fast_inference, t3)
    free, total = torch.cuda.mem_get_info()
    logger.info(f"fast_t3 installed (max_len {max_len}); GPU {(total - free) / 1e9:.2f} GB used")
    return t3._graph_decoder
