#!/usr/bin/env python3
"""
Layer-by-layer streaming inference for GLM-5.2-Channel-INT8-w8a8 (GlmMoeDsaForCausalLM)
on Hygon K100AI DCU (ROCm), under GPU VRAM constraints.

Strategy
--------
The model (705 GB checkpoint, INT8 channel-quantized) is too large to keep resident.
We construct the *full* model structure on the `meta` device (no memory cost), then
per forward pass we stream each decoder layer's weights from the safetensors shards,
dequantize int8->bf16 on the GPU, run the layer, and immediately move the layer back
to `meta` + `torch.cuda.empty_cache()` so only ~1 layer is resident at a time.

The forward is replicated manually (not via model.forward) so we can load/free layers
inside the loop. The manual loop is bit-identical to GlmMoeDsaModel.forward.

Generation: greedy argmax (temperature = 0), prompt "中国的首都是", 2 tokens.
Expected first token: "北京" (vocab id 99334).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

# Always use the repo's transformers source (site-packages ships an older version whose
# GlmMoeDsaConfig does not override `head_dim` -> mismatched MLA shapes for this checkpoint).
_REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from safetensors import safe_open  # noqa: E402

from transformers import AutoTokenizer, GlmMoeDsaConfig, GlmMoeDsaForCausalLM  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (  # noqa: E402
    GlmMoeDsaRotaryEmbedding,
)

MODEL_DIR = "/data/model/hygon/GLM-5.2-Channel-INT8-w8a8"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
EOS_IDS = {154820, 154827, 154829}
EXPECTED_FIRST_TOKEN_ID = 99334  # "北京"

PROMPT = "中国的首都是"
GEN_TOKENS = 2


def gpu_mem_gb():
    return torch.cuda.memory_allocated(DEVICE) / 1e9, torch.cuda.memory_reserved(DEVICE) / 1e9


def hy_smi():
    try:
        out = subprocess.run(["hy-smi"], capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return out.stderr.strip() or "(hy-smi error)"
        return "\n".join(out.stdout.strip().splitlines()[:40])
    except Exception as e:  # noqa: BLE001
        return f"(hy-smi unavailable: {e})"


def load_weight_map():
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def get_tensor(shard, name):
    path = os.path.join(MODEL_DIR, shard)
    with safe_open(path, framework="pt", backend="mmap") as f:
        return f.get_tensor(name)


def build_model(config):
    """Construct the full model structure on the meta device (no GPU cost)."""
    with torch.device("meta"):
        model = GlmMoeDsaForCausalLM(config)
    model.eval()
    # Critical: without requires_grad=False the forward builds an autograd graph that retains each
    # layer's params, so `free_layer` cannot release their GPU memory (accumulates to OOM).
    model.requires_grad_(False)
    # Force the experts implementation to the portable eager path (grouped_mm crashes on ROCm).
    model.set_experts_implementation("eager")
    # Rebuild rotary buffers on the real device (meta construction leaves them as meta tensors).
    with torch.device(DEVICE):
        model.model.rotary_emb = GlmMoeDsaRotaryEmbedding(config)
    return model


def load_persistent_weights(model, weight_map):
    """Keep the small, shared modules resident: embeddings, final norm, lm_head."""
    # Replace the meta Parameters with real GPU tensors. Direct `.data =` assignment
    # fails on meta Parameters in this torch build, so we swap in fresh Parameters.
    embed_t = get_tensor(weight_map["model.embed_tokens.weight"], "model.embed_tokens.weight").to(DEVICE)
    norm_t = get_tensor(weight_map["model.norm.weight"], "model.norm.weight").to(DEVICE)
    lm_t = get_tensor(weight_map["lm_head.weight"], "lm_head.weight").to(DEVICE)
    model.model.embed_tokens.weight = torch.nn.Parameter(embed_t, requires_grad=False)
    model.model.norm.weight = torch.nn.Parameter(norm_t, requires_grad=False)
    model.lm_head.weight = torch.nn.Parameter(lm_t, requires_grad=False)
    for name, t in [
        ("model.embed_tokens.weight", embed_t),
        ("model.norm.weight", norm_t),
        ("lm_head.weight", lm_t),
    ]:
        print(f"  resident {name}: {tuple(t.shape)} {t.dtype}")
    torch.cuda.empty_cache()


EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def dequantize(w, scale):
    """Per-channel int8 -> bf16: (w8.to(fp32) * scale.to(fp32)).to(bf16)."""
    return (w.to(DEVICE).to(torch.float32) * scale.to(DEVICE).to(torch.float32)).to(DTYPE)


def load_layer(model, config, layer_idx, weight_map):
    """Stream one decoder layer's weights from its shards onto the GPU."""
    prefix = f"model.layers.{layer_idx}."
    keys = [k for k in weight_map if k.startswith(prefix)]
    by_shard = {}
    for k in keys:
        by_shard.setdefault(weight_map[k], []).append(k)

    sd = {}
    n_experts = config.n_routed_experts
    gate_up = down = None
    expert_seen = False

    with torch.no_grad():
        for shard, skeys in by_shard.items():
            path = os.path.join(MODEL_DIR, shard)
            with safe_open(path, framework="pt", backend="mmap") as f:
                for k in skeys:
                    rest = k[len(prefix):]
                    if rest.endswith(".weight_scale"):
                        continue
                    m = EXPERT_RE.match(rest)
                    t = f.get_tensor(k)
                    if m:
                        n, proj = int(m.group(1)), m.group(2)
                        if not expert_seen:
                            expert_seen = True
                            h, inter = config.hidden_size, config.moe_intermediate_size
                            gate_up = torch.empty((n_experts, 2 * inter, h), dtype=DTYPE, device=DEVICE)
                            down = torch.empty((n_experts, h, inter), dtype=DTYPE, device=DEVICE)
                        if t.dtype == torch.int8:
                            wt = dequantize(t, f.get_tensor(k + "_scale"))
                        else:
                            wt = t.to(DEVICE)
                        if proj == "gate_proj":
                            gate_up[n, : config.moe_intermediate_size, :] = wt
                        elif proj == "up_proj":
                            gate_up[n, config.moe_intermediate_size:, :] = wt
                        else:  # down_proj
                            down[n] = wt
                    else:
                        if t.dtype == torch.int8:
                            wt = dequantize(t, f.get_tensor(k + "_scale"))
                        else:
                            wt = t.to(DEVICE)
                        if rest == "mlp.gate.e_score_correction_bias":
                            wt = wt.to(torch.float32)
                        sd[rest] = wt

        if expert_seen:
            sd["mlp.experts.gate_up_proj"] = gate_up
            sd["mlp.experts.down_proj"] = down

        layer = model.model.layers[layer_idx]
        res = layer.load_state_dict(sd, assign=True)
        if res.missing_keys:
            raise RuntimeError(f"layer {layer_idx} missing keys: {res.missing_keys}")
        if res.unexpected_keys:
            raise RuntimeError(f"layer {layer_idx} unexpected keys: {res.unexpected_keys[:10]}")
        # Safety net: any parameter still on meta (should not happen) -> move to GPU zeros.
        for p in layer.parameters():
            if p.is_meta:
                p.data = torch.zeros(p.shape, dtype=DTYPE, device=DEVICE)
    return layer


def free_layer(model, layer_idx):
    model.model.layers[layer_idx].to("meta")
    torch.cuda.empty_cache()


def forward_pass(model, config, cache, input_ids, load_cb, free_cb, max_layers=None):
    """Replicates GlmMoeDsaModel.forward + lm_head for the last position."""
    if max_layers is None:
        max_layers = config.num_hidden_layers
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        past_len = cache.get_seq_length()
        position_ids = (torch.arange(input_ids.shape[1], device=DEVICE) + past_len).unsqueeze(0)
        mask_dict = {
            "deepseek_sparse_attention": create_causal_mask(
                config=config,
                inputs_embeds=hidden,
                attention_mask=None,
                past_key_values=cache,
                position_ids=position_ids,
            )
        }
        position_embeddings = model.model.rotary_emb(hidden, position_ids=position_ids)
        topk_indices = None
        for i in range(max_layers):
            layer = model.model.layers[i]
            load_cb(i)
            hidden, topk_indices = layer(
                hidden,
                attention_mask=mask_dict["deepseek_sparse_attention"],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                prev_topk_indices=topk_indices,
            )
            free_cb(i)
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden[:, -1:, :])
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--tokens", type=int, default=GEN_TOKENS)
    ap.add_argument("--max-layers", type=int, default=78, help="for smoke test only")
    ap.add_argument("--no-decode", action="store_true", help="prefill only")
    args = ap.parse_args()

    print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)} total_mem={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.set_device(DEVICE)
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    config = GlmMoeDsaConfig.from_pretrained(MODEL_DIR)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    print(f"config: layers={config.num_hidden_layers} experts={config.n_routed_experts} "
          f"indexer_topk={config.index_topk} head_dim={config.head_dim}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    print(f"prompt ids ({prompt_ids.shape[1]} tokens): {prompt_ids.tolist()}")
    print(f"decoded prompt: {tokenizer.decode(prompt_ids[0])!r}")

    model = build_model(config)
    weight_map = load_weight_map()

    print("\n[1/3] loading persistent weights ...")
    load_persistent_weights(model, weight_map)

    cache = DynamicCache(config=config)
    print("cache layers:", len(cache.layers), type(cache.layers[0]).__name__)

    # Warmup: a dummy bf16 GEMM to trigger rocBLAS init outside the timed loop.
    _ = torch.empty((512, 512), device=DEVICE, dtype=DTYPE).normal_()
    _ = _ @ torch.empty((512, 512), device=DEVICE, dtype=DTYPE).normal_()
    torch.cuda.synchronize()

    loaded = set()
    n_loads = 0

    def load_cb(i):
        nonlocal n_loads
        if i not in loaded:
            t0 = time.time()
            load_layer(model, config, i, weight_map)
            n_loads += 1
            loaded.add(i)
            if args.max_layers < 78 or n_loads % 10 == 0 or i in (0, 3, config.num_hidden_layers - 1):
                a, r = gpu_mem_gb()
                print(f"    loaded layer {i:2d} in {time.time() - t0:5.1f}s | gpu alloc={a:.1f}GB resv={r:.1f}GB")

    def free_cb(i):
        if i in loaded:
            free_layer(model, i)
            loaded.discard(i)
            if args.max_layers < 78:
                a, r = gpu_mem_gb()
                print(f"    freed  layer {i:2d} | gpu alloc={a:.1f}GB resv={r:.1f}GB")

    generated = []
    start = time.time()
    try:
        for step in range(args.tokens):
            cur = prompt_ids if step == 0 else torch.tensor([[generated[-1]]], device=DEVICE)
            t0 = time.time()
            logits = forward_pass(model, config, cache, cur, load_cb, free_cb, max_layers=args.max_layers)
            torch.cuda.synchronize()
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())  # temperature = 0 (greedy)
            generated.append(next_id)
            a, r = gpu_mem_gb()
            print(f"step {step}: token={next_id} decoded={tokenizer.decode([next_id])!r} "
                  f"| {time.time() - t0:6.1f}s | gpu alloc={a:.1f}GB resv={r:.1f}GB")
            if next_id in EOS_IDS:
                print("  (EOS reached)")
                break
            if args.no_decode:
                break
    finally:
        a, r = gpu_mem_gb()
        print(f"\n=== DONE in {(time.time() - start) / 60:.1f} min | {n_loads} layer loads | gpu alloc={a:.1f}GB ===")

    print("\n=== RESULTS ===")
    print(f"prompt: {args.prompt!r}")
    print(f"generated token ids: {generated}")
    print(f"generated text: {tokenizer.decode(generated)!r}")
    if generated:
        first_ok = generated[0] == EXPECTED_FIRST_TOKEN_ID
        print(f"first token decoded: {tokenizer.decode([generated[0]])!r} (expected '北京', id {EXPECTED_FIRST_TOKEN_ID}) -> {'OK' if first_ok else 'MISMATCH'}")
        if len(generated) >= 2:
            print(f"second token decoded: {tokenizer.decode([generated[1]])!r}")

    print("\n=== hy-smi (final GPU state) ===")
    print(hy_smi())


if __name__ == "__main__":
    main()
