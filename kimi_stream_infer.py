#!/usr/bin/env python3
"""
Layer-by-layer streaming inference for KIMI K2.7 Code (KimiK25ForConditionalGeneration)
on Hygon K100AI DCU (ROCm), under GPU VRAM constraints.

Strategy
--------
The checkpoint (595 GB, 64 shards, INT4 group-quantized sparse experts) is too large to
keep resident. We construct the full text backbone structure (`DeepseekV3ForCausalLM`,
matching the checkpoint's `language_model` text subnetwork) on the `meta` device (no
memory cost), then per forward pass we stream each decoder layer's weights from the
safetensors shards, run the layer, and immediately move the layer back to `meta` +
`torch.cuda.empty_cache()` so only ~1 layer is resident at a time.

The checkpoint stores keys under a `language_model.` prefix because it is the text
backbone of a multimodal wrapper:
    language_model.model.layers.N.*   ->  model.layers.N.*
    language_model.model.embed_tokens.weight -> model.embed_tokens.weight
    language_model.model.norm.weight  ->  model.norm.weight
    language_model.lm_head.weight     ->  lm_head.weight
The repo's `DeepseekV3ForCausalLM` (which we build from the checkpoint `text_config`)
expects the stripped names, so the driver strips `language_model.` from every key.

Only the routed experts are quantized (compressed-tensors "pack-quantized", INT4,
group_size=32, symmetric). Every other tensor (attention, layernorms, router, shared
experts, dense layer-0 MLP, embeddings, lm_head) is bf16. Expert int4 dequant was
verified bit-identical to `compressed_tensors.compressors...dequantize`.

Two expert-loading modes (--experts):
  * `ondemand` (default): for sparse MoE layers, first load only the small non-expert
    weights + the top-k router, run the layer's attention + router to discover exactly
    which experts the router selects (8/token), then dequantize/load *only those* experts
    into the fused `gate_up_proj`/`down_proj` rows. ~48x less NFS I/O per sparse layer.
    Output is bit-identical to `full` because the eager expert forward only touches the
    router-selected expert rows.
  * `full`: load all 384 experts every time (kept for verification / comparison).

A tmpfs ramdisk (--ramcache, default /dev/shm/kimi_ram) caches the bf16 non-expert
weights of every layer (~20 GB total) so repeated decode passes skip the NFS read.

Attention cache: MLA compressed latents (kv_nope [B,1,S,512] + k_rot [B,1,S,64]) are
stored in a plain `DynamicCache(config=...)` (one `DynamicLayer` per layer) — exactly what
the repo `DeepseekV3Attention` writes via `past_key_values.update(kv_nope, k_rot, idx)`.

Generation: greedy argmax (temperature = 0), chat-template prompt "中国的首都是", N tokens.
Expected answer: "北京". Every decoded token is printed live.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

# Always use the repo's transformers source.
_REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from safetensors import safe_open  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config  # noqa: E402
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (  # noqa: E402
    DeepseekV3ForCausalLM,
    DeepseekV3RotaryEmbedding,
)

MODEL_DIR = "/data/model/Kimi-K2.7-Code"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
# The chat template ends the generation prompt with "<|im_assistant|>assistant<|im_middle|><think>"
# The model reasons inside <think>...</think> before answering, so the first generated token
# is the start of reasoning, not "北京". Correctness is judged by the decoded text containing 北京.
EOS_IDS = {163585, 163586, 163593}  # [EOS], <|im_end|>, [EOT]

PROMPT = "中国的首都是"
GEN_TOKENS = 12  # enough to see reasoning + answer

RAM_DIR = "/dev/shm/kimi_ram"
EXPERT_FUSED = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
INT4_GROUP = 32


def gpu_mem_gb():
    return torch.cuda.memory_allocated(DEVICE) / 1e9, torch.cuda.memory_reserved(DEVICE) / 1e9


def hy_smi():
    try:
        out = subprocess.run(["hy-smi"], capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return out.stderr.strip() or "(hy-smi error)"
        return "\n".join(out.stdout.strip().splitlines()[:30])
    except Exception:  # noqa: BLE001
        return "(hy-smi unavailable)"


TEXT_PREFIX = "language_model."


def load_weight_map():
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def build_model(config):
    """Construct the full text-backbone structure on the meta device (no GPU cost)."""
    with torch.device("meta"):
        model = DeepseekV3ForCausalLM(config)
    model.eval()
    # Without requires_grad=False the forward builds an autograd graph that retains each
    # layer's params, so `free_layer` cannot release their GPU memory (accumulates to OOM).
    model.requires_grad_(False)
    # Force the portable eager experts path (grouped_mm crashes on ROCm).
    model.set_experts_implementation("eager")
    # Rebuild rotary buffers on the real device (meta construction leaves them as meta tensors).
    with torch.device(DEVICE):
        model.model.rotary_emb = DeepseekV3RotaryEmbedding(config)
    return model


def load_persistent_weights(model, weight_map):
    """Keep the small shared modules resident: embeddings, final norm, lm_head."""
    embed_t = get_tensor(weight_map[TEXT_PREFIX + "model.embed_tokens.weight"], TEXT_PREFIX + "model.embed_tokens.weight").to(DEVICE)
    norm_t = get_tensor(weight_map[TEXT_PREFIX + "model.norm.weight"], TEXT_PREFIX + "model.norm.weight").to(DEVICE)
    lm_t = get_tensor(weight_map[TEXT_PREFIX + "lm_head.weight"], TEXT_PREFIX + "lm_head.weight").to(DEVICE)
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


def get_tensor(shard, name):
    path = os.path.join(MODEL_DIR, shard)
    with safe_open(path, framework="pt", backend="mmap") as f:
        return f.get_tensor(name)


EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight_packed$")


_NIBBLE_SHIFTS = None


def dequant_int4(wpack, wscale, wshape):
    """INT4 pack-quantized -> bf16, bit-identical to compressed_tensors (verified).

    wpack:   [out, in//8] int32, 8 int4 codes packed per int32 (code i in bits 4i..4i+3)
    wscale:  [out, in//group_size] bf16 per-group scale (symmetric, zero_point=0)
    wshape:  [2] int32 = [out, in]
    """
    global _NIBBLE_SHIFTS
    out, inp = wshape.tolist()
    if _NIBBLE_SHIFTS is None:
        # dtype=int32 keeps wpack >> shifts int32 (avoids int64 promotion that doubles
        # the [out, nw, 8] intermediate); nibble values 0-15 still fit exactly.
        _NIBBLE_SHIFTS = torch.arange(8, dtype=torch.int32, device=wpack.device) * 4
    # One broadcast-shift instead of 8 strided scatters: element [o, w*8+i] = nibble i of word w,
    # which is exactly the interleaving produced by the naive `unpacked[:, i::8] = ...` loop.
    unpacked = ((wpack.unsqueeze(-1) >> _NIBBLE_SHIFTS) & 0xF).reshape(out, wpack.shape[1] * 8)
    unpacked = unpacked[:, :inp]
    unpacked = (unpacked - 8).to(torch.int8)  # symmetric int4 range [-8, 7]
    groups = inp // INT4_GROUP
    deq = unpacked.unflatten(-1, (groups, INT4_GROUP)).float() * wscale.unsqueeze(-1).float()
    return deq.flatten(-2).to(DTYPE)


def ensure_ramdisk(path=RAM_DIR, size_gb=32):
    """Mount a tmpfs ramdisk at `path` if not already mounted. Returns path or None."""
    if os.path.ismount(path):
        print(f"ramdisk already mounted at {path}")
        return path
    os.makedirs(path, exist_ok=True)
    r = subprocess.run(
        ["mount", "-t", "tmpfs", "-o", f"size={size_gb}g", "tmpfs", path],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"ramdisk mount failed ({r.stderr.strip()}); continuing without ramcache")
        return None
    print(f"ramdisk mounted at {path} ({size_gb}G tmpfs)")
    return path


def _apply_sd(layer, sd, allow_missing_experts=False):
    """load_state_dict(assign=True, strict=False) + safety net that materializes any remaining meta param."""
    res = layer.load_state_dict(sd, assign=True, strict=False)
    if res.missing_keys:
        if allow_missing_experts:
            unexpected_missing = [k for k in res.missing_keys if k not in EXPERT_FUSED]
            if unexpected_missing:
                raise RuntimeError(f"layer missing keys: {unexpected_missing}")
        else:
            raise RuntimeError(f"layer missing keys: {res.missing_keys}")
    if res.unexpected_keys:
        raise RuntimeError(f"layer unexpected keys: {res.unexpected_keys[:10]}")
    # Safety net: materialize any remaining meta parameter as a fresh GPU zero Parameter.
    for name, p in layer.named_parameters():
        if p.is_meta:
            parts = name.split(".")
            mod = layer
            for part in parts[:-1]:
                mod = getattr(mod, part)
            setattr(mod, parts[-1], torch.nn.Parameter(torch.zeros(p.shape, dtype=DTYPE, device=DEVICE), requires_grad=False))


def load_non_expert(model, config, layer_idx, keys, ramcache):
    """Load a layer's non-expert weights (attention + norms + router + shared experts),
    optionally from a ramdisk cache. Expert params are left as GPU zeros for the on-demand
    path (filled later by load_experts_used). `keys` carries full checkpoint keys (with the
    `language_model.` prefix); `rest` strips it to match the repo model's state dict."""
    prefix = f"{TEXT_PREFIX}model.layers.{layer_idx}."
    sd = None
    if ramcache:
        p = os.path.join(ramcache, f"layer{layer_idx}_ne.pt")
        if os.path.exists(p):
            try:
                sd = torch.load(p, map_location=DEVICE, weights_only=True)
            except Exception:  # noqa: BLE001
                sd = None

    if sd is None:
        by_shard = {}
        for k, shard in keys:
            rest = k[len(prefix):]
            if EXPERT_RE.match(rest) or rest.endswith((".weight_scale", ".weight_shape")):
                continue
            by_shard.setdefault(shard, []).append((k, rest))
        sd = {}
        with torch.no_grad():
            for shard, items in by_shard.items():
                path = os.path.join(MODEL_DIR, shard)
                with safe_open(path, framework="pt", backend="mmap") as f:
                    for k, rest in items:
                        t = f.get_tensor(k)
                        sd[rest] = t.to(DEVICE)
        if ramcache:
            try:
                os.makedirs(ramcache, exist_ok=True)
                torch.save(sd, os.path.join(ramcache, f"layer{layer_idx}_ne.pt"))
            except OSError:
                pass  # ramdisk full/unavailable -> run without the cache
    # Sparse layers: materialize the expert params as GPU zeros so the (unused) rows are
    # defined; load_experts_used fills only the router-selected rows afterwards.
    if layer_idx >= config.first_k_dense_replace:
        n_exp, inter, h = config.num_local_experts, config.moe_intermediate_size, config.hidden_size
        sd["mlp.experts.gate_up_proj"] = torch.zeros((n_exp, 2 * inter, h), dtype=DTYPE, device=DEVICE)
        sd["mlp.experts.down_proj"] = torch.zeros((n_exp, h, inter), dtype=DTYPE, device=DEVICE)
    layer = model.model.layers[layer_idx]
    _apply_sd(layer, sd, allow_missing_experts=True)
    return layer


def load_experts_used(model, config, layer_idx, keys, used):
    """Dequantize only the router-selected experts and write them into the layer's
    gate_up_proj / down_proj rows (already materialized as zeros by load_non_expert)."""
    prefix = f"{TEXT_PREFIX}model.layers.{layer_idx}."
    layer = model.model.layers[layer_idx]
    gate_up = layer.mlp.experts.gate_up_proj
    down = layer.mlp.experts.down_proj
    inter = config.moe_intermediate_size
    used_set = set(used)
    by_shard = {}
    for k, shard in keys:
        m = EXPERT_RE.match(k[len(prefix):])
        if m and int(m.group(1)) in used_set:
            by_shard.setdefault(shard, []).append((k, int(m.group(1)), m.group(2)))
    with torch.no_grad():
        for shard, items in by_shard.items():
            path = os.path.join(MODEL_DIR, shard)
            with safe_open(path, framework="pt", backend="mmap") as f:
                for k, n, proj in items:
                    wpack = f.get_tensor(k)
                    wscale = f.get_tensor(k[:-len("weight_packed")] + "weight_scale")
                    wshape = f.get_tensor(k[:-len("weight_packed")] + "weight_shape")
                    wt = dequant_int4(wpack, wscale, wshape)
                    if proj == "gate_proj":
                        gate_up[n, :inter, :] = wt
                    elif proj == "up_proj":
                        gate_up[n, inter:, :] = wt
                    else:  # down_proj
                        down[n] = wt


def load_layer(model, config, layer_idx, keys):
    """Full layer load (all 384 experts) -- the `full` mode, kept for verification."""
    prefix = f"{TEXT_PREFIX}model.layers.{layer_idx}."
    by_shard = {}
    for k, shard in keys:
        by_shard.setdefault(shard, []).append(k)

    sd = {}
    n_experts = config.num_local_experts
    gate_up = down = None
    expert_seen = False
    inter = config.moe_intermediate_size
    h = config.hidden_size

    with torch.no_grad():
        for shard, skeys in by_shard.items():
            path = os.path.join(MODEL_DIR, shard)
            with safe_open(path, framework="pt", backend="mmap") as f:
                for k in skeys:
                    rest = k[len(prefix):]
                    if rest.endswith((".weight_scale", ".weight_shape")):
                        continue
                    m = EXPERT_RE.match(rest)
                    if m:
                        n, proj = int(m.group(1)), m.group(2)
                        if not expert_seen:
                            expert_seen = True
                            gate_up = torch.empty((n_experts, 2 * inter, h), dtype=DTYPE, device=DEVICE)
                            down = torch.empty((n_experts, h, inter), dtype=DTYPE, device=DEVICE)
                        wpack = f.get_tensor(k)
                        wscale = f.get_tensor(k[:-len("weight_packed")] + "weight_scale")
                        wshape = f.get_tensor(k[:-len("weight_packed")] + "weight_shape")
                        wt = dequant_int4(wpack, wscale, wshape)
                        if proj == "gate_proj":
                            gate_up[n, :inter, :] = wt
                        elif proj == "up_proj":
                            gate_up[n, inter:, :] = wt
                        else:  # down_proj
                            down[n] = wt
                    else:
                        sd[rest] = f.get_tensor(k).to(DEVICE)

        if expert_seen:
            sd["mlp.experts.gate_up_proj"] = gate_up
            sd["mlp.experts.down_proj"] = down

        layer = model.model.layers[layer_idx]
        _apply_sd(layer, sd, allow_missing_experts=False)
    return layer


def free_layer(model, layer_idx):
    model.model.layers[layer_idx].to("meta")
    torch.cuda.empty_cache()


def forward_pass(model, config, cache, input_ids, keys_by_layer, experts_mode, ramcache, max_layers=None):
    """Replicates DeepseekV3Model.forward + lm_head for the last position, streaming layers.
    For `ondemand` sparse layers the router is run first and only its selected experts load."""
    if max_layers is None:
        max_layers = config.num_hidden_layers
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        past_len = cache.get_seq_length()
        position_ids = (torch.arange(input_ids.shape[1], device=DEVICE) + past_len).unsqueeze(0)
        mask = create_causal_mask(
            config=config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=cache,
            position_ids=position_ids,
        )
        position_embeddings = model.model.rotary_emb(hidden, position_ids=position_ids)
        for i in range(max_layers):
            layer = model.model.layers[i]
            is_moe = i >= config.first_k_dense_replace
            if is_moe and experts_mode == "ondemand":
                t0 = time.time()
                load_non_expert(model, config, i, keys_by_layer[i], ramcache)
                ne_t = time.time() - t0
                residual = hidden
                ln = layer.input_layernorm(hidden)
                attn_out, _ = layer.self_attn(
                    hidden_states=ln,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
                post_attn = residual + attn_out
                mlp_in = layer.post_attention_layernorm(post_attn)
                # Route first, then load exactly the experts the router selected.
                _, topk_w, topk_i = layer.mlp.gate(mlp_in)
                used = topk_i.reshape(-1).unique().tolist()
                t1 = time.time()
                load_experts_used(model, config, i, keys_by_layer[i], used)
                exp_t = time.time() - t1
                orig = post_attn.shape
                mlp_out = layer.mlp.experts(mlp_in.reshape(-1, config.hidden_size), topk_i, topk_w)
                mlp_out = mlp_out.reshape(orig) + layer.mlp.shared_experts(mlp_in)
                hidden = post_attn + mlp_out
                a, _ = gpu_mem_gb()
                print(f"    layer {i:2d} (moe, {len(used)} experts): ne={ne_t:4.1f}s exp={exp_t:4.1f}s | gpu alloc={a:.1f}GB", flush=True)
            else:
                t0 = time.time()
                load_layer(model, config, i, keys_by_layer[i])
                print(f"    loaded layer {i:2d} in {time.time() - t0:5.1f}s", flush=True)
                hidden = layer(
                    hidden,
                    attention_mask=mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
            free_layer(model, i)
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden[:, -1:, :])
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--tokens", type=int, default=GEN_TOKENS)
    ap.add_argument("--max-layers", type=int, default=61, help="for smoke test only")
    ap.add_argument("--no-decode", action="store_true", help="prefill only")
    ap.add_argument("--experts", choices=["ondemand", "full"], default="ondemand",
                    help="load only router-selected experts (ondemand) or all 384 (full)")
    ap.add_argument("--ramcache", default=RAM_DIR, nargs="?", const=RAM_DIR,
                    help="tmpfs ramdisk dir caching non-expert layer weights (default /dev/shm/kimi_ram)")
    ap.add_argument("--no-ramcache", action="store_true", help="disable ramdisk cache")
    args = ap.parse_args()
    ramcache = None if args.no_ramcache else ensure_ramdisk(args.ramcache)

    print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)} total_mem={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.set_device(DEVICE)
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        text_config = json.load(f)["text_config"]
    config = DeepseekV3Config(**text_config)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    print(f"config: layers={config.num_hidden_layers} experts={config.num_local_experts} "
          f"moe_intermediate={config.moe_intermediate_size} hidden={config.hidden_size} "
          f"experts_mode={args.experts} ramcache={'off' if ramcache is None else ramcache}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    print(f"prompt ids ({prompt_ids.shape[1]} tokens): {prompt_ids.tolist()}")
    print(f"decoded prompt: {tokenizer.decode(prompt_ids[0].tolist())!r}")

    model = build_model(config)
    weight_map = load_weight_map()

    # Precompute per-layer key lists once (full checkpoint keys, `language_model.` prefix intact).
    keys_by_layer = {}
    for k, shard in weight_map.items():
        m = re.match(rf"^{re.escape(TEXT_PREFIX)}model\.layers\.(\d+)\.", k)
        if m and int(m.group(1)) < config.num_hidden_layers:
            keys_by_layer.setdefault(int(m.group(1)), []).append((k, shard))

    print("\n[1/3] loading persistent weights ...")
    load_persistent_weights(model, weight_map)

    cache = DynamicCache(config=config)
    print("cache layers:", len(cache.layers), type(cache.layers[0]).__name__)

    # Warmup a dummy bf16 GEMM to trigger rocBLAS init outside the timed loop.
    _ = torch.empty((512, 512), device=DEVICE, dtype=DTYPE).normal_()
    _ = _ @ torch.empty((512, 512), device=DEVICE, dtype=DTYPE).normal_()
    torch.cuda.synchronize()

    generated = []
    gen_pieces = []
    n_loads = 0
    start = time.time()
    try:
        for step in range(args.tokens):
            cur = prompt_ids if step == 0 else torch.tensor([[generated[-1]]], device=DEVICE)
            t0 = time.time()
            logits = forward_pass(
                model, config, cache, cur, keys_by_layer,
                experts_mode=args.experts, ramcache=ramcache, max_layers=args.max_layers,
            )
            torch.cuda.synchronize()
            n_loads += args.max_layers
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())  # temperature = 0 (greedy)
            generated.append(next_id)
            piece = tokenizer.decode([next_id])
            gen_pieces.append(piece)
            dt = time.time() - t0
            a, r = gpu_mem_gb()
            print(f"\n>>> STEP {step}: token_id={next_id} token={piece!r} "
                  f"text_so_far={''.join(gen_pieces)!r} | {dt:6.1f}s | gpu alloc={a:.1f}GB resv={r:.1f}GB", flush=True)
            if next_id in EOS_IDS:
                print("  (EOS reached)")
                break
            if args.no_decode:
                break
    finally:
        a, r = gpu_mem_gb()
        print(f"\n=== DONE in {(time.time() - start) / 60:.1f} min | ~{n_loads} layer loads | gpu alloc={a:.1f}GB ===")

    print("\n=== RESULTS ===")
    print(f"prompt: {args.prompt!r}")
    print(f"generated token ids: {generated}")
    print(f"generated text: {tokenizer.decode(generated)!r}")
    full = tokenizer.decode(prompt_ids[0].tolist() + generated)
    print(f"prompt+generated: {full!r}")
    if generated:
        text_so_far = "".join(gen_pieces)
        print(f"contains 北京: {'北京' in text_so_far or '北京' in full}")

    print("\n=== hy-smi (final GPU state) ===")
    print(hy_smi())


if __name__ == "__main__":
    main()
