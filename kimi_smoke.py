#!/usr/bin/env python3
"""Smoke test for kimi_stream_infer.py using real layer weights (shards 1-2) + random embeds.
Exercises: config load, meta build, layer-0 dense forward, layer-1 MoE on-demand forward,
DynamicCache MLA latent update, rotary, causal mask. No embed_tokens/norm/lm_head needed.
"""
import os, sys, json, re, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from safetensors import safe_open
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM, DeepseekV3RotaryEmbedding

MODEL_DIR = "/data/model/Kimi-K2.7-Code"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight_packed$")

def get_tensor(shard, name):
    with safe_open(os.path.join(MODEL_DIR, shard), framework="pt", backend="mmap") as f:
        return f.get_tensor(name)

def dequant_int4(wpack, wscale, wshape):
    out, inp = wshape.tolist()
    u = torch.zeros((out, wpack.shape[1]*8), dtype=torch.int32, device=wpack.device)
    for i in range(8):
        u[:, i::8] = (wpack >> (4*i)) & 0xF
    u = u[:, :inp]; u = (u - 8).to(torch.int8)
    g = inp // 32
    deq = u.unflatten(-1, (g, 32)).float() * wscale.unsqueeze(-1).float()
    return deq.flatten(-2).to(DTYPE)

# --- load weight map (full checkpoint keys) ---
TP = "language_model."
with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
    wm = json.load(f)["weight_map"]
with open(f"{MODEL_DIR}/config.json") as f:
    cfg = DeepseekV3Config(**json.load(f)["text_config"])
cfg._attn_implementation = "eager"; cfg._experts_implementation = "eager"
print("cfg:", cfg.num_hidden_layers, "layers,", cfg.hidden_size, "hidden,", cfg.first_k_dense_replace, "dense", flush=True)

with torch.device("meta"):
    model = DeepseekV3ForCausalLM(cfg)
model.eval(); model.requires_grad_(False)
model.set_experts_implementation("eager")
with torch.device(DEVICE):
    model.model.rotary_emb = DeepseekV3RotaryEmbedding(cfg)

# random embeddings [B, S, H]
torch.manual_seed(0)
embeds = torch.randn(1, 4, cfg.hidden_size, device=DEVICE, dtype=DTYPE)
position_ids = torch.arange(4, device=DEVICE).unsqueeze(0)
mask = create_causal_mask(config=cfg, inputs_embeds=embeds, attention_mask=None,
                          past_key_values=None, position_ids=position_ids)
print("mask type:", type(mask).__name__, "shape:", tuple(mask.shape) if torch.is_tensor(mask) else None, flush=True)
pe = model.model.rotary_emb(embeds, position_ids=position_ids)
print("position_embeddings cos:", tuple(pe[0].shape), "sin:", tuple(pe[1].shape), flush=True)

cache = DynamicCache(config=cfg)
print("cache layers:", len(cache.layers), flush=True)

# ===== Layer 0 (dense) =====
keys0 = [(k, s) for k, s in wm.items() if re.match(rf"^{re.escape(TP)}model\.layers\.0\.", k)]
sd = {}
for k, shard in keys0:
    rest = k[len(TP + "model.layers.0."):]
    sd[rest] = get_tensor(shard, k).to(DEVICE)
model.model.layers[0].load_state_dict(sd, assign=True, strict=False)
print("layer0 loaded. missing:", [m for m in model.model.layers[0].load_state_dict(sd, assign=True, strict=False).missing_keys][:3] if False else "", flush=True)

h = embeds
layer = model.model.layers[0]
h = layer(h, attention_mask=mask, position_embeddings=pe, position_ids=position_ids, past_key_values=cache, use_cache=True)
print("layer0 out:", tuple(h.shape), "finite:", torch.isfinite(h).all().item(), "std:", h.std().item(), flush=True)
# cache updated?
l0 = cache.layers[0]
print("cache layer0 keys:", tuple(l0.keys.shape), "values:", tuple(l0.values.shape), flush=True)
print("  (expect [1,1,4,512] latent kv_nope + [1,1,4,64] k_rot)", flush=True)

# ===== Layer 1 (MoE on-demand) =====
keys1 = [(k, s) for k, s in wm.items() if re.match(rf"^{re.escape(TP)}model\.layers\.1\.", k)]
prefix = TP + "model.layers.1."
sd_ne = {}
for k, shard in keys1:
    rest = k[len(prefix):]
    if EXPERT_RE.match(rest) or rest.endswith((".weight_scale", ".weight_shape")):
        continue
    sd_ne[rest] = get_tensor(shard, k).to(DEVICE)
n_exp, inter, hdim = cfg.num_local_experts, cfg.moe_intermediate_size, cfg.hidden_size
sd_ne["mlp.experts.gate_up_proj"] = torch.zeros((n_exp, 2*inter, hdim), dtype=DTYPE, device=DEVICE)
sd_ne["mlp.experts.down_proj"] = torch.zeros((n_exp, hdim, inter), dtype=DTYPE, device=DEVICE)
model.model.layers[1].load_state_dict(sd_ne, assign=True, strict=False)
print("layer1 non-expert loaded", flush=True)

layer1 = model.model.layers[1]
residual = h
ln = layer1.input_layernorm(h)
attn_out, _ = layer1.self_attn(hidden_states=ln, attention_mask=mask, position_ids=position_ids,
                               past_key_values=cache, use_cache=True, position_embeddings=pe)
post_attn = residual + attn_out
print("attn out:", tuple(attn_out.shape), "finite:", torch.isfinite(attn_out).all().item(), flush=True)
l1 = cache.layers[1]
print("cache layer1 keys:", tuple(l1.keys.shape), "values:", tuple(l1.values.shape), flush=True)

mlp_in = layer1.post_attention_layernorm(post_attn)
_, topk_w, topk_i = layer1.mlp.gate(mlp_in)
print("router topk_i:", topk_i.tolist(), "topk_w:", topk_w.flatten().tolist()[:8], flush=True)
used = topk_i.reshape(-1).unique().tolist()
print("used experts:", used, flush=True)

# load selected experts (dequant int4)
gate_up = layer1.mlp.experts.gate_up_proj
down = layer1.mlp.experts.down_proj
used_set = set(used)
by_shard = {}
for k, shard in keys1:
    m = EXPERT_RE.match(k[len(prefix):])
    if m and int(m.group(1)) in used_set:
        by_shard.setdefault(shard, []).append((k, int(m.group(1)), m.group(2)))
t0 = time.time()
for shard, items in by_shard.items():
    with safe_open(os.path.join(MODEL_DIR, shard), framework="pt", backend="mmap") as f:
        for k, n, proj in items:
            wpack = f.get_tensor(k); wscale = f.get_tensor(k[:-len("weight_packed")] + "weight_scale")
            wshape = f.get_tensor(k[:-len("weight_packed")] + "weight_shape")
            wt = dequant_int4(wpack, wscale, wshape)
            if proj == "gate_proj": gate_up[n, :inter, :] = wt
            elif proj == "up_proj": gate_up[n, inter:, :] = wt
            else: down[n] = wt
print(f"loaded {len(used)} experts (dequant) in {time.time()-t0:.2f}s", flush=True)

orig = post_attn.shape
mlp_out = layer1.mlp.experts(mlp_in.reshape(-1, hdim), topk_i, topk_w)
mlp_out = mlp_out.reshape(orig) + layer1.mlp.shared_experts(mlp_in)
h1 = post_attn + mlp_out
print("layer1 out:", tuple(h1.shape), "finite:", torch.isfinite(h1).all().item(), "std:", h1.std().item(), flush=True)
print("SMOKE OK", flush=True)
