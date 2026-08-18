#!/usr/bin/env python3
"""Reproducible golden runner for a trimmed GLM-5.2 W8A8 DSA checkpoint.

Semantic tensors are saved as fp32 CPU tensors with explicit ``[B,S,H]``
layout. Int8 weights are always manually dequantized; incomplete state fails.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from transformers import AutoTokenizer, GlmMoeDsaConfig, GlmMoeDsaForCausalLM  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.masking_utils import create_causal_mask  # noqa: E402
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding  # noqa: E402


MODEL_DIR = "/data/model/hygon/GLM-5.2-Channel-INT8-w8a8-trimed-fake-tp8/"
DEVICE, DTYPE = "cuda:0", torch.bfloat16
PERSISTENT = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
EOS_IDS = {154820, 154827, 154829}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor):
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproducible GLM-5.2 fake-TP8 DSA golden runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--input-mode", choices=("raw", "prompt"), default="raw")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="中国的首都是")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--num-layers", "--max-layers", dest="num_layers", type=int, default=2)
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument("--experts", choices=("ondemand", "full"), default="ondemand")
    parser.add_argument("--expert-validation-limit", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--dump-dir", default="/tmp/golden_trim_dumps")
    parser.add_argument(
        "--no-ramcache", action="store_true", help="Accepted for compatibility; cache is always disabled"
    )
    parser.add_argument("--ramcache", default=None, help="Accepted for compatibility; cache is always disabled")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def raw_ids(seq_len, seed, vocab_size):
    if seq_len < 1:
        raise ValueError("--seq-len must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    low = min(100, vocab_size - 1)
    ids = torch.randint(low, vocab_size, (1, seq_len), generator=generator)
    if ids.min() < 0 or ids.max() >= vocab_size:
        raise RuntimeError("raw generator produced an invalid token ID")
    return ids


def input_ids(args, config):
    if args.input_mode == "raw":
        return raw_ids(args.seq_len, args.seed, config.vocab_size), None
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt").input_ids, tokenizer


def open_checkpoint(model_dir):
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    for path in (config_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(f"required file not found: {path}")
    config = GlmMoeDsaConfig.from_pretrained(model_dir, local_files_only=True)
    with index_path.open() as handle:
        weight_map = json.load(handle).get("weight_map")
    if not weight_map:
        raise RuntimeError("checkpoint index has no weight_map")
    return config, weight_map, config_path, index_path


def validate_config(config, num_layers):
    if not 1 <= num_layers <= config.num_hidden_layers:
        raise ValueError(f"--num-layers must be in [1,{config.num_hidden_layers}]")
    for name in ("mlp_layer_types", "indexer_types", "layer_types"):
        value = getattr(config, name, None)
        if not isinstance(value, list) or len(value) != config.num_hidden_layers:
            raise RuntimeError(f"config.{name} length does not equal num_hidden_layers")
    if config.indexer_types[0] == "shared":
        raise RuntimeError("layer 0 cannot use a shared DSA index")


def inspect_manifest(model_dir, weight_map, num_layers):
    selected = set(PERSISTENT)
    selected.update(
        name for name in weight_map if (match := LAYER_RE.match(name)) and int(match.group(1)) < num_layers
    )
    absent = selected - set(weight_map)
    if absent:
        raise RuntimeError(f"missing selected keys: {sorted(absent)}")
    by_shard = {}
    for name in selected:
        by_shard.setdefault(weight_map[name], []).append(name)
    manifest = {}
    for shard, names in sorted(by_shard.items()):
        path = model_dir / shard
        if not path.is_file():
            raise FileNotFoundError(f"missing shard: {path}")
        with safe_open(path, framework="pt", backend="mmap") as handle:
            missing = set(names) - set(handle.keys())
            if missing:
                raise RuntimeError(f"indexed keys absent from {shard}: {sorted(missing)[:10]}")
            for name in names:
                tensor = handle.get_tensor(name)
                manifest[name] = {
                    "shard": shard,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                }
    for name, info in manifest.items():
        if info["dtype"] == "torch.int8" and name + "_scale" not in manifest:
            raise RuntimeError(f"int8 tensor lacks selected scale: {name}")
        if name.endswith(".weight_scale"):
            base = name.removesuffix("_scale")
            if base not in manifest or manifest[base]["dtype"] != "torch.int8":
                raise RuntimeError(f"orphan/invalid scale: {name}")
            if info["numel"] not in (1, manifest[base]["shape"][0]):
                raise RuntimeError(f"unsupported scale dimensions: {name} {info['shape']}")
    return manifest


def expected_shapes(config, num_layers):
    with torch.device("meta"):
        model = GlmMoeDsaForCausalLM(config)
    expected = {}
    for index in range(num_layers):
        prefix = f"model.layers.{index}."
        for name, tensor in model.model.layers[index].state_dict().items():
            if name not in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                expected[prefix + name] = list(tensor.shape)
        if config.mlp_layer_types[index] == "sparse":
            for expert in range(config.n_routed_experts):
                expected[f"{prefix}mlp.experts.{expert}.gate_proj.weight"] = [
                    config.moe_intermediate_size,
                    config.hidden_size,
                ]
                expected[f"{prefix}mlp.experts.{expert}.up_proj.weight"] = [
                    config.moe_intermediate_size,
                    config.hidden_size,
                ]
                expected[f"{prefix}mlp.experts.{expert}.down_proj.weight"] = [
                    config.hidden_size,
                    config.moe_intermediate_size,
                ]
    return expected


def preflight(args):
    model_dir = Path(args.model_dir).resolve()
    config, weight_map, config_path, index_path = open_checkpoint(model_dir)
    validate_config(config, args.num_layers)
    ids, tokenizer = input_ids(args, config)
    if ids.shape[1] > config.max_position_embeddings:
        raise ValueError("input exceeds max_position_embeddings")
    manifest = inspect_manifest(model_dir, weight_map, args.num_layers)
    expected = expected_shapes(config, args.num_layers)
    actual = {name for name in manifest if LAYER_RE.match(name) and not name.endswith(".weight_scale")}
    if actual != set(expected):
        raise RuntimeError(
            f"selected parameter mismatch: missing={sorted(set(expected) - actual)[:10]} unexpected={sorted(actual - set(expected))[:10]}"
        )
    wrong = {
        name: (manifest[name]["shape"], shape) for name, shape in expected.items() if manifest[name]["shape"] != shape
    }
    if wrong:
        raise RuntimeError(f"selected shape mismatch: {list(wrong.items())[:10]}")
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    metadata = {
        "script_sha256": file_hash(__file__),
        "transformers_git_commit": commit,
        "config_sha256": file_hash(config_path),
        "checkpoint_index_sha256": file_hash(index_path),
        "torch_version": torch.__version__,
        "rocm_version": getattr(torch.version, "hip", None),
        "seed": args.seed,
        "input_ids_sha256": tensor_hash(ids),
        "model_path": str(model_dir),
        "input_mode": args.input_mode,
        "seq_len": ids.shape[1],
        "num_layers": args.num_layers,
    }
    keys = {
        i: sorted((name, weight_map[name]) for name in weight_map if name.startswith(f"model.layers.{i}."))
        for i in range(args.num_layers)
    }
    report = {
        "status": "PASS",
        "metadata": metadata,
        "config": {
            "num_hidden_layers": config.num_hidden_layers,
            "selected_num_layers": args.num_layers,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "index_topk": config.index_topk,
            "mlp_layer_types": config.mlp_layer_types,
            "indexer_types": config.indexer_types,
            "layer_types": config.layer_types,
        },
        "input": {
            "shape": list(ids.shape),
            "dtype": str(ids.dtype),
            "min": int(ids.min()),
            "max": int(ids.max()),
            "sha256": tensor_hash(ids),
        },
        "checkpoint_manifest": manifest,
        "selected_checkpoint_tensor_count": len(manifest),
    }
    return config, weight_map, report, ids, tokenizer, keys


def checkpoint_tensor(model_dir, weight_map, name):
    if name not in weight_map:
        raise RuntimeError(f"unindexed checkpoint tensor: {name}")
    with safe_open(Path(model_dir) / weight_map[name], framework="pt", backend="mmap") as handle:
        return handle.get_tensor(name)


def materialize(model_dir, weight_map, name):
    value = checkpoint_tensor(model_dir, weight_map, name)
    if value.dtype == torch.int8:
        scale = checkpoint_tensor(model_dir, weight_map, name + "_scale")
        if scale.numel() not in (1, value.shape[0]):
            raise RuntimeError(f"unsupported scale for {name}")
        value = (value.to(DEVICE, torch.float32) * scale.to(DEVICE, torch.float32)).to(DTYPE)
    else:
        value = value.to(DEVICE)
    if not torch.isfinite(value).all():
        raise RuntimeError(f"non-finite loaded tensor: {name}")
    return value


def build_model(config, weight_map, model_dir):
    with torch.device("meta"):
        model = GlmMoeDsaForCausalLM(config)
    model.eval().requires_grad_(False)
    model.set_experts_implementation("eager")
    with torch.device(DEVICE):
        model.model.rotary_emb = GlmMoeDsaRotaryEmbedding(config)
    embed, norm, head = (materialize(model_dir, weight_map, name) for name in PERSISTENT)
    model.model.embed_tokens.weight = torch.nn.Parameter(embed, requires_grad=False)
    model.model.norm.weight = torch.nn.Parameter(norm, requires_grad=False)
    model.lm_head.weight = torch.nn.Parameter(head, requires_grad=False)
    expected = [
        (config.vocab_size, config.hidden_size),
        (config.hidden_size,),
        (config.vocab_size, config.hidden_size),
    ]
    for name, tensor, shape in zip(PERSISTENT, (embed, norm, head), expected):
        if tensor.is_meta or tuple(tensor.shape) != shape:
            raise RuntimeError(f"invalid post-init dimensions for {name}")
    return model


def load_layer(model, config, index, keys, model_dir, weight_map):
    layer = model.model.layers[index]
    prefix = f"model.layers.{index}."
    state = {}
    for full_name, _ in keys:
        name = full_name[len(prefix) :]
        if name.endswith(".weight_scale") or EXPERT_RE.match(name):
            continue
        state[name] = materialize(model_dir, weight_map, full_name)
        if name == "mlp.gate.e_score_correction_bias":
            state[name] = state[name].float()
    expected = {
        name for name in layer.state_dict() if name not in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
    }
    if set(state) != expected:
        raise RuntimeError(
            f"layer {index} non-expert mismatch: missing={sorted(expected - set(state))} unexpected={sorted(set(state) - expected)}"
        )
    result = layer.load_state_dict(state, assign=True, strict=False)
    allowed = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    if set(result.missing_keys) - allowed or result.unexpected_keys:
        raise RuntimeError(f"layer {index} load mismatch: {result}")
    if config.mlp_layer_types[index] == "sparse":
        n, inter, hidden = config.n_routed_experts, config.moe_intermediate_size, config.hidden_size
        layer.mlp.experts.gate_up_proj = torch.nn.Parameter(
            torch.empty(n, 2 * inter, hidden, device=DEVICE, dtype=DTYPE), requires_grad=False
        )
        layer.mlp.experts.down_proj = torch.nn.Parameter(
            torch.empty(n, hidden, inter, device=DEVICE, dtype=DTYPE), requires_grad=False
        )
    meta = [name for name, value in layer.named_parameters() if value.is_meta]
    if meta:
        raise RuntimeError(f"layer {index} has meta parameters: {meta}")
    return layer


def load_experts(layer, config, index, keys, used, model_dir, weight_map):
    used = {int(x) for x in used}
    if not used or min(used) < 0 or max(used) >= config.n_routed_experts:
        raise RuntimeError(f"invalid expert selection: {used}")
    prefix = f"model.layers.{index}."
    available = {name[len(prefix) :] for name, _ in keys if EXPERT_RE.match(name[len(prefix) :])}
    expected = {
        f"mlp.experts.{expert}.{proj}.weight" for expert in used for proj in ("gate_proj", "up_proj", "down_proj")
    }
    if not expected <= available:
        raise RuntimeError(f"selected expert tensors missing: {sorted(expected - available)[:10]}")
    loaded, inter = set(), config.moe_intermediate_size
    with torch.no_grad():
        for name in sorted(expected):
            match = EXPERT_RE.match(name)
            expert, proj = int(match.group(1)), match.group(2)
            value = materialize(model_dir, weight_map, prefix + name)
            target = (
                layer.mlp.experts.gate_up_proj[expert, :inter]
                if proj == "gate_proj"
                else layer.mlp.experts.gate_up_proj[expert, inter:]
                if proj == "up_proj"
                else layer.mlp.experts.down_proj[expert]
            )
            if target.shape != value.shape:
                raise RuntimeError(f"expert shape mismatch: {name}")
            target.copy_(value)
            if not torch.equal(target, value):
                raise RuntimeError(f"expert copy verification failed: {name}")
            loaded.add(name)
    if loaded != expected:
        raise RuntimeError("not every selected expert tensor was proven loaded")
    return {"status": "verified", "experts": sorted(used), "tensor_count": len(loaded)}


def save_boundary(dump, name, tensor, semantic):
    value = tensor.detach().float().cpu()
    if value.ndim != 3:
        raise RuntimeError(f"{name} is not [B,S,H]: {value.shape}")
    dump[name] = value
    dump.setdefault("tensor_metadata", {})[name] = {
        "shape": list(value.shape),
        "layout": "[B,S,H]",
        "semantic": semantic,
        "sha256": tensor_hash(value),
    }


def dsa_record(indices, positions, topk):
    value = indices.detach().cpu().long()
    selected, unique, candidates = [], [], []
    for batch in range(value.shape[0]):
        for query in range(value.shape[1]):
            row, valid = value[batch, query], int(positions[batch, query]) + 1
            valid_row = row[row < valid]
            selected.append(valid_row.numel())
            unique.append(valid_row.unique().numel())
            candidates.append(valid)
            if (
                valid_row.numel() != min(topk, valid)
                or valid_row.unique().numel() != valid_row.numel()
                or valid_row.min() < 0
                or valid_row.max() >= valid
            ):
                raise RuntimeError(f"invalid DSA selection at query {query}")
    return {
        "indices": value.int(),
        "shape": list(value.shape),
        "indices_sha256": tensor_hash(value.int()),
        "selected_count_min": min(selected),
        "selected_count_max": max(selected),
        "unique_count_min": min(unique),
        "unique_count_max": max(unique),
        "valid_candidates_min": min(candidates),
        "valid_candidates_max": max(candidates),
        "valid_range": [0, max(candidates) - 1],
        "queries_over_topk_candidates": sum(x > topk for x in candidates),
        "cutoff": {"topk": topk, "scores_available": False, "margin": None},
    }


def forward(model, config, cache, ids, keys, args, dump):
    hidden = model.model.embed_tokens(ids)
    save_boundary(dump, "embed", hidden, "token embedding output")
    past = cache.get_seq_length()
    positions = torch.arange(past, past + ids.shape[1], device=DEVICE).unsqueeze(0).expand(ids.shape[0], -1)
    attention_2d = torch.ones(ids.shape[0], past + ids.shape[1], dtype=torch.long, device=DEVICE)
    mask = create_causal_mask(
        config=config, inputs_embeds=hidden, attention_mask=attention_2d, past_key_values=cache, position_ids=positions
    )
    if mask is None or mask.shape[-2:] != (ids.shape[1], past + ids.shape[1]):
        raise RuntimeError(f"bad causal mask shape: {None if mask is None else mask.shape}")
    dump["positions"], dump["causal_mask_shape"] = positions.cpu(), list(mask.shape)
    rope = model.model.rotary_emb(hidden, position_ids=positions)
    previous = None
    with torch.no_grad():
        for index in range(args.num_layers):
            layer = load_layer(model, config, index, keys[index], args.model_dir, model.weight_map)
            save_boundary(dump, f"layer{index}_in", hidden, f"layer {index} input")
            normalized = layer.input_layernorm(hidden)
            attn, _, indices = layer.self_attn(
                hidden_states=normalized,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=rope,
                prev_topk_indices=previous,
            )
            record = dsa_record(indices, positions, config.index_topk)
            record["indexer_type"] = config.indexer_types[index]
            record["exact_reuse"] = config.indexer_types[index] == "shared"
            if record["exact_reuse"] and (previous is None or not torch.equal(indices, previous)):
                raise RuntimeError(f"layer {index} did not exactly reuse DSA indices")
            dump[f"layer{index}_dsa"] = record
            post = hidden + attn
            save_boundary(dump, f"layer{index}_attn", post, f"layer {index} input + attention output")
            mlp_in = layer.post_attention_layernorm(post)
            save_boundary(dump, f"layer{index}_mlp_in", mlp_in, f"layer {index} post-attention RMSNorm")
            if config.mlp_layer_types[index] == "sparse":
                _, weights, expert_ids = layer.mlp.gate(mlp_in)
                used = list(range(config.n_routed_experts)) if args.experts == "full" else expert_ids.unique().tolist()
                if args.expert_validation_limit and len(used) > args.expert_validation_limit:
                    raise RuntimeError("selected experts exceed --expert-validation-limit")
                dump[f"layer{index}_experts"] = load_experts(
                    layer, config, index, keys[index], used, args.model_dir, model.weight_map
                )
                output = layer.mlp.experts(mlp_in.reshape(-1, config.hidden_size), expert_ids, weights).reshape(
                    post.shape
                )
                output += layer.mlp.shared_experts(mlp_in)
            else:
                output = layer.mlp(mlp_in)
            hidden = post + output
            save_boundary(dump, f"layer{index}_out", hidden, f"layer {index} post-MLP residual")
            previous = indices
            model.model.layers[index].to("meta")
            torch.cuda.empty_cache()
        hidden = model.model.norm(hidden)
        save_boundary(dump, "norm", hidden, "final RMSNorm output")
        logits = model.lm_head(hidden[:, -1:])
        save_boundary(dump, "logits", logits, "last-position LM-head logits")
    return logits


def run_once(args, config, weight_map, ids, tokenizer, keys, number):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_model(config, weight_map, args.model_dir)
    model.weight_map = weight_map
    cache, generated, steps = DynamicCache(config=config), [], {}
    start = time.time()
    for step in range(args.tokens):
        current = ids.to(DEVICE) if step == 0 else torch.tensor([[generated[-1]]], device=DEVICE)
        dump = {}
        logits = forward(model, config, cache, current, keys, args, dump)
        generated.append(int(logits[:, -1].argmax(-1)))
        dump["input_ids"], dump["generated_ids"] = ids.clone(), list(generated)
        steps[f"step{step}"] = dump
        if args.no_decode or generated[-1] in EOS_IDS:
            break
    result = {"run_index": number, "generated_ids": generated, "steps": steps, "elapsed_seconds": time.time() - start}
    if tokenizer:
        result["generated_text"] = tokenizer.decode(generated)
    del model, cache
    torch.cuda.empty_cache()
    return result


def compare(first, second):
    mismatches = []

    def visit(left, right, path):
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
                mismatches.append(path)
        elif isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                mismatches.append(path + ".keys")
            for key in set(left) & set(right):
                if key not in ("elapsed_seconds", "run_index"):
                    visit(left[key], right[key], path + "." + key)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatches.append(path + ".length")
            for i, pair in enumerate(zip(left, right)):
                visit(*pair, f"{path}[{i}]")
        elif left != right:
            mismatches.append(path)

    visit(first, second, "runs")
    return {"status": "PASS" if not mismatches else "FAIL", "exact": not mismatches, "mismatches": mismatches}


def build_semantic_artifact(steps, num_layers):
    normalized = {}
    for step_name, dump in steps.items():
        step = "prefill" if step_name == "step0" else step_name.replace("step", "decode", 1)
        normalized[step] = {
            "embedding": dump["embed"],
            "layer_input": [dump[f"layer{index}_in"] for index in range(num_layers)],
            "post_attention_residual": [
                dump[f"layer{index}_attn"] for index in range(num_layers)
            ],
            "mlp_input": [dump[f"layer{index}_mlp_in"] for index in range(num_layers)],
            "layer_output": [dump[f"layer{index}_out"] for index in range(num_layers)],
            "final_norm": dump["norm"],
            "logits": dump["logits"],
            "dsa_topk": [
                dump[f"layer{index}_dsa"]["indices"] for index in range(num_layers)
            ],
        }
    return {
        "format": "glm52_transformers_semantic_v1",
        "producer": "transformers",
        "steps": normalized,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.repeat < 1 or args.tokens < 1:
        raise ValueError("--repeat and --tokens must be positive")
    config, weight_map, report, ids, tokenizer, keys = preflight(args)
    short = {key: value for key, value in report.items() if key != "checkpoint_manifest"}
    print(json.dumps(short, indent=2, sort_keys=True))
    if args.preflight_only:
        print("FINAL PASS: CPU-only preflight completed; no CUDA model tensors constructed")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required")
    torch.cuda.set_device(DEVICE)
    torch.use_deterministic_algorithms(True)
    config._attn_implementation = config._experts_implementation = "eager"
    runs = [run_once(args, config, weight_map, ids, tokenizer, keys, i) for i in range(args.repeat)]
    comparisons = [compare(runs[0], run) for run in runs[1:]]
    deterministic = all(item["exact"] for item in comparisons)
    sparse_required = ids.shape[1] > config.index_topk
    records = [
        value
        for value in runs[0]["steps"]["step0"].values()
        if isinstance(value, dict) and "queries_over_topk_candidates" in value
    ]
    sparse_proved = not sparse_required or any(
        x["queries_over_topk_candidates"] and x["selected_count_max"] == config.index_topk for x in records
    )
    status = "PASS" if deterministic and sparse_proved else "FAIL"
    artifact = {
        "format_version": 2,
        "status": status,
        "boundary_layout": "All semantic activation/logit tensors are [B,S,H]",
        "preflight": report,
        "runs": runs,
        "repeat_comparisons": comparisons,
        "summary": {
            "status": status,
            "deterministic": deterministic,
            "sparse_selection_required": sparse_required,
            "sparse_selection_proved": sparse_proved,
        },
    }
    output_dir = Path(args.dump_dir)
    output = output_dir / "golden_dumps.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    semantic = build_semantic_artifact(runs[0]["steps"], args.num_layers)
    semantic["metadata"] = report["metadata"]
    semantic_output = output_dir / "transformers_semantic.pt"
    torch.save(semantic, semantic_output)
    print(f"saved artifact: {output}")
    print(f"saved semantic artifact: {semantic_output}\nFINAL {status}")
    return status != "PASS"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FINAL FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise
