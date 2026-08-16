export const meta = {
  name: 'kimi-k2.7-scout',
  description: 'Parallel exploration of KIMI K2.7 streaming-inference port: modeling compat, config, runtime env',
  phases: [
    { title: 'Explore', detail: '3 parallel analysis agents' },
    { title: 'Synthesize', detail: 'merge findings, surface blockers' },
  ],
}

const REPORT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    class_hierarchy: { type: 'string', description: 'exact class names from KimiK25ForConditionalGeneration down through text backbone to decoder layers' },
    language_model_attr: { type: 'string', description: 'attribute/module for text backbone inside multimodal wrapper and its class' },
    weight_prefix: { type: 'string', description: 'exact state_dict prefix for transformer layers as repo builds them' },
    mla_shapes_match: { type: 'boolean', description: 'do repo MLA Linear shapes match checkpoint' },
    moe_routing: { type: 'string', description: 'exact router formula a driver must replicate to pick experts' },
    expert_module: { type: 'string', description: 'how MoE expert weights are structured in repo (per-expert separate proj, shapes)' },
    cache_class: { type: 'string', description: 'cache class used (DynamicCache? DSA/indexer? custom)' },
    nextn: { type: 'number', description: 'num_nextn_predict_layers and whether MTP/nextn is used' },
    yarn_rope: { type: 'string', description: 'how YARN rotary is applied' },
    load_compatibility: { type: 'string', description: 'verdict: can repo modeling load this checkpoint keys as-is?' },
    issues: { type: 'array', items: { type: 'string' }, description: 'blockers/gotchas for streaming driver' },
  },
  required: ['class_hierarchy', 'language_model_attr', 'weight_prefix', 'mla_shapes_match', 'moe_routing', 'expert_module', 'cache_class', 'nextn', 'yarn_rope', 'load_compatibility', 'issues'],
}

const CONFIG_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    auto_map: { type: 'string', description: 'what auto_map says and whether repo-native classes resolve it' },
    repo_loads_directly: { type: 'boolean', description: 'can repo (PYTHONPATH=src) load config.json without model-dir custom files' },
    needed_custom_classes: { type: 'array', items: { type: 'string' }, description: 'classes/files that must come from model dir' },
    tokenizer_present: { type: 'boolean', description: 'is a tokenizer present in model dir; how to obtain offline' },
    chat_template: { type: 'string', description: 'template format, special tokens, assistant prefix' },
    bos_eos_pad_ids: { type: 'string', description: 'special ids and any turn-boundary tokens' },
    vision_required: { type: 'boolean', description: 'can text-only inference skip vision tower/mm_projector' },
    config_diff: { type: 'string', description: 'key diffs between model-dir and repo configuration_kimi_k25.py' },
    issues: { type: 'array', items: { type: 'string' }, description: 'blockers/gotchas' },
  },
  required: ['auto_map', 'repo_loads_directly', 'needed_custom_classes', 'tokenizer_present', 'chat_template', 'bos_eos_pad_ids', 'vision_required', 'config_diff', 'issues'],
}

const ENV_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    download_gb_done: { type: 'number', description: 'GB downloaded so far' },
    download_gb_total: { type: 'number', description: 'total model size in GB (595.2)' },
    rate_gb_per_min: { type: 'number', description: 'measured download rate' },
    eta_minutes: { type: 'number', description: 'estimated minutes until download completes' },
    disk_free_gb: { type: 'number', description: 'free space on /data NFS mount' },
    vram_free_gb: { type: 'number', description: 'free VRAM on cuda:0' },
    ram_free_gb: { type: 'number', description: 'free host RAM' },
    ramdisk_free_gb: { type: 'number', description: 'free space on /mnt/glm_ram or /dev/shm tmpfs' },
    kimi_checkpoint_gb_on_ram: { type: 'string', description: 'estimate of non-expert bf16 weights per layer for ramcache' },
    notes: { type: 'string' },
  },
  required: ['download_gb_done', 'download_gb_total', 'rate_gb_per_min', 'eta_minutes', 'disk_free_gb', 'vram_free_gb', 'ram_free_gb', 'ramdisk_free_gb', 'kimi_checkpoint_gb_on_ram', 'notes'],
}

phase('Explore')

const [modeling, config, env] = await parallel([
  () => agent(
    `You are analyzing HuggingFace transformers repo support for a KIMI K2.7 checkpoint to build a layer-by-layer streaming inference driver (load weights layer-by-layer onto GPU, compute, free). READ-ONLY research - do not write/edit files; report structured output only.

CONTEXT: Checkpoint at /data/model/Kimi-K2.7-Code (config.json, model_type kimi_k25, KimiK25ForConditionalGeneration, multimodal). text_config.model_type = kimi_k2, DeepseekV3-style MoE text backbone: hidden 7168, 61 layers, 64 heads, MLA (kv_lora_rank 512, q_lora_rank 1536, qk_nope_head_dim 128, qk_rope_head_dim 64, v_head_dim 128), MoE 384 routed / 8 per tok / 1 shared, first_k_dense_replace 1, moe_layer_freq 1, topk_method noaux_tc, scoring_func sigmoid, norm_topk_prob true, n_group 1, routed_scaling_factor 2.827, yarn rope (factor 64, original_max 4096), vocab 163840, num_nextn_predict_layers 0, tie_word_embeddings false.

CHECKPOINT WEIGHT NAMES (probed from real safetensors shards) use prefix language_model.model.layers.N.*, e.g.:
  language_model.model.layers.0.input_layernorm.weight (bf16 7168)
  language_model.model.layers.1.self_attn.kv_a_proj_with_mqa.weight (576,7168)
  language_model.model.layers.1.self_attn.kv_b_proj.weight (16384,512)
  language_model.model.layers.1.self_attn.q_a_proj.weight (1536,7168) q_b_proj (12288,1536) o_proj (7168,8192)
  language_model.model.layers.1.mlp.gate.weight (384,7168) bf16
  language_model.model.layers.1.mlp.gate.e_score_correction_bias (384,) f32
  language_model.model.layers.1.mlp.shared_experts.{gate,up,down}_proj.weight (2048x7168 / 7168x2048) bf16
  language_model.model.layers.1.mlp.experts.{0..383}.{gate,up}_proj.weight_packed (2048,896) int32 + weight_scale (2048,224) bf16 + weight_shape
  language_model.model.layers.1.mlp.experts.{0..383}.down_proj.weight_packed (7168,256) int32 + weight_scale (7168,64) bf16

TASK: Read /data/mmh/kernel_zoo_ref_projs/transformers/src/transformers/models/kimi_k25/modeling_kimi_k25.py IN FULL (37KB). Also read modular_kimi_k25.py and configuration_kimi_k25.py as needed. Answer:
1. class_hierarchy: exact class names from KimiK25ForConditionalGeneration down through text backbone (language_model) to transformer model and decoder layers; namespacing in __init__.
2. language_model_attr: class used for self.language_model and how wrapper calls it in forward.
3. weight_prefix: exact state_dict key prefix repo model expects for transformer layers (should be language_model.model.layers.N. - confirm).
4. mla_shapes_match: do repo MLA linears match checkpoint shapes? kv_b_proj (16384,512) = num_kv_heads*(qk_nope+v_head), q_b_proj (12288,1536) = num_heads*qk_head_dim.
5. moe_routing: read the router (MoEGate). topk_method noaux_tc with scoring_func sigmoid, norm_topk_prob. Exact formula a driver must replicate to know selected experts and topk_weights. With n_group=1 is group-mask a no-op?
6. expert_module: how experts are stored (per-expert separate gate/up/down Linear? ModuleList? stacked tensor?). Attribute structure and forward reading gate_proj.
7. cache_class: what cache the model uses in forward with past_key_values. DynamicCache? DSA/indexer like GLM-5.2 (I believe NO)? report actual.
8. nextn: num_nextn_predict_layers=0 => no MTP/nextn?
9. yarn_rope: how rotary_emb handles yarn (DeepseekV3RotaryEmbedding?). inv_freq construction and mscale.
10. load_compatibility: can repo modeling load checkpoint state_dict keys directly (prefixes/shapes)?
11. issues: concrete gotchas for a streaming driver (custom Linear, quantized linears, no_grad, layernorms).

Be precise, cite file:line. Return ONLY the structured object.`,
    { label: 'explore:modeling', phase: 'Explore', schema: REPORT_SCHEMA }
  ),
  () => agent(
    `You are checking config/custom-class compatibility for a KIMI K2.7 checkpoint to build a streaming inference driver. READ-ONLY research - do not edit files.

CONTEXT: Model dir /data/model/Kimi-K2.7-Code contains: config.json (model_type kimi_k25), custom files configuration_kimi_k25.py, configuration_deepseek.py, kimi_k25_processor.py, kimi_k25_vision_processing.py, media_utils.py, chat_template.jinja, generation_config.json, LICENSE, docs/, figures/. NO tokenizer file appears present. Repo at /data/mmh/kernel_zoo_ref_projs/transformers/src/transformers/models/kimi_k25/ has native configuration_kimi_k25.py, modeling_kimi_k25.py, modular_kimi_k25.py, processing_kimi_k25.py, image_processing_kimi_k25.py, video_processing_kimi_k25.py.

TASKS:
1. Read /data/model/Kimi-K2.7-Code/config.json auto_map and generation_config.json. Read repo configuration_kimi_k25.py and diff against model-dir configuration_kimi_k25.py. Which config class does the checkpoint need and can repo native kimi_k25 config load config.json directly (top-level multimodal KimiK25Config with text_config and vision_config subconfigs)?
2. repo_loads_directly: with PYTHONPATH=/data/mmh/kernel_zoo_ref_projs/transformers/src, can transformers auto-load via AutoConfig/AutoModelForCausalLM? Or need model-dir custom files (configuration_deepseek dependency)? Determine whether repo kimi_k25 imports from transformers.models.deepseek_v3 or model-dir configuration_deepseek.py.
3. tokenizer_present: check /data/model/Kimi-K2.7-Code for tokenizer files (tokenizer.json, tokenizer_config.json, *.model, vocab). If absent check /root/.cache/huggingface or /data for cached MoonshotAI Kimi-K2.7 tokenizer. Report whether we can build tokenizer offline or must download from modelscope (modelscope 1.35.4 installed; HF_HUB_OFFLINE=1). What tokenizer_class does config specify?
4. chat_template: read chat_template.jinja, summarize format (special tokens, user turn wrapping, system prompt, thinking mode toggle, exact assistant prefix). Note special control token ids if visible.
5. bos_eos_pad_ids: confirm bos 163584 eos 163586 pad 163839 and additional special ids used by template.
6. vision_required: read repo modeling_kimi_k25.py to determine if vision tower + mm_projector + image processor are REQUIRED to instantiate the model and run text-only forward, or can be skipped/None. Report if text-only forward needs inputs_embeds and can bypass vision. Report how wrapper computes text embeddings (embed_tokens on language_model).
7. Report the exact command line a driver should use to build the model with text-only weights, skipping vision if possible.

Be precise. Return ONLY the structured object.`,
    { label: 'explore:config', phase: 'Explore', schema: CONFIG_SCHEMA }
  ),
  () => agent(
    `You are measuring the runtime environment and download progress for a KIMI K2.7 streaming-inference port on a Hygon K100AI DCU. READ-ONLY research - do not edit files.

CONTEXT: Model downloading to /data/model/Kimi-K2.7-Code via 'modelscope download --model moonshotai/Kimi-K2.7-Code --local_dir Kimi-K2.7-Code' (already running). Total = 595.2 GB, 64 shards (shard1=1.00GB, shards2-64=9.81GB) + model.safetensors.index.json. /data is NFS v3 (11.10.3.160:/mnt/htxjj). Completed shards appear directly in main dir; in-flight to ._____temp/. Reference driver /data/mmh/kernel_zoo_ref_projs/transformers/glm_dsa_stream_infer.py has gpu_mem_gb() and hy_smi() helpers.

TASKS:
1. Compute download progress: sum sizes of *.safetensors in main dir + ._____temp/. Measure rate by sampling total now and ~25s later. Estimate ETA.
2. disk_free_gb on /data (df -h). Confirm it can hold 595GB.
3. vram_free_gb: run 'hy-smi' (plain; --showmem invalid here) or torch.cuda. 4 x ~68.7GB VRAM. Report cuda:0 free. Confirm no other heavy GPU process.
4. ram_free_gb: free -g (125GB RAM).
5. ramdisk_free_gb: df -h /mnt/glm_ram (40GB tmpfs from GLM) and /dev/shm (63GB tmpfs).
6. kimi_checkpoint_gb_on_ram: estimate bf16 size of NON-EXPERT tensors per MoE layer for a ramdisk cache, using these shapes (bf16=2 bytes, f32=4): input_layernorm 7168, post_attention_layernorm 7168, gate.weight 384x7168, gate.e_score_correction_bias 384, shared_experts gate_proj 2048x7168 + up_proj 2048x7168 + down_proj 7168x2048, self_attn q_a_proj 1536x7168 + q_a_layernorm 1536 + kv_a_layernorm 512 + kv_a_proj_with_mqa 576x7168 + kv_b_proj 16384x512 + q_b_proj 12288x1536 + o_proj 7168x8192. Total per MoE layer and whole 61-layer model. Layer 0 dense: 2 layernorms + 3 dense MLP 18432x7168/7168x18432 + same self_attn. Also estimate a layer-1 FULL dequant: int4 packed bytes = 384 * (gate_up 2*2048*7168/8 + down 7168*2048/8) * 4 bytes; dequantized bf16 = 384 * (2*2048*7168 + 7168*2048) * 2 bytes. Report both per layer.
7. notes: contention, /mnt/glm_ram mounted?, download PID alive.

Use helpers by importing glm_dsa_stream_infer.py (sys.path insert /data/mmh/kernel_zoo_ref_projs/transformers). Do NOT start downloads or kill processes. Return ONLY the structured object.`,
    { label: 'explore:env', phase: 'Explore', schema: ENV_SCHEMA }
  ),
])

phase('Synthesize')

const SYNTH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    summary: { type: 'string', description: '2-3 sentence overall readiness assessment' },
    blockers: { type: 'array', items: { type: 'string' }, description: 'concrete blockers with the agent finding that surfaced each' },
    decisions: { type: 'array', items: { type: 'string' }, description: 'key design decisions for kimi_stream_infer.py' },
    open_questions: { type: 'array', items: { type: 'string' }, description: 'things still unknown for the next workflow' },
  },
  required: ['summary', 'blockers', 'decisions', 'open_questions'],
}

const synth = await agent(
  `You are the synthesis step of a scouting workflow for porting GLM-5.2-style layer-by-layer streaming inference to KIMI K2.7. Three exploration agents reported. Merge into a coherent readiness assessment for writing kimi_stream_infer.py (meta-device build, per-layer streaming, int4 dequant, on-demand experts, ramdisk cache).

MODELING FINDINGS: ${JSON.stringify(modeling)}
CONFIG FINDINGS: ${JSON.stringify(config)}
ENV FINDINGS: ${JSON.stringify(env)}

Produce: overall readiness, concrete blockers (preventing first end-to-end run), design decisions for the driver, open questions for the design/implement workflow. Be specific. Return ONLY the structured object.`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return { merged: { modeling, config, env }, synth }
