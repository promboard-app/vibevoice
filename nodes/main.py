"""promboard.vibevoice — VibeVoice-1.5B(Microsoft) 긴 대화 목소리 노드 코드.

노드 하나(vibevoice)가 대본을 읽어 최대 4명이 주고받는 목소리 WAV(24kHz 모노)를 낸다. 화자마다 참조 목소리를 이으면 그 목소리로
복제하고, 잇지 않으면 모델이 목소리를 고른다.

공식 경로
  Microsoft 는 GitHub 의 VibeVoice-TTS 추론 코드를 2025-09-05 지웠다. 지금 쓸 수 있는 공식 경로는 transformers(5.x)에 들어간
  VibeVoiceForConditionalGeneration 이다(Hugging Face, 2026-08). transformers 문서가 가리키는 변환본 vibevoice/VibeVoice-1.5B-hf 는
  비공식 커뮤니티 조직이 올린 것이라 쓰지 않는다. 대신 Microsoft 공식 가중치 microsoft/VibeVoice-1.5B 를 받아, transformers 의
  공식 변환 스크립트(models/vibevoice/convert_vibevoice_to_hf.py)와 같은 규칙으로 실행할 때 메모리에서 이름만 바꿔 올린다
  (아래 STATE_DICT_MAPPING · _hf_config · _processor 는 그 스크립트를 옮긴 것이다. Apache-2.0, Copyright 2026 The HuggingFace Inc. team).
  글 토크나이저는 원본 설정(language_model_pretrained_name)이 가리키는 Qwen/Qwen2.5-1.5B 의 토크나이저 파일만 받는다.

대본 쓰는 법: 줄마다 'Speaker 1: 글' 또는 '1: 글'(화자 1~4). 화자 표시가 없는 줄은 바로 앞 화자, 처음부터 없으면 화자 1 이다.
설정: guidance(classifier-free guidance, 변환 스크립트 기본 1.3) · steps(확산 단계, 기본 10) · seed. 생성은 늘 argmax 다(transformers
  구현이 샘플링을 쓰지 않는다).
노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import json
import os
import re
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 받은 파일만 쓴다(실행 환경은 인터넷이 막혀 있다)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_FILE = "microsoft/VibeVoice-1.5B/config.json"
TOKENIZER_FILE = "Qwen/Qwen2.5-1.5B/tokenizer.json"
SAMPLE_RATE = 24000
MAX_SPEAKERS = 4
MAX_TEXT_CHARS = 6000  # 한 번에 읽을 대본 상한(약 7분 분량) — 앱이 노드 한 번에 주는 시간(만들기 15분)에 끝나게. 공식 최대 90분보다 훨씬 짧다
MAX_REF_SEC = 60.0
TOKENS_PER_SEC = 7.5  # 소리 토큰 하나 = 1/7.5초(공식 카드 7.5Hz)
MAX_NEW_TOKENS = 40500  # 변환 스크립트의 1.5B 기본값

# transformers convert_vibevoice_to_hf.py 의 STATE_DICT_MAPPING 그대로(순서 중요)
STATE_DICT_MAPPING = {
    r"semantic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.": r"semantic_tokenizer_encoder.stem.conv.conv.",
    r"semantic_tokenizer\.encoder\.stages\.0\.": r"semantic_tokenizer_encoder.stem.stage.",
    r"semantic_tokenizer\.encoder\.downsample_layers\.(\d+)\.0\.conv\.": r"semantic_tokenizer_encoder.conv_layers.PLACEHOLDER.conv.conv.",
    r"semantic_tokenizer\.encoder\.stages\.(\d+)\.": r"semantic_tokenizer_encoder.conv_layers.PLACEHOLDER.stage.",
    r"semantic_tokenizer\.encoder\.head\.conv\.": r"semantic_tokenizer_encoder.head.",
    r"acoustic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.": r"audio_tower.encoder.stem.conv.conv.",
    r"acoustic_tokenizer\.encoder\.stages\.0\.": r"audio_tower.encoder.stem.stage.",
    r"acoustic_tokenizer\.encoder\.downsample_layers\.(\d+)\.0\.conv\.": r"audio_tower.encoder.conv_layers.PLACEHOLDER.conv.conv.",
    r"acoustic_tokenizer\.encoder\.stages\.(\d+)\.": r"audio_tower.encoder.conv_layers.PLACEHOLDER.stage.",
    r"acoustic_tokenizer\.encoder\.head\.conv\.": r"audio_tower.encoder.head.",
    r"acoustic_tokenizer\.decoder\.upsample_layers\.0\.0\.conv\.conv\.": r"audio_tower.decoder.stem.conv.conv.",
    r"acoustic_tokenizer\.decoder\.stages\.0\.": r"audio_tower.decoder.stem.stage.",
    r"acoustic_tokenizer\.decoder\.upsample_layers\.(\d+)\.0\.convtr\.convtr\.": r"audio_tower.decoder.conv_layers.PLACEHOLDER.convtr.convtr.",
    r"acoustic_tokenizer\.decoder\.stages\.(\d+)\.": r"audio_tower.decoder.conv_layers.PLACEHOLDER.stage.",
    r"acoustic_tokenizer\.decoder\.head\.conv\.": r"audio_tower.decoder.head.",
    r"acoustic_tokenizer\.": r"audio_tower.",
    r"prediction_head\.t_embedder\.mlp\.0\.": r"diffusion_head.timestep_proj.fc1.",
    r"prediction_head\.t_embedder\.mlp\.2\.": r"diffusion_head.timestep_proj.fc2.",
    r"prediction_head\.layers\.(\d+)\.adaLN_modulation\.1\.": r"diffusion_head.layers.\1.linear.",
    r"prediction_head\.final_layer\.adaLN_modulation\.1\.": r"diffusion_head.final_layer.linear_1.",
    r"prediction_head\.final_layer\.linear\.": r"diffusion_head.final_layer.linear_2.",
    r"prediction_head\.": r"diffusion_head.",
    r"acoustic_connector\.fc1\.": r"multi_modal_projector.linear_1.",
    r"acoustic_connector\.norm\.": r"multi_modal_projector.act.",
    r"acoustic_connector\.fc2\.": r"multi_modal_projector.linear_2.",
    r"semantic_connector\.fc1\.": r"semantic_connector.linear_1.",
    r"semantic_connector\.norm\.": r"semantic_connector.act.",
    r"semantic_connector\.fc2\.": r"semantic_connector.linear_2.",
    r"^model\.speech_scaling_factor": r"model.latent_scaling_factor",
    r"^model\.speech_bias_factor": r"model.latent_bias_factor",
    r"mixer\.conv\.conv\.conv\.": r"mixer.conv.",
    r"\.conv\.conv\.conv\.": r".conv.conv.",
}

# convert_vibevoice_to_hf.py 의 CHAT_TEMPLATE 그대로
CHAT_TEMPLATE = """{%- set system_prompt = system_prompt | default(" Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\\n") -%}
{{ system_prompt -}}
{%- set audio_bos_token = audio_bos_token | default("<|vision_start|>") %}
{%- set audio_eos_token = audio_eos_token | default("<|vision_end|>") %}
{%- set audio_token = audio_token | default("<|vision_pad|>") %}
{%- set eos_token = eos_token | default("<|endoftext|>") %}
{%- set ns = namespace(num_audio=0, num_seen=0, speakers_with_audio="") %}
{%- for message in messages %}
    {%- set ns.num_audio = ns.num_audio + (message['content'] | selectattr('type', 'equalto', 'audio') | list | length) %}
{%- endfor %}
{%- set has_target_audio = not add_generation_prompt and ns.num_audio > 0 %}
{%- set num_voice_prompts = ns.num_audio - 1 if has_target_audio else ns.num_audio %}
{%- for message in messages %}
    {%- set role = message['role'] %}
    {%- set audio_count = message['content'] | selectattr('type', 'equalto', 'audio') | list | length %}
    {%- if audio_count > 0 and ns.num_seen < num_voice_prompts and role not in ns.speakers_with_audio %}
        {%- set ns.speakers_with_audio = ns.speakers_with_audio + role + "," %}
    {%- endif %}
    {%- set ns.num_seen = ns.num_seen + audio_count %}
{%- endfor %}

{%- if ns.speakers_with_audio %}
{{ " Voice input:\\n" }}
{%- for speaker in ns.speakers_with_audio.rstrip(',').split(',') %}
{%- if speaker %}
 Speaker {{ speaker }}:{{ audio_bos_token }}{{ audio_token }}{{ audio_eos_token }}{{ "\\n" }}
{%- endif %}
{%- endfor %}
{%- endif %}
 Text input:{{ "\\n" }}

{%- for message in messages %}
    {%- set role = message['role'] %}
    {%- set text_items = message['content'] | selectattr('type', 'equalto', 'text') | list %}
    {%- for item in text_items %}
 Speaker {{ role }}: {{ item['text'] }}{{ "\\n" }}
    {%- endfor %}
{%- endfor %}
 Speech output:{{ "\\n" }}{{ audio_bos_token }}
{%- if not add_generation_prompt %}
{%- if has_target_audio %}{{ audio_token }}{{ audio_eos_token }}{% endif %}{{ eos_token }}
{%- endif %}"""

_STATE = {}


def _map_key(key):
    for pattern, replacement in STATE_DICT_MAPPING.items():
        match = re.search(pattern, key)
        if match:
            if "PLACEHOLDER" in replacement and match.groups():
                replacement = replacement.replace("PLACEHOLDER", str(int(match.group(1)) - 1))
            key = re.sub(pattern, replacement, key)
    return key


def _tokenizer_config(d, keys_to_remove):
    """변환 스크립트 process_tokenizer_config 그대로."""
    if "encoder_depths" in d and isinstance(d["encoder_depths"], str):
        d["encoder_depths"] = list(map(int, d["encoder_depths"].split("-")))
    if "layernorm_eps" in d:
        d["rms_norm_eps"] = d.pop("layernorm_eps")
    if "encoder_ratios" in d:
        d["downsampling_ratios"] = list(reversed(d.pop("encoder_ratios")))
    if "encoder_n_filters" in d:
        d["num_filters"] = d.pop("encoder_n_filters")
    if "encoder_depths" in d:
        d["depths"] = d.pop("encoder_depths")
    if "vae_dim" in d:
        d["hidden_size"] = d.pop("vae_dim")
    for k in keys_to_remove:
        d.pop(k, None)
    return d


def _hf_config(folder):
    """원본 config.json → VibeVoiceConfig(변환 스크립트 4 · 6 · 7 · 9단계)."""
    from transformers import VibeVoiceAcousticTokenizerConfig, VibeVoiceAcousticTokenizerEncoderConfig, VibeVoiceConfig

    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    remove = [
        "decoder_depths", "decoder_n_filters", "decoder_ratios", "std_dist_type", "fix_std", "pad_mode", "conv_bias",
        "causal", "mixer_layer", "layernorm", "disable_last_norm", "conv_norm", "corpus_normalize",
        "layernorm_elementwise_affine",
    ]
    semantic = _tokenizer_config(cfg.pop("semantic_tokenizer_config", {}).copy(), remove)
    acoustic = _tokenizer_config(cfg.pop("acoustic_tokenizer_config", {}).copy(), remove)
    if "fix_std" in acoustic:  # 변환 스크립트에도 있는 줄 — fix_std 는 위에서 이미 빠져 늘 건너뛴다(스크립트와 같은 결과를 내려고 그대로 둔다)
        acoustic["vae_std"] = acoustic.pop("fix_std") / 0.8
    diffusion = cfg.pop("diffusion_head_config")
    cfg["diffusion_head_config"] = {
        "hidden_size": diffusion["hidden_size"],
        "latent_size": acoustic["hidden_size"],
        "num_hidden_layers": diffusion["head_layers"],
        "intermediate_size": int(diffusion["head_ffn_ratio"] * diffusion["hidden_size"]),
        "rms_norm_eps": diffusion["rms_norm_eps"],
    }
    cfg["text_config"] = cfg.pop("decoder_config")
    cfg["text_config"]["dtype"] = cfg["text_config"].pop("torch_dtype")
    cfg["dtype"] = cfg.pop("torch_dtype", None)
    for k in ["acoustic_vae_dim", "semantic_vae_dim", "tie_word_embeddings"]:
        cfg.pop(k, None)
    cfg["audio_config"] = VibeVoiceAcousticTokenizerConfig(**acoustic).to_dict()
    cfg["semantic_model_config"] = VibeVoiceAcousticTokenizerEncoderConfig(**semantic).to_dict()
    return VibeVoiceConfig(**cfg)


def _processor(folder, tokenizer_folder):
    """변환 스크립트 3 · 8단계 — 특징 추출기 + Qwen2.5 토크나이저 + 대화 틀."""
    from transformers import Qwen2TokenizerFast, VibeVoiceAcousticTokenizerFeatureExtractor, VibeVoiceProcessor

    audio_config = {}
    path = os.path.join(folder, "preprocessor_config.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            audio_config = json.load(f).get("audio_processor", {})
    audio_config.pop("feature_extractor_type", None)
    audio_config.setdefault("sampling_rate", SAMPLE_RATE)
    audio_config.setdefault("normalize_audio", True)
    audio_config.setdefault("target_dB_FS", -25)
    audio_config.setdefault("eps", 1e-6)
    audio_config.setdefault("pad_to_multiple_of", 3200)
    tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_folder, padding=True, padding_side="left")
    return VibeVoiceProcessor(
        feature_extractor=VibeVoiceAcousticTokenizerFeatureExtractor(**audio_config),
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
    )


def _generation_config():
    """변환 스크립트가 정하는 생성 기본값(1.5B)."""
    from transformers import GenerationConfig

    return GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        guidance_scale=1.3,
        num_diffusion_steps=10,
        noise_scheduler_class="DPMSolverMultistepScheduler",
        noise_scheduler_config={"beta_schedule": "squaredcos_cap_v2", "prediction_type": "v_prediction"},
        bos_token_id=None,
        eos_token_id=151643,
        pad_token_id=151643,
    )


def _converted(ctx, folder):
    """원본 가중치를 변환 스크립트 규칙으로 바꿔 플러그인 캐시 폴더에 한 번 써 둔다(bf16 약 5.4GB). 다 쓴 뒤에 표식을 남긴다.
    표식에는 원본 폴더 이름(저장소@커밋)을 적어, 모델이 바뀌면 다시 바꾼다.
    원본 조각마다 읽고 바로 같은 번호의 조각으로 쓰며, 텐서를 읽는 사이사이 진행을 알린다 — 느린 디스크에서도 앱이 기다리는
    무응답 상한(load-model 10분)을 넘지 않게."""
    import glob

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    out = os.path.join(ctx.cache_dir, "vibevoice-1.5b-hf")
    marker = os.path.join(out, "converted-from.txt")
    source = os.path.basename(folder)
    if os.path.isfile(marker) and open(marker, encoding="utf-8").read().strip() == source:
        return out
    ctx.log("converting microsoft/VibeVoice-1.5B to the transformers layout (once)")
    os.makedirs(out, exist_ok=True)
    for old in [marker, *glob.glob(os.path.join(out, "*.safetensors")), os.path.join(out, "model.safetensors.index.json")]:
        if os.path.exists(old):
            os.remove(old)
    config = _hf_config(folder)
    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    weight_map, total = {}, 0
    for i, shard in enumerate(shards):
        state = {}
        with safe_open(os.path.join(folder, shard), framework="pt") as f:
            names = list(f.keys())
            for j, k in enumerate(names):
                if j % 25 == 0:
                    ctx.check_cancel()
                    ctx.progress(0.02 + 0.03 * (i + j / len(names)) / len(shards), "load-model")
                state[_map_key(k)] = f.get_tensor(k).to(torch.bfloat16).contiguous()
        # lm_head 는 embed_tokens 와 묶여 있어(1.5B) 따로 쓰지 않는다 — 불러올 때 다시 묶인다.
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        ctx.progress(0.02 + 0.03 * (i + 1) / len(shards), "load-model")
        save_file(state, os.path.join(out, name), metadata={"format": "pt"})
        weight_map.update({k: name for k in state})
        total += sum(v.numel() * v.element_size() for v in state.values())
        del state
    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=2)
    config.save_pretrained(out)
    _generation_config().save_pretrained(out)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(source)
    return out


def _load(ctx):
    device = ctx.device
    if _STATE.get("device") == device:
        return _STATE["model"], _STATE["processor"]
    import torch
    from transformers import VibeVoiceForConditionalGeneration

    if not ctx.cache_dir:
        raise RuntimeError("플러그인 캐시 폴더가 없어 VibeVoice 가중치를 바꿔 둘 곳이 없습니다. 로컬 AI 모델 폴더를 확인해 주세요.")
    ctx.progress(0.02, "load-model")  # 올리는 동안 파이썬이 오래 멈춘다 — 바로 앞에 알린다
    start = time.monotonic()
    _STATE.clear()
    folder = os.path.dirname(ctx.model_path(MODEL_FILE))
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    path = _converted(ctx, folder)
    ctx.progress(0.06, "load-model")
    try:
        model = VibeVoiceForConditionalGeneration.from_pretrained(path, dtype=dtype, device_map=device).eval()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError("그래픽카드 메모리가 모자라 모델을 올리지 못했습니다(약 6GB 필요). 다른 AI 작업 · 프로그램을 닫고 다시 해 주세요.") from None
    processor = _processor(folder, os.path.dirname(ctx.model_path(TOKENIZER_FILE)))
    _STATE.update(device=device, model=model, processor=processor)
    ctx.log(f"VibeVoice-1.5B loaded on {device} in {time.monotonic() - start:.1f}s")
    return model, processor


def _read_audio(path):
    """참조 목소리 → float32 모노 24kHz. wav · flac · ogg · mp3 는 soundfile, 그 밖은 PyAV."""
    import numpy as np

    data, sr = None, None
    try:
        import soundfile as sf

        d, sr = sf.read(path, dtype="float32", always_2d=True)
        data = d.mean(axis=1)
    except Exception:
        import av

        try:
            with av.open(path) as container:
                if not container.streams.audio:
                    raise ValueError("참조 목소리 파일에 소리가 없습니다.")
                resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
                chunks = [o.to_ndarray().reshape(-1) for fr in container.decode(audio=0) for o in resampler.resample(fr)]
                chunks += [o.to_ndarray().reshape(-1) for o in resampler.resample(None)]
        except av.error.FFmpegError as exc:
            raise ValueError(f"참조 목소리 파일을 읽지 못했습니다. wav · mp3 · flac · ogg · m4a 소리인지 확인해 주세요. ({exc})") from None
        data, sr = (np.concatenate(chunks) if chunks else np.zeros(0, np.float32)), SAMPLE_RATE
    if sr != SAMPLE_RATE and data.size:
        import torch
        import torchaudio

        data = torchaudio.functional.resample(torch.from_numpy(data), sr, SAMPLE_RATE).numpy()
    seconds = data.size / SAMPLE_RATE
    if seconds < 1.0:
        raise ValueError("참조 목소리가 너무 짧습니다. 3~10초 길이의 말소리를 이어 주세요.")
    if seconds > MAX_REF_SEC:
        raise ValueError(f"참조 목소리가 너무 깁니다({seconds:.0f}초). {MAX_REF_SEC:.0f}초까지 받습니다.")
    return data.astype(np.float32)


SPEAKER_RE = re.compile(r"^\s*(?:speaker\s*)?([1-4])\s*[:：]\s*(.*)$", re.IGNORECASE)


def parse_script(text):
    """대본 → [(화자 번호 문자열 '0'~'3', 글)]. 화자 표시 없는 줄은 앞 화자에 붙인다."""
    turns = []
    speaker = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = SPEAKER_RE.match(line)
        if m:
            speaker = str(int(m.group(1)) - 1)
            if m.group(2).strip():
                turns.append([speaker, m.group(2).strip()])
            continue
        if speaker is None:
            speaker = "0"
        if turns and turns[-1][0] == speaker:
            turns[-1][1] += " " + line
        else:
            turns.append([speaker, line])
    return [(s, t) for s, t in turns]


def vibevoice(ctx, inputs, settings):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    text = (inputs.get("text") or "").strip()
    if not text:
        raise ValueError("읽을 대본을 이어 주세요.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"대본이 너무 깁니다({len(text)}자). 한 번에 {MAX_TEXT_CHARS}자까지 읽습니다 — 나눠서 이어 주세요.")
    turns = parse_script(text)
    if not turns:
        raise ValueError("대본에서 읽을 글을 찾지 못했습니다. 'Speaker 1: 글' 처럼 적어 주세요.")

    refs = {}
    for n in range(1, MAX_SPEAKERS + 1):
        path = inputs.get(f"voice-{n}")
        if path:
            refs[str(n - 1)] = _read_audio(path)
    model, processor = _load(ctx)
    ctx.check_cancel()

    # 화자마다 처음 나오는 차례에 참조 목소리를 붙인다(transformers 문서 「Generating a podcast from a script」).
    conversation, seen, audios = [], set(), []
    for speaker, line in turns:
        content = [{"type": "text", "text": line}]
        if speaker in refs and speaker not in seen:
            content.append({"type": "audio", "path": f"voice-{speaker}"})
            audios.append(refs[speaker])
            seen.add(speaker)
        conversation.append({"role": speaker, "content": content})
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[prompt], audio=audios or None, return_tensors="pt")
    enc = {k: (v.to(model.device, model.dtype) if v.is_floating_point() else v.to(model.device)) for k, v in enc.items()}

    # 글 길이로 어림한 소리 토큰 수(영어 약 15자/초)를 진행률 분모로 쓴다 — 넘으면 0.95 에서 멈춘다.
    expected = max(30, int(len(text) / 15 * TOKENS_PER_SEC))
    prompt_len = enc["input_ids"].shape[1]

    class Cancel(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            ctx.check_cancel()
            made = input_ids.shape[1] - prompt_len
            if made % 15 == 0:
                ctx.progress(0.1 + 0.85 * min(1.0, made / expected), "generate")
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    if ctx.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(int(settings["seed"]))
    start = time.monotonic()
    try:
        with torch.inference_mode():
            audio = model.generate(
                **enc,
                guidance_scale=float(settings["guidance"]),
                num_diffusion_steps=int(settings["steps"]),
                stopping_criteria=StoppingCriteriaList([Cancel()]),
            )[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError("그래픽카드 메모리가 모자랍니다. 다른 AI 작업 · 프로그램을 닫고 다시 하거나, 대본을 나눠 짧게 읽혀 주세요.") from None
    took = time.monotonic() - start
    if audio is None or audio.numel() == 0:
        raise RuntimeError("목소리가 만들어지지 않았습니다. 대본을 바꿔 다시 해 주세요.")
    wav = audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    out = ctx.out_path("vibevoice.wav")
    sf.write(out, wav, SAMPLE_RATE, subtype="PCM_16")
    peak = torch.cuda.max_memory_reserved() / 2**20 if ctx.device == "cuda" else 0
    ctx.log(
        f"{len(turns)} turns, {len(refs)} voices → {wav.size / SAMPLE_RATE:.1f}s audio in {took:.1f}s, "
        f"peak VRAM reserved {peak:.0f} MiB"
    )
    ctx.progress(1.0, "done")
    return {"audio": out}


NODES = {"vibevoice": vibevoice}
