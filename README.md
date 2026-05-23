# AIMS - AI Media Storage

Decouple video generation into two phases: Encode -> Cache -> Generate
Save 4-5GB VRAM, batch-generate without re-encoding, hot-swap prompts instantly.

> **作者注（必读）**  
> 我不会编程，不懂原理，出问题也不会修。  
> 全程只靠：我的思路 + AI 讨论论证 + Hermes Agent 执行。  
> 为解决自己爆显存问题做的方案，实测可用。  
> 代码、工作流、文档全在这里，能用你就用。  
>  
> — 作者

---

## Concept

Standard: LoadImage -> CLIP -> VAE -> [14B Model + Sampler] -> Decode
AIMS Phase 1 (Encode): LoadImage -> CLIP -> VAE -> T5 -> Save .pth cache
AIMS Phase 2 (Generate): Load .pth cache -> [14B Model + Sampler] -> Decode

## Installation

1. Copy to ComfyUI/custom_nodes/ComfyUI-AIMS/
2. Restart ComfyUI
3. Find 5 nodes under AIMS category

Dependencies: None beyond ComfyUI built-ins (torch).
Required: ComfyUI-WanVideoWrapper (for WANVIDIMAGE_EMBEDS type)

## Nodes

| Node | Input | Output | Purpose |
|------|-------|--------|---------|
| AIMS Save Image Embeds | WANVIDIMAGE_EMBEDS | - | Save encoded image tensors |
| AIMS Load Image Embeds | file_name, char_id | WANVIDIMAGE_EMBEDS | Load cached image tensors |
| AIMS Save Text Embeds | WANVIDEOTEXTEMBEDS | - | Save encoded text tensors |
| AIMS Load Text Embeds | file_name, char_id | WANVIDEOTEXTEMBEDS | Load cached text tensors |
| AIMS Force VRAM Release | trigger | - | Unload all models, clear GPU |

## Cache Structure

ComfyUI/aims_cache/{character_id}/{file_name}.pth
Delete or re-run Phase 1 to overwrite.

## Verified Hardware

- RTX 3080 20GB | Wan2.1 I2V 14B GGUF Q6_K | 81f 640x480 | 20step | block_swap=12
- Phase 1: seconds, ~8GB VRAM
- Phase 2: ~18min, 15.4GB VRAM (vs 19.6GB standard)



## Compatibility / 兼容性

### Supported Models

| Model | I2V | T2V | Notes |
|-------|-----|-----|-------|
| Wan2.1 I2V 14B | ✅ Tested | ❌ | Image embeds: ✅ / Text embeds: ✅ |
| Wan2.1 Fun / FL2V | ✅ Likely | ❌ | Re-encode needed per model type |
| Wan2.2 Rapid | ✅ Likely | ❌ | Same WANVIDIMAGE_EMBEDS type |
| Wan2.2 Animate | ✅ Likely | ❌ | Same WANVIDIMAGE_EMBEDS type |

### Known Limitations

1. **I2V only for image embeds** — AIMSSaveImageEmbeds / AIMSLoadImageEmbeds require WanVideoImageToVideoEncode, which only exists in I2V pipelines. T2V (text-to-video) does not have image encoding.

2. **Cache is WanVideoWrapper-specific** — WANVIDIMAGE_EMBEDS and WANVIDEOTEXTEMBEDS are custom types defined by ComfyUI-WanVideoWrapper. They won't work with other video node packs (e.g. Kijai's nodes).

3. **Cache is NOT cross-version** — The internal dict keys (image_embeds, clip_context, num_frames, etc.) may differ between model versions. Re-run Phase 1 (encode) when switching models.

4. **File size** — Each .pth file is 8-16MB. Cache directory: ComfyUI/aims_cache/{character_id}/

5. **Wan2.3** — Not yet supported by WanVideoWrapper. Compatibility TBD when it arrives.

#

## Workflow Templates

Two example workflows are included in `workflows/`:

- **`wan21_aims_encode.json`** — Phase 1: Encode & cache image/text embeds
- **`wan21_aims_generate.json`** — Phase 2: Load cache & generate video

Based on the original `wan21_i2v_base.json` workflow.  
Designed for **Wan2.1 I2V 14B GGUF** (RTX 3080 20GB verified).

Created by [瑶光城](https://github.com/wenfeng11413583) — concept by user, implementation by Hermes Agent + AI collaboration.


## 中文说明

| 模型版本 | 图生视频 | 文生视频 | 说明 |
|----------|---------|---------|------|
| Wan2.1 I2V 14B | ✅ 实测通过 | ❌ | 图片+文本缓存都可用 |
| Wan2.1 Fun/FL2V | ✅ 理论上通 | ❌ | 换模型类型需重新编码 |
| Wan2.2 Rapid/Animate | ✅ 理论上通 | ❌ | 同一套 WANVIDIMAGE_EMBEDS |

**重要限制：**
- 图片缓存节点（Save/Load Image Embeds）**只能用于 I2V**，T2V 没有图片编码阶段
- 缓存格式是 WanVideoWrapper **专有**的，换别的插件不能用
- **换模型版本必须重新跑编码**，不通用
- 每个 .pth 文件 8-16MB




## Workflow Templates

Two example workflows are included in `workflows/`:

- **`wan21_aims_encode.json`** — Phase 1: Encode & cache image/text embeds
- **`wan21_aims_generate.json`** — Phase 2: Load cache & generate video

Based on the original `wan21_i2v_base.json` workflow.  
Designed for **Wan2.1 I2V 14B GGUF** (RTX 3080 20GB verified).

Created by [瑶光城](https://github.com/wenfeng11413583) — concept by user, implementation by Hermes Agent + AI collaboration.


## 中文说明

拆成两步：编码->存.pth(不加载14B模型)->读缓存->采样+解码，省4-5GB显存。
5个节点在AIMS分类下，缓存文件在ComfyUI/aims_cache/。
