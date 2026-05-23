"""
AIMS - AI Media Storage
=======================

A ComfyUI custom node pack for decoupling video generation into two phases:

  Phase 1 (Encode):  Run CLIP + VAE + T5 encoding ONCE, save results to disk
  Phase 2 (Generate): Load cached embeddings, run only sampler + decode

Benefits:
  - Save 4-5GB VRAM during generation (no encoder overhead)
  - Change seeds / batch-generate without re-encoding
  - Hot-swap prompts without re-encoding (save text embeds separately)
  - Resume from cache if ComfyUI crashes mid-generation

Requirements:
  - ComfyUI (any recent version)
  - torch (bundled with ComfyUI)
  - WanVideoWrapper custom nodes (for WANVIDIMAGE_EMBEDS / WANVIDEOTEXTEMBEDS types)

Cache Structure:
  ComfyUI/
  └── aims_cache/
      └── {character_id}/
          ├── {file_name}.pth            (image_embeds — from WanVideoImageToVideoEncode)
          └── {file_name}_text.pth       (text_embeds — from WanVideoTextEncode)

中文说明:
  本插件将 WanVideo I2V 流程拆成"编码→缓存"和"读缓存→生成"两步。
  编码阶段只跑 VAE/CLIP/T5，不加载 14B 扩散模型；
  生成阶段读缓存只跑采样+解码，显存从 ~19.6GB 降到 ~15.4GB。
"""

import os
import gc
import torch
import folder_paths
import comfy.model_management as mm

# ─── Cache directory ────────────────────────────────────────
# All cached .pth files live under ComfyUI/aims_cache/{character_id}/
AIMS_CACHE_DIR = os.path.join(folder_paths.base_path, "aims_cache")
os.makedirs(AIMS_CACHE_DIR, exist_ok=True)

# ─── Helper ─────────────────────────────────────────────────
def _ensure_char_dir(character_id: str) -> str:
    """Create and return the per-character cache subdirectory."""
    char_dir = os.path.join(AIMS_CACHE_DIR, character_id)
    os.makedirs(char_dir, exist_ok=True)
    return char_dir


# ═══════════════════════════════════════════════════════════
#  Node 1: AIMSSaveImageEmbeds
# ═══════════════════════════════════════════════════════════
class AIMSSaveImageEmbeds:
    """
    Save WANVIDIMAGE_EMBEDS to a .pth file on disk.

    Connect this node AFTER WanVideoImageToVideoEncode.
    The saved file can later be loaded by AIMSLoadImageEmbeds,
    skipping the entire VAE + CLIP encoding pipeline.

    INPUTS:
      image_embeds  : WANVIDIMAGE_EMBEDS  — output of WanVideoImageToVideoEncode
      file_name     : STRING              — base name for the .pth file (no extension)
      character_id  : STRING              — subdirectory name under aims_cache/

    OUTPUTS: (none — this is a terminal / output node)

    Example:
      LoadImage → WanVideoClipVisionEncode → WanVideoImageToVideoEncode → [AIMSSaveImageEmbeds]
                                                                         → AIMSForceVRAMRelease
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                "file_name": ("STRING", {"default": "aims_image_embeds"}),
                "character_id": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "AIMS"

    def save(self, image_embeds, file_name, character_id):
        """Serialize relevant keys from image_embeds dict to disk as .pth."""
        char_dir = _ensure_char_dir(character_id)
        save_dict = {}
        # Only persist keys that are actually present — skip VAE object references
        for key in [
            "image_embeds", "clip_context", "negative_clip_context",
            "max_seq_len", "num_frames", "lat_h", "lat_w",
            "control_embeds", "end_image",
            "fun_or_fl2v_model", "has_ref", "mask",
        ]:
            if key in image_embeds:
                save_dict[key] = image_embeds[key]
        path = os.path.join(char_dir, f"{file_name}.pth")
        torch.save(save_dict, path, pickle_protocol=4)
        print(f"[AIMS] Image embeds saved → {path}  ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")
        return ()


# ═══════════════════════════════════════════════════════════
#  Node 2: AIMSLoadImageEmbeds
# ═══════════════════════════════════════════════════════════
class AIMSLoadImageEmbeds:
    """
    Load a previously saved WANVIDIMAGE_EMBEDS from a .pth file.

    Connect this node INSTEAD of WanVideoImageToVideoEncode in the
    generation phase. The loaded tensor dict feeds directly into
    WanVideoSampler.image_embeds.

    INPUTS:
      file_name     : STRING  — base name (same as used in AIMSSaveImageEmbeds)
      character_id  : STRING  — subdirectory name (same as used in AIMSSaveImageEmbeds)

    OUTPUTS:
      image_embeds  : WANVIDIMAGE_EMBEDS — identical shape to the original encoding output

    Format validation checks for required keys on load.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_name": ("STRING", {"default": "aims_image_embeds"}),
                "character_id": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "load"
    CATEGORY = "AIMS"

    def load(self, file_name, character_id):
        path = os.path.join(AIMS_CACHE_DIR, character_id, f"{file_name}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[AIMS] Cache not found: {path}\n"
                f"  Run the encoding workflow first (AIMSSaveImageEmbeds)."
            )
        data = torch.load(path, map_location="cpu", weights_only=True)
        # Validate that the essential keys are present
        required_keys = {"image_embeds", "num_frames", "lat_h", "lat_w", "mask"}
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(
                f"[AIMS] Corrupted or incompatible cache: {path}\n"
                f"  Missing required keys: {missing}"
            )
        print(f"[AIMS] Image embeds loaded ← {path}")
        return (data,)


# ═══════════════════════════════════════════════════════════
#  Node 3: AIMSSaveTextEmbeds
# ═══════════════════════════════════════════════════════════
class AIMSSaveTextEmbeds:
    """
    Save WANVIDEOTEXTEMBEDS to a .pth file on disk.

    Connect this node AFTER WanVideoTextEncode.
    Caching text embeds lets you hot-swap prompts without
    re-running the T5 encoder.

    INPUTS:
      text_embeds   : WANVIDEOTEXTEMBEDS  — output of WanVideoTextEncode
      file_name     : STRING              — base name for the .pth file
      character_id  : STRING              — subdirectory name

    OUTPUTS: (none — terminal / output node)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "file_name": ("STRING", {"default": "aims_text_embeds"}),
                "character_id": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "AIMS"

    def save(self, text_embeds, file_name, character_id):
        char_dir = _ensure_char_dir(character_id)
        save_dict = {}
        for key in ["prompt_embeds", "negative_prompt_embeds", "echoshot"]:
            if key in text_embeds:
                save_dict[key] = text_embeds[key]
        path = os.path.join(char_dir, f"{file_name}.pth")
        torch.save(save_dict, path, pickle_protocol=4)
        print(f"[AIMS] Text embeds saved → {path}  ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")
        return ()


# ═══════════════════════════════════════════════════════════
#  Node 4: AIMSLoadTextEmbeds
# ═══════════════════════════════════════════════════════════
class AIMSLoadTextEmbeds:
    """
    Load previously saved WANVIDEOTEXTEMBEDS from a .pth file.

    Connect this node INSTEAD of WanVideoTextEncode in the
    generation phase to skip T5 encoding entirely.

    INPUTS:
      file_name     : STRING  — base name (same as used in AIMSSaveTextEmbeds)
      character_id  : STRING  — subdirectory name

    OUTPUTS:
      text_embeds   : WANVIDEOTEXTEMBEDS
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_name": ("STRING", {"default": "aims_text_embeds"}),
                "character_id": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS",)
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "load"
    CATEGORY = "AIMS"

    def load(self, file_name, character_id):
        path = os.path.join(AIMS_CACHE_DIR, character_id, f"{file_name}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[AIMS] Cache not found: {path}\n"
                f"  Run the encoding workflow first (AIMSSaveTextEmbeds)."
            )
        data = torch.load(path, map_location="cpu", weights_only=True)
        required_keys = {"prompt_embeds", "negative_prompt_embeds"}
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(
                f"[AIMS] Corrupted or incompatible cache: {path}\n"
                f"  Missing required keys: {missing}"
            )
        print(f"[AIMS] Text embeds loaded ← {path}")
        return (data,)


# ═══════════════════════════════════════════════════════════
#  Node 5: AIMSForceVRAMRelease
# ═══════════════════════════════════════════════════════════
class AIMSForceVRAMRelease:
    """
    Force-unload all models and clear GPU VRAM.

    Place this node at the end of your workflow to reclaim
    VRAM after encoding or generation is done. Useful for
    chaining multiple workflows without restarting ComfyUI.

    Just click "run" when you want to clean up.

    INPUTS:
      trigger  : ["run"]  — click the button to fire

    OUTPUTS: (none — output node)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"trigger": (["run"],)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "release"
    CATEGORY = "AIMS"

    def release(self, trigger):
        print("[AIMS] Releasing VRAM...")
        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()
        mm.soft_empty_cache()
        print("[AIMS] VRAM released — all models unloaded, cache cleared")
        return ()


# ═══════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════
NODE_CLASS_MAPPINGS = {
    "AIMSSaveImageEmbeds": AIMSSaveImageEmbeds,
    "AIMSLoadImageEmbeds": AIMSLoadImageEmbeds,
    "AIMSSaveTextEmbeds": AIMSSaveTextEmbeds,
    "AIMSLoadTextEmbeds": AIMSLoadTextEmbeds,
    "AIMSForceVRAMRelease": AIMSForceVRAMRelease,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIMSSaveImageEmbeds": "AIMS Save Image Embeds",
    "AIMSLoadImageEmbeds": "AIMS Load Image Embeds",
    "AIMSSaveTextEmbeds": "AIMS Save Text Embeds",
    "AIMSLoadTextEmbeds": "AIMS Load Text Embeds",
    "AIMSForceVRAMRelease": "AIMS Force VRAM Release",
}
