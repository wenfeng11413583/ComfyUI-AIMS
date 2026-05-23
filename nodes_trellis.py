"""
AIMS TRELLIS 3D Generator
ComfyUI node for Microsoft TRELLIS 3D generation via NVIDIA NIM API.
Runs in the cloud - no local VRAM required.
"""

import os
import base64
import requests
import urllib3
import folder_paths

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TRELLIS_OUTPUT_DIR = os.path.join(folder_paths.get_output_directory(), "trellis_3d")
os.makedirs(TRELLIS_OUTPUT_DIR, exist_ok=True)

class AIMSTrellisTextTo3D:
    """
    Generate a 3D model (GLB) from a text prompt using TRELLIS API.
    Inputs: prompt, api_key, filename, seed
    Outputs: glb_path (STRING), model_name (STRING)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "a beautiful ancient Chinese woman in red hanbok, standing elegantly", "multiline": True}),
                "api_key": ("STRING", {"default": "", "password": True}),
                "filename": ("STRING", {"default": "trellis_model"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("glb_path", "model_name",)
    FUNCTION = "generate"
    CATEGORY = "AIMS"

    def generate(self, prompt, api_key, filename):
        if not api_key or not api_key.startswith("nvapi-"):
            raise ValueError("Valid NVIDIA API Key (nvapi-...) required. Get one from https://build.nvidia.com/microsoft/trellis")

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"prompt": prompt}

        print(f"[AIMS TRELLIS] Generating: {prompt[:80]}... (2-3 min)")

        response = requests.post(
            "https://ai.api.nvidia.com/v1/genai/microsoft/trellis",
            headers=headers, json=payload, timeout=600, verify=False
        )

        if response.status_code == 200:
            glb_data = base64.b64decode(response.json()["artifacts"][0]["base64"])
            safe_name = "".join(c for c in filename if c.isalnum() or c in "_-").strip() or "trellis_model"
            glb_path = os.path.join(TRELLIS_OUTPUT_DIR, f"{safe_name}.glb")
            with open(glb_path, "wb") as f:
                f.write(glb_data)
            print(f"[AIMS TRELLIS] OK -> {glb_path} ({len(glb_data)/1024/1024:.1f}MB)")
            return (glb_path, safe_name)
        else:
            raise ValueError(f"TRELLIS API Error ({response.status_code}): {response.text[:300]}")


NODE_CLASS_MAPPINGS = {"AIMSTrellisTextTo3D": AIMSTrellisTextTo3D}
NODE_DISPLAY_NAME_MAPPINGS = {"AIMSTrellisTextTo3D": "AIMS TRELLIS Text to 3D"}
