import torch
from diffusers import AutoencoderKL, DiffusionPipeline, ControlNetModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)


CONTROLC_CACHE = "control-canny-cache"
SDXL_MODEL_CACHE = "./sdxl-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_CACHE = "feature-cache"
FEATURE_NAME = "Intel/dpt-hybrid-midas"

better_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

pipe = DiffusionPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    vae=better_vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.save_pretrained(SDXL_MODEL_CACHE, safe_serialization=True)

safety = StableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker",
    #torch_dtype=torch.float16,
)
safety.save_pretrained(SAFETY_CACHE)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    #torch_dtype=torch.float16
)
controlnet.save_pretrained(CONTROLC_CACHE)
