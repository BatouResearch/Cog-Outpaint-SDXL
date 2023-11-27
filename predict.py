import hashlib
import json
import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from weights import WeightsDownloadCache
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from safetensors.torch import load_file
from transformers import CLIPImageProcessor
from dataset_and_utils import TokenEmbeddingsHandler

CONTROL_CACHE = "control-cache"
SDXL_MODEL_CACHE = "./sdxl-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def load_trained_weights(self, weights, pipe):
        print("loading custom weights")
        from no_init import no_init_or_tensor

        weights = str(weights)

        self.tuned_weights = weights

        local_weights_cache = self.weights_cache.ensure(weights)

        # load UNET
        print("Loading fine-tuned model")
        self.is_lora = False

        maybe_unet_path = os.path.join(local_weights_cache, "unet.safetensors")
        if not os.path.exists(maybe_unet_path):
            print("Does not have Unet. assume we are using LoRA")
            self.is_lora = True

        if not self.is_lora:
            print("Loading Unet")

            new_unet_params = load_file(
                os.path.join(local_weights_cache, "unet.safetensors")
            )
            # this should return _IncompatibleKeys(missing_keys=[...], unexpected_keys=[])
            pipe.unet.load_state_dict(new_unet_params, strict=False)

        else:
            print("Loading Unet LoRA")

            unet = pipe.unet

            tensors = load_file(os.path.join(local_weights_cache, "lora.safetensors"))

            unet_lora_attn_procs = {}
            name_rank_map = {}
            for tk, tv in tensors.items():
                # up is N, d
                if tk.endswith("up.weight"):
                    proc_name = ".".join(tk.split(".")[:-3])
                    r = tv.shape[1]
                    name_rank_map[proc_name] = r

            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                with no_init_or_tensor():
                    module = LoRAAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=name_rank_map[name],
                    )
                unet_lora_attn_procs[name] = module.to("cuda", non_blocking=True)

            unet.set_attn_processor(unet_lora_attn_procs)
            unet.load_state_dict(tensors, strict=False)

        # load text
        handler = TokenEmbeddingsHandler(
            [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
        )
        handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

        # load params
        with open(os.path.join(local_weights_cache, "special_params.json"), "r") as f:
            params = json.load(f)
        self.token_map = params

        self.tuned_model = True

    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        self.tuned_model = False
        self.tuned_weights = None
        if str(weights) == "weights":
            weights = None

        self.weights_cache = WeightsDownloadCache()

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        if not os.path.exists(SDXL_MODEL_CACHE):
            download_weights(SDXL_URL, SDXL_MODEL_CACHE)

        controlnet = ControlNetModel.from_pretrained(
            CONTROL_CACHE,
            torch_dtype=torch.float16,
        )

        print("Loading SDXL Controlnet pipeline...")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.pipe.to("cuda")
        self.is_lora = False
        if weights or os.path.exists("./trained-model"):
            self.load_trained_weights(weights, self.pipe)


        print("setup took: ", time.time() - start)

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")
    
    def resize_image(self, image):
        image_width, image_height = image.size
        print("Original width:"+str(image_width)+", height:"+str(image_height))
        new_width, new_height = self.resize_to_allowed_dimensions(image_width, image_height)
        print("new_width:"+str(new_width)+", new_height:"+str(new_height))
        image = image.resize((new_width, new_height))
        return image, new_width, new_height
    
    def resize_to_allowed_dimensions(self, width, height):
        """
        Function re-used from Lucataco's implementation of SDXL-Controlnet for Replicate
        """
        # List of SDXL dimensions
        allowed_dimensions = [
            (512, 2048), (512, 1984), (512, 1920), (512, 1856),
            (576, 1792), (576, 1728), (576, 1664), (640, 1600),
            (640, 1536), (704, 1472), (704, 1408), (704, 1344),
            (768, 1344), (768, 1280), (832, 1216), (832, 1152),
            (896, 1152), (896, 1088), (960, 1088), (960, 1024),
            (1024, 1024), (1024, 960), (1088, 960), (1088, 896),
            (1152, 896), (1152, 832), (1216, 832), (1280, 768),
            (1344, 768), (1408, 704), (1472, 704), (1536, 640),
            (1600, 640), (1664, 576), (1728, 576), (1792, 576),
            (1856, 512), (1920, 512), (1984, 512), (2048, 512)
        ]
        # Calculate the aspect ratio
        aspect_ratio = width / height
        print(f"Aspect Ratio: {aspect_ratio:.2f}")
        # Find the closest allowed dimensions that maintain the aspect ratio
        closest_dimensions = min(
            allowed_dimensions,
            key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
        )
        return closest_dimensions

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    def make_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1]
        image[image_mask > 0.5] = 1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    def add_outpaint_pixels(self, image, outpaint_direction, outpaint_size, color):
        """
        Outpaints the given PIL image in the specified direction by the given size with white pixels.
        """
        original_width, original_height = image.size

        if outpaint_direction == 'left':
            new_size = (original_width + outpaint_size, original_height)
            paste_position = (outpaint_size, 0)
        elif outpaint_direction == 'right':
            new_size = (original_width + outpaint_size, original_height)
            paste_position = (0, 0)
        elif outpaint_direction == 'up':
            new_size = (original_width, original_height + outpaint_size)
            paste_position = (0, outpaint_size)
        else:  # 'down'
            new_size = (original_width, original_height + outpaint_size)
            paste_position = (0, 0)

        new_image = Image.new("RGB", new_size, color)
        new_image.paste(image, paste_position)
        return new_image

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        outpaint_direction: str = Input(
            description="In what direction to outpaint",
            choices=["left", "right", "up", "down"],
            default="down",
        ),
        outpaint_size: int = Input(
            description="How many pixels the mask should grow in the chosen direction",
            ge=512,
            le=128,
            default=128
        ),
        image: Path = Input(
            description="Input image to inpaint",
            default=None,
        ),
        mask: Path = Input(
            description="Binary image, white pixels will be inpainted",
            default=None,
        ),
        condition_scale: float = Input(
            description="The bigger this number is, the more ControlNet interferes",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        lora_weights: str = Input(
            description="Replicate LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        ),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),

    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if lora_weights:
            self.load_trained_weights(lora_weights, self.pipe)

        # OOMs can leave vae in bad state
        if self.pipe.vae.dtype == torch.float32:
            self.pipe.vae.to(dtype=torch.float16)

        sdxl_kwargs = {}
        if self.tuned_model:
            # consistency with fine-tuning API
            for k, v in self.token_map.items():
                prompt = prompt.replace(k, v)


        image = self.add_outpaint_pixels(self.load_image(image), outpaint_direction, outpaint_size, "black")
        mask  = self.add_outpaint_pixels(self.load_image(mask), outpaint_direction, outpaint_size, "white")
        image, width, height = self.resize_image(image)

        sdxl_kwargs["image"] = image
        sdxl_kwargs["mask"] = mask
        sdxl_kwargs["control_image"] = self.make_inpaint_condition(image, mask)
        sdxl_kwargs["controlnet_conditioning_scale"] = condition_scale
        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height
        pipe = self.pipe

        if not apply_watermark:
            # toggles watermark for this prediction
            watermark_cache = pipe.watermark
            pipe.watermark = None

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        if self.is_lora:
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        output = pipe(**common_args, **sdxl_kwargs)

        if not apply_watermark:
            pipe.watermark = watermark_cache

        _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, nsfw in enumerate(has_nsfw_content):
            if nsfw:
                print(f"NSFW content detected in image {i}")
                continue
            output_path = f"/tmp/out-{i}.png"
            output.images[i].save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
