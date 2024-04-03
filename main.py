import random
import sys
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import mediapy as media

# Parameters
num_inference_steps = 4
use_lora = False
model_type = "lora" if use_lora else "unet"
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = f"sdxl_lightning_{num_inference_steps}step_{model_type}.safetensors"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Main code
def main():
    # Load UNet model
    unet_config = UNet2DConditionModel.load_config(base, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config).to(device)

    unet.load_state_dict(
        load_file(
            hf_hub_download(
                repo,
                ckpt,
                ),
            device=device,
            ),
        )

    # Instantiate Stable Diffusion XL Pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base,
        unet=unet,
        use_safetensors=True,
        variant="fp16",
        ).to(device)

    if use_lora:
        pipe.load_lora_weights(hf_hub_download(repo, ckpt))
        pipe.fuse_lora()

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        )

    # Additional code snippet
    prompt = "helicopter is landing in the forest "
    seed = random.randint(0, sys.maxsize)

    images = pipe(
        prompt=prompt,
        guidance_scale=0.0,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device).manual_seed(seed),
        ).images

    print(f"Prompt:\t{prompt}\nSeed:\t{seed}")
    media.show_images(images)
    images[0].save("output.jpg")

if __name__ == "__main__":
    main()
