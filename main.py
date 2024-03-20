import argparse
import random
import sys
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
import mediapy as media

def load_model(num_inference_steps: int, use_lora: bool, device: str) -> StableDiffusionXLPipeline:
    model_type = "lora" if use_lora else "unet"
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = f"sdxl_lightning_{num_inference_steps}step_{model_type}.safetensors"

    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float32)

    unet.load_state_dict(
        load_file(
            hf_hub_download(repo, ckpt),
            device=device,
        ),
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base,
        unet=unet,
        torch_dtype=torch.float32,
        use_safetensors=True,
        variant="fp16",
    ).to(device)

    if use_lora:
        pipe.load_lora_weights(hf_hub_download(repo, ckpt))
        pipe.fuse_lora()

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    return pipe

def generate_images(prompt: str, num_inference_steps: int, use_lora: bool, device: str) -> None:
    seed = random.randint(0, sys.maxsize)

    pipe = load_model(num_inference_steps, use_lora, device)

    images = pipe(
        prompt=prompt,
        guidance_scale=0.0,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device).manual_seed(seed),
    ).images

    print(f"Prompt:\t{prompt}\nSeed:\t{seed}")
    media.show_images(images)
    images[0].save("output.jpg")

def main():
    parser = argparse.ArgumentParser(description="Generate images from text prompt")
    parser.add_argument("prompt", type=str, help="Text prompt")
    parser.add_argument("--num-inference-steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    args = parser.parse_args()

    generate_images(args.prompt, args.num_inference_steps, args.use_lora, args.device)

if __name__ == "__main__":
    main()
