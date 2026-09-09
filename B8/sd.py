import torch
from diffusers import StableDiffusionPipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fp16 NaNs on the T500 (Turing): latents overflow to NaN mid-run -> black images.
# The fix: run in fp32 with sequential CPU offload (fits in 4 GB, ~1-2 min/image).
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.enable_sequential_cpu_offload()


prompts = [
    "a photorealistic cat sitting on a windowsill, soft light",
    "an oil painting of two moons over a mountain lake",
    "a tiny frog wearing a knitted sweater, studio photo",
]
for i, p in enumerate(prompts):
    img = pipe(p, guidance_scale=7.5, num_inference_steps=25).images[0]
    img.save(f'sd_sample_{i}.png')
    
