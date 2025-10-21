import os
from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

pipeline = QwenImageEditPipeline.from_pretrained("/data/nvme1/gao/qwen_image_edit")
transformer = pipeline.transformer
print("pipeline loaded")
pipeline.to(torch.bfloat16) 
pipeline.to("cuda:0")
pipeline.set_progress_bar_config(disable=None)

image = Image.open("/home/nlab/gao/code/qwen_image_as_prompt/test.png").convert("RGB")
prompt = "make the stop sing an animal sign."


inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit.png")
    print("image saved at", os.path.abspath("output_image_edit.png"))