import copy
import os
import re
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoReCamMasterPipeline, ModelManager, load_state_dict, WanVideoPipeline,save_video
from diffsynth.models.wan_mix_transformers import VAP_MoT
from diffusers import QwenImageEditPipeline
import torchvision
from PIL import Image
import numpy as np
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import shutil
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Optional

def inject_lora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    bias: str = "none",
):
    """
    给 model 插入 LoRA，不额外加任何新层。
    target_modules=None 时，对所有 nn.Linear / nn.Conv* 生效。
    """
    if target_modules is None:
        target_modules = []
        for n, m in model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                target_modules.append(n)

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        modules_to_save=None,  # 不额外保留任何全参模块
    )
    return get_peft_model(model, lora_config)


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=496, width=496, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        while True:
            try:
                text = self.text[data_id]
                path = self.path[data_id]
                if self.is_image(path):
                    if self.is_i2v:
                        raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
                    video = self.load_image(path)
                else:
                    video = self.load_video(path)
                if self.is_i2v:
                    video, first_frame = video
                    data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
                else:
                    data = {"text": text, "video": video, "path": path}
                break
            except:
                data_id += 1
        return data
    

    def __len__(self):
        return len(self.path)

def parse_args(): 
    parser = argparse.ArgumentParser(description="ReCamMaster Inference")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/nvme1/gao/dataset_root",
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/data1/gao/video_edit/checkpoint/lightning_logs/version_1/checkpoints/step66000.safetensors",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_full__66000",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--cam_type",
        type=str,
        default=1,
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    qwen_pipe = QwenImageEditPipeline.from_pretrained("/data/nvme1/gao/qwen_image_edit").to(torch.bfloat16) .to("cuda:0")
    qwen = qwen_pipe.transformer.to(dtype=torch.bfloat16).to(device="cuda")
    prepare_latents_fn = qwen_pipe.prepare_latents
    encode_prompt = qwen_pipe.encode_prompt
    vae_scale_factor = qwen_pipe.vae_scale_factor
    del qwen_pipe
    # 1. 加载原始 pipeline（与训练完全一致）
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    # model_manager.load_models([
    #     "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    #     "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    #     "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    # ])
    model_manager.load_models([
       "/data/nvme1/gao/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "/data/nvme1/gao/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "/data/nvme1/gao/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    model_manager.to(device="cuda")

    # for name, model in model_manager.model:
    #     model.to(device=device, dtype=torch.bfloat16)
    pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda")
    wan = pipe.dit

    pipe.dit = VAP_MoT(backbone=wan, expert=qwen,prepare_latents_fn=prepare_latents_fn,encode_prompt =encode_prompt, vae_scale_factor = vae_scale_factor,cross_attn_heads=8, temporal_bias_delta=8).to("cuda").to(dtype=torch.bfloat16)
    del qwen
    del wan
    pipe.scheduler.set_timesteps(1000, training=True)
    pipe.dit.eval()
    
    # # 2. 获取原始 denoising model
    # dit = pipe.dit

    # # 3. 加载 checkpoint
    # state_dict = torch.load(args.ckpt_path, map_location="cuda")
    # dit.load_state_dict(state_dict)

    # # 4. 替换 pipeline
    # pipe.dit = dit.to("cuda").eval()

    # source_video

    # 5. dataset
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        args,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    os.makedirs(args.output_dir, exist_ok=True)
    for batch_idx, batch in enumerate(dataloader):
        target_text = batch["text"]
        source_video = batch["video"]
        file_name = os.path.basename(batch['path'][0])
        # target_camera = batch["camera"]

        video = pipe(
            prompt=target_text,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            source_video=source_video,
            cfg_scale=args.cfg_scale,
            num_inference_steps=50,
            seed=0, tiled=True
        )
        img = video[0]          # [C, T, H, W] 或 [B, C, T, H, W]

        img.save(os.path.join(args.output_dir, file_name))

    # 5. 推理
    # video_frames = pipe(
    #     prompt="a cat wearing sunglasses, cyberpunk style",
    #     negative_prompt="blurry, low quality",
    #     cfg_scale=5.0,
    #     num_inference_steps=50,
    #     seed=0, tiled=True
    # )
    # # 6. 保存
    # save_video(video_frames, "output.mp4", fps=30, quality=5)