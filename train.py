import copy
import os
import re
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoReCamMasterPipeline, ModelManager, load_state_dict, WanVideoPipeline
from diffsynth.models.wan_mix_transformers_debug import VAP_MoT

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
from safetensors.torch import save_file
import gc
from lightning.pytorch.utilities import rank_zero_only


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



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        
        self.pipe.device = self.device
        if video is not None:
            pth_path = path + ".tensors.pth"
            if not os.path.exists(pth_path):
                # prompt
                prompt_emb = self.pipe.encode_prompt(text)
                # video
                video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
                # image
                if "first_frame" in batch:
                    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                    _, _, num_frames, height, width = video.shape
                    image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
                else:
                    image_emb = {}
                data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}
                torch.save(data, pth_path)
            else:
                print(f"File {pth_path} already exists, skipping.")

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)



class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch,height=496, width=496):
        self.metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in self.metadata["file_name"]]
        print(len(self.path), "videos in metadata.")

        self.path = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth") and "output" in i]
        print(len(self.path), "tensors cached in metadata (only output).")
        assert len(self.path) > 0
        self.steps_per_epoch = steps_per_epoch
        self.metadata .set_index("file_name", inplace=True)
        self.height = height
        self.width = width
        # self.max_num_frames = max_num_frames
        # self.frame_interval = frame_interval
        # self.num_frames = num_frames
        # self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame
    
    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses


    def __getitem__(self, index):
        while True:
            try:
                data = {}
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path)  # 保持随机+可复现
                path_tgt = self.path[data_id]                 # 形如 …/34-output1.png.tensors.pth

                # 1. 取出 uid 和扩展名
                base_name = os.path.basename(path_tgt)        # 34-output1.png.tensors.pth
                uid = base_name.split('-')[0]                 # 34

                # 2. 拼出对应的 cond 文件
                path_cond = os.path.join(
                    os.path.dirname(path_tgt),
                    f"{uid}-input.png.tensors.pth"
                )
                image_path = os.path.join(os.path.dirname(path_tgt),f"{uid}-input.png")
                ref_text = self.metadata.loc[f"{uid}-input.png", "text"]
                if not os.path.exists(path_cond):
                    raise FileNotFoundError(f"{path_cond} not found")

                # 3. 正常加载
                data_tgt  = torch.load(path_tgt,  weights_only=True, map_location="cpu")
                data_cond = torch.load(path_cond, weights_only=True, map_location="cpu")
                
                # 4. 组装
                data['latents'] = torch.cat(
                    (data_tgt['latents'], data_cond['latents']), dim=1
                )                                              # [C, 2*T, H, W]
                data['prompt_emb'] = data_tgt['prompt_emb']
                data['image_emb']  = {}
                data['ref_image'] = self.load_image(image_path)
                data["ref_prompt"] = ref_text
                break

            except Exception as e:
                print(f"ERROR WHEN LOADING: {e}")
                index = random.randrange(len(self.path))
        return data

    def __len__(self):
        return self.steps_per_epoch



class LightningFullModel(pl.LightningModule):
    # def __init__(
    #     self,
    #     qwen_path: str,
    #     wan_dit_path: str,
    #     wan_encoder_path: str,
    #     wan_vae_path: str,
    #     learning_rate: float = 1e-5,
    #     use_gradient_checkpointing: bool = True,
    #     use_gradient_checkpointing_offload: bool = False,
    # ):
    #     super().__init__()
    #     self.save_hyperparameters()

    #     qwen_pipe = QwenImageEditPipeline.from_pretrained(qwen_path)
    #     qwen = qwen_pipe.transformer
    #     self.text_encoder = qwen_pipe.text_encoder
    #     self.vae = qwen_pipe.vae
    #     prepare_latents_fn = qwen_pipe.prepare_latents
    #     encode_prompt = qwen_pipe.encode_prompt
    #     vae_scale_factor = qwen_pipe.vae_scale_factor

    #     del qwen_pipe
    #     # 1. 加载原始 pipeline（与训练完全一致）
    #     model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")

    #     model_manager.load_models([
    #         wan_dit_path,
    #         # wan_encoder_path,
    #         # wan_vae_path,
    #     ])
    #     # model_manager.to(device="cuda")

    #     # for name, model in model_manager.model:
    #     #     model.to(device=device, dtype=torch.bfloat16)
    #     self.pipe = WanVideoPipeline.from_model_manager(model_manager)
    #     wan = self.pipe.dit

    #     self.pipe.dit = VAP_MoT(backbone=wan, expert=qwen,prepare_latents_fn=prepare_latents_fn,encode_prompt =encode_prompt, vae_scale_factor = vae_scale_factor,cross_attn_heads=8, temporal_bias_delta=8)
    #     # del qwen
    #     # del wan
    #     # torch.cuda.empty_cache()

    #     self.pipe.scheduler.set_timesteps(1000, training=True)


    #     # 3. 确认所有参数都可训练
    #     # for p in self.model.parameters():
    #     #     p.requires_grad = True

    #     # 先冻结所有参数
    #     for p in self.pipe.denoising_model().parameters():
    #         p.requires_grad = False

    #     # 只让 self.expert2backbone 这一层可训练
    #     for p in self.pipe.denoising_model().expert2backbone.parameters():
    #         p.requires_grad = True

    #     self.pipe.denoising_model().train()
    #     total_params = sum(p.numel() for p in self.pipe.denoising_model().parameters() if p.requires_grad)
    #     print(f"Total trainable parameters (full fine-tune): {total_params}")

    #     # 4. 其余参数
    #     self.learning_rate = learning_rate
    #     self.use_gradient_checkpointing = use_gradient_checkpointing
    #     self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
    def __init__(
        self,
        qwen_path: str,
        wan_dit_path: str,
        wan_encoder_path: str,
        wan_vae_path: str,
        learning_rate: float = 1e-5,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 只保存路径，不加载模型！
        self.qwen_path = qwen_path
        self.wan_dit_path = wan_dit_path
        self.wan_encoder_path = wan_encoder_path
        self.wan_vae_path = wan_vae_path
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.pipe = None  # 模型占位

    def on_train_start(self):
        engine = getattr(self.trainer.strategy, "deepspeed_engine", None)
        if engine is None:
            return
        # 1. 阻止 ZeRO 分段保存
        engine.save_non_zero_checkpoint = lambda *a, **kw: None
        # 2. 阻止 DeepSpeed 的完整 checkpoint
        engine.save_checkpoint = lambda *a, **kw: None

    def configure_sharded_model(self):
        if self.pipe is not None:
            return  # 防止重复初始化

        # 安全加载 Qwen（不使用 device_map！）
        qwen_pipe = QwenImageEditPipeline.from_pretrained(
            self.qwen_path,
            torch_dtype=torch.bfloat16,
        )
        print(f"Device:-----------------------{qwen_pipe.device}")
        qwen = qwen_pipe.transformer
        self.text_encoder = qwen_pipe.text_encoder
        self.vae = qwen_pipe.vae
        prepare_latents_fn = qwen_pipe.prepare_latents
        encode_prompt = qwen_pipe.encode_prompt
        vae_scale_factor = qwen_pipe.vae_scale_factor
        del qwen_pipe

        # 加载 Wan 模型
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([self.wan_dit_path])
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        
        wan = self.pipe.dit
        self.pipe.dit = VAP_MoT(
            backbone=wan,
            expert=qwen,
            prepare_latents_fn=prepare_latents_fn,
            encode_prompt=encode_prompt,
            vae_scale_factor=vae_scale_factor,
            cross_attn_heads=8,
            temporal_bias_delta=8
        )

        # 冻结逻辑
        for p in self.pipe.denoising_model().parameters():
            p.requires_grad = False
        # 冻结 VAE
        for p in self.vae.parameters():
            p.requires_grad = False

        # 冻结 text encoder
        for p in self.text_encoder.parameters():
            p.requires_grad = False
            
        # for p in self.pipe.denoising_model().expert2backbone.parameters():
        #     # p.requires_grad = True 
        #     p.requires_grad = False

        for p in self.pipe.denoising_model().backbone.parameters():
            p.requires_grad = True 
        # for name, module in self.pipe.denoising_model().backbone.named_modules():
        #     if any(keyword in name for keyword in ["projector", "self_attn"]):
        #         print(f"Trainable: {name}")
        #         for param in module.parameters():
        #             param.requires_grad = True
                 
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.text_encoder.train()
        self.vae.train()
        self.pipe.denoising_model().train()
    # -------------------------------------------------------
    def forward(self, *args, **kwargs):
        return self.pipe.denoising_model()(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # if batch_idx % 100 == 0:  # 每100步打印一次
        #     for i in range(torch.cuda.device_count()):
        #         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        #         print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024**2:.1f} MB")
        #         print(f"  Cached:    {torch.cuda.memory_reserved(i)/1024**2:.1f} MB")
        #         print(f"  Total:     {torch.cuda.get_device_properties(i).total_memory/1024**2:.1f} MB")
        #         print("-" * 40)
        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        ref_image = batch["ref_image"]
        ref_prompt = batch["ref_prompt"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)
        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )
        extra_input = self.pipe.prepare_extra_input(latents) 
        origin_latents = latents.clone()
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        tgt_latent_len = noisy_latents.shape[2] // 2
        noisy_latents[:, :, tgt_latent_len:, ...] = origin_latents[:, :, tgt_latent_len:, ...]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        # Forward
        noise_pred = self.pipe.denoising_model()(
            ref_image_tokens = ref_image,
            ref_img_shapes = [(1, 64, 64)],
            ref_text_tokens = ref_prompt,
            target_video=noisy_latents,
            target_timestep=timestep,
            target_text_tokens = prompt_emb["context"],
            clip_feature=None,
            y = None,
        )

        loss = torch.nn.functional.mse_loss(
            noise_pred[:, :, :tgt_latent_len, ...].float(),
            training_target[:, :, :tgt_latent_len, ...].float(),
        )
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    # -------------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.pipe.denoising_model().backbone.parameters(), lr=self.learning_rate)

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint_dir = self.trainer.checkpoint_callback.dirpath
    #     print(f"Checkpoint directory: {checkpoint_dir}")
    #     current_step = self.global_step
    #     print(f"Current step: {current_step}")

    #     checkpoint.clear()
    #     state_dict = self.pipe.denoising_model().state_dict()
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    #     # torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))
    #     save_path = os.path.join(checkpoint_dir, f"step{current_step}.safetensors")
    #     save_file(state_dict, save_path)
    # @rank_zero_only
    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint_dir = self.trainer.checkpoint_callback.dirpath
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    #     print(f"Checkpoint directory: {checkpoint_dir}")
    #     current_step = self.global_step
    #     print(f"Current step: {current_step}")

    #     checkpoint.clear()
    #     trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
    #     trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
    #     state_dict = self.pipe.denoising_model().state_dict()
    #     torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))
    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")

        checkpoint.clear()

        # 只选 requires_grad=True 的参数
        # trainable_param_names = {
        #     name for name, param in self.pipe.denoising_model().named_parameters()
        #     if param.requires_grad
        # }

        # state_dict = self.pipe.denoising_model().state_dict()

        # # 过滤字典
        # trainable_state_dict = {
        #     name: param for name, param in state_dict.items()
        #     if name in trainable_param_names
        # }
        # save_file(trainable_state_dict, os.path.join(checkpoint_dir, f"step{current_step}.safetensors"))
        backbone_state_dict = self.pipe.denoising_model().backbone.state_dict()
        save_path = os.path.join(checkpoint_dir, f"backbone_step{current_step}.safetensors")
        save_file(backbone_state_dict, save_path)
        # proj_state_dict = self.pipe.denoising_model().expert2backbone.state_dict()
        # save_path = os.path.join(checkpoint_dir, f"proj_step{current_step}.safetensors")
        # save_file(proj_state_dict, save_path)
        # torch.save(trainable_state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))
def parse_args():
    parser = argparse.ArgumentParser(description="Train ReCamMaster")
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./dataset_root",
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./checkpoint_debug_expert_mapped_to_back_nograd",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--qwen_path",
        type=str,
        default="./qwen_image_edit",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--wan_dit_path",
        type=str,
        default="./models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--wan_encoder_path",
        type=str,
        default="./models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--wan_vae_path",
        type=str,
        default="./models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=8000,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=496,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=496,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_3",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata.csv",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--edit_model_path",
        type=str,
        default=None,
    )

    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--strategy", type=str, default="deepspeed_stage_2")

    args = parser.parse_args()
    return args

def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_file_name),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)

def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    # model = LightningModelForTrain(
    #     dit_path=args.dit_path,
    #     learning_rate=args.learning_rate,
    #     use_gradient_checkpointing=args.use_gradient_checkpointing,
    #     use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
    #     resume_ckpt_path=args.resume_ckpt_path,
    # )
    # qwen_pipe = QwenImageEditPipeline.from_pretrained(args.qwen_path,torch_dtype=torch.bfloat16,device_map="balanced")
    model = LightningFullModel(
        qwen_path=args.qwen_path,
        wan_dit_path=args.wan_dit_path,
        wan_encoder_path=args.wan_encoder_path,
        wan_vae_path=args.wan_vae_path,
        learning_rate=1e-4,
    )
    # if torch.cuda.is_available():
    #     print("\n===== GPU Memory Usage =====")
    # for i in range(torch.cuda.device_count()):
    #     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    #     print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024**2:.1f} MB")
    #     print(f"  Cached:    {torch.cuda.memory_reserved(i)/1024**2:.1f} MB")
    #     print(f"  Total:     {torch.cuda.get_device_properties(i).total_memory/1024**2:.1f} MB")
    #     print("-" * 40)
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        # master_addr=args.master_addr,
        # master_port=args.master_port,
        precision="bf16",
        # strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_weights_only=True,every_n_epochs=5,save_top_k=1)],
        logger=logger,
    )

    # trainer.strategy.deepspeed_engine.save_non_zero_checkpoint = lambda *a, **kw: None
    trainer.fit(model, dataloader)
    
    



if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
