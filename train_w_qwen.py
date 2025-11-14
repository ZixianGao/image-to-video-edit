import copy
import os
import re
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoReCamMasterPipeline, ModelManager, load_state_dict, WanVideoPipeline
# from diffsynth.models.wan_mix_transformers_debug2 import VAP_MoT
import math
from typing import Any, Callable, Dict, List, Optional, Union
import inspect

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

import torch, os, json
from diffsynth import load_state_dict


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

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None

class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.qwen_pipe = QwenImageEditPipeline.from_pretrained("/work/gj40/j40001/code/image-to-video-edit/qwen_image_edit",torch_dtype=torch.bfloat16).to("cuda")
        self.vae_scale_factor = 2 ** len(self.qwen_pip.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        self.pipe.device = self.device
        if video is not None: 
            pth_path = path + ".tensors.pth"
            if not os.path.exists(pth_path):
                # prompt
                prompt_emb = self.pipe.encode_prompt(text)

                #qwen image:
                image =  Image.open(path).convert("RGB")
                image_size = image[0].size if isinstance(image, list) else image.size
                calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
                height = None
                width = None
                height = height or calculated_height
                width = width or calculated_width
                multiple_of = self.vae_scale_factor * 2
                width = width // multiple_of * multiple_of
                height = height // multiple_of * multiple_of
                image = self.qwen_pipe.image_processor.resize(image, calculated_height, calculated_width)
                prompt_image = image
                image = self.qwen_pipe.image_processor.preprocess(image, calculated_height, calculated_width)
                image = image.unsqueeze(2)
                ref_prompt_emb, ref_prompt_embeds_mask = self.qwen_pipe.encode_prompt(prompt = text,image = prompt_image,device=torch.device("cuda"),num_images_per_prompt=1,prompt_embeds =None,prompt_embeds_mask = None,max_sequence_length=512)
                num_channels_latents = self.qwen_pipe.transformer.config.in_channels // 4
                ref_latents, image_latents = self.qwen_pipe.prepare_latents(
                    image,
                    1 * 1,
                    num_channels_latents,
                    height,
                    width,
                    ref_prompt_emb.dtype,
                    torch.device("cuda"),
                    torch.manual_seed(0),
                    None,
                )
                img_shapes = [
                            [
                                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
                            ]
                        ] * 1


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
                data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb, "ref_prompt_emb": ref_prompt_emb, "ref_prompt_mask": ref_prompt_embeds_mask, "ref_latents": ref_latents,"ref_img_latents": image_latents,"image_shapes": img_shapes}
                torch.save(data, pth_path)
            else:
                # os.remove(pth_path)
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
                # image_path = os.path.join(os.path.dirname(path_tgt),f"{uid}-input.png")
                # ref_text = self.metadata.loc[f"{uid}-input.png", "text"]
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
                data['ref_image_latents'] = data_cond['ref_img_latents']
                data["ref_prompt"] = data_cond['ref_prompt_emb']
                data["ref_prompt_mask"] = data_cond['ref_prompt_mask']
                data["img_shapes"] = data_cond['image_shapes']
                data['ref_latents'] = data_cond['ref_latents']

                break

            except Exception as e:
                print(f"ERROR WHEN LOADING: {e}")
                index = random.randrange(len(self.path))
        return data

    def __len__(self):
        return self.steps_per_epoch

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def tensor_to_tuple(x):
    if isinstance(x, torch.Tensor):
        return int(x.item())
    elif isinstance(x, list):
        # 如果这一层是 [tensor, tensor, tensor] 这样的列表，就转成 tuple
        if all(isinstance(i, torch.Tensor) for i in x):
            return tuple(int(i.item()) for i in x)
        else:
            return [tensor_to_tuple(i) for i in x]
    else:
        return x

class LayerAdapter(nn.Module):
    """Simple linear adapter + optional LayerNorm to map between dims."""
    def __init__(self, in_dim: int, out_dim: int, use_norm: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.norm(self.lin(x))

class LightningFullModel(pl.LightningModule):
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
        self.scheduler = qwen_pipe.scheduler
        self.progress_bar = qwen_pipe.progress_bar
        self.attention_kwargs = None
        self.interrupt = False
        print(f"Device:-----------------------{qwen_pipe.device}")
        self.qwen_transformer = qwen_pipe.transformer
        vae_scale_factor = qwen_pipe.vae_scale_factor
        del qwen_pipe

        # 加载 Wan 模型
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([self.wan_dit_path])
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.dit.expert2backbone = LayerAdapter(3072, 1536)
        # wan = self.pipe.dit
        # self.pipe.dit = VAP_MoT(
        #     backbone=wan
        # )

        # 冻结逻辑
        for p in self.pipe.denoising_model().parameters():
            p.requires_grad = False
        for p in self.qwen_transformer.parameters():
            p.requires_grad = False
        # for p in self.pipe.denoising_model().expert2backbone.parameters():
        #     # p.requires_grad = True 
        #     p.requires_grad = False
        for p in self.pipe.denoising_model().parameters():
            p.requires_grad = True 
        for p in self.pipe.dit.expert2backbone.parameters():
            p.requires_grad = True 
        # for name, module in self.pipe.denoising_model().backbone.named_modules():
        #     if any(keyword in name for keyword in ["projector", "self_attn"]):
        #         print(f"Trainable: {name}")
        #         for param in module.parameters():
        #             param.requires_grad = True
                 
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.denoising_model().train()
        self.qwen_transformer.eval()
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
        # ref_image = batch["ref_image"]
        # ref_prompt = batch["ref_prompt"]

        ref_image_latents = batch['ref_image_latents'][0].to(self.device)
        ref_prompt = batch["ref_prompt"][0].to(self.device)
        ref_prompt_mask = batch["ref_prompt_mask"][0].to(self.device)
        img_shapes = batch["img_shapes"]
        img_shapes = tensor_to_tuple(img_shapes)
        ref_latents = batch["ref_latents"][0].to(self.device)
        #qwen :
        num_inference_steps = 50
        sigmas = None
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = ref_latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        ref_timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            self.pipe.device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(ref_timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(ref_timesteps)
        guidance_scale = 1.0
        if self.qwen_transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=self.pipe.device, dtype=torch.float32)
            guidance = guidance.expand(ref_latents.shape[0])
        else:
            guidance = None

        # if self.attention_kwargs is None:
        #     self._attention_kwargs = {}
        txt_seq_lens = ref_prompt_mask.sum(dim=1).tolist() if ref_prompt_mask is not None else None
        self.scheduler.set_begin_index(0)
        hidden_states_last = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(ref_timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = ref_latents
                if ref_image_latents is not None:
                    latent_model_input = torch.cat([ref_latents, ref_image_latents], dim=1)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(ref_latents.shape[0]).to(ref_latents.dtype).to(self.device)
                with self.qwen_transformer.cache_context("cond"):
                    noise_pred,hidden_states_list = self.qwen_transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=ref_prompt_mask,
                        encoder_hidden_states=ref_prompt,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )
                    noise_pred = noise_pred[:, : ref_latents.size(1)]

                if i == len(ref_timesteps) - 1:
                    hidden_states_last = hidden_states_list[-30:]
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = ref_latents.dtype
                ref_latents = self.scheduler.step(noise_pred, t, ref_latents, return_dict=False)[0]
                if ref_latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        ref_latents = ref_latents.to(latents_dtype)


                # call the callback, if provided
                if i == len(ref_timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)
        # Loss
        # wan video
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
            ref_image_latents = hidden_states_last,
            x=noisy_latents,
            timestep=timestep,
            context = prompt_emb["context"],
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
        return torch.optim.AdamW(self.pipe.denoising_model().parameters(), lr=self.learning_rate)

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")

        checkpoint.clear()


        backbone_state_dict = self.pipe.denoising_model().state_dict()
        torch.save(backbone_state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))

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
        default="./dataset_root_debug",
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./checkpoint_with_qwen",
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
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_weights_only=True,every_n_epochs=1,save_top_k=1)],
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
