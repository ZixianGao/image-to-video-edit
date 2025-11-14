# vap_mot.py
import math
from typing import Optional, Tuple, List, Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
# from diffsynth import ModelManager, load_state_dict, WanVideoPipeline,save_video
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffusers import QwenImageEditPipeline
from torchvision.transforms.functional import to_pil_image

# ---------- Assumptions ----------
# - Your WanModel and QwenImageTransformer2DModel classes are importable.
# - Utilities used by WanModel (sinusoidal_embedding_1d, precompute_freqs_cis_3d) are accessible.
# - QwenImageTransformer2DModel exposes .img_in, .time_text_embed, .pos_embed and accepts the same
#   arguments as in your pasted class.
#
# If names differ in your workspace, import accordingly:
# from your_wan_module import WanModel, sinusoidal_embedding_1d
# from your_qwen_module import QwenImageTransformer2DModel
#
# For now we reference them as types in typing hints only.

# ---------- Helper modules ----------


device = torch.device("cuda")

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


class LayerAdapter(nn.Module):
    """Simple linear adapter + optional LayerNorm to map between dims."""
    def __init__(self, in_dim: int, out_dim: int, use_norm: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.norm(self.lin(x))


# ---------- Mixture-of-Transformers wrapper ----------

class VAP_MoT(nn.Module):
    """
    VAP Mixture-of-Transformers wrapper.
    backbone: WanModel (video DiT) -> frozen (generate)
    expert: QwenImageTransformer2DModel (image transformer) -> trainable (reference understanding)
    """

    def __init__(
        self,
        backbone,  
    ):
        super().__init__()
        self.backbone = backbone


        # dims
        self.backbone_dim = getattr(backbone, "dim", None)
        if self.backbone_dim is None:
            raise ValueError("Backbone must expose attribute .dim (e.g., WanModel.dim)")

        # self.expert_dim = getattr(expert, "inner_dim", None)
        # if self.expert_dim is None:
        #     raise ValueError("Expert must expose attribute .inner_dim (e.g., QwenImageTransformer2DModel.inner_dim)")
        self.expert_dim = 3072

        # # Adapters for mapping token dims for cross-attention
        # # expert -> backbone space (used when backbone queries expert)
        self.expert2backbone = LayerAdapter(self.expert_dim, self.backbone_dim)
        # backbone -> expert space (used when expert queries backbone)
        # self.backbone2expert = LayerAdapter(self.backbone_dim, self.expert_dim)



        # metadata
        self.backbone_layers = len(self.backbone.blocks)
        # self.expert_layers = len(self.expert.transformer_blocks)

        # if layer counts mismatch we simply iterate up to max and skip absent blocks
        # Note: we *do not* modify inner blocks; we only call them and then apply our cross-attn.

    def _compute_backbone_time_emb(self, timestep: torch.Tensor):
        # Use backbone.time_embedding + time_projection as in WanModel
        t = self.backbone.time_embedding(sinusoidal_embedding_1d(self.backbone.freq_dim, timestep))
        t_mod = self.backbone.time_projection(t).unflatten(1, (6, self.backbone.dim))
        return t, t_mod


    def forward(
        self,
        ref_image_latents: List,
        target_video: torch.Tensor,
        target_timestep: torch.Tensor,
        target_text_tokens: Optional[torch.Tensor],
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs
    ):
        """
        Inputs:
        - ref_image_tokens: (B, N_ref, in_channels_for_expert_img_in)  (raw image patch embeddings or flattened patch pixels)
        - ref_img_shapes: image shapes list as used by expert.pos_embed (list of tuples)
        - ref_text_tokens: tokens that correspond to reference caption (if any) (B, L_ref, joint_attention_dim)
        - target_video: (B, C, F, H, W) video frames in pixel or latent form that WanModel.patchify expects
        - target_timestep: tensor (B,) indicating timesteps used by WanModel
        - target_text_tokens: (B, L_tar, text_dim) textual context for backbone
        - clip_feature, y: optional WanModel args (same semantics as WanModel.forward)
        """
        B = target_video.shape[0]
        # --- Prepare backbone side (WanModel) ---
        # Compute time embeddings and patchify target video exactly like WanModel.forward
        t = self.backbone.time_embedding(sinusoidal_embedding_1d(self.backbone.freq_dim, target_timestep))
        t_mod = self.backbone.time_projection(t).unflatten(1, (6, self.backbone.dim))
        # prepare text context for backbone: embed via backbone.text_embedding if provided
        if target_text_tokens is not None:
            backbone_context = self.backbone.text_embedding(target_text_tokens)
        else:
            # fallback to zeros
            device = target_video.device
            backbone_context = torch.zeros(B, 1, self.backbone.dim, device=device)

        if self.backbone.has_image_input and clip_feature is not None:
            clip_embdding = self.backbone.img_emb(clip_feature)
            backbone_context = torch.cat([clip_embdding, backbone_context], dim=1)

        # patchify target video
        backbone_tokens, grid = self.backbone.patchify(target_video)
        # backbone_tokens, grid = self.backbone.patchify(target_video, control_camera_latents_input=y)  # (B, N_t, D_back)
        f, h, w = grid
        # freqs used in WanModel -- we re-create using backbone.freqs (precomputed) to be consistent
        freqs = torch.cat([
            self.backbone.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.backbone.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.backbone.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(backbone_tokens.device)


        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        for idx in range(self.backbone_layers):
            bblock = self.backbone.blocks[idx]

            # 映射专家后半部分
            # expert_idx = idx + (self.expert_layers - self.backbone_layers)  # 31-60 层
            hidden_expert_to_back = ref_image_latents[idx]
            expert_mapped_to_back = self.expert2backbone(hidden_expert_to_back)
            # expert_mapped_to_back = None
            try:
                # backbone_tokens = bblock(backbone_tokens, backbone_context, t_mod, freqs, expert_token=expert_mapped_to_back)
                if self.training and use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            backbone_tokens = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(bblock),
                                backbone_tokens, backbone_context, t_mod, freqs,
                                expert_token=expert_mapped_to_back,
                                use_reentrant=False,
                            )
                    else:
                        backbone_tokens = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(bblock),
                            backbone_tokens, backbone_context, t_mod, freqs,
                            expert_token=expert_mapped_to_back,
                            use_reentrant=False,
                        )
                else:
                    backbone_tokens = bblock(backbone_tokens, backbone_context, t_mod, freqs, expert_token=expert_mapped_to_back)
            except TypeError:
                backbone_tokens = bblock(backbone_tokens, backbone_context, t_mod, freqs)


        # --- final backbone head to produce video output ---
        # WanModel expects x (B, N, D) and t (original t) for head
        x_out = self.backbone.head(backbone_tokens, t)  # returns patchized output
        x_out = self.backbone.unpatchify(x_out, (f, h, w))
        # Return generated video (and optionally expert features)
        return x_out



if __name__ == "__main__":
    # model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    # model_manager.load_models([
    #    "/data/nvme1/gao/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    #     "/data/nvme1/gao/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    #     "/data/nvme1/gao/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    # ])
    # model_manager.to(device="cuda")    
    # pipe = WanVideoPipeline.from_model_manager(model_manager, device="cuda").to(dtype=torch.bfloat16)
    # wan = pipe.dit

    # qwen_pipe = QwenImageEditPipeline.from_pretrained("/data/nvme1/gao/qwen_image_edit")
    # qwen = qwen_pipe.transformer.to(dtype=torch.bfloat16).to(device="cuda")
    # # If you have checkpoints, load them first:
    # # wan.load_state_dict(torch.load("wan_checkpoint.pth"))
    # # qwen.load_state_dict(torch.load("qwen_checkpoint.pth"))

    # # Build Mixture-of-Transformers wrapper
    # mot = VAP_MoT(backbone=wan, expert=qwen, cross_attn_heads=8, temporal_bias_delta=8).to(device="cuda").to(dtype=torch.bfloat16)
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "/data/nvme1/gao/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "/data/nvme1/gao/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "/data/nvme1/gao/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])

    # 2. 从 CPU 模型构建 backbone，但不搬到 GPU
    wan = WanVideoPipeline.from_model_manager(model_manager, device="cpu").dit

    # 3. Qwen transformer 也只在 CPU
    qwen_pipe = QwenImageEditPipeline.from_pretrained("/data/nvme1/gao/qwen_image_edit")
    qwen = qwen_pipe.transformer.to(dtype=torch.bfloat16).to(device="cuda")
    prepare_latents_fn = qwen_pipe.prepare_latents
    encode_prompt = qwen_pipe.encode_prompt
    # 4. 创建 VAP_MoT，并一次性搬到 GPU
    mot = VAP_MoT(backbone=wan, expert=qwen,prepare_latents_fn=prepare_latents_fn, encode_prompt=encode_prompt,cross_attn_heads=8, temporal_bias_delta=8).to("cuda").to(dtype=torch.bfloat16)
    del qwen


    # Prepare dummy inputs (replace with your real tensors)
    B = 1
    # example reference image tokens: shape (B, N_ref, in_channels_for_qwen.img_in)
    C_in = 64  # in_channels
    # H, W = 500,500
    # patch_size = 2
    # seq_len_img = (H // patch_size) * (W // patch_size)
    # ref_image_tokens = torch.randn(B, 4096, C_in).to(dtype=torch.bfloat16).to(device)
    ref_image_tokens = torch.randn(1, 3, 1, 496, 496).to(device)

    # ref_image_tokens = torch.randn(1, 3, 1, 1024, 1024).to(device)
    # ref image shapes for qwen.pos_embed (list of tuples)
    ref_img_shapes = [(1, 64, 64)]

    # reference text tokens (joint_attention_dim)
    ref_text_tokens = torch.randn(B, 32, 3584).to(dtype=torch.bfloat16).to(device)


    # target video: (B, C, F, H, W) (e.g., latent or pixel format expected by WanModel.patch_embedding)
    target_video = torch.randn(B, 16, 16, 128, 128).to(dtype=torch.bfloat16).to(device)

    # timesteps
    target_timestep = torch.tensor([10], dtype=torch.long).to(dtype=torch.bfloat16).to(device)

    # target text tokens for backbone (text_dim)
    target_text_tokens = torch.randn(B, 32, 4096).to(dtype=torch.bfloat16).to(device)  
    ref_text_shapes = [1,32]
    # run forward
    with torch.no_grad():
        out_video = mot(
            ref_image_tokens=ref_image_tokens,
            ref_img_shapes=ref_img_shapes,
            ref_text_tokens=ref_text_tokens,
            target_video=target_video,
            target_timestep=target_timestep,
            target_text_tokens=target_text_tokens,
            ref_text_shapes = ref_text_shapes,
        )
        print("out_video shape:", out_video.shape)