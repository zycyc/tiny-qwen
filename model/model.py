import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from .vision import Qwen2VLVisionEncoder, VisionConfig


@dataclass
class ModelConfig:
    # llm_config (Qwen3-1.7B-Base):  {'head_dim': 128, 'n_embed': 2048, 'n_mlp': 6144, 'n_heads': 16, 'n_layer': 28, 'n_kv_heads': 8, 'rms_norm_eps': 1e-06, 'rope_theta': 1000000, 'tie_word_embeddings': True, 'vocab_size': 151936}
    # llm_config (Qwen3-14B-Base): {'head_dim': 128, 'n_embed': 5120, 'n_mlp': 17408, 'n_heads': 40, 'n_layer': 40, 'n_kv_heads': 8, 'rms_norm_eps': 1e-06, 'rope_theta': 1000000, 'tie_word_embeddings': False, 'vocab_size': 151936}
    n_embed: int  # example embedding size: 2048
    n_heads: int  # example number of heads: 16
    n_kv_heads: int  # example number of key/value heads: 8
    n_layer: int  # example number of layers: 28
    n_mlp: int  # example MLP size: 6144
    rope_theta: float  # example RoPE theta: 1000000
    rms_norm_eps: float  # example RMSNorm epsilon: 1e-06
    vocab_size: int  # example vocabulary size: 151936
    tie_word_embeddings: (
        bool  # example tie word embeddings: True in 1.7B-Base, False in 14B-Base
    )
    vision_config: Optional[VisionConfig] = None
    head_dim: Optional[int] = None  # example head dimension: 128

    # MoE parameters
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_size: Optional[int] = None


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use explicit head_dim if provided, otherwise calculate
        d = (
            config.head_dim
            if config.head_dim is not None
            else (config.n_embed // config.n_heads)
        )
        t = config.rope_theta
        r = torch.arange(0, d, 2)
        self.inv_freq = 1.0 / (t ** (r / d)).float()

    def forward(self, x, position_ids):
        # Check the dimensionality of position_ids to decide shape
        # shape is typically B x T (2D) for text, or B x 3 x T (3D) for multimodal
        if position_ids.dim() == 3:
            inv_freq = self.inv_freq.unsqueeze(0).unsqueeze(0).to(x.device)
        else:
            inv_freq = self.inv_freq.to(x.device)

        position_ids = position_ids.unsqueeze(-1)
        freqs = position_ids * inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads

        self.n_embed = config.n_embed
        self.n_embed_per_head = config.n_embed // config.n_heads
        self.n_kv_embed = config.n_kv_heads * self.n_embed_per_head

        self.q_proj = nn.Linear(self.n_embed, self.n_embed, bias=True)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=True)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=True)
        self.o_proj = nn.Linear(self.n_embed, self.n_embed, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.n_embed_per_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.n_embed_per_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.n_embed_per_head).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin):
        if cos.dim() == 4:
            # shape [B, 3, T, D] -> multi-modal
            cos = CausalSelfAttention._process_rotary_component(cos)
            sin = CausalSelfAttention._process_rotary_component(sin)
        else:
            # shape [B, T, D] -> text-only
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (CausalSelfAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (CausalSelfAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _process_rotary_component(x):
        # Split into sections and select appropriate indices
        sections = x.split([16, 24, 24, 16, 24, 24], dim=-1)
        processed = [m[i % 3] for i, m in enumerate(sections)]
        # Combine and add dimension
        return torch.cat(processed, dim=-1).unsqueeze(1)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class Qwen3Attention(nn.Module):
    """Qwen3 attention with q_norm and k_norm layers"""

    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_embed = config.n_embed

        # Use explicit head_dim from config if provided, otherwise calculate
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else (config.n_embed // config.n_heads)
        )

        # Calculate dimensions using explicit head_dim
        self.q_embed = self.n_heads * self.head_dim
        self.kv_embed = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.n_embed, self.q_embed, bias=False)
        self.k_proj = nn.Linear(self.n_embed, self.kv_embed, bias=False)
        self.v_proj = nn.Linear(self.n_embed, self.kv_embed, bias=False)
        self.o_proj = nn.Linear(self.q_embed, self.n_embed, bias=False)

        # Qwen3 specific: q_norm and k_norm use explicit head_dim
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply normalization to q and k before RoPE (Qwen3 specific)
        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.q_embed)
        y = self.o_proj(y)
        return y

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin):
        if cos.dim() == 4:
            # shape [B, 3, T, D] -> multi-modal
            cos = Qwen3Attention._process_rotary_component(cos)
            sin = Qwen3Attention._process_rotary_component(sin)
        else:
            # shape [B, T, D] -> text-only
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (Qwen3Attention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen3Attention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _process_rotary_component(x):
        # Split into sections and select appropriate indices
        sections = x.split([16, 24, 24, 16, 24, 24], dim=-1)
        processed = [m[i % 3] for i, m in enumerate(sections)]
        # Combine and add dimension
        return torch.cat(processed, dim=-1).unsqueeze(1)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class Qwen3MoeAttention(nn.Module):
    """Qwen3 MoE attention with explicit head_dim support"""

    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_embed = config.n_embed

        # Use explicit head_dim if provided, otherwise calculate
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else (config.n_embed // config.n_heads)
        )

        self.q_proj = nn.Linear(self.n_embed, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.n_embed, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.n_embed, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.n_embed, bias=False)

        # Qwen3 specific: q_norm and k_norm on head dimension
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply normalization to q and k before RoPE (Qwen3 specific)
        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        y = self.o_proj(y)
        return y

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin):
        if cos.dim() == 4:
            # shape [B, 3, T, D] -> multi-modal
            cos = Qwen3MoeAttention._process_rotary_component(cos)
            sin = Qwen3MoeAttention._process_rotary_component(sin)
        else:
            # shape [B, T, D] -> text-only
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (Qwen3MoeAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen3MoeAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _process_rotary_component(x):
        # Split into sections and select appropriate indices
        sections = x.split([16, 24, 24, 16, 24, 24], dim=-1)
        processed = [m[i % 3] for i, m in enumerate(sections)]
        # Combine and add dimension
        return torch.cat(processed, dim=-1).unsqueeze(1)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.up_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.down_proj = nn.Linear(config.n_mlp, config.n_embed, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.gate = nn.Linear(config.n_embed, config.num_experts, bias=False)

        # Expert layers with proper naming to match checkpoint
        self.experts = nn.ModuleList()
        for _ in range(config.num_experts):
            expert = nn.Module()
            expert.gate_proj = nn.Linear(
                config.n_embed, config.moe_intermediate_size, bias=False
            )
            expert.up_proj = nn.Linear(
                config.n_embed, config.moe_intermediate_size, bias=False
            )
            expert.down_proj = nn.Linear(
                config.moe_intermediate_size, config.n_embed, bias=False
            )
            self.experts.append(expert)

    def forward(self, x):
        b, seq_len, embed_dim = x.shape
        scores = self.gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        expert_outputs = []
        for e in range(self.num_experts):
            expert = self.experts[e]
            hidden = F.silu(expert.gate_proj(x)) * expert.up_proj(x)
            out = expert.down_proj(hidden)
            expert_outputs.append(out.unsqueeze(-2))
        expert_outputs = torch.cat(
            expert_outputs, dim=-2
        )  # (b, t, num_experts, emb_dim)

        gating_probs = torch.zeros_like(scores)

        for i in range(self.num_experts_per_tok):
            indices = topk_indices[..., i : i + 1]
            prob = topk_probs[..., i : i + 1]
            gating_probs.scatter_(dim=-1, index=indices, src=prob)
        gating_probs = gating_probs.unsqueeze(-1)  # (b, t, num_experts, 1)

        # Weighted sum over experts
        y = (gating_probs * expert_outputs).sum(dim=-2)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(n_embed=n_embed, eps=eps)

        # Use MoE if experts are configured, otherwise regular MLP
        if config.num_experts and config.num_experts > 0:
            self.mlp = MoEFeedForward(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = Qwen3Attention(config)
        self.post_attention_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3MoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = Qwen3MoeAttention(config)
        self.post_attention_layernorm = RMSNorm(n_embed=n_embed, eps=eps)

        # Use MoE if experts are configured, otherwise regular MLP
        if config.num_experts and config.num_experts > 0:
            self.mlp = MoEFeedForward(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(self, x, position_ids):
        cos, sin = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return x


class Qwen2VL(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen2VLVisionEncoder(config.vision_config)
        self.model = Qwen2Model(config)
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.vision_start_token_id = 151652
        self.image_pad_token_id = 151655
        self.video_pad_token_id = -1  # placeholder

    def _get_position_ids(
        self,
        input_ids: torch.LongTensor,
        d_image: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        B, T = input_ids.shape
        device = input_ids.device
        all_pos_ids = torch.zeros(B, 3, T, dtype=torch.long, device=device)

        for batch_idx in range(B):
            seq = input_ids[batch_idx]
            seq_idx = 0
            image_idx = 0
            pos_chunks = []
            position_id = 0

            while seq_idx < T:
                token_id = seq[seq_idx].item()
                if token_id == self.image_pad_token_id:
                    t, h, w = d_image[image_idx]
                    h = h // self.config.vision_config.spatial_merge_size
                    w = w // self.config.vision_config.spatial_merge_size

                    t_idx = torch.arange(t).view(t, 1).expand(t, h * w).flatten()
                    h_idx = torch.arange(h).view(1, h, 1).expand(t, h, w).flatten()
                    w_idx = torch.arange(w).view(1, 1, w).expand(t, h, w).flatten()

                    pos_vision = torch.stack([t_idx, h_idx, w_idx]) + position_id
                    pos_chunks.append(pos_vision)
                    position_id = pos_vision.max().item() + 1
                    seq_idx += t * h * w
                    image_idx += 1
                else:
                    pos_text = torch.tensor([position_id])
                    pos_text = pos_text.unsqueeze(0).expand(3, 1)  # shape (3,1)
                    pos_chunks.append(pos_text)

                    position_id += 1
                    seq_idx += 1

            # Concatenate all chunks for this example => shape [3, seq_len]
            pos_ids_example = torch.cat(pos_chunks, dim=1).to(device)
            all_pos_ids = pos_ids_example.unsqueeze(1).expand(-1, B, -1)

        return all_pos_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_embeds = self.model.embed_tokens(input_ids)

        if pixels is not None:
            # encode images through the vision encoder.
            image_embeds = self.visual(pixels=pixels, d_image=d_image)
            # create a mask for the image tokens of shape (B, T)
            image_mask = input_ids == self.image_pad_token_id
            # expand the mask along embedding dimension to shape (B, T, C)
            image_mask = image_mask.unsqueeze(-1).expand_as(input_embeds)
            # replace image pad token embeddings with actual image embeddings
            input_embeds = input_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image)
        x = self.model(x=input_embeds, position_ids=position_ids)

        if self.lm_head is None:
            logits = torch.matmul(x, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixels: torch.Tensor,
        d_image: torch.Tensor,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
        stream: bool = False,
    ):
        if stop_tokens is None:
            stop_tokens = [
                151645,
                151644,
                151643,
            ]  # <|im_end|>, <|im_start|>, <|endoftext|>

        for _ in range(max_new_tokens):
            logits = self.forward(input_ids=input_ids, pixels=pixels, d_image=d_image)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = probs.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # If streaming, yield the new token
            if stream:
                yield next_token.item()

            # Check if we hit a stop token
            if next_token.item() in stop_tokens:
                break

        # If not streaming, return the full input_ids
        if not stream:
            return input_ids

    @classmethod
    def from_pretrained(cls, repo_id: str, device_map: str = "auto"):
        from .util import load_pretrained_model

        return load_pretrained_model(cls, repo_id, device_map=device_map)


class Qwen2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)

        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(T, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        return position_ids

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(input_ids)
        position_ids = self._get_position_ids(input_ids)
        x = self.model(x=x, position_ids=position_ids)
        if self.lm_head is None:
            logits = torch.matmul(x, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
        stream: bool = False,
    ):
        if stop_tokens is None:
            stop_tokens = [
                151645,
                151644,
                151643,
            ]  # <|im_end|>, <|im_start|>, <|endoftext|>

        for _ in range(max_new_tokens):
            logits = self.forward(input_ids=input_ids)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = probs.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # If streaming, yield the new token
            if stream:
                yield next_token.item()

            # Check if we hit a stop token
            if next_token.item() in stop_tokens:
                break

        # If not streaming, return the full input_ids
        if not stream:
            return input_ids

    @classmethod
    def from_pretrained(cls, repo_id: str, device_map: str = "auto"):
        from .util import load_pretrained_model

        return load_pretrained_model(cls, repo_id, device_map=device_map)


class Qwen3Model(nn.Module):
    """Qwen3 model with proper attention normalization"""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        # Use Qwen3Block with proper attention
        self.layers = nn.ModuleList(Qwen3Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

        # Store config for convenience
        self.config = config

    def forward(self, x, position_ids):
        cos, sin = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return x


class Qwen3MoeModel(nn.Module):
    """Qwen3 MoE model with proper attention normalization and MoE layers"""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        # Use Qwen3MoeBlock with proper attention and MoE
        self.layers = nn.ModuleList(
            Qwen3MoeBlock(config) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

        # Store config for convenience
        self.config = config

    def forward(self, x, position_ids):
        cos, sin = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return x


class Qwen3(nn.Module):
    """Qwen3 dense model - text-only version"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(T, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        return position_ids

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(input_ids)
        position_ids = self._get_position_ids(input_ids)
        x = self.model(x=x, position_ids=position_ids)

        if self.lm_head is None:
            logits = torch.matmul(x, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
        stream: bool = False,
    ):
        if stop_tokens is None:
            stop_tokens = [
                151645,
                151644,
                151643,
            ]  # <|im_end|>, <|im_start|>, <|endoftext|>

        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids=input_ids)
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                next_token = probs.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # If streaming, yield the new token
                if stream:
                    yield next_token.item()

                # Check if we hit a stop token
                if next_token.item() in stop_tokens:
                    break

        # If not streaming, return the full input_ids
        if not stream:
            return input_ids

    @classmethod
    def from_pretrained(cls, repo_id: str, device_map: str = "auto"):
        from .util import load_pretrained_model

        return load_pretrained_model(cls, repo_id, device_map=device_map)


class Qwen3MoE(nn.Module):
    """Qwen3 MoE model - text-only version with mixture of experts"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config)

        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(T, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        return position_ids

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(input_ids)
        position_ids = self._get_position_ids(input_ids)
        x = self.model(x=x, position_ids=position_ids)

        if self.lm_head is None:
            logits = torch.matmul(x, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
        stream: bool = False,
    ):
        if stop_tokens is None:
            stop_tokens = [
                151645,
                151644,
                151643,
            ]  # <|im_end|>, <|im_start|>, <|endoftext|>

        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids=input_ids)
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                next_token = probs.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # If streaming, yield the new token
                if stream:
                    yield next_token.item()

                # Check if we hit a stop token
                if next_token.item() in stop_tokens:
                    break

        # If not streaming, return the full input_ids
        if not stream:
            return input_ids

    @classmethod
    def from_pretrained(cls, repo_id: str, device_map: str = "auto"):
        from .util import load_pretrained_model

        return load_pretrained_model(cls, repo_id, device_map=device_map)
