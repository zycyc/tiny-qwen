from typing import Optional
from dataclasses import dataclass


@dataclass
class VisionConfig:
    """
    "depth": 32,
    "embed_dim": 1280,
    "mlp_ratio": 4,
    "num_heads": 16,
    "in_chans": 3,
    "hidden_size": 1536,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "temporal_patch_size": 2
    """
    n_embed: int
    n_layer: int
    n_heads: int
    
    output_n_embed: int # same as n_embed of the downstream model/LLM

    in_channels: int
    spatial_merge_size: int
    spatial_patch_size: int
    temporal_patch_size: int


@dataclass
class ModelConfig:
    n_embed: int
    n_heads: int
    n_kv_heads: int
    n_layer: int
    n_mlp: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    tie_word_embeddings: bool
    vision_config: Optional[VisionConfig] = None
