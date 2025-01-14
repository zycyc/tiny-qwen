import os
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any
import torch
from dataclasses import dataclass

from models.config import ModelConfig, VisionConfig
from safetensors.torch import save_file, safe_open
from huggingface_hub import snapshot_download, HfApi


LOAD_IN_BF16 = False
# TODO: vision encoder requires load in full while llm requires load in bf16
# make this configurable.


def _rename_dict_keys(original_dict: dict, key_mapping: dict) -> dict:
    """
    Renames keys in a dictionary according to a provided mapping.

    Args:
        original_dict (dict): The original dictionary whose keys need to be renamed.
        key_mapping (dict): A mapping from old key names to new key names (old_key_name: new_key_name).

    Returns:
        dict: A new dictionary with keys renamed according to the mapping.
              Keys not present in the mapping remain unchanged.
    """
    new_dict = {}
    for key, value in original_dict.items():
        new_key_name = key_mapping[key] if key in key_mapping else key
        new_dict[new_key_name] = value
    return new_dict


def _convert_llm_config(hf_config: dict):
    llm_config_key_name_mapping = {
        "hidden_size": "n_embed",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads",
        "num_hidden_layers": "n_layer",
        "intermediate_size": "n_mlp",
        "rms_norm_eps": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "rope_theta": "rope_theta",
        "tie_word_embeddings": "tie_word_embeddings",
    }
    return _rename_dict_keys(
        original_dict=hf_config,
        key_mapping=llm_config_key_name_mapping,
    )


def _convert_vision_config(hf_config: dict):
    vision_config_key_name_mapping = {
        "depth": "n_layer",
        "embed_dim": "n_embed",
        "num_heads": "n_heads",
        "in_chans": "in_channels",
        "hidden_size": "output_n_embed",
    }
    vision_config = hf_config["vision_config"]
    return _rename_dict_keys(
        original_dict=vision_config,
        key_mapping=vision_config_key_name_mapping,
    )


def _filter_dict_by_dataclass(params: dict, dataclass_type: dataclass) -> dict:
    return {k: v for k, v in params.items() if k in dataclass_type.__annotations__}


class ModelHubMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        force_download: bool = False,
        **kwargs,
    ):
        # Determine model path
        if os.path.isdir(pretrained_model_name_or_path):
            model_path = Path(pretrained_model_name_or_path)
        else:
            model_path = Path(
                snapshot_download(
                    repo_id=pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    force_download=force_download,
                )
            )

        # Load config
        with open(model_path / "config.json", "r") as f:
            config_data = json.load(f)

        llm_config = _convert_llm_config(config_data)
        llm_config = _filter_dict_by_dataclass(llm_config, ModelConfig)
        model_config = ModelConfig(**llm_config)
        if "vision_config" in config_data:
            vision_config = _convert_vision_config(config_data)
            vision_config = _filter_dict_by_dataclass(vision_config, VisionConfig)
            model_config.vision_config = VisionConfig(**vision_config)

        # Initialize model
        if LOAD_IN_BF16:
            model = cls(model_config).to(torch.bfloat16)
        else:
            model = cls(model_config)

        # Load weights
        model_files = sorted(model_path.glob("*.safetensors"))
        state_dict = model.state_dict()
        for file_path in model_files:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in state_dict:
                        tensor = f.get_tensor(key)
                        state_dict[key].copy_(tensor)

        return model

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        config: Optional[ModelConfig] = None,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
    ):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save model weights
        self._save_pretrained(save_directory)

        # Save config
        if config is None and hasattr(self, "config"):
            config = self.config
        if config is not None:
            with open(save_directory / "config.json", "w") as f:
                json.dump(config.__dict__, f, indent=2)

        # Optionally push to hub
        if push_to_hub and repo_id is not None:
            api = HfApi(token=token)
            api.upload_folder(
                folder_path=str(save_directory),
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )

    def _save_pretrained(self, save_directory: Path):
        # Save the model using safetensors
        state_dict = self.state_dict()
        # Determine split size (e.g., 10GB per file)
        split_size = 10 * 1024 * 1024 * 1024  # 10GB in bytes
        current_size = 0
        file_index = 1
        tensors = {}
        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            if current_size + tensor_size > split_size:
                # Save current tensors to a file
                save_path = (
                    save_directory / f"model-{file_index:05d}-of-xxxx.safetensors"
                )
                save_file(tensors, save_path)
                tensors = {}
                current_size = 0
                file_index += 1
            tensors[key] = tensor
            current_size += tensor_size

        # Save remaining tensors
        if tensors:
            save_path = save_directory / f"model-{file_index:05d}-of-xxxx.safetensors"
            save_file(tensors, save_path)

        # Update 'xxxx' in filenames with total number of files
        total_files = file_index
        for idx in range(1, total_files + 1):
            old_name = save_directory / f"model-{idx:05d}-of-xxxx.safetensors"
            new_name = (
                save_directory / f"model-{idx:05d}-of-{total_files:05d}.safetensors"
            )
            old_name.rename(new_name)
