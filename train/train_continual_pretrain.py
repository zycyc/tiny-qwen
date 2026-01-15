"""
Continual Pretraining Script for Catastrophic Forgetting Experiments

This script trains models in two phases:
- Phase 1 (Part A): Pretrain on wikitext-103-raw-v1
- Phase 2 (Part B): Continue training on codeparrot/codeparrot-clean

Throughout both phases, validation loss is measured on BOTH datasets
to quantify catastrophic forgetting.

Usage:
    python train/train_continual_pretrain.py --model dense
    python train/train_continual_pretrain.py --model cortex-small
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import lightning as L
import matplotlib.pyplot as plt
from datetime import datetime
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset, Dataset, DataLoader
from datasets import load_dataset
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from model.model import Qwen3, Qwen3MoE, ModelConfig, MoEFeedForward


# =============================================================================
# Dataset Classes for Wikitext and CodeParrot
# =============================================================================


class WikitextDataset(IterableDataset):
    """
    Iterable dataset for wikitext-103-raw-v1.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_length: int = 2048,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        # Load wikitext-103-raw-v1 with streaming
        self.dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split=split,
            streaming=True,
        )

        self.pad_id = tokenizer.token_to_id("<|pad|>")
        if self.pad_id is None:
            self.pad_id = tokenizer.token_to_id("<|endoftext|>")

    def __iter__(self):
        buffer = []

        for example in self.dataset:
            text = example["text"]
            if not text.strip():  # Skip empty lines
                continue
            tokens = self.tokenizer.encode(text).ids
            buffer.extend(tokens)

            while len(buffer) >= self.seq_length + 1:
                chunk = buffer[: self.seq_length + 1]
                buffer = buffer[self.seq_length :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}


class WikitextMapDataset(Dataset):
    """
    Map-style dataset for wikitext-103-raw-v1 validation.
    Loads a subset into memory for deterministic validation.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_length: int = 2048,
        split: str = "validation",
        max_samples: int = 500,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.samples = []

        self.pad_id = tokenizer.token_to_id("<|pad|>")
        if self.pad_id is None:
            self.pad_id = tokenizer.token_to_id("<|endoftext|>")

        # Load wikitext-103-raw-v1
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split=split,
            streaming=True,
        )

        buffer = []
        for example in dataset:
            text = example["text"]
            if not text.strip():
                continue
            tokens = self.tokenizer.encode(text).ids
            buffer.extend(tokens)

            while len(buffer) >= self.seq_length + 1:
                chunk = buffer[: self.seq_length + 1]
                buffer = buffer[self.seq_length :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                self.samples.append({"input_ids": input_ids, "labels": labels})

                if len(self.samples) >= max_samples:
                    return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CodeParrotDataset(IterableDataset):
    """
    Iterable dataset for codeparrot/codeparrot-clean.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_length: int = 2048,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        # Load codeparrot-clean with streaming
        self.dataset = load_dataset(
            "codeparrot/codeparrot-clean",
            split=split,
            streaming=True,
        )

        self.pad_id = tokenizer.token_to_id("<|pad|>")
        if self.pad_id is None:
            self.pad_id = tokenizer.token_to_id("<|endoftext|>")

    def __iter__(self):
        buffer = []
        skipped = 0

        for example in self.dataset:
            try:
                text = example["content"]  # codeparrot uses "content" field
                if not text or not text.strip():
                    continue
                tokens = self.tokenizer.encode(text).ids
                buffer.extend(tokens)

                while len(buffer) >= self.seq_length + 1:
                    chunk = buffer[: self.seq_length + 1]
                    buffer = buffer[self.seq_length :]

                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)

                    yield {"input_ids": input_ids, "labels": labels}
            except Exception:
                skipped += 1
                if skipped % 1000 == 0:
                    print(f"[CodeParrot] Skipped {skipped} corrupted records")
                continue


class CodeParrotMapDataset(Dataset):
    """
    Map-style dataset for codeparrot-clean validation.
    Loads a subset into memory for deterministic validation.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_length: int = 2048,
        split: str = "train",  # codeparrot-clean only has train split
        max_samples: int = 500,
        skip_examples: int = 0,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.samples = []

        self.pad_id = tokenizer.token_to_id("<|pad|>")
        if self.pad_id is None:
            self.pad_id = tokenizer.token_to_id("<|endoftext|>")

        # Load codeparrot-clean with streaming
        dataset = load_dataset(
            "codeparrot/codeparrot-clean",
            split=split,
            streaming=True,
        )

        # Skip some examples for validation (to not overlap with training start)
        if skip_examples > 0:
            dataset = dataset.skip(skip_examples)

        buffer = []
        for example in dataset:
            try:
                text = example["content"]
                if not text or not text.strip():
                    continue
                tokens = self.tokenizer.encode(text).ids
                buffer.extend(tokens)

                while len(buffer) >= self.seq_length + 1:
                    chunk = buffer[: self.seq_length + 1]
                    buffer = buffer[self.seq_length :]

                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)

                    self.samples.append({"input_ids": input_ids, "labels": labels})

                    if len(self.samples) >= max_samples:
                        return
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Simple collator that stacks sequences."""
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


# =============================================================================
# Model Configurations (same as train_pretrain.py)
# =============================================================================

# Dense model: Qwen3 (matched to cortex-small param count)
# n_mlp = 2 × n_mlp_moe for exact active param matching
QWEN3_CONFIG = ModelConfig(
    n_embed=384,
    n_heads=6,
    n_kv_heads=2,
    n_layer=2,
    n_mlp=1536,  # 4*n_embed
    vocab_size=151936,
    rope_theta=1000000,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    head_dim=64,
    disable_attention=True,
)

# MoE model: Traditional sparse mixture of experts
# Sized so active params (2 experts) ≈ dense model total params
# Active per layer: router (6,144) + 2 experts × 3 × 384 × 2015 = 9,291,264
QWEN3_MOE_CONFIG = ModelConfig(
    n_embed=384,
    n_heads=6,
    n_kv_heads=2,
    n_layer=2,
    n_mlp=1536,  # placeholder, not used
    vocab_size=151936,
    rope_theta=1000000,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    head_dim=64,
    # MoE parameters
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=766,
    disable_attention=True,
)

# Cortex E2E: Meta-learning with inner/outer loop
# Optimizes post-TTT loss with 2nd-order gradients (gradients through adaptation)
QWEN3_CORTEX_E2E_NEURON_CONFIG = ModelConfig(
    n_embed=384,
    n_heads=6,
    n_kv_heads=2,
    n_layer=2,
    n_mlp=1536,
    vocab_size=151936,
    rope_theta=1000000,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    head_dim=64,
    # Cortex parameters
    use_cortex=True,
    cortex_hidden_size=4957,
    cortex_k_ratio=4,
    cortex_topk_softmax=True,
    cortex_soft_kwta=False,
    cortex_temperature=1.0,
    cortex_low_rank_gate=True,  # False for normal gate, True for low-rank gate
    cortex_gate_rank=64,  # 64 for low-rank gate
    cortex_gate_norm=True,  # normalize gate logits to prevent scale drift
    # TTT-E2E: sequential gate and/or neuron weight adaptation
    cortex_use_ttt_e2e=True,
    cortex_ttt_lr=1.0,
    cortex_ttt_batch_size=16,
    cortex_ttt_adapt_gate=False,
    cortex_ttt_adapt_neurons=True,
    disable_attention=True,
)

QWEN3_CORTEX_E2E_GATE_CONFIG = ModelConfig(
    n_embed=384,
    n_heads=6,
    n_kv_heads=2,
    n_layer=2,
    n_mlp=1536,
    vocab_size=151936,
    rope_theta=1000000,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    head_dim=64,
    # Cortex parameters
    use_cortex=True,
    # to match total param of dense -- 1536*3/4 = 1152, or (3*1536-64) * 384 / (3*384+64) = 1435 when using lor gate
    cortex_hidden_size=4957,  # 2633 for normal gate, 4957 for low-rank gate
    cortex_k_ratio=4,
    cortex_topk_softmax=False,
    cortex_soft_kwta=True,
    cortex_temperature=1.0,
    cortex_low_rank_gate=True,  # False for normal gate, True for low-rank gate
    cortex_gate_rank=64,  # 64 for low-rank gate
    cortex_gate_norm=True,  # normalize gate logits to prevent scale drift
    # TTT-E2E: sequential gate and/or neuron weight adaptation
    cortex_use_ttt_e2e=True,
    cortex_ttt_lr=1.0,
    cortex_ttt_batch_size=16,
    cortex_ttt_adapt_gate=True,
    cortex_ttt_adapt_neurons=False,
    disable_attention=False,
)

# Dense TTT-E2E: Meta-learning with MLP weight adaptation
# Same as dense but with test-time MLP adaptation (sequential mini-batch)
QWEN3_DENSE_TTT_CONFIG = ModelConfig(
    n_embed=384,
    n_heads=6,
    n_kv_heads=2,
    n_layer=2,
    n_mlp=1536,
    vocab_size=151936,
    rope_theta=1000000,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    head_dim=64,
    disable_attention=True,
    # Dense TTT-E2E parameters
    dense_use_ttt_e2e=True,
    dense_ttt_lr=1.0,
    dense_ttt_batch_size=16,  # Adapt weights every 16 tokens
)

# MoE TTT-E2E (gate only): Adapt router gate weights at test time
QWEN3_MOE_TTT_GATE_CONFIG = ModelConfig(
    n_embed=384,
    n_heads=6,
    n_kv_heads=2,
    n_layer=2,
    n_mlp=1536,  # placeholder, not used
    vocab_size=151936,
    rope_theta=1000000,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    head_dim=64,
    disable_attention=False,
    # MoE parameters
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=766,
    # MoE TTT-E2E parameters
    moe_use_ttt_e2e=True,
    moe_ttt_lr=1.0,
    moe_ttt_batch_size=16,
    moe_ttt_adapt_gate=True,
    moe_ttt_adapt_experts=False,
)

# MoE TTT-E2E (experts only): Adapt expert MLP weights at test time
QWEN3_MOE_TTT_EXPERT_CONFIG = ModelConfig(
    n_embed=384,
    n_heads=6,
    n_kv_heads=2,
    n_layer=2,
    n_mlp=1536,  # placeholder, not used
    vocab_size=151936,
    rope_theta=1000000,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    head_dim=64,
    disable_attention=False,
    # MoE parameters
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=766,
    # MoE TTT-E2E parameters
    moe_use_ttt_e2e=True,
    moe_ttt_lr=1.0,
    moe_ttt_batch_size=16,
    moe_ttt_adapt_gate=False,
    moe_ttt_adapt_experts=True,
)

MODEL_CONFIGS = {
    "dense": ("dense", QWEN3_CONFIG),
    "dense-ttt": ("dense", QWEN3_DENSE_TTT_CONFIG),
    "moe": ("moe", QWEN3_MOE_CONFIG),
    "moe-ttt-gate": ("moe", QWEN3_MOE_TTT_GATE_CONFIG),
    "moe-ttt-expert": ("moe", QWEN3_MOE_TTT_EXPERT_CONFIG),
    "cortex-ttt-neuron": ("cortex", QWEN3_CORTEX_E2E_NEURON_CONFIG),
    "cortex-ttt-gate": ("cortex", QWEN3_CORTEX_E2E_GATE_CONFIG),
}

# =============================================================================
# Dataset size constants (approximate token counts)
# =============================================================================

# wikitext-103-raw-v1: ~100 million tokens
WIKITEXT_TOKENS = 100_000_000

# codeparrot-clean: ~27 billion tokens (conservative estimate)
CODEPARROT_TOKENS = 27_000_000_000


# =============================================================================
# Training Module with Multi-Validation Support
# =============================================================================


class Qwen3ForContinualPretrain(L.LightningModule):
    """Lightning module for continual pretraining with multi-chunk validation."""

    def __init__(
        self,
        config: ModelConfig,
        model_type: str = "dense",
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.model_type = model_type
        self.learning_rate = learning_rate

        # Create model with random weights
        if model_type == "moe" or model_type == "cortex":
            self.qwen_model = Qwen3MoE(config)
        else:
            self.qwen_model = Qwen3(config)

        self.loss_fct = nn.CrossEntropyLoss()

        # Track losses for plotting
        self.train_losses = []
        self.val_losses_part_a = []
        self.val_losses_part_b = []

        # Per-position loss tracking for wandb
        self.val_per_position_losses_part_a = []
        self.val_per_position_losses_part_b = []

        # Current phase (for logging)
        self.current_phase = 1

        # Cortex TTT-E2E (adapt gate and/or neuron weights at test time)
        self.use_ttt_e2e = getattr(config, "cortex_use_ttt_e2e", False)
        if self.use_ttt_e2e:
            self.ttt_lr = config.cortex_ttt_lr
            self.ttt_batch_size = config.cortex_ttt_batch_size
            self.cortex_ttt_adapt_gate = getattr(config, "cortex_ttt_adapt_gate", True)
            self.cortex_ttt_adapt_neurons = getattr(
                config, "cortex_ttt_adapt_neurons", False
            )

        # Dense TTT-E2E (adapt MLP weights at test time)
        self.use_dense_ttt_e2e = getattr(config, "dense_use_ttt_e2e", False)
        if self.use_dense_ttt_e2e:
            self.dense_ttt_lr = config.dense_ttt_lr
            self.dense_ttt_batch_size = getattr(config, "dense_ttt_batch_size", 16)

        # MoE TTT-E2E (adapt gate and/or expert weights at test time)
        self.use_moe_ttt_e2e = getattr(config, "moe_use_ttt_e2e", False)
        if self.use_moe_ttt_e2e:
            self.moe_ttt_lr = config.moe_ttt_lr
            self.moe_ttt_batch_size = getattr(config, "moe_ttt_batch_size", 16)
            self.moe_ttt_adapt_gate = getattr(config, "moe_ttt_adapt_gate", True)
            self.moe_ttt_adapt_experts = getattr(config, "moe_ttt_adapt_experts", False)

    def forward(self, input_ids):
        return self.qwen_model(input_ids)

    def compute_per_position_loss(self, logits, labels):
        """Compute loss at each token position, averaged over batch."""
        batch_size, seq_len, vocab_size = logits.shape
        per_token_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            reduction="none",
        )  # (batch * seq_len,)
        per_token_loss = per_token_loss.view(batch_size, seq_len)  # (batch, seq_len)
        per_position_loss = per_token_loss.mean(dim=0)  # (seq_len,)
        return per_position_loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        if self.use_ttt_e2e:
            # Cortex TTT-E2E: Sequential mini-batch adaptation of gate and/or neuron weights
            batch_size, seq_len = input_ids.shape
            chunk_size = self.ttt_batch_size

            # 1. Collect parameters to adapt
            gate_params = []  # (layer_idx, weight or (gate_down_w, gate_up_w))
            neuron_params = []  # (layer_idx, {'gate_proj': w, 'up_proj': w, 'down_proj': w})
            is_low_rank = False

            for i, layer in enumerate(self.qwen_model.model.layers):
                if self.cortex_ttt_adapt_gate:
                    if hasattr(layer.mlp, "gate"):
                        gate_params.append((i, layer.mlp.gate.weight))
                    elif getattr(layer.mlp, "low_rank_gate", False):
                        is_low_rank = True
                        gate_params.append(
                            (i, (layer.mlp.gate_down.weight, layer.mlp.gate_up.weight))
                        )

                if self.cortex_ttt_adapt_neurons:
                    neuron_params.append(
                        (
                            i,
                            {
                                "gate_proj": layer.mlp.gate_proj.weight,
                                "up_proj": layer.mlp.up_proj.weight,
                                "down_proj": layer.mlp.down_proj.weight,
                            },
                        )
                    )

            # 2. Initialize adapted weights as clones
            current_gate_weights = {}
            for layer_idx, weight in gate_params:
                if is_low_rank:
                    gate_down_w, gate_up_w = weight
                    current_gate_weights[layer_idx] = (
                        gate_down_w.clone(),
                        gate_up_w.clone(),
                    )
                else:
                    current_gate_weights[layer_idx] = weight.clone()

            current_neuron_weights = {}  # {layer_idx: {'gate_proj': w, ...}}
            for layer_idx, param_dict in neuron_params:
                current_neuron_weights[layer_idx] = {
                    key: param.clone() for key, param in param_dict.items()
                }

            # 3. Process sequence in chunks
            all_logits = []
            total_loss = 0.0
            num_chunks = (seq_len + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, seq_len)

                chunk_input = input_ids[:, start:end]
                chunk_labels = labels[:, start:end]

                # Position IDs for this chunk
                chunk_position_ids = (
                    torch.arange(start, end, device=input_ids.device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

                # Forward with adapted weights
                logits_chunk = self.qwen_model(
                    chunk_input,
                    position_ids=chunk_position_ids,
                    gate_weight_overrides=current_gate_weights
                    if self.cortex_ttt_adapt_gate
                    else None,
                    neuron_weight_overrides_by_layer=current_neuron_weights
                    if self.cortex_ttt_adapt_neurons
                    else None,
                )
                all_logits.append(logits_chunk)

                # Compute chunk loss
                chunk_loss = self.loss_fct(
                    logits_chunk.reshape(-1, logits_chunk.size(-1)),
                    chunk_labels.reshape(-1),
                )
                total_loss = total_loss + chunk_loss * (end - start)

                # Adapt weights (except for last chunk)
                if chunk_idx < num_chunks - 1:
                    # Collect all params for gradient computation
                    all_params = []
                    if self.cortex_ttt_adapt_gate:
                        if is_low_rank:
                            for i, _ in gate_params:
                                all_params.extend(
                                    [
                                        current_gate_weights[i][0],
                                        current_gate_weights[i][1],
                                    ]
                                )
                        else:
                            all_params.extend(
                                [current_gate_weights[i] for i, _ in gate_params]
                            )
                    if self.cortex_ttt_adapt_neurons:
                        for layer_idx, _ in neuron_params:
                            for key in ["gate_proj", "up_proj", "down_proj"]:
                                all_params.append(
                                    current_neuron_weights[layer_idx][key]
                                )

                    # Compute gradients
                    grads = torch.autograd.grad(
                        chunk_loss,
                        all_params,
                        create_graph=True,
                        retain_graph=True,
                    )

                    # Update weights
                    grad_idx = 0
                    if self.cortex_ttt_adapt_gate:
                        if is_low_rank:
                            for layer_idx, _ in gate_params:
                                gd_grad, gu_grad = grads[grad_idx], grads[grad_idx + 1]
                                current_gate_weights[layer_idx] = (
                                    current_gate_weights[layer_idx][0]
                                    - self.ttt_lr * gd_grad,
                                    current_gate_weights[layer_idx][1]
                                    - self.ttt_lr * gu_grad,
                                )
                                grad_idx += 2
                        else:
                            for layer_idx, _ in gate_params:
                                current_gate_weights[layer_idx] = (
                                    current_gate_weights[layer_idx]
                                    - self.ttt_lr * grads[grad_idx]
                                )
                                grad_idx += 1

                    if self.cortex_ttt_adapt_neurons:
                        for layer_idx, _ in neuron_params:
                            for key in ["gate_proj", "up_proj", "down_proj"]:
                                current_neuron_weights[layer_idx][key] = (
                                    current_neuron_weights[layer_idx][key]
                                    - self.ttt_lr * grads[grad_idx]
                                )
                                grad_idx += 1

            # 4. Combine logits and compute final loss
            logits_post = torch.cat(all_logits, dim=1)
            loss_post = total_loss / seq_len

            self.log(
                "train_loss", loss_post, prog_bar=True, on_step=True, on_epoch=False
            )
            self.log("phase", float(self.current_phase), on_step=True, on_epoch=False)
            self.train_losses.append(loss_post.detach().cpu())

            # Log per-position loss periodically
            if self.global_step % 100 == 0:
                per_pos_loss = self.compute_per_position_loss(logits_post, labels)
                seq_len = per_pos_loss.size(0)
                self.logger.experiment.log(
                    {
                        "train_loss_token": wandb.plot.line_series(
                            xs=list(range(seq_len)),
                            ys=[per_pos_loss.detach().cpu().tolist()],
                            keys=["loss"],
                            title="Train Loss by Token Position",
                            xname="position",
                        )
                    }
                )

            return loss_post  # Optimize POST-TTT loss

        elif self.use_dense_ttt_e2e:
            # Dense TTT-E2E: Sequential mini-batch MLP weight adaptation
            # Process tokens in chunks, adapting weights after each chunk
            # This allows loss to decrease with position even without attention

            _, seq_len = input_ids.shape
            chunk_size = self.dense_ttt_batch_size

            # 1. Collect original MLP params by layer index
            mlp_params = []
            for i, layer in enumerate(self.qwen_model.model.layers):
                mlp_params.append(
                    (
                        i,
                        {
                            "gate_proj": layer.mlp.gate_proj.weight,
                            "up_proj": layer.mlp.up_proj.weight,
                            "down_proj": layer.mlp.down_proj.weight,
                        },
                    )
                )

            # Initialize adapted weights as original weights
            current_weights = {}
            for layer_idx, param_dict in mlp_params:
                current_weights[layer_idx] = {
                    key: param.clone() for key, param in param_dict.items()
                }

            # 2. Process sequence in chunks, adapting after each
            all_logits = []
            total_loss = 0.0
            num_chunks = (seq_len + chunk_size - 1) // chunk_size

            batch_size = input_ids.size(0)

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, seq_len)

                # Get chunk inputs/labels
                chunk_input = input_ids[:, start:end]
                chunk_labels = labels[:, start:end]

                # Create correct position IDs for this chunk (not starting from 0!)
                chunk_position_ids = torch.arange(
                    start, end, device=input_ids.device, dtype=torch.long
                )
                chunk_position_ids = chunk_position_ids.unsqueeze(0).expand(
                    batch_size, -1
                )

                # Forward with current adapted weights and correct position IDs
                logits_chunk = self.qwen_model(
                    chunk_input,
                    position_ids=chunk_position_ids,
                    mlp_overrides_by_layer=current_weights,
                )
                all_logits.append(logits_chunk)

                # Compute loss for this chunk
                chunk_loss = self.loss_fct(
                    logits_chunk.reshape(-1, logits_chunk.size(-1)),
                    chunk_labels.reshape(-1),
                )
                total_loss = total_loss + chunk_loss * (end - start)

                # Adapt weights based on this chunk's loss (except for last chunk)
                if chunk_idx < num_chunks - 1:
                    # Flatten current weights for gradient computation
                    all_params = [
                        current_weights[i][key]
                        for i, _ in mlp_params
                        for key in ["gate_proj", "up_proj", "down_proj"]
                    ]

                    # Compute gradients with graph tracking
                    grads = torch.autograd.grad(
                        chunk_loss,
                        all_params,
                        create_graph=True,
                        retain_graph=True,
                    )

                    # Update adapted weights
                    grad_idx = 0
                    for layer_idx, _ in mlp_params:
                        for key in ["gate_proj", "up_proj", "down_proj"]:
                            current_weights[layer_idx][key] = (
                                current_weights[layer_idx][key]
                                - self.dense_ttt_lr * grads[grad_idx]
                            )
                            grad_idx += 1

            # Combine all chunk logits
            logits_post = torch.cat(all_logits, dim=1)
            loss_post = total_loss / seq_len  # Normalize by sequence length

            self.log(
                "train_loss", loss_post, prog_bar=True, on_step=True, on_epoch=False
            )
            self.log("phase", float(self.current_phase), on_step=True, on_epoch=False)
            self.train_losses.append(loss_post.detach().cpu())

            # Log per-position loss every 100 steps
            if self.global_step % 100 == 0:
                per_pos_loss = self.compute_per_position_loss(logits_post, labels)
                seq_len = per_pos_loss.size(0)
                self.logger.experiment.log(
                    {
                        "train_loss_token": wandb.plot.line_series(
                            xs=list(range(seq_len)),
                            ys=[per_pos_loss.detach().cpu().tolist()],
                            keys=["loss"],
                            title="Train Loss by Token Position",
                            xname="position",
                        )
                    }
                )

            return loss_post  # Optimize POST-TTT loss

        elif self.use_moe_ttt_e2e:
            # MoE TTT-E2E: Sequential mini-batch adaptation of gate and/or expert weights
            batch_size, seq_len = input_ids.shape
            chunk_size = self.moe_ttt_batch_size

            # 1. Collect parameters to adapt
            gate_params = []  # (layer_idx, gate_weight)
            expert_params = []  # (layer_idx, expert_idx, {'gate_proj': w, ...})

            for i, layer in enumerate(self.qwen_model.model.layers):
                mlp = layer.mlp
                if not isinstance(mlp, MoEFeedForward):
                    continue

                if self.moe_ttt_adapt_gate:
                    gate_params.append((i, mlp.gate.weight))

                if self.moe_ttt_adapt_experts:
                    for e in range(mlp.num_experts):
                        expert = mlp.experts[e]
                        expert_params.append(
                            (
                                i,
                                e,
                                {
                                    "gate_proj": expert.gate_proj.weight,
                                    "up_proj": expert.up_proj.weight,
                                    "down_proj": expert.down_proj.weight,
                                },
                            )
                        )

            # 2. Initialize adapted weights as clones
            current_gate_weights = {}
            for layer_idx, weight in gate_params:
                current_gate_weights[layer_idx] = weight.clone()

            current_expert_weights = {}  # {layer_idx: {expert_idx: {'gate_proj': w, ...}}}
            for layer_idx, expert_idx, param_dict in expert_params:
                if layer_idx not in current_expert_weights:
                    current_expert_weights[layer_idx] = {}
                current_expert_weights[layer_idx][expert_idx] = {
                    key: param.clone() for key, param in param_dict.items()
                }

            # 3. Process sequence in chunks
            all_logits = []
            total_loss = 0.0
            num_chunks = (seq_len + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, seq_len)

                chunk_input = input_ids[:, start:end]
                chunk_labels = labels[:, start:end]

                # Position IDs for this chunk
                chunk_position_ids = (
                    torch.arange(start, end, device=input_ids.device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

                # Forward with adapted weights
                logits_chunk = self.qwen_model(
                    chunk_input,
                    position_ids=chunk_position_ids,
                    gate_weight_overrides=current_gate_weights
                    if self.moe_ttt_adapt_gate
                    else None,
                    expert_weight_overrides_by_layer=current_expert_weights
                    if self.moe_ttt_adapt_experts
                    else None,
                )
                all_logits.append(logits_chunk)

                # Compute chunk loss
                chunk_loss = self.loss_fct(
                    logits_chunk.reshape(-1, logits_chunk.size(-1)),
                    chunk_labels.reshape(-1),
                )
                total_loss = total_loss + chunk_loss * (end - start)

                # Adapt weights (except for last chunk)
                if chunk_idx < num_chunks - 1:
                    # Collect all params for gradient computation
                    all_params = []
                    if self.moe_ttt_adapt_gate:
                        all_params.extend(
                            [current_gate_weights[i] for i, _ in gate_params]
                        )
                    if self.moe_ttt_adapt_experts:
                        for layer_idx, expert_idx, _ in expert_params:
                            for key in ["gate_proj", "up_proj", "down_proj"]:
                                all_params.append(
                                    current_expert_weights[layer_idx][expert_idx][key]
                                )

                    # Compute gradients
                    grads = torch.autograd.grad(
                        chunk_loss,
                        all_params,
                        create_graph=True,
                        retain_graph=True,
                    )

                    # Update weights
                    grad_idx = 0
                    if self.moe_ttt_adapt_gate:
                        for layer_idx, _ in gate_params:
                            current_gate_weights[layer_idx] = (
                                current_gate_weights[layer_idx]
                                - self.moe_ttt_lr * grads[grad_idx]
                            )
                            grad_idx += 1

                    if self.moe_ttt_adapt_experts:
                        for layer_idx, expert_idx, _ in expert_params:
                            for key in ["gate_proj", "up_proj", "down_proj"]:
                                current_expert_weights[layer_idx][expert_idx][key] = (
                                    current_expert_weights[layer_idx][expert_idx][key]
                                    - self.moe_ttt_lr * grads[grad_idx]
                                )
                                grad_idx += 1

            # 4. Combine logits and compute final loss
            logits_post = torch.cat(all_logits, dim=1)
            loss_post = total_loss / seq_len

            self.log(
                "train_loss", loss_post, prog_bar=True, on_step=True, on_epoch=False
            )
            self.log("phase", float(self.current_phase), on_step=True, on_epoch=False)
            self.train_losses.append(loss_post.detach().cpu())

            # Log per-position loss periodically
            if self.global_step % 100 == 0:
                per_pos_loss = self.compute_per_position_loss(logits_post, labels)
                seq_len = per_pos_loss.size(0)
                self.logger.experiment.log(
                    {
                        "train_loss_token": wandb.plot.line_series(
                            xs=list(range(seq_len)),
                            ys=[per_pos_loss.detach().cpu().tolist()],
                            keys=["loss"],
                            title="Train Loss by Token Position",
                            xname="position",
                        )
                    }
                )

            return loss_post

        else:
            # Standard forward pass
            logits = self(input_ids)
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
            self.log("phase", float(self.current_phase), on_step=True, on_epoch=False)
            self.train_losses.append(loss.detach().cpu())

            # Log per-position loss every 100 steps
            if self.global_step % 100 == 0:
                per_pos_loss = self.compute_per_position_loss(logits, labels)
                seq_len = per_pos_loss.size(0)
                self.logger.experiment.log(
                    {
                        "train_loss_token": wandb.plot.line_series(
                            xs=list(range(seq_len)),
                            ys=[per_pos_loss.detach().cpu().tolist()],
                            keys=["loss"],
                            title="Train Loss by Token Position",
                            xname="position",
                        )
                    }
                )

            return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        if self.use_ttt_e2e:
            # Cortex TTT-E2E: Sequential mini-batch adaptation of gate and/or neuron weights
            with torch.inference_mode(False), torch.enable_grad():
                input_ids = input_ids.clone()
                labels = labels.clone()

                batch_size, seq_len = input_ids.shape
                chunk_size = self.ttt_batch_size

                # Collect parameters to adapt
                gate_params = []
                neuron_params = []
                is_low_rank = False

                for i, layer in enumerate(self.qwen_model.model.layers):
                    if self.cortex_ttt_adapt_gate:
                        if hasattr(layer.mlp, "gate"):
                            gate_params.append((i, layer.mlp.gate.weight))
                        elif getattr(layer.mlp, "low_rank_gate", False):
                            is_low_rank = True
                            gate_params.append(
                                (
                                    i,
                                    (
                                        layer.mlp.gate_down.weight,
                                        layer.mlp.gate_up.weight,
                                    ),
                                )
                            )

                    if self.cortex_ttt_adapt_neurons:
                        neuron_params.append(
                            (
                                i,
                                {
                                    "gate_proj": layer.mlp.gate_proj.weight,
                                    "up_proj": layer.mlp.up_proj.weight,
                                    "down_proj": layer.mlp.down_proj.weight,
                                },
                            )
                        )

                # Initialize adapted weights as clones
                current_gate_weights = {}
                for layer_idx, weight in gate_params:
                    if is_low_rank:
                        gate_down_w, gate_up_w = weight
                        current_gate_weights[layer_idx] = (
                            gate_down_w.clone(),
                            gate_up_w.clone(),
                        )
                    else:
                        current_gate_weights[layer_idx] = weight.clone()

                current_neuron_weights = {}
                for layer_idx, param_dict in neuron_params:
                    current_neuron_weights[layer_idx] = {
                        key: param.clone() for key, param in param_dict.items()
                    }

                # Process sequence in chunks
                all_logits = []
                num_chunks = (seq_len + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, seq_len)

                    chunk_input = input_ids[:, start:end]
                    chunk_labels = labels[:, start:end]

                    chunk_position_ids = (
                        torch.arange(
                            start, end, device=input_ids.device, dtype=torch.long
                        )
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                    )

                    logits_chunk = self.qwen_model(
                        chunk_input,
                        position_ids=chunk_position_ids,
                        gate_weight_overrides=current_gate_weights
                        if self.cortex_ttt_adapt_gate
                        else None,
                        neuron_weight_overrides_by_layer=current_neuron_weights
                        if self.cortex_ttt_adapt_neurons
                        else None,
                    )
                    all_logits.append(logits_chunk)

                    # Adapt weights (except for last chunk)
                    if chunk_idx < num_chunks - 1:
                        chunk_loss = self.loss_fct(
                            logits_chunk.reshape(-1, logits_chunk.size(-1)),
                            chunk_labels.reshape(-1),
                        )

                        # Collect all params for gradient computation
                        all_params = []
                        if self.cortex_ttt_adapt_gate:
                            if is_low_rank:
                                for i, _ in gate_params:
                                    all_params.extend(
                                        [
                                            current_gate_weights[i][0],
                                            current_gate_weights[i][1],
                                        ]
                                    )
                            else:
                                all_params.extend(
                                    [current_gate_weights[i] for i, _ in gate_params]
                                )
                        if self.cortex_ttt_adapt_neurons:
                            for layer_idx, _ in neuron_params:
                                for key in ["gate_proj", "up_proj", "down_proj"]:
                                    all_params.append(
                                        current_neuron_weights[layer_idx][key]
                                    )

                        grads = torch.autograd.grad(
                            chunk_loss, all_params, retain_graph=False
                        )

                        # Update weights
                        grad_idx = 0
                        if self.cortex_ttt_adapt_gate:
                            if is_low_rank:
                                for layer_idx, _ in gate_params:
                                    gd_grad, gu_grad = (
                                        grads[grad_idx],
                                        grads[grad_idx + 1],
                                    )
                                    current_gate_weights[layer_idx] = (
                                        current_gate_weights[layer_idx][0]
                                        - self.ttt_lr * gd_grad,
                                        current_gate_weights[layer_idx][1]
                                        - self.ttt_lr * gu_grad,
                                    )
                                    grad_idx += 2
                            else:
                                for layer_idx, _ in gate_params:
                                    current_gate_weights[layer_idx] = (
                                        current_gate_weights[layer_idx]
                                        - self.ttt_lr * grads[grad_idx]
                                    )
                                    grad_idx += 1

                        if self.cortex_ttt_adapt_neurons:
                            for layer_idx, _ in neuron_params:
                                for key in ["gate_proj", "up_proj", "down_proj"]:
                                    current_neuron_weights[layer_idx][key] = (
                                        current_neuron_weights[layer_idx][key]
                                        - self.ttt_lr * grads[grad_idx]
                                    )
                                    grad_idx += 1

                # Combine logits and compute final loss
                logits_post = torch.cat(all_logits, dim=1)
                loss = self.loss_fct(
                    logits_post.reshape(-1, logits_post.size(-1)),
                    labels.reshape(-1),
                )
                logits_for_pos_loss = logits_post

        elif self.use_dense_ttt_e2e:
            # Dense TTT-E2E: Sequential mini-batch MLP weight adaptation
            with torch.inference_mode(False), torch.enable_grad():
                input_ids = input_ids.clone()
                labels = labels.clone()

                batch_size, seq_len = input_ids.shape
                chunk_size = self.dense_ttt_batch_size

                # Collect original MLP weights by layer
                mlp_params = []
                for i, layer in enumerate(self.qwen_model.model.layers):
                    mlp_params.append(
                        (
                            i,
                            {
                                "gate_proj": layer.mlp.gate_proj.weight,
                                "up_proj": layer.mlp.up_proj.weight,
                                "down_proj": layer.mlp.down_proj.weight,
                            },
                        )
                    )

                # Initialize adapted weights as clones of original
                current_weights = {}
                for layer_idx, param_dict in mlp_params:
                    current_weights[layer_idx] = {
                        key: param.clone() for key, param in param_dict.items()
                    }

                # Process sequence in chunks
                all_logits = []
                num_chunks = (seq_len + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, seq_len)

                    chunk_input = input_ids[:, start:end]
                    chunk_labels = labels[:, start:end]

                    # Correct position IDs for this chunk
                    chunk_position_ids = torch.arange(
                        start, end, device=input_ids.device, dtype=torch.long
                    )
                    chunk_position_ids = chunk_position_ids.unsqueeze(0).expand(
                        batch_size, -1
                    )

                    # Forward with adapted weights
                    logits_chunk = self.qwen_model(
                        chunk_input,
                        position_ids=chunk_position_ids,
                        mlp_overrides_by_layer=current_weights,
                    )
                    all_logits.append(logits_chunk)

                    # Adapt weights (except for last chunk)
                    if chunk_idx < num_chunks - 1:
                        chunk_loss = self.loss_fct(
                            logits_chunk.reshape(-1, logits_chunk.size(-1)),
                            chunk_labels.reshape(-1),
                        )

                        all_params = [
                            current_weights[i][key]
                            for i, _ in mlp_params
                            for key in ["gate_proj", "up_proj", "down_proj"]
                        ]

                        grads = torch.autograd.grad(
                            chunk_loss, all_params, retain_graph=False
                        )

                        # Update adapted weights (no graph needed for validation)
                        grad_idx = 0
                        for layer_idx, _ in mlp_params:
                            for key in ["gate_proj", "up_proj", "down_proj"]:
                                current_weights[layer_idx][key] = (
                                    current_weights[layer_idx][key]
                                    - self.dense_ttt_lr * grads[grad_idx]
                                )
                                grad_idx += 1

                # Combine logits and compute final loss
                logits_post = torch.cat(all_logits, dim=1)
                loss = self.loss_fct(
                    logits_post.reshape(-1, logits_post.size(-1)),
                    labels.reshape(-1),
                )
                logits_for_pos_loss = logits_post

        elif self.use_moe_ttt_e2e:
            # MoE TTT-E2E: Sequential mini-batch adaptation of gate and/or expert weights
            with torch.inference_mode(False), torch.enable_grad():
                input_ids = input_ids.clone()
                labels = labels.clone()

                batch_size, seq_len = input_ids.shape
                chunk_size = self.moe_ttt_batch_size

                # Collect parameters to adapt
                gate_params = []
                expert_params = []

                for i, layer in enumerate(self.qwen_model.model.layers):
                    mlp = layer.mlp
                    if not isinstance(mlp, MoEFeedForward):
                        continue

                    if self.moe_ttt_adapt_gate:
                        gate_params.append((i, mlp.gate.weight))

                    if self.moe_ttt_adapt_experts:
                        for e in range(mlp.num_experts):
                            expert = mlp.experts[e]
                            expert_params.append(
                                (
                                    i,
                                    e,
                                    {
                                        "gate_proj": expert.gate_proj.weight,
                                        "up_proj": expert.up_proj.weight,
                                        "down_proj": expert.down_proj.weight,
                                    },
                                )
                            )

                # Initialize adapted weights as clones
                current_gate_weights = {}
                for layer_idx, weight in gate_params:
                    current_gate_weights[layer_idx] = weight.clone()

                current_expert_weights = {}
                for layer_idx, expert_idx, param_dict in expert_params:
                    if layer_idx not in current_expert_weights:
                        current_expert_weights[layer_idx] = {}
                    current_expert_weights[layer_idx][expert_idx] = {
                        key: param.clone() for key, param in param_dict.items()
                    }

                # Process sequence in chunks
                all_logits = []
                num_chunks = (seq_len + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, seq_len)

                    chunk_input = input_ids[:, start:end]
                    chunk_labels = labels[:, start:end]

                    chunk_position_ids = (
                        torch.arange(
                            start, end, device=input_ids.device, dtype=torch.long
                        )
                        .unsqueeze(0)
                        .expand(batch_size, -1)
                    )

                    logits_chunk = self.qwen_model(
                        chunk_input,
                        position_ids=chunk_position_ids,
                        gate_weight_overrides=current_gate_weights
                        if self.moe_ttt_adapt_gate
                        else None,
                        expert_weight_overrides_by_layer=current_expert_weights
                        if self.moe_ttt_adapt_experts
                        else None,
                    )
                    all_logits.append(logits_chunk)

                    # Adapt weights (except for last chunk)
                    if chunk_idx < num_chunks - 1:
                        chunk_loss = self.loss_fct(
                            logits_chunk.reshape(-1, logits_chunk.size(-1)),
                            chunk_labels.reshape(-1),
                        )

                        all_params = []
                        if self.moe_ttt_adapt_gate:
                            all_params.extend(
                                [current_gate_weights[i] for i, _ in gate_params]
                            )
                        if self.moe_ttt_adapt_experts:
                            for layer_idx, expert_idx, _ in expert_params:
                                for key in ["gate_proj", "up_proj", "down_proj"]:
                                    all_params.append(
                                        current_expert_weights[layer_idx][expert_idx][
                                            key
                                        ]
                                    )

                        grads = torch.autograd.grad(
                            chunk_loss, all_params, retain_graph=False
                        )

                        # Update weights (no graph needed for validation)
                        grad_idx = 0
                        if self.moe_ttt_adapt_gate:
                            for layer_idx, _ in gate_params:
                                current_gate_weights[layer_idx] = (
                                    current_gate_weights[layer_idx]
                                    - self.moe_ttt_lr * grads[grad_idx]
                                )
                                grad_idx += 1

                        if self.moe_ttt_adapt_experts:
                            for layer_idx, expert_idx, _ in expert_params:
                                for key in ["gate_proj", "up_proj", "down_proj"]:
                                    current_expert_weights[layer_idx][expert_idx][
                                        key
                                    ] = (
                                        current_expert_weights[layer_idx][expert_idx][
                                            key
                                        ]
                                        - self.moe_ttt_lr * grads[grad_idx]
                                    )
                                    grad_idx += 1

                # Combine logits and compute final loss
                logits_post = torch.cat(all_logits, dim=1)
                loss = self.loss_fct(
                    logits_post.reshape(-1, logits_post.size(-1)),
                    labels.reshape(-1),
                )
                logits_for_pos_loss = logits_post

        else:
            logits = self(input_ids)
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            logits_for_pos_loss = logits

        # Accumulate per-position loss for val_loss_token (separate for part_a and part_b)
        per_pos_loss = self.compute_per_position_loss(logits_for_pos_loss, labels)
        if dataloader_idx == 0:
            self.val_per_position_losses_part_a.append(per_pos_loss.detach())
        else:
            self.val_per_position_losses_part_b.append(per_pos_loss.detach())

        # Log with part-specific name (part_a = wikitext, part_b = codeparrot)
        part_name = ["part_a_wikitext", "part_b_codeparrot"][dataloader_idx]
        self.log(
            f"val_loss_{part_name}",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )

        # Track for plotting
        if dataloader_idx == 0:
            self.val_losses_part_a.append(loss.detach().cpu())
        else:
            self.val_losses_part_b.append(loss.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        """Log per-position validation loss to wandb (separate for part_a and part_b)."""
        # Part A (wikitext)
        if self.val_per_position_losses_part_a:
            avg_per_pos = torch.stack(self.val_per_position_losses_part_a).mean(dim=0)
            seq_len = avg_per_pos.size(0)
            self.logger.experiment.log(
                {
                    "val_loss_token_part_a": wandb.plot.line_series(
                        xs=list(range(seq_len)),
                        ys=[avg_per_pos.cpu().tolist()],
                        keys=["loss"],
                        title="Val Loss by Token Position (Part A - Wikitext)",
                        xname="position",
                    )
                }
            )
            self.val_per_position_losses_part_a = []

        # Part B (codeparrot)
        if self.val_per_position_losses_part_b:
            avg_per_pos = torch.stack(self.val_per_position_losses_part_b).mean(dim=0)
            seq_len = avg_per_pos.size(0)
            self.logger.experiment.log(
                {
                    "val_loss_token_part_b": wandb.plot.line_series(
                        xs=list(range(seq_len)),
                        ys=[avg_per_pos.cpu().tolist()],
                        keys=["loss"],
                        title="Val Loss by Token Position (Part B - CodeParrot)",
                        xname="position",
                    )
                }
            )
            self.val_per_position_losses_part_b = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        return optimizer

    def set_phase(self, phase: int):
        """Set current training phase for logging."""
        self.current_phase = phase


class PhaseMarkerCallback(Callback):
    """Callback to log phase transitions to wandb."""

    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger

    def on_train_start(self, trainer, pl_module):
        phase = pl_module.current_phase
        if self.wandb_logger:
            self.wandb_logger.experiment.log(
                {"phase_start": phase, "global_step": trainer.global_step}
            )


def init_weights(module):
    """Initialize weights for pretraining."""
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def calculate_steps_for_tokens(
    tokens: int, batch_size: int, grad_accum: int, seq_length: int
) -> int:
    """Calculate number of training steps needed for a given token count."""
    effective_batch_size = batch_size * grad_accum
    tokens_per_step = effective_batch_size * seq_length
    return tokens // tokens_per_step


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continual pretraining for catastrophic forgetting experiments"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dense",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model configuration to use",
    )
    parser.add_argument(
        "--phase1-steps",
        type=int,
        default=None,
        help="Training steps for phase 1 (wikitext). Default: full dataset",
    )
    parser.add_argument(
        "--phase2-steps",
        type=int,
        default=None,
        help="Training steps for phase 2 (codeparrot). Default: full dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per GPU (default: 4)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Sequence length (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate - constant throughout (default: 1e-4)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=10000,
        help="Number of validation samples per dataset (default: 10000)",
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        default=500,
        help="Validation check interval in steps (default: 500)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="tiny-qwen-continual",
        help="Wandb project name",
    )
    return parser.parse_args()


def create_trainer(
    phase: int,
    max_steps: int,
    model_name: str,
    wandb_logger: WandbLogger,
    args,
    checkpoint_dir: str,
):
    """Create a Lightning Trainer for a specific phase."""

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{model_name}-phase{phase}"
        + "-{step:06d}-{val_loss_part_a_wikitext:.4f}",
        save_top_k=2,
        monitor="val_loss_part_a_wikitext",
        mode="min",
        save_last=True,
        every_n_train_steps=5000,
    )

    trainer = L.Trainer(
        max_steps=max_steps,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.grad_accum,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[lr_monitor, checkpoint_callback],
        default_root_dir=checkpoint_dir,
        enable_progress_bar=True,
    )

    return trainer, checkpoint_callback


if __name__ == "__main__":
    args = parse_args()

    # Timestamp for unique checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get model config
    model_type, config = MODEL_CONFIGS[args.model]
    model_name = args.model

    # Checkpoint directory with timestamp
    checkpoint_dir = f"cache/checkpoints/continual/{model_name}/{timestamp}"

    # Initialize tokenizer
    tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.add_special_tokens(["<|pad|>"])

    # Calculate default steps if not specified
    effective_batch_size = args.batch_size * args.grad_accum
    tokens_per_step = effective_batch_size * args.seq_length

    phase1_steps = args.phase1_steps
    if phase1_steps is None:
        phase1_steps = calculate_steps_for_tokens(
            WIKITEXT_TOKENS, args.batch_size, args.grad_accum, args.seq_length
        )
        print(
            f"Phase 1 (wikitext-103): Auto-calculated {phase1_steps} steps for ~{WIKITEXT_TOKENS / 1e6:.0f}M tokens"
        )

    phase2_steps = args.phase2_steps
    if phase2_steps is None:
        phase2_steps = calculate_steps_for_tokens(
            CODEPARROT_TOKENS, args.batch_size, args.grad_accum, args.seq_length
        )
        print(
            f"Phase 2 (codeparrot): Auto-calculated {phase2_steps} steps for ~{CODEPARROT_TOKENS / 1e9:.0f}B tokens"
        )

    # Disable Flash Attention for E2E mode (2nd-order gradients not supported)
    if getattr(config, "cortex_use_ttt_e2e", False) or getattr(
        config, "dense_use_ttt_e2e", False
    ) or getattr(config, "moe_use_ttt_e2e", False):
        print("\nDisabling Flash/Memory-efficient attention for 2nd-order gradients...")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    # Create model with random weights
    print(f"\nInitializing {model_name} ({model_type}) model with random weights...")
    model = Qwen3ForContinualPretrain(
        config=config,
        model_type=model_type,
        learning_rate=args.lr,
    )

    # Initialize weights properly
    model.apply(init_weights)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    if model_type == "cortex":
        k = config.cortex_hidden_size // config.cortex_k_ratio
        print(f"Cortex hidden size: {config.cortex_hidden_size}")
        print(
            f"Active units per token: {k} / {config.cortex_hidden_size} ({100 * k / config.cortex_hidden_size:.1f}%)"
        )
        if getattr(config, "cortex_use_ttt_e2e", False):
            print("Cortex TTT-E2E: ENABLED")
            adapt_gate = getattr(config, "cortex_ttt_adapt_gate", True)
            adapt_neurons = getattr(config, "cortex_ttt_adapt_neurons", False)
            print(f"  Adapt gate: {adapt_gate}, Adapt neurons: {adapt_neurons}")
            print(f"  TTT learning rate: {config.cortex_ttt_lr}")
            print(f"  TTT batch size: {config.cortex_ttt_batch_size} tokens")

    if getattr(config, "dense_use_ttt_e2e", False):
        print("Dense TTT-E2E: ENABLED")
        print("  Adapts MLP weights (gate_proj, up_proj, down_proj) at test time")
        print(f"  TTT learning rate: {config.dense_ttt_lr}")
        print(f"  TTT batch size: {config.dense_ttt_batch_size} tokens")
        print("  Sequential mini-batch adaptation (loss should decrease with position)")

    # ==========================================================================
    # Create Dataloaders
    # ==========================================================================

    print("\nLoading datasets...")

    # Training dataloader for Part A (wikitext-103)
    print("  Loading wikitext-103-raw-v1 training data...")
    train_dataset_part_a = WikitextDataset(
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        split="train",
    )
    train_loader_part_a = DataLoader(
        train_dataset_part_a,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Training dataloader for Part B (codeparrot-clean)
    print("  Loading codeparrot/codeparrot-clean training data...")
    train_dataset_part_b = CodeParrotDataset(
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        split="train",
    )
    train_loader_part_b = DataLoader(
        train_dataset_part_b,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Validation dataloaders for both datasets (used in BOTH phases)
    print(f"  Loading wikitext-103 validation data ({args.val_samples} samples)...")
    val_dataset_part_a = WikitextMapDataset(
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        split="validation",
        max_samples=args.val_samples,
    )
    val_loader_part_a = DataLoader(
        val_dataset_part_a,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # For codeparrot, skip some examples for validation to avoid overlap
    print(f"  Loading codeparrot validation data ({args.val_samples} samples)...")
    val_dataset_part_b = CodeParrotMapDataset(
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        split="train",  # codeparrot only has train split
        max_samples=args.val_samples,
        skip_examples=100000,  # Skip first 100k to use different data for val
    )
    val_loader_part_b = DataLoader(
        val_dataset_part_b,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    val_loaders = [val_loader_part_a, val_loader_part_b]

    # ==========================================================================
    # Configure Wandb Logger (single run for both phases)
    # ==========================================================================

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"{model_name}-{timestamp}",
        save_dir="cache/wandb",
        log_model=False,
        config={
            "model": model_name,
            "model_type": model_type,
            "experiment": "continual_pretraining_wikitext_codeparrot",
            "phase1_dataset": "wikitext-103-raw-v1",
            "phase2_dataset": "codeparrot/codeparrot-clean",
            "phase1_tokens": WIKITEXT_TOKENS,
            "phase2_tokens": CODEPARROT_TOKENS,
            "phase1_steps": phase1_steps,
            "phase2_steps": phase2_steps,
            "total_steps": phase1_steps + phase2_steps,
            "batch_size": args.batch_size,
            "effective_batch_size": effective_batch_size,
            "tokens_per_step": tokens_per_step,
            "seq_length": args.seq_length,
            "learning_rate": args.lr,
            "lr_schedule": "constant",
            "total_params": total_params,
            "timestamp": timestamp,
            **(
                {
                    "cortex_hidden_size": config.cortex_hidden_size,
                    "cortex_k_ratio": config.cortex_k_ratio,
                    "cortex_active_units": config.cortex_hidden_size
                    // config.cortex_k_ratio,
                    "cortex_use_ttt_e2e": getattr(config, "cortex_use_ttt_e2e", False),
                    "cortex_ttt_lr": getattr(config, "cortex_ttt_lr", None),
                    "cortex_ttt_batch_size": getattr(
                        config, "cortex_ttt_batch_size", None
                    ),
                    "cortex_ttt_adapt_gate": getattr(
                        config, "cortex_ttt_adapt_gate", True
                    ),
                    "cortex_ttt_adapt_neurons": getattr(
                        config, "cortex_ttt_adapt_neurons", False
                    ),
                }
                if model_type == "cortex"
                else {}
            ),
        },
    )

    # ==========================================================================
    # Phase 1: Train on wikitext-103 (Part A)
    # ==========================================================================

    print(f"\n{'=' * 60}")
    print(f"PHASE 1: Training on wikitext-103-raw-v1 for {phase1_steps} steps")
    print(f"{'=' * 60}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    model.set_phase(1)

    trainer_phase1, ckpt_callback_phase1 = create_trainer(
        phase=1,
        max_steps=phase1_steps,
        model_name=model_name,
        wandb_logger=wandb_logger,
        args=args,
        checkpoint_dir=checkpoint_dir,
    )

    # Log phase start
    wandb_logger.experiment.log({"phase": 1, "phase_event": "start"})

    # Run validation before training to get baseline metrics
    print("Running initial validation to get baseline metrics...")
    trainer_phase1.validate(model=model, dataloaders=val_loaders)

    trainer_phase1.fit(
        model=model,
        train_dataloaders=train_loader_part_a,
        val_dataloaders=val_loaders,
    )

    # Save phase 1 final checkpoint
    phase1_checkpoint = f"{checkpoint_dir}/{model_name}-phase1-final.ckpt"
    trainer_phase1.save_checkpoint(phase1_checkpoint)
    print(f"\nPhase 1 complete. Checkpoint saved: {phase1_checkpoint}")

    # Log phase 1 end metrics
    wandb_logger.experiment.log({"phase": 1, "phase_event": "end"})

    # ==========================================================================
    # Phase 2: Continue Training on codeparrot-clean (Part B)
    # ==========================================================================

    print(f"\n{'=' * 60}")
    print(f"PHASE 2: Training on codeparrot-clean for {phase2_steps} steps")
    print(f"{'=' * 60}")

    model.set_phase(2)

    # Create new trainer for phase 2
    # Note: We continue with the same model state (no reload needed)
    trainer_phase2, ckpt_callback_phase2 = create_trainer(
        phase=2,
        max_steps=phase1_steps + phase2_steps,  # Total steps from start
        model_name=model_name,
        wandb_logger=wandb_logger,
        args=args,
        checkpoint_dir=checkpoint_dir,
    )

    # Log phase start
    wandb_logger.experiment.log({"phase": 2, "phase_event": "start"})

    # Continue training from phase 1 checkpoint
    trainer_phase2.fit(
        model=model,
        train_dataloaders=train_loader_part_b,
        val_dataloaders=val_loaders,
        ckpt_path=phase1_checkpoint,
    )

    # Save phase 2 final checkpoint
    phase2_checkpoint = f"{checkpoint_dir}/{model_name}-phase2-final.ckpt"
    trainer_phase2.save_checkpoint(phase2_checkpoint)
    print(f"\nPhase 2 complete. Checkpoint saved: {phase2_checkpoint}")

    # Log phase 2 end
    wandb_logger.experiment.log({"phase": 2, "phase_event": "end"})

    # ==========================================================================
    # Final Summary and Plots
    # ==========================================================================

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Phase 1 (wikitext-103): {phase1_steps} steps")
    print(f"Phase 2 (codeparrot): {phase2_steps} steps")
    print(f"Total steps: {phase1_steps + phase2_steps}")
    print(f"Checkpoints saved in: {checkpoint_dir}")

    # Plot losses
    if model.train_losses:
        plt.figure(figsize=(12, 5))

        # Plot training loss
        plt.subplot(1, 2, 1)
        train_loss_values = [loss.item() for loss in model.train_losses]
        plt.plot(train_loss_values, alpha=0.7, label="train_loss")
        plt.axvline(x=phase1_steps, color="r", linestyle="--", label="Phase 2 start")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()

        # Plot validation losses
        plt.subplot(1, 2, 2)
        val_interval = args.val_check_interval

        if model.val_losses_part_a:
            val_indices = [
                i * val_interval for i in range(len(model.val_losses_part_a))
            ]
            plt.plot(
                val_indices,
                [loss.item() for loss in model.val_losses_part_a],
                label="val_loss_wikitext",
                marker="o",
                markersize=3,
            )
        if model.val_losses_part_b:
            val_indices = [
                i * val_interval for i in range(len(model.val_losses_part_b))
            ]
            plt.plot(
                val_indices,
                [loss.item() for loss in model.val_losses_part_b],
                label="val_loss_codeparrot",
                marker="s",
                markersize=3,
            )

        plt.axvline(x=phase1_steps, color="r", linestyle="--", label="Phase 2 start")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Validation Losses (Forgetting Analysis)")
        plt.legend()

        plt.tight_layout()
        plot_path = f"cache/{model_name}_continual_loss_{timestamp}.png"
        plt.savefig(plot_path)
        print(f"Loss plot saved: {plot_path}")

        # Log plot to wandb
        import wandb

        wandb.log({"loss_curves": wandb.Image(plot_path)})
        plt.show()

    # Finish wandb run
    wandb_logger.experiment.finish()

    print("\nDone!")
