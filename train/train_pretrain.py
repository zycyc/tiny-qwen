import argparse
import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from model.model import Qwen3, Qwen3MoE, ModelConfig
from data.slimpajama import init_slimpajama_dataloader, init_slimpajama_val_dataloader
from huggingface_hub import PyTorchModelHubMixin


# =============================================================================
# Model Configurations
# =============================================================================

# Dense model: Qwen3
QWEN3_CONFIG = ModelConfig(
    n_embed=768,  # was 1024
    n_heads=12,  # was 16
    n_kv_heads=4,  # was 8
    n_layer=12,  # was 28
    n_mlp=4033,  # was 3072 (or 4034)
    vocab_size=151936,
    rope_theta=1000000,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,  # was True
    head_dim=64,  # was 128
)

# MoE model: Small MoE (~600M total, ~200M active)
QWEN3_MOE_SMALL_CONFIG = ModelConfig(
    n_embed=768,
    n_heads=12,
    n_kv_heads=4,
    n_layer=12,
    n_mlp=2048,
    vocab_size=151936,
    rope_theta=1000000,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    head_dim=64,
    # MoE parameters
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=2048,
)

# Cortex model: Continuous expert with learned masks
QWEN3_CORTEX_SMALL_CONFIG = ModelConfig(
    n_embed=768,
    n_heads=12,
    n_kv_heads=4,
    n_layer=12,
    n_mlp=2048,
    vocab_size=151936,
    rope_theta=1000000,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    head_dim=64,
    # Cortex parameters
    use_cortex=True,
    cortex_hidden_size=3025,  # 55x55 grid (matches MoE total params)
    cortex_k_ratio=3,  # k=1008 active units (matches MoE's 1024)
    cortex_soft_kwta=True,
    cortex_temperature=1.0,
    cortex_use_lateral=False,
    cortex_lateral_steps=0,
)

MODEL_CONFIGS = {
    "dense": ("dense", QWEN3_CONFIG),
    "moe-small": ("moe", QWEN3_MOE_SMALL_CONFIG),
    "cortex-small": ("cortex", QWEN3_CORTEX_SMALL_CONFIG),
}


# =============================================================================
# Training Module
# =============================================================================


class Qwen3ForPretrain(L.LightningModule, PyTorchModelHubMixin):
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
        self.train_loss = []
        self.val_loss = []

    def forward(self, input_ids):
        return self.qwen_model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = self(input_ids)
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_loss.append(loss.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = self(input_ids)
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_loss.append(loss.detach().cpu())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.learning_rate * 0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def init_weights(module):
    """Initialize weights for pretraining."""
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Qwen3 models on SlimPajama")
    parser.add_argument(
        "--model",
        type=str,
        default="dense-0.6b",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model configuration to use",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum training steps (-1 for unlimited)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Maximum epochs (default: 1)",
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
        default=2048,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
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
        default=500,
        help="Number of validation samples (default: 500)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="tiny-qwen-pretrain",
        help="Wandb project name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get model config
    model_type, config = MODEL_CONFIGS[args.model]
    model_name = args.model

    # Initialize tokenizer
    tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.add_special_tokens(["<|pad|>"])

    # Create model with random weights
    print(f"Initializing {model_name} ({model_type}) model with random weights...")
    model = Qwen3ForPretrain(
        config=config,
        model_type=model_type,
        learning_rate=args.lr,
    )

    # Initialize weights properly
    model.apply(init_weights)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    if model_type == "moe":
        # Calculate active params for MoE
        # Active = total - (inactive experts' params)
        expert_params = (
            config.n_embed
            * config.moe_intermediate_size
            * 3  # gate, up, down per expert
            * config.num_experts
        )
        active_expert_params = (
            config.n_embed
            * config.moe_intermediate_size
            * 3
            * config.num_experts_per_tok
        )
        non_expert_params = total_params - expert_params * config.n_layer
        active_params = non_expert_params + active_expert_params * config.n_layer
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Active parameters per forward: {active_params / 1e6:.2f}M")
        print(
            f"Experts: {config.num_experts} total, {config.num_experts_per_tok} active per token"
        )
    elif model_type == "cortex":
        # Cortex uses all params but sparse activations
        k = config.cortex_hidden_size // config.cortex_k_ratio
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Cortex hidden size: {config.cortex_hidden_size}")
        print(
            f"Active units per token: {k} / {config.cortex_hidden_size} ({100 * k / config.cortex_hidden_size:.1f}%)"
        )
        active_params = total_params
    else:
        print(f"Model parameters: {total_params / 1e6:.2f}M")
        active_params = total_params

    # Create dataloaders
    print("Loading SlimPajama dataset (chunk1)...")
    train_dataloader = init_slimpajama_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        split="train",
        data_dir="train/chunk1",
    )

    val_dataloader = init_slimpajama_val_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        max_samples=args.val_samples,
    )

    # Configure wandb logger
    effective_batch_size = args.batch_size * args.grad_accum
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"{model_name}-slimpajama",
        save_dir="cache/wandb",
        log_model=False,
        config={
            "model": model_name,
            "model_type": model_type,
            "dataset": "SlimPajama-627B/chunk1",
            "batch_size": args.batch_size,
            "effective_batch_size": effective_batch_size,
            "seq_length": args.seq_length,
            "learning_rate": args.lr,
            "max_epochs": args.max_epochs,
            "total_params": total_params,
            "active_params": active_params,
            **(
                {
                    "num_experts": config.num_experts,
                    "num_experts_per_tok": config.num_experts_per_tok,
                }
                if model_type == "moe"
                else {
                    "cortex_hidden_size": config.cortex_hidden_size,
                    "cortex_k_ratio": config.cortex_k_ratio,
                    "cortex_active_units": config.cortex_hidden_size
                    // config.cortex_k_ratio,
                }
                if model_type == "cortex"
                else {}
            ),
        },
    )

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"cache/checkpoints/pretrain/{model_name}",
        filename=f"{model_name}" + "-{step:06d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        every_n_train_steps=5000,
    )

    # Configure trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.grad_accum,
        val_check_interval=1000,
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[lr_monitor, checkpoint_callback],
        default_root_dir=f"cache/checkpoints/pretrain/{model_name}",
    )

    # Train
    print(f"Starting pretraining for {args.max_epochs} epoch(s)...")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Tokens per step: {effective_batch_size * args.seq_length:,}")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Plot losses
    if model.train_loss:
        plt.figure(figsize=(10, 5))
        plt.plot(
            [loss.item() for loss in model.train_loss], label="train_loss", alpha=0.7
        )
        if model.val_loss:
            val_interval = 1000
            val_indices = [i * val_interval for i in range(len(model.val_loss))]
            plt.plot(
                val_indices,
                [loss.item() for loss in model.val_loss],
                label="val_loss",
                marker="o",
            )
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"{model_name} Pretraining on SlimPajama")
        plt.savefig(f"cache/{model_name}_pretrain_loss.png")

        # Log plot to wandb
        import wandb

        wandb.log({"loss_curve": wandb.Image(f"cache/{model_name}_pretrain_loss.png")})
        plt.show()

    # Finish wandb run
    wandb_logger.experiment.finish()

    # Save model
    # model.qwen_model.save_pretrained(f"cache/model/{model_name}-Pretrained")
    # model.push_to_hub(f"your-username/{model_name}-Pretrained")
