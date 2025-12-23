import torch
from typing import List
from torch.utils.data import Dataset, IterableDataset
from tokenizers import Tokenizer
from datasets import load_dataset


class SlimPajamaDataset(IterableDataset):
    """
    Iterable dataset for SlimPajama-627B pretraining.
    Uses streaming to avoid downloading the entire 895GB dataset.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_length: int = 2048,
        split: str = "train",
        data_dir: str = None,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.split = split

        # Load dataset with streaming to avoid downloading everything
        # For chunk1 specifically, use data_dir parameter
        self.dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split=split,
            streaming=True,
            data_dir=data_dir,  # e.g., "train/chunk1" for specific chunk
        )

        self.pad_id = tokenizer.token_to_id("<|pad|>")
        if self.pad_id is None:
            self.pad_id = tokenizer.token_to_id("<|endoftext|>")

    def __iter__(self):
        buffer = []

        for example in self.dataset:
            text = example["text"]
            tokens = self.tokenizer.encode(text).ids
            buffer.extend(tokens)

            # Yield complete sequences from buffer
            while len(buffer) >= self.seq_length + 1:
                # +1 because we need seq_length for input and seq_length for labels (shifted by 1)
                chunk = buffer[: self.seq_length + 1]
                buffer = buffer[self.seq_length:]  # Keep some overlap for context

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}


class SlimPajamaMapDataset(Dataset):
    """
    Map-style dataset for SlimPajama - loads a subset into memory.
    Use this for validation or when you want deterministic batching.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        seq_length: int = 2048,
        split: str = "validation",
        max_samples: int = 1000,
        data_dir: str = None,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.samples = []

        self.pad_id = tokenizer.token_to_id("<|pad|>")
        if self.pad_id is None:
            self.pad_id = tokenizer.token_to_id("<|endoftext|>")

        # Load dataset with streaming
        dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split=split,
            streaming=True,
            data_dir=data_dir,
        )

        buffer = []
        for example in dataset:
            text = example["text"]
            tokens = self.tokenizer.encode(text).ids
            buffer.extend(tokens)

            while len(buffer) >= self.seq_length + 1:
                chunk = buffer[: self.seq_length + 1]
                buffer = buffer[self.seq_length:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                self.samples.append({"input_ids": input_ids, "labels": labels})

                if len(self.samples) >= max_samples:
                    return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def data_collator_for_pretrain(batch: List[dict]):
    """Simple collator that stacks pre-padded sequences."""
    return {
        "input_ids": torch.stack([example["input_ids"] for example in batch]),
        "labels": torch.stack([example["labels"] for example in batch]),
    }


def init_slimpajama_dataloader(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_length: int = 2048,
    split: str = "train",
    data_dir: str = None,
    num_workers: int = 0,
):
    """
    Initialize a DataLoader for SlimPajama pretraining.

    Args:
        tokenizer: The tokenizer to use
        batch_size: Batch size
        seq_length: Sequence length for each training example
        split: Dataset split ("train", "validation", "test")
        data_dir: Specific data directory (e.g., "train/chunk1")
        num_workers: Number of worker processes
    """
    dataset = SlimPajamaDataset(
        tokenizer=tokenizer,
        seq_length=seq_length,
        split=split,
        data_dir=data_dir,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator_for_pretrain,
        num_workers=num_workers,
    )

    return dataloader


def init_slimpajama_val_dataloader(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_length: int = 2048,
    max_samples: int = 1000,
    data_dir: str = None,
):
    """
    Initialize a validation DataLoader for SlimPajama.
    Uses map-style dataset for deterministic evaluation.
    """
    dataset = SlimPajamaMapDataset(
        tokenizer=tokenizer,
        seq_length=seq_length,
        split="validation",
        max_samples=max_samples,
        data_dir=data_dir,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator_for_pretrain,
        shuffle=False,
    )

    return dataloader


if __name__ == "__main__":
    # PYTHONPATH=. python data/slimpajama.py
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.add_special_tokens(["<|pad|>"])

    print("Testing SlimPajama dataloader...")
    dataloader = init_slimpajama_dataloader(
        tokenizer,
        batch_size=2,
        seq_length=512,
        split="train",
        data_dir="train/chunk1",
    )

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        if i >= 2:
            break
