import re
import torch
from typing import List
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from datasets import load_dataset


class AnthropicDataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        ignore_index: int = -100,
        cache_dir: str = None,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.im_start_id = tokenizer.token_to_id("<|im_start|>")
        self.im_end_id = tokenizer.token_to_id("<|im_end|>")
        self.pad_id = tokenizer.token_to_id("<|pad|>")
        self.ignore_index = ignore_index

        dataset = load_dataset(
            path="Unified-Language-Model-Alignment/Anthropic_HH_Golden",
            cache_dir=cache_dir,
        )[split]
        self.examples = dataset.map(self._process_example)

    def _process_example(self, example: dict) -> dict:
        for key in ["chosen", "rejected"]:
            text = example[key]
            pattern = (
                r"(?:Human:|Assistant:)\s*(.*?)(?=(?:\n\nHuman:|\n\nAssistant:|$))"
            )
            messages = re.findall(pattern, text, flags=re.DOTALL)
            formatted = "".join(
                [f"<|im_start|>{message.strip()}<|im_end|>" for message in messages]
            )
            number_of_turns = len(messages)
            input_ids = self.tokenizer.encode(formatted).ids

            # get a equal length label sequence by masking out non-ai tokens and performing a right shift.
            # e.g. [S, h1, h2, h3, E, S, a1, a2, a3, a4, E] -> [-, -, -, -, -, a1, a2, a3, a4, E, -]
            # where - is the ignore_index, S and E are the start and end tokens, h's and a's are the human and assistant tokens
            labels = [self.ignore_index] * len(input_ids)
            im_start_idx = [i for i, token_id in enumerate(input_ids) if token_id == self.im_start_id]
            im_end_idx = [i for i, token_id in enumerate(input_ids) if token_id == self.im_end_id]
            for idx, (start_idx, end_idx) in enumerate(zip(im_start_idx, im_end_idx)):
                if idx % 2 == 1:
                    for i in range(start_idx + 1, end_idx):
                        labels[i] = input_ids[i]
            labels = labels[1:] + [self.ignore_index]

            # Remove fixed-length padding
            example[f"{key}_input_ids"] = input_ids
            example[f"{key}_labels"] = labels
            example[f"{key}_number_of_turns"] = number_of_turns

        return example

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "chosen_input_ids": example["chosen_input_ids"],
            "chosen_labels": example["chosen_labels"],
            "rejected_input_ids": example["rejected_input_ids"],
            "rejected_labels": example["rejected_labels"],
        }


def data_collator_for_sft_example(batch: List[dict], tokenizer: Tokenizer):
    max_length = max(len(example["chosen_input_ids"]) for example in batch)
    pad_id = tokenizer.token_to_id("<|pad|>")
    ignore_index = -100
    for example in batch:
        # dynamically pad to max length in batch
        padding_length = max_length - len(example["chosen_input_ids"])
        example["chosen_input_ids"] += [pad_id] * padding_length
        example["chosen_labels"] += [ignore_index] * padding_length

        # only taking the chosen input_ids and labels for this sft example
        example["input_ids"] = torch.tensor(example["chosen_input_ids"])
        example["labels"] = torch.tensor(example["chosen_labels"])

    return {
        "input_ids": torch.stack([example["input_ids"] for example in batch]),
        "labels": torch.stack([example["labels"] for example in batch]),
    }


def init_anthropic_dataloader_for_sft_example(
    tokenizer: Tokenizer, batch_size: int, split: str = "train"
):
    dataset = AnthropicDataset(tokenizer, cache_dir="cache/data/Anthropic", split=split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: data_collator_for_sft_example(batch, tokenizer),
    )
    return dataloader


if __name__ == "__main__":
    # PYTHONPATH=. python data/anthropic.py
    tokenizer = Tokenizer.from_file("cache/Qwen2.5-3B/tokenizer.json")
    tokenizer.add_special_tokens(["<|pad|>"])
    dataloader = init_anthropic_dataloader_for_sft_example(tokenizer, 16)
    for batch in dataloader:
        print(batch)
        break
