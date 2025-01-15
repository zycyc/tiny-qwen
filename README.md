
# 🦋 Quick Start

For text-only Qwen2 models

```python
from models.model import Qwen2
from models.processor import MultimodalProcessor

model_name = "Qwen/Qwen2.5-3B"

model = Qwen2.from_pretrained(repo_id=model_name)
processor = MultimodalProcessor(tokenizer_name_or_path=model_name)

context = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nhello :)<|im_end|>\n<|im_start|>assistant\n"]

inputs = processor(context, device="cuda")
output = model.generate(input_ids=inputs["input_ids"], max_new_tokens=64)

text = processor.tokenizer.decode(output[0].tolist())
print(text)
```

For text + vision Qwen2 models

```python
from models.model import Qwen2VL
from models.processor import MultimodalProcessor

model_name = "Qwen/Qwen2-VL-2B-Instruct"

model = Qwen2VL.from_pretrained(repo_id=model_name)
processor = MultimodalProcessor(tokenizer_name_or_path=model_name)

# Arrange context in a list of strings and images
context_1 = [
    "<|im_start|>user\n<|vision_start|>",
    Image.open("test-images/test-image.jpeg"),
    "<|vision_end|>What's on the flower and what does it say about the meaning of life?<|im_end|>\n<|im_start|>assistant\n",
]

# You can also use multiple images
context_2 = [
    "<|im_start|>user\nhere is an image\n<|vision_start|>",
    Image.open("test-images/test-image.jpeg"),
    "<|vision_end|>\nhere is another image\n<|vision_start|>",
    Image.open("test-images/test-image.jpeg"),
    "<|vision_end|>\nare these 2 images the same?<|im_end|>\n<|im_start|>assistant\n",
]

inputs = processor(context_1, device="cuda")
output = model.generate(
    input_ids=inputs["input_ids"],
    pixels=inputs["pixels"],
    d_image=inputs["d_image"],
    max_new_tokens=64,
)

text = processor.tokenizer.decode(output[0].tolist())
print(text)
```

# 🍀 Choose ANY Qwen2 / Qwen2-VL models

We support both text-only (Instruct, Coder, Math, etc.) and text+vision versions. Just pick its Hugging Face repo ID:

| Model Variant        | Sample Sizes                 | Text + Vision? | Example Repo ID                  |
| -------------------- | ---------------------------- | -------------: | -------------------------------- |
| **Qwen2.5-Instruct** | 0.5B, 1.5B, 3B, 7B, 14B, 32B |              ❌ | `Qwen/Qwen2.5-7B-Instruct`       |
| **Qwen2.5-Coder**    | 0.5B, 1.5B, 3B, 7B...        |              ❌ | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| **Qwen2.5-Math**     | 1.5B, 7B, 72B                |              ❌ | `Qwen/Qwen2.5-Math-7B-Instruct`  |
| **Qwen2-Instruct**   | 0.5B, 7B, 14B, 32B           |              ❌ | `Qwen/Qwen2-7B-Instruct`         |
| **Qwen2-VL**         | 2B, 7B, 72B                  |              ✅ | `Qwen/Qwen2-VL-2B-Instruct`      |

I’ve tested up to **32B** (single GPU, full precision). Stay tuned for FSDP support in the coming days. If you run into any issues, open a PR or create an issue.

# 🛠️ Fine-tune / Post-train Your Own

See `train/train_sft.py` for simple example on how to fine-tune your own Qwen model. Any library compatible with `torch.nn.Module` wourld work. Here I used [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/index.html) for its flexibility and simplcity. Also see `train/train_mnist.py` for inspiration on how to use this library.

To run any of the training scripts, just do:

```bash
PYTHONPATH=. python train/train_mnist.py
```

or

```bash
PYTHONPATH=. python train/train_sft.py
```
