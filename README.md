# ✨ Tiny Qwen

A minimal, easy-to-read PyTorch reimplementation of the Qwen2 and Qwen2.5, the open source multi-modal LLM. 

We support both text-only (Instruct, Coder, Math, etc.) and text+vision, and any full prevision Qwen2+ model at any size. Just choose a repo id from Hugging Face [here](https://huggingface.co/Qwen). 

Keep in mind you'll likely need multiple GPU for models bigger than 32B. Stay tuned for FSDP support in the coming days. If you run into any issues, open a PR or create an issue.

# 🦋 Quick Start

```python
from models.model import Qwen2, Qwen2VL
from models.processor import MultimodalProcessor

processor = MultimodalProcessor(tokenizer_name_or_path=model_name)

# text-only models
model = Qwen2.from_pretrained(repo_id="Qwen/Qwen2.5-3B")

context = ["<|im_start|>user\nhello:)<|im_end|>\n<|im_start|>assistant\n"]
inputs = processor(context, device="cuda")
output = model.generate(input_ids=inputs["input_ids"], max_new_tokens=64)
output_text = processor.tokenizer.decode(output[0].tolist())

# text + vision models
model = Qwen2VL.from_pretrained(repo_id="Qwen/Qwen2-VL-2B-Instruct")

context = [
    "<|im_start|>user\n<|vision_start|>", 
    Image.open("test-images/test-image.jpeg"), 
    "<|vision_end|>What's on this image?<|im_end|>\n<|im_start|>assistant\n"
]
inputs = processor(context, device="cuda")
output = model.generate(
    input_ids=inputs["input_ids"],
    pixels=inputs["pixels"],
    d_image=inputs["d_image"],
    max_new_tokens=64,
)
output_text = processor.tokenizer.decode(output[0].tolist())
```

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
