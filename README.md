
# 🦋 Quick Start

```python
import torch
from tokenizers import Tokenizer
from tiny_transformers.qwen import Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained(
    repo_id="Qwen/Qwen2-VL-7B-Instruct",
    local_dir="cache/Qwen2-VL-7B-Instruct",
)

tokenizer = Tokenizer.from_file("cache/Qwen2-VL-7B-Instruct/tokenizer.json")

text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nhello :)<|im_end|>\n<|im_start|>assistant\n"

token_ids = tokenizer.encode(text).ids
token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to("cuda")
generation_ids = model.generate(tokens=token_ids, max_new_tokens=16)
print(tokenizer.decode(generation_ids[0].tolist()))
```
