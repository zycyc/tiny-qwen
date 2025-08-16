from model import Qwen2VL
from processor import Processor
from PIL import Image

# Test vision model - use original working approach
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2VL.from_pretrained(repo_id=model_name, device_map="auto")
processor = Processor(repo_id=model_name, vision_config=model.config.vision_config)

context = [
    "<|im_start|>user\n<|vision_start|>",
    Image.open("images/test-img-1.jpg"),
    "<|vision_end|>What's on this image?<|im_end|>\n<|im_start|>assistant\n",
]

inputs = processor(context, device="cuda")

# Use built-in generate method
output = model.generate(
    input_ids=inputs["input_ids"],
    pixels=inputs["pixels"],
    d_image=inputs["d_image"],
    max_new_tokens=64,
)

# Decode only the new tokens (skip original input)
original_length = inputs["input_ids"].shape[1]
response_tokens = output[0, original_length:].tolist()
output_text = processor.tokenizer.decode(response_tokens)
print(output_text)
