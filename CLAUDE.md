# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tiny Qwen is a minimal PyTorch re-implementation of Qwen3 and Qwen2.5-VL models, supporting both text and vision capabilities, including dense and mixture of experts architectures.

## Key Architecture

### Model Classes
- `Qwen3`: Text-only model for standard Qwen3 variants (0.6B-32B)
- `Qwen3MoE`: Mixture of Experts model for Instruct and Thinking variants
- `Qwen2VL`: Vision-language model supporting image inputs

### Core Components
- **model/model.py**: Main model implementations with attention, RoPE, and MoE layers
- **model/processor.py**: Handles tokenization and input preprocessing for both text and vision
- **model/vision.py**: Vision encoder for Qwen2.5-VL models
- **run.py**: Interactive chat interface with model selection

## Development Commands

```bash
# Set up environment with uv
pip install uv && uv venv
source .venv/bin/activate  # Linux/macOS
uv pip install -r requirements.txt

# Run interactive chat
python run.py

# For vision models, reference images with @path/to/image.jpg syntax
```

## Model Loading Pattern

Models are loaded using HuggingFace repository IDs mapped to specific model classes in `REPO_ID_TO_MODEL_CLASS`. The processor is initialized with vision config for VL models:

```python
model = ModelClass.from_pretrained(repo_id)
processor = Processor(repo_id=repo_id, vision_config=vision_config)  # vision_config only for VL models
```

## Key Implementation Details

- Models use Flash Attention via `F.scaled_dot_product_attention`
- RoPE (Rotary Position Embeddings) implementation supports both 2D (text) and 3D (multimodal) position IDs
- MoE models include expert routing with top-k selection
- Streaming generation supported via generator pattern
- Stop tokens: `[151645, 151644, 151643]` for `<|im_end|>`, `<|im_start|>`, `<|endoftext|>`