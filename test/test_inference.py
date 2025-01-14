import unittest
import torch
from huggingface_hub import list_models
from PIL import Image

from models.model import Qwen2, Qwen2VL
from models.processor import MultimodalProcessor

class TestQwen2Inference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Gather all Qwen model IDs that we intend to test."""
        cls.forbidden_endings = ["Int4", "Int8", "AWQ", "GGUF", "MLX"]
        cls.forbidden_keywords = ["Audio", "72B"]
        all_models = list_models(author="Qwen")

        cls.model_ids = []
        for model_info in all_models:
            model_id = model_info.modelId
            if (
                model_id.startswith("Qwen/Qwen2")
                and not any(model_id.endswith(end) for end in cls.forbidden_endings)
                and not any(keyword in model_id for keyword in cls.forbidden_keywords)
            ):
                cls.model_ids.append(model_id)
        # cls.model_ids = [
        #     "Qwen/Qwen2-VL-2B-Instruct",  # VL model
        #     "Qwen/Qwen2.5-3B",            # Text-only model
        # ]
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"



    def test_inference_on_qwen2_variants(self):
        """
        Loop through the filtered Qwen2 model IDs.
        - Use Qwen2 for all non-VL models with a simple text prompt
        - Use Qwen2VL with a text+image prompt
        """
        total_models = len(self.model_ids)
        for idx, model_name in enumerate(self.model_ids, 1):
            with self.subTest(model_name=model_name):
                # Decide whether to load Qwen2 or Qwen2VL
                if "VL" in model_name:
                    model_cls = Qwen2VL
                    test_prompt = [
                        "<|im_start|>user\n<|vision_start|>",
                        Image.open("test-images/test-image.jpeg"),
                        "<|vision_end|>What's on the flower and what does it say about the meaning of life?<|im_end|>\n<|im_start|>assistant\n",
                    ]
                else:
                    model_cls = Qwen2
                    test_prompt = ["what is the meaning of life?"]

                print(f"\n[TEST] [{idx}/{total_models}] Loading: {model_name}")
                model = model_cls.from_pretrained(model_name).to(self.device)
                model.eval()

                # Create processor (with vision config for VL models)
                if "VL" in model_name:
                    processor = MultimodalProcessor(
                        tokenizer_name_or_path=model_name,
                        vision_config=model.config.vision_config,
                    )
                else:
                    processor = MultimodalProcessor(tokenizer_name_or_path=model_name)

                inputs = processor(test_prompt, device=self.device)

                with torch.no_grad():
                    # Generate a small number of new tokens
                    if "VL" in model_name:
                        output = model.generate(
                            input_ids=inputs["input_ids"],
                            pixels=inputs["pixels"],
                            d_image=inputs["d_image"],
                            max_new_tokens=8,
                        )
                    else:
                        output = model.generate(
                            input_ids=inputs["input_ids"],
                            max_new_tokens=8,
                        )

                # Decode the output to check if we got valid text
                decoded = processor.tokenizer.decode(output[0].tolist())
                print(f"Model: {model_name} => Output: {decoded[:100]}...")
                # Basic check: ensure we got something back
                self.assertTrue(len(decoded) > 0, f"Decoded text is empty for model {model_name}")


if __name__ == "__main__":
    unittest.main()
