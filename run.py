import os
import sys
import re
import base64
import mimetypes
from pathlib import Path
from PIL import Image
import torch

import anthropic
import dotenv

import typer
import questionary
from questionary import Choice, Style
from rich.console import Console
from rich.text import Text

from model.processor import Processor
from model.model import Qwen2VL, Qwen3, Qwen3MoE

dotenv.load_dotenv()

ASCII_LOGO = """
██╗    ████████╗██╗███╗   ██╗██╗   ██╗    ██████╗ ██╗    ██╗███████╗███╗   ██╗
╚██╗   ╚══██╔══╝██║████╗  ██║╚██╗ ██╔╝   ██╔═══██╗██║    ██║██╔════╝████╗  ██║
 ╚██╗     ██║   ██║██╔██╗ ██║ ╚████╔╝    ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║
 ██╔╝     ██║   ██║██║╚██╗██║  ╚██╔╝     ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║
██╔╝      ██║   ██║██║ ╚████║   ██║      ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║
╚═╝       ╚═╝   ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝
"""

STARTING_HELP_TEXT = """
Welcome to Tiny-Qwen Interactive Chat!

Tips:
1. /help for more information.
2. /exit or Ctrl+C to exit.
"""

HELP_TEXT = """
Available commands:
/help - Show this help message
/exit - Exit the application

For vision models, use @path/to/image.jpg to include images in your messages.
"""

# Mapping of all models: generation -> variant -> HF repo id
ALL_MODELS = {
    "Qwen2.5-VL": {
        "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-32B-Instruct": "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen2.5-VL-72B-Instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    },
    "Qwen3": {
        "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
        "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
        "Qwen3-4B": "Qwen/Qwen3-4B",
        "Qwen3-8B": "Qwen/Qwen3-8B",
        "Qwen3-14B": "Qwen/Qwen3-14B",
        "Qwen3-32B": "Qwen/Qwen3-32B",
        "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen3-30B-A3B-Instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen3-235B-A22B-Instruct-2507": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen3-4B-Thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen3-30B-A3B-Thinking-2507": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen3-235B-A22B-Thinking-2507": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    },
}

REPO_ID_TO_MODEL_CLASS = {
    "Qwen/Qwen2.5-VL-3B-Instruct": Qwen2VL,
    "Qwen/Qwen2.5-VL-7B-Instruct": Qwen2VL,
    "Qwen/Qwen2.5-VL-32B-Instruct": Qwen2VL,
    "Qwen/Qwen2.5-VL-72B-Instruct": Qwen2VL,
    
    "Qwen/Qwen3-0.6B": Qwen3,
    "Qwen/Qwen3-1.7B": Qwen3,
    "Qwen/Qwen3-4B": Qwen3,
    "Qwen/Qwen3-8B": Qwen3,
    "Qwen/Qwen3-14B": Qwen3,
    "Qwen/Qwen3-32B": Qwen3,

    "Qwen/Qwen3-4B-Instruct-2507": Qwen3MoE,
    "Qwen/Qwen3-30B-A3B-Instruct-2507": Qwen3MoE,
    "Qwen/Qwen3-235B-A22B-Instruct-2507": Qwen3MoE,
    "Qwen/Qwen3-4B-Thinking-2507": Qwen3MoE,
    "Qwen/Qwen3-30B-A3B-Thinking-2507": Qwen3MoE,
    "Qwen/Qwen3-235B-A22B-Thinking-2507": Qwen3MoE,
}




STYLE = Style(
    [
        ("question", "bold"),
        ("selected", "fg:#000000 bg:#face0a bold"),
        ("highlighted", "fg:#face0a bold"),
        ("instruction", "fg:#888888"),
        ("separator", "fg:#666666"),
        ("text", ""),
        ("qmark", "fg:#face0a"),
    ]
)

console = Console(highlight=False)
app = typer.Typer(add_completion=False)


def parse_message_with_images(text):
    """Parse a message that may contain @path/to/image.jpg references and convert to Claude format."""
    # Find all @path patterns
    image_pattern = r'@([^\s]+\.(?:jpg|jpeg|png|gif|webp))'
    matches = re.finditer(image_pattern, text, re.IGNORECASE)
    
    content_blocks = []
    last_end = 0
    
    for match in matches:
        # Add text before the image reference
        if match.start() > last_end:
            text_part = text[last_end:match.start()].strip()
            if text_part:
                content_blocks.append({
                    "type": "text",
                    "text": text_part
                })
        
        # Process the image
        image_path = match.group(1)
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                console.print(f"Warning: Image file not found: {image_path}", style="yellow")
                continue
                
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith('image/'):
                console.print(f"Warning: Invalid image type: {image_path}", style="yellow")
                continue
            
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_data
                }
            })
            
            console.print(f"✓ Loaded image: {image_path}", style="green")
            
        except Exception as e:
            console.print(f"Error loading image {image_path}: {e}", style="red")
            continue
            
        last_end = match.end()
    
    # Add remaining text
    if last_end < len(text):
        remaining_text = text[last_end:].strip()
        if remaining_text:
            content_blocks.append({
                "type": "text",
                "text": remaining_text
            })
    
    # If no images found, return simple text format
    if not content_blocks:
        return text
    
    # If only text, return as string
    if len(content_blocks) == 1 and content_blocks[0]["type"] == "text":
        return content_blocks[0]["text"]
    
    return content_blocks


@app.command()
def main():

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))



    try:
        # Clear the terminal
        os.system("cls" if os.name == "nt" else "clear")

        # Show logo and help message
        yellow_logo = Text(ASCII_LOGO, style="#face0a")
        console.print(yellow_logo)
        console.print(STARTING_HELP_TEXT)

        # Select model generation e.g. Qwen2, Qwen2.5, Qwen2.5-VL, Qwen3, etc.
        selected_model_generation = questionary.select(
            message="Select model",
            choices=[Choice(generation, generation) for generation in ALL_MODELS],
            pointer=">",
            qmark="",
            style=STYLE,
        ).ask()

        if not selected_model_generation:
            return

        # Select model variant e.g. Qwen2-0.5B-Instruct, Qwen2.5-1.5B-Instruct, etc.
        selected_model_variant = questionary.select(
            message="Select model variant",
            choices=[
                Choice(variant, variant)
                for variant in ALL_MODELS[selected_model_generation]
            ],
            pointer=">",
            qmark="",
            style=STYLE,
        ).ask()

        if not selected_model_variant:
            return

        hf_repo_id = ALL_MODELS[selected_model_generation][selected_model_variant]
        console.print(f"\nLoading model: {hf_repo_id}")

        model_class = REPO_ID_TO_MODEL_CLASS.get(hf_repo_id)
        if not model_class:
            console.print("Invalid model variant", style="red")
            return

        model = model_class.from_pretrained(hf_repo_id)
        processor = Processor(repo_id=hf_repo_id)
        if not model or not processor:
            console.print("Failed to load model. Exiting...", style="red")
            return

        # Start the interactive chat loop
        messages = []
        while True:
            # Get user input
            user_input = input("> ").strip()

            # Handle special commands
            if user_input == "/exit":
                console.print("Goodbye!")
                break
            elif user_input == "/help":
                console.print(HELP_TEXT)
                continue
            elif not user_input:
                continue

            # Parse message for images
            parsed_content = parse_message_with_images(user_input)
            messages.append({"role": "user", "content": parsed_content})
            
            try:
                with client.messages.stream(
                    model="claude-sonnet-4-20250514",
                    messages=messages,
                    max_tokens=1024,
                ) as stream:
                    assistant_response = ""
                    for text in stream.text_stream:
                        print(text, end="", flush=True)
                        assistant_response += text
                    print()
                    
                # Add assistant's response to messages
                messages.append({"role": "assistant", "content": assistant_response})
                
            except Exception as e:
                console.print(f"Error generating response: {e}", style="red")
                # Remove the failed user message
                messages.pop()

    except KeyboardInterrupt:
        console.print("\nGoodbye!")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
        raise


if __name__ == "__main__":
    app()
