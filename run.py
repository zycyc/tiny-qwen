# #!/usr/bin/env python3
# import sys
# import re
# from pathlib import Path
# from PIL import Image
# import torch

# import typer
# from rich.console import Console
# from rich.panel import Panel
# from rich.table import Table
# from rich.prompt import Prompt, Confirm
# from rich.text import Text

# from util import load_pretrained_model, generate_text_stream, generate_multimodal_stream
# from processor import Processor

# console = Console()
# app = typer.Typer(help="> Tiny-Qwen Interactive Chat")


# class QwenREPL:
#     def __init__(self):
#         self.model = None
#         self.tokenizer = None
#         self.processor = None
#         self.model_type = None
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#     def show_welcome(self):
#         """Show welcome screen and model selection"""
#         console.print(Panel.fit(">Tiny-Qwen Interactive Chat", style="bold blue"))

#         # Available models table
#         table = Table(
#             title="Available Models", show_header=True, header_style="bold magenta"
#         )
#         table.add_column("Model ID", style="cyan")
#         table.add_column("Type", style="green")
#         table.add_column("Description", style="white")

#         # Text-only models
#         table.add_row("Qwen/Qwen2-0.5B-Instruct", "Text", "Qwen2 0.5B chat model")
#         table.add_row("Qwen/Qwen2-1.5B-Instruct", "Text", "Qwen2 1.5B chat model")
#         table.add_row("Qwen/Qwen2-7B-Instruct", "Text", "Qwen2 7B chat model")

#         # Vision models
#         table.add_row(
#             "Qwen/Qwen2.5-VL-2B-Instruct", "Vision+Text", "Qwen2.5-VL 2B multimodal"
#         )
#         table.add_row(
#             "Qwen/Qwen2.5-VL-7B-Instruct", "Vision+Text", "Qwen2.5-VL 7B multimodal"
#         )

#         console.print(table)

#     def select_model(self) -> str:
#         """Interactive model selection"""
#         default_models = [
#             "Qwen/Qwen2.5-VL-3B-Instruct",
#             "Qwen/Qwen2.5-3B",
#         ]

#         console.print("\n[bold]Select a model:[/bold]")
#         for i, model in enumerate(default_models, 1):
#             console.print(f"  {i}. {model}")
#         console.print("  0. Enter custom model ID")

#         choice = Prompt.ask(
#             "Choose",
#             choices=[str(i) for i in range(len(default_models) + 1)],
#             default="1",
#         )

#         if choice == "0":
#             return Prompt.ask("Enter model ID")
#         else:
#             return default_models[int(choice) - 1]

#     def load_model(self, model_id: str):
#         """Load model and tokenizer"""
#         try:
#             console.print(f"\n= Loading {model_id}...")
#             self.model, self.tokenizer = load_pretrained_model(
#                 model_id, device=self.device
#             )

#             # Determine if this is a vision model
#             if "vl" in model_id.lower():
#                 self.model_type = "vision"
#                 # For vision models, we'll need the processor too
#                 self.processor = Processor(model_id, self.model.config.vision_config)
#             else:
#                 self.model_type = "text"

#             console.print(" Model loaded successfully!\n")
#             return True

#         except Exception as e:
#             console.print(f"[red]L Error loading model: {e}[/red]")
#             return False

#     def show_help(self):
#         """Show help information"""
#         if self.model_type == "vision":
#             help_text = """
# [bold]Commands:[/bold]
#   /help    - Show this help
#   /exit    - Exit the chat
#   /clear   - Clear conversation history
  
# [bold]Vision Model Usage:[/bold]
#   @path/image.jpg your question about the image
#   @img1.png @img2.jpg compare these images
  
# [bold]Examples:[/bold]
#   @receipt.jpg how much do I need to pay?
#   @diagram.png explain this flowchart
#   @photo1.jpg @photo2.jpg what's different between these?
#             """
#         else:
#             help_text = """
# [bold]Commands:[/bold]
#   /help    - Show this help  
#   /exit    - Exit the chat
#   /clear   - Clear conversation history
  
# [bold]Text Model Usage:[/bold]
#   Just type your message and press Enter
#             """

#         console.print(Panel(help_text, title="Help", style="blue"))

#     def parse_input(self, user_input: str) -> tuple[list, str]:
#         """
#         Parse input like '@path/image.jpg this is my receipt...' into [image, text]
#         Returns: ([inputs], input_type)
#         """
#         if self.model_type != "vision":
#             # Text-only model - no image parsing
#             return [user_input], "text"

#         # Find all @path patterns for vision models
#         image_pattern = r"@([^\s]+\.(jpg|jpeg|png|webp|gif|bmp))"
#         matches = re.findall(image_pattern, user_input, re.IGNORECASE)

#         if not matches:
#             # Pure text
#             return [user_input], "text"

#         # Process mixed input
#         inputs = []
#         remaining_text = user_input

#         for match in matches:
#             image_path = match[0]  # match[1] is the extension

#             # Load image
#             try:
#                 if not Path(image_path).exists():
#                     console.print(f"[yellow] Image not found: {image_path}[/yellow]")
#                     continue

#                 image = Image.open(image_path)
#                 inputs.append(image)

#                 # Remove @path from text
#                 remaining_text = remaining_text.replace(f"@{image_path}", "", 1)
#                 console.print(f"=� Loaded image: {image_path}")

#             except Exception as e:
#                 console.print(f"[red]L Error loading image {image_path}: {e}[/red]")
#                 return [user_input], "text"  # Fallback to text-only

#         # Add remaining text (cleaned)
#         cleaned_text = remaining_text.strip()
#         if cleaned_text:
#             inputs.append(cleaned_text)

#         return inputs, "multimodal"

#     def chat(self, user_inputs: list, input_type: str) -> str:
#         """Process user input and generate response"""
#         try:
#             if input_type == "multimodal" and self.processor:
#                 # Use processor for multimodal inputs
#                 processed = self.processor(user_inputs, device=self.device)
#                 response_tokens = []

#                 # Generate with multimodal input
#                 for token_text in generate_text_stream(
#                     model=self.model,
#                     tokenizer=self.tokenizer,
#                     prompt="",  # Prompt is already processed
#                     max_new_tokens=200,
#                 ):
#                     response_tokens.append(token_text)

#                 return "".join(response_tokens)

#             else:
#                 # Text-only generation
#                 prompt = user_inputs[0] if user_inputs else ""
#                 response_tokens = []

#                 for token_text in generate_text_stream(
#                     model=self.model,
#                     tokenizer=self.tokenizer,
#                     prompt=prompt,
#                     max_new_tokens=200,
#                 ):
#                     response_tokens.append(token_text)

#                 return "".join(response_tokens)

#         except Exception as e:
#             return f"Error generating response: {e}"

#     def run(self):
#         """Main REPL loop"""
#         self.show_welcome()

#         # Model selection loop
#         while True:
#             model_id = self.select_model()
#             if self.load_model(model_id):
#                 break
#             else:
#                 if not Confirm.ask("Try another model?"):
#                     return

#         # Show usage hint for VL models
#         if self.model_type == "vision":
#             console.print(
#                 Panel(
#                     "=� [bold]Tip:[/bold] Use @path/image.jpg to include images in your messages\nType /help for more info",
#                     style="yellow",
#                 )
#             )

#         console.print("Type /help for commands or start chatting!")
#         console.print("-" * 50)

#         # Main chat loop
#         while True:
#             try:
#                 user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

#                 if user_input.startswith("/"):
#                     if user_input == "/exit":
#                         console.print("=K Goodbye!")
#                         break
#                     elif user_input == "/help":
#                         self.show_help()
#                         continue
#                     elif user_input == "/clear":
#                         console.clear()
#                         continue
#                     else:
#                         console.print(
#                             "[yellow]Unknown command. Type /help for available commands.[/yellow]"
#                         )
#                         continue

#                 # Process and respond
#                 inputs, input_type = self.parse_input(user_input)

#                 console.print(f"\n[bold green]Assistant[/bold green]: ", end="")

#                 # Stream the response
#                 response_parts = []
#                 try:
#                     if input_type == "multimodal" and self.processor:
#                         # Handle multimodal input
#                         processed = self.processor(inputs, device=self.device)
#                         # Use proper multimodal generation
#                         for token in generate_multimodal_stream(
#                             self.model, self.tokenizer, processed, max_new_tokens=200
#                         ):
#                             console.print(token, end="", style="green")
#                             response_parts.append(token)
#                     else:
#                         # Text-only generation
#                         for token in generate_text_stream(
#                             self.model, self.tokenizer, inputs[0], max_new_tokens=200
#                         ):
#                             console.print(token, end="", style="green")
#                             response_parts.append(token)

#                 except KeyboardInterrupt:
#                     console.print("\n[yellow] Generation stopped[/yellow]")
#                 except Exception as e:
#                     console.print(f"\n[red]L Generation error: {e}[/red]")

#                 console.print()  # New line after response

#             except KeyboardInterrupt:
#                 if Confirm.ask("\nExit chat?"):
#                     break
#             except Exception as e:
#                 console.print(f"[red]Error: {e}[/red]")


# @app.command()
# def main():
#     """Launch the Tiny-Qwen interactive chat interface"""
#     try:
#         repl = QwenREPL()
#         repl.run()
#     except KeyboardInterrupt:
#         console.print("\n=K Goodbye!")
#     except Exception as e:
#         console.print(f"[red]Fatal error: {e}[/red]")
#         raise


# if __name__ == "__main__":
#     app()



from rich.console import Console
from rich.text import Text

ascii_logo = """
██╗    ████████╗██╗███╗   ██╗██╗   ██╗    ██████╗ ██╗    ██╗███████╗███╗   ██╗
╚██╗   ╚══██╔══╝██║████╗  ██║╚██╗ ██╔╝   ██╔═══██╗██║    ██║██╔════╝████╗  ██║
 ╚██╗     ██║   ██║██╔██╗ ██║ ╚████╔╝    ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║
 ██╔╝     ██║   ██║██║╚██╗██║  ╚██╔╝     ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║
██╔╝      ██║   ██║██║ ╚████║   ██║      ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║
╚═╝       ╚═╝   ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝
"""


console = Console()
yellow_logo = Text(ascii_logo, style="cyan")
console.print(yellow_logo)
