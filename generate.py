import argparse
import torch

from config import GPTConfig
from model import GPT
from tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate poems with sunset-llm")
    parser.add_argument("--prompt", type=str, default="El mar", help="Starting prompt")
    parser.add_argument("--checkpoint", type=str, default="model.pt", help="Model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Tokenizer path")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = Tokenizer()
    tokenizer.load(args.tokenizer)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    print("-" * 50)

    # Encode prompt
    tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    # Generate
    print(f"Prompt: {args.prompt}")
    print("-" * 50)

    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=tokenizer.eos_id
        )

    # Decode and print
    text = tokenizer.decode(output[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
