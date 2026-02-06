import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import GPTConfig
from model import GPT
from tokenizer import Tokenizer
from dataset import PoemDataset, collate_fn, SAMPLE_POEMS, load_poems


def train(
    model: GPT,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    log_interval: int = 10
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        _, loss = model(input_ids, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % log_interval == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    return total_loss / num_batches


@torch.no_grad()
def generate_sample(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50
) -> str:
    """Generate a sample from the model."""
    model.eval()

    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    output = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_id
    )

    return tokenizer.decode(output[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Train sunset-llm")
    parser.add_argument("--data", type=str, default=None, help="Path to poems file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="model.pt", help="Checkpoint path")
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

    # Load or use sample data
    if args.data and os.path.exists(args.data):
        print(f"Loading poems from {args.data}")
        poems = load_poems(args.data)
    else:
        print("Using sample poems")
        poems = SAMPLE_POEMS

    print(f"Loaded {len(poems)} poems")

    # Train tokenizer
    print("Training tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.train(poems, vocab_size=1000)  # Smaller vocab for demo
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Save tokenizer
    tokenizer.save("tokenizer.json")

    # Create config and model
    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPT(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create dataset and dataloader
    dataset = PoemDataset(poems, tokenizer, max_length=config.context_len)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    print(f"Training examples: {len(dataset)}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    total_steps = args.epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    # Training loop
    print("\nStarting training...")
    print("-" * 50)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")

        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config,
            }, args.checkpoint)
            print(f"  Saved checkpoint (loss: {avg_loss:.4f})")

        # Generate sample every 10 epochs
        if epoch % 10 == 0:
            print("\n--- Sample generation ---")
            prompt = "El mar"
            output = generate_sample(model, tokenizer, prompt, device)
            print(f"Prompt: {prompt}")
            print(f"Output: {output}")
            print("-" * 50 + "\n")

    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {args.checkpoint}")
    print(f"Tokenizer saved to: tokenizer.json")


if __name__ == "__main__":
    main()
