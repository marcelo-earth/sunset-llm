<p align="center">
  <h1 align="center">🌅 Sunset LLM 📑</h1>
</p>

<p align="center">
  A mini-LLM that generates Spanish poems about sunsets and the sea. 📑<br />
</p>

## Architecture

```
Input: "El sol se hunde en"
            │
            ▼
    ┌───────────────┐
    │ Token Embed   │
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ Pos Embed     │
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ Transformer   │ ×6 layers
    │ Block         │
    │  - Attention  │
    │  - FFN        │
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ LM Head       │
    └───────────────┘
            │
            ▼
Output: "el mar"
```

## Installation

```bash
pip install torch
```

## Usage

```python
from model import GPT
from config import GPTConfig
from tokenizer import Tokenizer

# Create model
config = GPTConfig()
model = GPT(config)

# Load tokenizer
tokenizer = Tokenizer()
tokenizer.load("tokenizer.json")

# Generate text
prompt = "El mar en calma"
output = model.generate(prompt, tokenizer, max_tokens=50)
print(output)
```

## Training

```bash
python train.py
```

## Configuration

| Parameter | Value |
|-----------|-------|
| vocab_size | ~5000 |
| d_model | 384 |
| n_heads | 6 |
| n_layers | 6 |
| context_len | 256 |
| batch_size | 64 |

## Example Output

```
Input:  "El mar en calma"
Output: "El mar en calma espera la noche,
         mientras el sol derrama su oro
         sobre las olas que suspiran..."
```
