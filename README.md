# sunset-llm

Mini-LLM que genera poemas en español sobre atardeceres y el mar.

## Arquitectura

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
    │ Transformer   │ ×6 capas
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

## Instalación

```bash
pip install torch
```

## Uso

```python
from model import GPT
from config import GPTConfig
from tokenizer import Tokenizer

# Crear modelo
config = GPTConfig()
model = GPT(config)

# Cargar tokenizer
tokenizer = Tokenizer()
tokenizer.load("tokenizer.json")

# Generar texto
prompt = "El mar en calma"
output = model.generate(prompt, tokenizer, max_tokens=50)
print(output)
```

## Entrenamiento

```python
python train.py
```

## Configuración

| Parámetro | Valor |
|-----------|-------|
| vocab_size | ~5000 |
| d_model | 384 |
| n_heads | 6 |
| n_layers | 6 |
| context_len | 256 |
| batch_size | 64 |

## Ejemplo de Salida

```
Input:  "El mar en calma"
Output: "El mar en calma espera la noche,
         mientras el sol derrama su oro
         sobre las olas que suspiran..."
```
