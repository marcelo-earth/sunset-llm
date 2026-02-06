import torch
from torch.utils.data import Dataset
from typing import Optional

from tokenizer import Tokenizer


class PoemDataset(Dataset):
    """Dataset for loading and tokenizing poems."""

    def __init__(
        self,
        texts: list[str],
        tokenizer: Tokenizer,
        max_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            # Split long texts into chunks
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i + max_length + 1]
                if len(chunk) > 1:
                    self.examples.append(chunk)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.examples[idx]

        # Pad if necessary
        if len(tokens) < self.max_length + 1:
            tokens = tokens + [self.tokenizer.pad_id] * (self.max_length + 1 - len(tokens))

        tokens = torch.tensor(tokens, dtype=torch.long)

        # Input is all but last, target is all but first
        x = tokens[:-1]
        y = tokens[1:]

        # Mask padding in targets
        y[y == self.tokenizer.pad_id] = -100

        return {"input_ids": x, "labels": y}


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for DataLoader."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


def load_poems(path: str) -> list[str]:
    """Load poems from a text file (one poem per paragraph, separated by blank lines)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by double newlines
    poems = [p.strip() for p in content.split("\n\n") if p.strip()]
    return poems


# Sample poems for testing/demo
SAMPLE_POEMS = [
    """El sol se hunde en el mar,
tiñendo el cielo de oro y carmesí,
las olas susurran secretos al azar,
mientras la tarde muere aquí.""",

    """Sobre la playa desierta camino,
buscando en las olas tu recuerdo,
el atardecer pinta mi destino
con colores que nunca pierdo.""",

    """El horizonte se viste de fuego,
las gaviotas regresan a su nido,
el mar en calma, como en un juego,
refleja un sol que se ha dormido.""",

    """Crepúsculo marino, hora de ensueño,
cuando el día le dice adiós al mundo,
el mar guarda celosamente el empeño
de un amor profundo.""",

    """Las olas besan la arena dorada,
el cielo arde en tonos de violeta,
cada tarde es una pincelada
del pintor que la naturaleza interpreta.""",

    """En el ocaso el mar se vuelve espejo,
refleja nubes teñidas de coral,
un pescador regresa desde lejos
con la luz del día final.""",

    """Atardecer de sal y de espuma,
el viento canta entre las palmeras,
el sol desciende envuelto en bruma
tiñendo de oro las costeras.""",

    """El mar suspira con la tarde,
guardando memorias de otros días,
mientras el sol lentamente arde
consumiendo todas las porfías.""",

    """Cae la noche sobre el oleaje,
el faro enciende su luz primera,
el atardecer deja su mensaje
escrito en la costa marinera.""",

    """Entre las rocas el sol se esconde,
dejando estelas de luz rosada,
el mar pregunta pero nadie responde
en esta playa abandonada.""",
]
