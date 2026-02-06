import json
import re
from collections import Counter
from typing import Optional


class Tokenizer:
    """Simple character-level tokenizer with optional BPE merges."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []

        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        special = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special):
            self.vocab[token] = i
            self.inverse_vocab[i] = token

    def train(self, texts: list[str], vocab_size: int = 5000):
        """Train tokenizer on texts using BPE-like algorithm."""
        self._init_special_tokens()

        # Start with character-level vocabulary
        all_text = " ".join(texts)
        chars = set(all_text)

        idx = len(self.vocab)
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
                idx += 1

        # Tokenize texts into characters
        words = []
        for text in texts:
            word_tokens = re.findall(r'\S+|\s+', text)
            for word in word_tokens:
                words.append(list(word))

        # BPE merges
        while len(self.vocab) < vocab_size:
            # Count pairs
            pairs = Counter()
            for word in words:
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += 1

            if not pairs:
                break

            # Find most common pair
            best_pair = pairs.most_common(1)[0][0]
            new_token = best_pair[0] + best_pair[1]

            # Add to vocabulary
            self.vocab[new_token] = idx
            self.inverse_vocab[idx] = new_token
            self.merges.append(best_pair)
            idx += 1

            # Merge in all words
            for word in words:
                i = 0
                while i < len(word) - 1:
                    if word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                        word[i] = new_token
                        del word[i + 1]
                    else:
                        i += 1

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        """Apply BPE merges to token list."""
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                    tokens[i] = merge[0] + merge[1]
                    del tokens[i + 1]
                else:
                    i += 1
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token ids."""
        # Start with characters
        tokens = list(text)

        # Apply merges
        tokens = self._apply_merges(tokens)

        # Convert to ids
        ids = []
        if add_special_tokens:
            ids.append(self.vocab[self.bos_token])

        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab[self.unk_token])

        if add_special_tokens:
            ids.append(self.vocab[self.eos_token])

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text."""
        tokens = []
        special = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}

        for idx in ids:
            token = self.inverse_vocab.get(idx, self.unk_token)
            if skip_special_tokens and token in special:
                continue
            tokens.append(token)

        return "".join(tokens)

    def save(self, path: str):
        """Save tokenizer to JSON file."""
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """Load tokenizer from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self.merges = [tuple(m) for m in data["merges"]]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.bos_token]
