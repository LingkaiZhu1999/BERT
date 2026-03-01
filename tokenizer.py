import regex as re
from typing import Iterable
import json

class Tokenizer:
    PAT = r"""\p{L}+|\p{N}+|[^\s\p{L}\p{N}]"""

    def __init__(self, vocab, merges=None, special_tokens=None, unk_token="[UNK]"):
        """Construct a WordPiece tokenizer from a vocabulary.

        Args:
            vocab (list[str] | dict[int, str | bytes]): Vocabulary tokens.
            merges: Kept for backward compatibility; ignored for WordPiece.
            special_tokens (list[str] | None): Optional list of string special tokens.
            unk_token (str): Token used for unknown pieces.
        """

        ids_to_token = {idx: token for idx, token in enumerate(vocab)}
        self.vocab = ids_to_token
        self.tokens_to_ids = {token: idx for idx, token in self.vocab.items()}
        self.special_tokens = special_tokens or []
        self.unk_token = unk_token
        self.merges = merges or []

        if self.unk_token not in self.tokens_to_ids:
            raise ValueError(f"Unknown token {self.unk_token} must be in vocab")


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath=None, special_tokens=None, unk_token="[UNK]"):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = [line.rstrip("\n") for line in f]

        if isinstance(vocab, list):
            for token in special_tokens or []:
                if token not in vocab:
                    vocab.append(token)
        else:
            max_id = max(vocab.keys()) if vocab else -1
            existing_tokens = set(vocab.values())
            for token in special_tokens or []:
                if token not in existing_tokens:
                    max_id += 1
                    vocab[max_id] = token

        return cls(vocab, merges=None, special_tokens=special_tokens, unk_token=unk_token)
    
    def encode(self, text: str) -> list[int]:
        if self.special_tokens is not None:
            # Sort special tokens by length (descending) so longer tokens take precedence in matching
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(tok) for tok in sorted_special_tokens) + ")"
            splits = re.split(pattern, text)
        else:
            splits = [text]

        ids = []
        for split in splits:
            if split == "":
                continue
            if self.special_tokens is not None and split in self.special_tokens:
                ids.append(self.tokens_to_ids[split])
                continue

            pretokenized_data = re.findall(self.PAT, split)

            for token in pretokenized_data:
                pieces = self.encode_word(token)
                for piece in pieces:
                    ids.append(self.tokens_to_ids.get(piece, self.tokens_to_ids[self.unk_token]))
        return ids
    
    def encode_word(self, word: str) -> list[str]:
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.tokens_to_ids:
                i -= 1
            if i == 0:
                return [self.unk_token]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f'##{word}'
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        # return (self.encode(text) for text in iterable)
        for text in iterable:
            ids = self.encode(text)
            for id_ in ids:
                yield id_
        
    
    def decode(self, ids: list[int]) -> str:
        text = ""
        for id_ in ids:
            token = self.vocab.get(id_, "")
            if token:
                if token in self.special_tokens:
                    if text and not text.endswith(" "):
                        text += " "
                    text += token
                    continue

                if token.startswith("##"):
                    text += token[2:]
                    continue

                if text and not text.endswith(" ") and not re.match(r"^[^\w\s]+$", token):
                    text += " "
                text += token
        return text
    
if __name__ == "__main__":
    import numpy as np
    tokenizer = Tokenizer.from_files(
        "TinyStoriesV2-GPT4-train-vocab.txt",
        None,
        special_tokens=["<|endoftext|>"]
    )
    corpus_path = "./data/TinyStoriesV2-GPT4-train.txt"
    all_ids = []
    with open(corpus_path) as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    # # save all_ids to numpy array of datatype uint16
    print(max(all_ids))
    np.save("./data/TinyStoriesV2-GPT4-train.npy", np.array(all_ids, dtype=np.uint16))