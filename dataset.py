import numpy as np
import torch
import random
from torch.utils.data import Dataset


class BertPretrainingDataset(Dataset):
    def __init__(
        self,
        data,
        context_length: int,
        vocab_size: int,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0,
        mask_token_id: int = 103,
        nsp_negative_prob: float = 0.5,
        mlm_probability: float = 0.15,
        split: str = "train",
        short_seq_prob: float = 0.1,
    ):
        if context_length < 5:
            raise ValueError("context_length must be at least 5")
        if split not in {"train", "valid"}:
            raise ValueError("split must be one of {'train', 'valid'}")

        self.data = data
        self.context_length = context_length
        self.payload_len = context_length
        self.vocab_size = vocab_size
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = max(0, min(int(mask_token_id), vocab_size - 1))
        self.nsp_negative_prob = nsp_negative_prob
        self.mlm_probability = mlm_probability
        self.split = split
        self.short_seq_prob = short_seq_prob

        self.sentence_ids = data["sentence_ids"]
        self.sentence_offsets = data["sentence_offsets"]
        self.doc_offsets = data["doc_offsets"]
        self.documents = self._build_documents(self.doc_offsets)
        self.positive_doc_indices = np.array(
            [i for i, doc in enumerate(self.documents) if len(doc) >= 2],
            dtype=np.int64,
        )
        if len(self.positive_doc_indices) == 0:
            raise ValueError("No valid documents with at least two sentences were found")
        self.num_sentences = len(self.sentence_offsets) - 1

    @staticmethod
    def _build_documents(doc_offsets: np.ndarray) -> list[np.ndarray]:
        documents = []
        for d in range(len(doc_offsets) - 1):
            s0 = int(doc_offsets[d])
            s1 = int(doc_offsets[d + 1])
            if s1 > s0:
                documents.append(np.arange(s0, s1, dtype=np.int64))
        if not documents:
            raise ValueError("No valid documents were found in the saved npz")
        return documents

    def __len__(self):
        return len(self.positive_doc_indices)

    def _sentence_from_index(self, sent_idx: int) -> list[int]:
        start = int(self.sentence_offsets[sent_idx])
        end = int(self.sentence_offsets[sent_idx + 1])
        return self.sentence_ids[start:end].astype(np.int64).tolist()

    def _collect_segment(self, document: np.ndarray, start_pos: int, target_tokens: int) -> tuple[list[int], int]:
        tokens = []
        pos = start_pos
        while pos < len(document) and (len(tokens) < target_tokens or not tokens):
            tokens.extend(self._sentence_from_index(int(document[pos])))
            pos += 1
        return tokens, pos

    def _truncate_seq_pair(self, left: list[int], right: list[int], max_num_tokens: int) -> None:
        while len(left) + len(right) > max_num_tokens:
            trunc_tokens = left if len(left) > len(right) else right
            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def _build_bert_pair(self, left: list[int], right: list[int]):
        if len(left) == 0 or len(right) == 0:
            raise ValueError("Encountered empty segment while building BERT pair")

        self._truncate_seq_pair(left, right, self.context_length - 3)
        combined = [self.cls_token_id] + left + [self.sep_token_id] + right + [self.sep_token_id]
        segment_boundary = len(left) + 2

        tokens = np.full((self.context_length,), self.pad_token_id, dtype=np.int64)
        token_type_ids = np.zeros((self.context_length,), dtype=np.int64)
        attention_mask = np.zeros((self.context_length,), dtype=np.int64)

        seq_len = len(combined)
        tokens[:seq_len] = combined
        attention_mask[:seq_len] = 1
        token_type_ids[segment_boundary:seq_len] = 1

        return tokens, token_type_ids, attention_mask

    def _random_doc_index(self, exclude_doc_idx: int | None = None) -> int:
        if len(self.documents) == 1:
            return 0
        for _ in range(10):
            doc_idx = random.randrange(len(self.documents))
            if doc_idx != exclude_doc_idx:
                return doc_idx
        doc_idx = random.randrange(len(self.documents) - 1)
        if exclude_doc_idx is not None and doc_idx >= exclude_doc_idx:
            doc_idx += 1
        return doc_idx

    def _sample_segment_pair(self, idx: int):
        doc_idx = int(self.positive_doc_indices[idx])
        document = self.documents[doc_idx]

        max_num_tokens = self.context_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)

        start_pos = random.randrange(len(document) - 1)
        target_a_length = random.randint(1, max(1, target_seq_length - 1))
        left, next_pos = self._collect_segment(document, start_pos, target_a_length)
        if next_pos >= len(document):
            next_pos = len(document) - 1

        is_negative = random.random() < self.nsp_negative_prob
        nsp_label = 1 if is_negative else 0
        target_b_length = max(1, target_seq_length - len(left))

        if is_negative:
            random_doc_idx = self._random_doc_index(exclude_doc_idx=doc_idx)
            random_doc = self.documents[random_doc_idx]
            random_start = random.randrange(len(random_doc))
            right, _ = self._collect_segment(random_doc, random_start, target_b_length)
        else:
            right, _ = self._collect_segment(document, next_pos, target_b_length)

        return left, right, nsp_label

    def _apply_mlm_masking(self, input_ids: torch.Tensor):
        """Creates the predictions for the Masked LM objective."""
        special_token_ids = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
        cand_indexes = []
        for (i, token) in enumerate(input_ids):
            token_id = int(token.item())
            if token_id in special_token_ids:
                continue
            cand_indexes.append(i)
        random.shuffle(cand_indexes)
        output_tokens = input_ids.clone()
        max_predictions = max(1, int(round(len(input_ids) * self.mlm_probability)))
        num_to_predict = min(
            len(cand_indexes),
            max_predictions,
            max(1, int(round(len(cand_indexes) * self.mlm_probability))),
        )

        masked_lms = []
        covered_indexes = set()
        
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            
            covered_indexes.add(index)
            
            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = self.mask_token_id
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = int(input_ids[index].item())
                # 10% of the time, replace with random word
                else:
                    masked_token = random.randint(0, self.vocab_size - 1)
                    while masked_token in special_token_ids:
                        masked_token = random.randint(0, self.vocab_size - 1)

            output_tokens[index] = masked_token

            masked_lms.append((index, int(input_ids[index].item()))) # (Position, Original Label)

        masked_lms = sorted(masked_lms, key=lambda x: x[0])
        masked_lm_positions = [p for p, l in masked_lms]
        masked_lm_labels = [l for p, l in masked_lms]
        masked_lm_positions += [0] * (max_predictions - len(masked_lm_positions))
        masked_lm_labels += [-100] * (max_predictions - len(masked_lm_labels))

        return (
            output_tokens,
            torch.tensor(masked_lm_positions, dtype=torch.long),
            torch.tensor(masked_lm_labels, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> dict:
        if self.split == "train": 
            idx = np.random.randint(0, len(self))

        left, right, nsp_label = self._sample_segment_pair(idx)

        input_ids, token_type_ids, attention_mask = self._build_bert_pair(left, right)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)
        nsp_label_t = torch.tensor(nsp_label, dtype=torch.long)
        mlm_inputs_t, mlm_mask_positions_t, mlm_labels_t = self._apply_mlm_masking(input_ids_t)

        return {
            "input_ids": input_ids_t,
            "token_type_ids": token_type_ids_t,
            "attention_mask": attention_mask_t,
            "nsp_labels": nsp_label_t,
            "mlm_inputs": mlm_inputs_t,
            "mlm_mask_positions": mlm_mask_positions_t,
            "mlm_labels": mlm_labels_t,
        }

if __name__ == "__main__":
    import numpy as np
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    from torch.utils.data import DataLoader

    data = np.load("data/wiki.train.sentences.npz")


    tokenizer = Tokenizer.from_file("tokenizer.json")

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    dataset = BertPretrainingDataset(
        data=data,
        context_length=128,
        vocab_size=25000,
        cls_token_id=wrapped_tokenizer.cls_token_id,
        sep_token_id=wrapped_tokenizer.sep_token_id,
        pad_token_id=wrapped_tokenizer.pad_token_id,
        mask_token_id=wrapped_tokenizer.mask_token_id,
        nsp_negative_prob=0.5,
        mlm_probability=0.15,
        split="train",
    )

    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    batch = next(iter(train_loader))
    print(wrapped_tokenizer.convert_ids_to_tokens(batch["mlm_inputs"][0].tolist()))
    print(batch["mlm_mask_positions"][0])
    print(batch["mlm_labels"][0])
    print(wrapped_tokenizer.convert_ids_to_tokens(batch["mlm_labels"][0][0].tolist()))
