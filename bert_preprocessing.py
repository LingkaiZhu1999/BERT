from dataclasses import dataclass

import numpy as np
import torch
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
    ):
        if context_length < 4:
            raise ValueError("context_length must be at least 4")
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

        self.sentence_ids = data["sentence_ids"]
        self.sentence_offsets = data["sentence_offsets"]
        self.doc_offsets = data["doc_offsets"]
        self.positive_starts = self._prepare_sentence_index(self.doc_offsets)
        self.num_sentences = len(self.sentence_offsets) - 1

    @staticmethod
    def _prepare_sentence_index(doc_offsets: np.ndarray) -> np.ndarray:
        positive_starts = []
        for d in range(len(doc_offsets) - 1):
            s0 = int(doc_offsets[d])
            s1 = int(doc_offsets[d + 1])
            if s1 - s0 >= 2:
                for s in range(s0, s1 - 1):
                    positive_starts.append(s)
        if not positive_starts:
            raise ValueError("No valid adjacent sentence pairs for NSP were found")
        return np.array(positive_starts, dtype=np.int64)

    def __len__(self):
        return len(self.positive_starts)

    def _sentence_from_index(self, sent_idx: int) -> np.ndarray:
        start = int(self.sentence_offsets[sent_idx])
        end = int(self.sentence_offsets[sent_idx + 1])
        return self.sentence_ids[start:end]

    def _build_bert_pair(self, left: np.ndarray, right: np.ndarray):
        # left/right are already special-tokenized as [CLS] ... [SEP].
        # Compose pair as: [CLS] A [SEP] B [SEP] by removing right-side [CLS].
        if len(left) == 0 or len(right) == 0:
            raise ValueError("Encountered empty sentence while building BERT pair")
        if right[0] == self.cls_token_id:
            right_payload = right[1:]
            right_type_len = len(right_payload)
        else:
            right_payload = right
            right_type_len = len(right)

        max_right = max(0, self.context_length - len(left))
        right_payload = right_payload[:max_right]
        right_type_len = min(right_type_len, max_right)
        combined = np.concatenate([left, right_payload], dtype=np.int64)[: self.context_length]

        tokens = np.full((self.context_length,), self.pad_token_id, dtype=np.int64)
        token_type_ids = np.zeros((self.context_length,), dtype=np.int64)
        attention_mask = np.zeros((self.context_length,), dtype=np.int64)

        seq_len = len(combined)
        tokens[:seq_len] = combined
        attention_mask[:seq_len] = 1

        right_start = min(len(left), self.context_length)
        right_end = min(right_start + right_type_len, self.context_length)
        token_type_ids[right_start:right_end] = 1

        return tokens, token_type_ids, attention_mask

    def _sample_sentence_pair(self, idx: int):
        sent_a_idx = int(self.positive_starts[idx])
        sent_b_pos_idx = sent_a_idx + 1

        is_negative = np.random.rand() < self.nsp_negative_prob
        nsp_label = 1 if is_negative else 0
        sent_b_idx = np.random.randint(0, self.num_sentences - 1) if is_negative else sent_b_pos_idx
        sent_b_idx = sent_b_idx + 1 if is_negative and sent_b_idx >= sent_b_pos_idx else sent_b_idx

        left = self._sentence_from_index(sent_a_idx)
        right = self._sentence_from_index(int(sent_b_idx))
        return left, right, nsp_label

    def _apply_mlm_masking(self, input_ids: torch.Tensor):
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_mask = (
            (input_ids == self.cls_token_id)
            | (input_ids == self.sep_token_id)
            | (input_ids == self.pad_token_id)
        )
        probability_matrix[special_mask] = 0.0

        masked_positions = torch.bernoulli(probability_matrix).bool()
        labels[~masked_positions] = -100

        masked_inputs = input_ids.clone()

        replace_with_mask = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_positions
        )
        masked_inputs[replace_with_mask] = self.mask_token_id

        replace_with_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_positions
            & ~replace_with_mask
        )
        random_tokens = torch.randint(5, self.vocab_size, labels.shape, dtype=torch.long)
        masked_inputs[replace_with_random] = random_tokens[replace_with_random]

        return masked_inputs, labels

    def __getitem__(self, idx: int) -> dict:
        if self.split == "train":
            idx = np.random.randint(0, len(self))

        left, right, nsp_label = self._sample_sentence_pair(idx)

        input_ids, token_type_ids, attention_mask = self._build_bert_pair(left, right)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids_t = torch.tensor(token_type_ids, dtype=torch.long)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)
        nsp_label_t = torch.tensor(nsp_label, dtype=torch.long)
        mlm_inputs_t, mlm_labels_t = self._apply_mlm_masking(input_ids_t)

        return {
        "input_ids": input_ids_t,
        "token_type_ids": token_type_ids_t,
        "attention_mask": attention_mask_t,
        "nsp_labels": nsp_label_t,
        "mlm_inputs": mlm_inputs_t,
        "mlm_labels": mlm_labels_t,
    }


