import os
import re
import numpy as np
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


tokenizer = Tokenizer.from_file("tokenizer.json")
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?![^()]*\))")
# Specifically matches the top-level article title
ARTICLE_RE = re.compile(r"^= [^=]+ =$")
# Matches ANY header (Article, Section, Sub-section)
ANY_HEADER_RE = re.compile(r"^=.*=$")


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def build_sentence_corpus(input_path: str):
    sentence_ids: list[int] = []
    sentence_offsets: list[int] = [0]
    doc_offsets: list[int] = [0]

    current_doc_sentence_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # 1. Check if the line is any kind of header
            if ANY_HEADER_RE.match(line):
                # 2. If it's a TOP-LEVEL header, split the document
                if ARTICLE_RE.match(line):
                    if current_doc_sentence_count > 0:
                        doc_offsets.append(doc_offsets[-1] + current_doc_sentence_count)
                        current_doc_sentence_count = 0
                
                # 3. Always skip headers so they aren't included in the sentences
                continue

            # 4. Only actual prose reaches this point

            for sent in split_into_sentences(line):
                ids = wrapped_tokenizer.encode(sent, add_special_tokens=False)
                if not ids:
                    continue
                sentence_ids.extend(ids)
                sentence_offsets.append(len(sentence_ids))
                current_doc_sentence_count += 1

    if current_doc_sentence_count > 0:
        doc_offsets.append(doc_offsets[-1] + current_doc_sentence_count)

    if len(sentence_offsets) <= 1:
        sentence_ids_arr = np.array([], dtype=np.uint32)
        sentence_offsets_arr = np.array([0], dtype=np.uint64)
        doc_offsets_arr = np.array([0], dtype=np.uint64)
    else:
        max_id = max(sentence_ids)
        token_dtype = np.uint16 if max_id <= np.iinfo(np.uint16).max else np.uint32
        sentence_ids_arr = np.array(sentence_ids, dtype=token_dtype)
        sentence_offsets_arr = np.array(sentence_offsets, dtype=np.uint64)
        doc_offsets_arr = np.array(doc_offsets, dtype=np.uint64)

    return sentence_ids_arr, sentence_offsets_arr, doc_offsets_arr


def main() -> None:
    split_to_file = {
        "train": "wiki.train.raw",
        "validation": "wiki.valid.raw",
        "test": "wiki.test.raw",
    }

    base_dir = os.path.join("data", "wikitext-103-raw")
    out_dir = "data"

    for split, filename in split_to_file.items():
        in_path = os.path.join(base_dir, filename)

        sentence_ids, sentence_offsets, doc_offsets = build_sentence_corpus(in_path)
        sent_out_path = os.path.join(out_dir, f"wiki.{split}.sentences.npz")
        np.savez(
            sent_out_path,
            sentence_ids=sentence_ids,
            sentence_offsets=sentence_offsets,
            doc_offsets=doc_offsets,
        )


if __name__ == "__main__":
    main()
