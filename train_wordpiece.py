from transformers import AutoTokenizer
from typing import Tuple, BinaryIO
from collections import defaultdict
import multiprocessing as mp
import regex as re
import time
import os
import json

def train_wordpiece(input_paths, vocab_size: int, special_tokens: list[str]) -> list[str]:
    """
    Train a WordPiece vocabulary from one or more text files.
    
    Args:
        input_paths: a single path (str) or a list of (path, split_token) tuples.
                     split_token is the byte-string used to find chunk boundaries.
                     Example: [("data/tiny.txt", b"<|endoftext|>"), ("data/wiki.txt", b"\n")]
        vocab_size:  target vocabulary size.
        special_tokens: tokens to strip from the text before counting.
    """
    PAT = r"""\p{L}+|\p{N}+|[^\s\p{L}\p{N}]"""
    num_processes = 16

    # Normalise input_paths to a list of (path, split_token) tuples
    if isinstance(input_paths, str):
        input_paths = [(input_paths, b"<|endoftext|>")]

    word_freqs = defaultdict(int)
    time1 = time.time()
    for input_path, split_token in input_paths:
        print(f"Processing {input_path} (delimiter={split_token}) ...")
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, split_token)
            args = [(start, end, input_path, special_tokens, PAT)
                    for start, end in zip(boundaries[:-1], boundaries[1:])]
            with mp.Pool(num_processes) as pool:
                results = pool.starmap(process_chunk, args)
            for res in results:
                for k, v in res.items():
                    word_freqs[k] += v
    time2 = time.time()
    print(f"Time taken for pretokenization processing: {time2 - time1:.1f}s  "
          f"({len(word_freqs)} unique words)")

    alphabet = []
    for word in word_freqs.keys():
        if word[0] not in alphabet:
            alphabet.append(word[0])
        for letter in word[1:]:
            if f"##{letter}" not in alphabet:
                alphabet.append(f"##{letter}")
    alphabet.sort()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
    splits = {word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
              for word in word_freqs.keys()
              }
    pair_scores = compute_pair_scores(splits, word_freqs=word_freqs)

    best_pair = ""
    max_score = None
    for pair, score in pair_scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score

    while len(vocab) < vocab_size:
        scores = compute_pair_scores(splits, word_freqs=word_freqs)
        best_pair, max_score = "", None
        for pair, score in scores.items():
            if max_score is None or max_score < score:
                best_pair = pair
                max_score = score
        splits = merge_pair(*best_pair, splits, word_freqs=word_freqs)
        new_token = (
            best_pair[0] + best_pair[1][2:]
            if best_pair[1].startswith("##")
            else best_pair[0] + best_pair[1]
        )
        vocab.append(new_token)
    
    return vocab
    

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(boundary1, boundary2, input_path, special_tokens, PAT):
    word_freqs = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(boundary1)
        chunk = f.read(boundary2 - boundary1).decode("utf-8", errors="ignore")
        # remove special tokens from chunk
        splits = re.split("|".join(re.escape(tok) for tok in special_tokens), chunk)
        for split in splits:
            pretokenized_data = re.findall(PAT, split)
            for token in pretokenized_data:
                word_freqs[token] += 1
    return word_freqs

def compute_pair_scores(splits, word_freqs):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq
    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }

    return scores

def merge_pair(a, b, splits, word_freqs):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split

    return splits

def encode_word(word, vocab):
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            print(i, word[:i])
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f'##{word}'
    return tokens




if __name__ == "__main__":
    
    path = "./data/"
    save_path = "./data/"

    input_paths = [
        (os.path.join(path, "TinyStoriesV2-GPT4-train.txt"), b"<|endoftext|>"),
        (os.path.join(path, "wikitext-103-raw", "wiki.train.raw"), b"\n"),
    ]

    vocab = train_wordpiece(input_paths, 10000, special_tokens=["<|endoftext|>"])
    vocab_txt_path = os.path.join(save_path, "combined-vocab.txt")
    with open(vocab_txt_path, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(f"{token}\n")
    print(f"Saved vocab ({len(vocab)} tokens) to {vocab_txt_path}")

#     corpus = [
#     "This is the Hugging Face Course.",
#     "This chapter is about tokenization.",
#     "This section shows several tokenizer algorithms.",
#     "Hopefully, you will be able to understand how they are trained and generate tokens.",
# ]

    # word_freqs1 = defaultdict(int)
    # for text in corpus:
    #     words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    #     new_words = [word for word, offset in words_with_offsets]
    #     for word in new_words:
    #         word_freqs1[word] +=1
    # print(word_freqs1)
#     word_freqs = defaultdict(int)
#     for text in corpus:
#         PAT = r"""\p{L}+|\p{N}+|[^\s\p{L}\p{N}]"""
#         pretokenized_data = re.findall(PAT, text)
#         for word in pretokenized_data:
#             word_freqs[word] += 1
#     print(word_freqs)

#     alphabet = []
#     for word in word_freqs.keys():
#         if word[0] not in alphabet:
#             alphabet.append(word[0])
#         for letter in word[1:]:
#             if f"##{letter}" not in alphabet:
#                 alphabet.append(f"##{letter}")
#     alphabet.sort()
#     print(alphabet)
#     vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
#     splits = {word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
#               for word in word_freqs.keys()
#               }
#     print(splits)
    
#     pair_scores = compute_pair_scores(splits)
#     for i, key in enumerate(pair_scores.keys()):
#         print(f"{key}: {pair_scores[key]}")
#         if i >= 5:
#             break

#     best_pair = ""
#     max_score = None
#     for pair, score in pair_scores.items():
#         if max_score is None or max_score < score:
#             best_pair = pair
#             max_score = score
#     print(best_pair, max_score)

#     demo_splits = {word: pieces.copy() for word, pieces in splits.items()}
#     demo_splits = merge_pair("a", "##b", demo_splits)
#     print(demo_splits["about"])

#     vocab_size = 70
#     while len(vocab) < vocab_size:
#         scores = compute_pair_scores(splits)
#         best_pair, max_score = "", None
#         for pair, score in scores.items():
#             if max_score is None or max_score < score:
#                 best_pair = pair
#                 max_score = score
#         splits = merge_pair(*best_pair, splits)
#         new_token = (
#             best_pair[0] + best_pair[1][2:]
#             if best_pair[1].startswith("##")
#             else best_pair[0] + best_pair[1]
#         )
#         vocab.append(new_token)
#     print(vocab)
#     assert vocab == ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '##a', '##b', '##c', '##d', '##e', '##f', '##g', '##h', '##i', '##k',
#  '##l', '##m', '##n', '##o', '##p', '##r', '##s', '##t', '##u', '##v', '##w', '##y', '##z', ',', '.', 'C', 'F', 'H',
#  'T', 'a', 'b', 'c', 'g', 'h', 'i', 's', 't', 'u', 'w', 'y', 'ab', '##fu', 'Fa', 'Fac', '##ct', '##ful', '##full', '##fully',
#  'Th', 'ch', '##hm', 'cha', 'chap', 'chapt', '##thm', 'Hu', 'Hug', 'Hugg', 'sh', 'th', 'is', '##thms', '##za', '##zat',
#  '##ut']
    
#     print(encode_word("Hugging"))
#     print(encode_word("HOgging"))
    

    