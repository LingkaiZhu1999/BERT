from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

def get_training_corpus(split="train"):
    for i in range(0, len(dataset), 1000):
        yield dataset[split][i, i + 1000]["text"]

dataset = load_dataset(
    "text",
    data_files={
        "train": "data/wikitext-103-raw/wiki.train.raw",
        "validation": "data/wikitext-103-raw/wiki.valid.raw",
        "test": "data/wikitext-103-raw/wiki.test.raw",
    },
)

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
tokenizer.train_from_iterator(dataset["train"]["text"], trainer=trainer)
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)

tokenizer.decoder = decoders.WordPiece(prefix="##")
print(tokenizer.decode(encoding.ids))

tokenizer.save("tokenizer.json")
