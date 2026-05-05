import argparse
import json

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from transformer import Transformer_Bert


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def strip_compile_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if all(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
    return state_dict


def load_model(args, vocab_size: int) -> Transformer_Bert:
    model = Transformer_Bert(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(strip_compile_prefix(model_state), strict=True)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model: Transformer_Bert, tokenizer: PreTrainedTokenizerFast, args) -> None:
    encoded = tokenizer(
        args.text,
        text_pair=args.text_pair,
        add_special_tokens=True,
        truncation=True,
        max_length=args.context_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)
    token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids)).to(args.device)

    outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask.bool())

    print("Input tokens:")
    print(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer does not define a [MASK] token.")

    mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
    if not isinstance(mask_positions, list):
        mask_positions = [mask_positions]

    if mask_positions:
        print("\n[MASK] predictions:")
        for pos in mask_positions:
            probs = F.softmax(outputs["mlm_logits"][0, pos], dim=-1)
            top_probs, top_ids = torch.topk(probs, k=args.top_k)
            predicted_tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())
            print(f"position {pos}:")
            for tok, p in zip(predicted_tokens, top_probs.tolist()):
                print(f"  {tok:>16s}  prob={p:.4f}")
    else:
        print("\nNo [MASK] token found in input, skipping MLM top-k output.")

    nsp_probs = F.softmax(outputs["nsp_logits"][0], dim=-1)
    print("\nNSP probabilities:")
    print(f"  is_next={nsp_probs[0].item():.4f}")
    print(f"  not_next={nsp_probs[1].item():.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run BERT MLM/NSP inference from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="bert_wikitext_pretrain_best.pt")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    parser.add_argument("--text", type=str, required=True, help="Input text. Include [MASK] for MLM prediction.")
    parser.add_argument("--text_pair", type=str, default=None, help="Optional second sentence for NSP.")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k token predictions per [MASK].")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path)
    with open(args.tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    vocab_size = len(tokenizer_json["model"]["vocab"])
    model = load_model(args, vocab_size=vocab_size)
    model.eval()
    run_inference(model, tokenizer, args)
