import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformer import Transformer_Bert
from data_loading import bert_pair_data_loading, valid_bert_pair_data_loading
from utils import save_checkpoint
from utils import learning_rate_schedule
from masked_lm import apply_mlm_masking

import numpy as np
import wandb
import torch
import torch.nn.functional as F
from torch.optim import AdamW
torch.set_float32_matmul_precision('high')


def train(run, args):
    train_data_path = os.path.join(os.path.dirname(__file__), "sample_text.npy")
    train_data = np.load(train_data_path, mmap_mode="r")
    valid_data_path = os.path.join(os.path.dirname(__file__), "sample_text.npy")
    valid_data = np.load(valid_data_path, mmap_mode="r")
    model = Transformer_Bert(
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.vocab_size,
        args.context_length,
        args.num_layers,
    ).to(args.device)
    model = torch.compile(model)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    iterations = args.total_tokens_processed // args.batch_size // args.context_length
    for iter in range(iterations):
        model.train()
        current_lr = learning_rate_schedule(iter+1, args.lr_max, args.lr_min, args.iter_warmup, t_cos_anneal=args.iter_cos_annealing)
        run.log({"lr": current_lr})
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad()
        input_ids, token_type_ids, attention_mask, nsp_labels = bert_pair_data_loading(
            train_data,
            args.batch_size,
            args.context_length,
            device=args.device,
            cls_token_id=args.cls_token_id,
            sep_token_id=args.sep_token_id,
            pad_token_id=args.pad_token_id,
            nsp_negative_prob=args.nsp_negative_prob,
        )
        mlm_inputs, mlm_labels = apply_mlm_masking(
            input_ids,
            args.vocab_size,
            args.mask_token_id,
            mlm_probability=args.mlm_probability,
        )
        outputs = model(mlm_inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)
        mlm_loss = F.cross_entropy(
            outputs["mlm_logits"].reshape(-1, args.vocab_size),
            mlm_labels.reshape(-1),
            ignore_index=-100,
        )
        nsp_loss = F.cross_entropy(outputs["nsp_logits"], nsp_labels)
        loss = mlm_loss + args.nsp_loss_weight * nsp_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_l2_norm)
        optimizer.step()
        run.log({
            "loss/train": loss.item(),
            "loss/train_mlm": mlm_loss.item(),
            "loss/train_nsp": nsp_loss.item(),
        })
        print("iter: ", iter, "loss: ", loss.item(), "mlm:", mlm_loss.item(), "nsp:", nsp_loss.item())
        if iter % args.eval_interval == 0:
            valid_loss = validate(valid_data, model, args)
            run.log({"loss/valid": valid_loss}) 
            print("iter: ", iter, "train loss: ", loss.item(), "valid loss: ", valid_loss)
    save_checkpoint(model, optimizer, iterations, "owt_final_model.pt")


def validate(valid_data, model, args):
    payload_len = args.context_length - 3
    num_sequences = len(valid_data) // payload_len
    if num_sequences <= 0:
        return float("nan")

    iters = (num_sequences + args.batch_size - 1) // args.batch_size
    model.eval()
    total_loss = 0.0
    total_mlm_loss = 0.0
    total_nsp_loss = 0.0
    total_sequences = 0
    with torch.no_grad():
        for i in range(iters):
            input_ids, token_type_ids, attention_mask, nsp_labels = valid_bert_pair_data_loading(
                valid_data,
                args.batch_size,
                args.context_length,
                args.device,
                i,
                cls_token_id=args.cls_token_id,
                sep_token_id=args.sep_token_id,
                pad_token_id=args.pad_token_id,
                nsp_negative_prob=args.nsp_negative_prob,
            )
            mlm_inputs, mlm_labels = apply_mlm_masking(
                input_ids,
                args.vocab_size,
                args.mask_token_id,
                mlm_probability=args.mlm_probability,
            )
            outputs = model(mlm_inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)
            mlm_loss = F.cross_entropy(
                outputs["mlm_logits"].reshape(-1, args.vocab_size),
                mlm_labels.reshape(-1),
                ignore_index=-100,
            )
            nsp_loss = F.cross_entropy(outputs["nsp_logits"], nsp_labels)
            loss = mlm_loss + args.nsp_loss_weight * nsp_loss
            current_batch_size = input_ids.shape[0]
            total_loss += loss.item() * current_batch_size
            total_mlm_loss += mlm_loss.item() * current_batch_size
            total_nsp_loss += nsp_loss.item() * current_batch_size
            total_sequences += current_batch_size
    avg_loss = total_loss / total_sequences
    avg_mlm_loss = total_mlm_loss / total_sequences
    avg_nsp_loss = total_nsp_loss / total_sequences
    print("valid loss:", avg_loss, "valid mlm:", avg_mlm_loss, "valid nsp:", avg_nsp_loss)
    return avg_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="BERT",
        description="Assignment 1 of cs336",
        epilog="N/A",
    )
    # parser.add_argument("--iterations", type=int, default=1000, help="total iterations for training")
    parser.add_argument("--total_tokens_processed", type=int, default=327680000, help="batch size * total step count * context length")
    parser.add_argument("--iter_warmup", type=int, default=200, help="warm up iterations")
    parser.add_argument("--lr_max", type=float, default=3e-3, help="learning rate max")
    parser.add_argument("--lr_min", type=float, default=3e-4, help="learning rate min")
    parser.add_argument("--iter_cos_annealing", type=int, default=20000, help="iteration for cosine annealing")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="betas for AdamW")
    parser.add_argument("--eps", type=float, default=1e-8, help="eps for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay (l1/l2 norm coefficient)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--context_length", type=int, default=256, help="max sequence length")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--d_ff", type=int, default=1344,)
    parser.add_argument("--vocab_size", type=int, default=32000, help="10000 for tinystories, 32000 for owt")
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="token masking probability for MLM")
    parser.add_argument("--mask_token_id", type=int, default=103, help="[MASK] token id")
    parser.add_argument("--cls_token_id", type=int, default=101, help="[CLS] token id")
    parser.add_argument("--sep_token_id", type=int, default=102, help="[SEP] token id")
    parser.add_argument("--pad_token_id", type=int, default=0, help="[PAD] token id")
    parser.add_argument("--nsp_negative_prob", type=float, default=0.5, help="probability of sampling a negative NSP pair")
    parser.add_argument("--nsp_loss_weight", type=float, default=1.0, help="weight for NSP loss in total objective")
    parser.add_argument("--eval_interval", type=int, default=1000, help="interval to run validation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_l2_norm", type=float, default=1.0, help="max L2 norm for gradient clipping")

    args = parser.parse_args()
    run = wandb.init(project=parser.prog, config=vars(args))
    print(args)
    train(run, args=args)


    # load data 
   

