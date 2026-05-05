import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformer import Transformer_Bert
from dataset import BertPretrainingDataset
from utils import save_checkpoint
from utils import learning_rate_schedule

import numpy as np
import wandb
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('high')


def train(run, args):
    train_data_path = os.path.join(os.path.dirname(__file__), "data", "wiki.train.sentences.npz")
    train_data = np.load(train_data_path)
    valid_data_path = os.path.join(os.path.dirname(__file__), "data", "wiki.validation.sentences.npz")
    valid_data = np.load(valid_data_path)
    model = Transformer_Bert(
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.vocab_size,
        args.context_length,
        args.num_layers,
        dropout=args.dropout,
    ).to(args.device)
    model = torch.compile(model)
    train_dataset = BertPretrainingDataset(
        train_data,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        cls_token_id=args.cls_token_id,
        sep_token_id=args.sep_token_id,
        pad_token_id=args.pad_token_id,
        mask_token_id=args.mask_token_id,
        nsp_negative_prob=args.nsp_negative_prob,
        mlm_probability=args.mlm_probability,
        split="train",
    )
    valid_dataset = BertPretrainingDataset(
        valid_data,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        cls_token_id=args.cls_token_id,
        sep_token_id=args.sep_token_id,
        pad_token_id=args.pad_token_id,
        mask_token_id=args.mask_token_id,
        nsp_negative_prob=args.nsp_negative_prob,
        mlm_probability=args.mlm_probability,
        split="valid",
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    train_iter = iter(train_loader)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # iterations = args.total_tokens_processed // args.batch_size // args.context_length
    best_valid_loss = float("inf")
    for step in range(args.iterations):
        model.train()
        current_lr = learning_rate_schedule(step + 1, args.lr_max, args.lr_min, args.iter_warmup, t_total=args.iterations)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        mlm_inputs = batch["mlm_inputs"].to(args.device, non_blocking=True)
        token_type_ids = batch["token_type_ids"].to(args.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(args.device, non_blocking=True)
        mlm_labels = batch["mlm_labels"].to(args.device, non_blocking=True)
        mlm_mask_positions = batch["mlm_mask_positions"].to(args.device, non_blocking=True)
        nsp_labels = batch["nsp_labels"].to(args.device, non_blocking=True)

        with torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16):
            outputs = model(mlm_inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)
            batch_indices = torch.arange(mlm_inputs.size(0), device=args.device).unsqueeze(1)
            masked_logits = outputs["mlm_logits"][batch_indices, mlm_mask_positions]
            mlm_loss = F.cross_entropy(
                masked_logits.reshape(-1, args.vocab_size),
                mlm_labels.reshape(-1),
                ignore_index=-100,
            ) 
            nsp_loss = F.cross_entropy(outputs["nsp_logits"], nsp_labels)
            loss = mlm_loss + args.nsp_loss_weight * nsp_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_l2_norm)
        optimizer.step()
        run.log({
            "lr": current_lr,
            "loss/train": loss.item(),
            "loss/train_mlm": mlm_loss.item(),
            "loss/train_nsp": nsp_loss.item(),
        })
        print(
            f"iter: {step}, lr: {current_lr:.9f}, "
            f"loss: {loss.item():.5f}, mlm: {mlm_loss.item():.5f}, nsp: {nsp_loss.item():.5f}"
        )
        if (step + 1) % args.eval_interval == 0:
            valid_metrics = validate(valid_loader, model, args)
            valid_loss = valid_metrics["loss"]
            run.log({
                "loss/valid": valid_loss,
                "loss/valid_mlm": valid_metrics["mlm_loss"],
                "loss/valid_nsp": valid_metrics["nsp_loss"],
                "accuracy/valid_mlm": valid_metrics["mlm_accuracy"],
            })
            print(
                f"iter: {step}, train loss: {loss.item()}, "
                f"valid loss: {valid_loss}, valid mlm acc: {valid_metrics['mlm_accuracy']:.5f}"
            )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_checkpoint(model, optimizer, step, "bert_wikitext_pretrain_best.pt")
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, step, "bert_wikitext_pretrain.pt")


def validate(valid_loader, model, args):
    model.eval()
    total_loss = 0.0
    total_mlm_loss = 0.0
    total_nsp_loss = 0.0
    total_batches = 0
    total_mlm_correct = 0
    total_mlm_tokens = 0

    with torch.no_grad():
        for batch in valid_loader:
            mlm_inputs = batch["mlm_inputs"].to(args.device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(args.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(args.device, non_blocking=True)
            mlm_labels = batch["mlm_labels"].to(args.device, non_blocking=True)
            mlm_mask_positions = batch["mlm_mask_positions"].to(args.device, non_blocking=True)
            nsp_labels = batch["nsp_labels"].to(args.device, non_blocking=True)
            with torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16):
                outputs = model(mlm_inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)
                batch_indices = torch.arange(mlm_inputs.size(0), device=args.device).unsqueeze(1)
                masked_logits = outputs["mlm_logits"][batch_indices, mlm_mask_positions]
                mlm_loss = F.cross_entropy(
                    masked_logits.reshape(-1, args.vocab_size),
                    mlm_labels.reshape(-1),
                    ignore_index=-100,
                )
                nsp_loss = F.cross_entropy(outputs["nsp_logits"], nsp_labels)
                loss = mlm_loss + args.nsp_loss_weight * nsp_loss
            
            valid_mlm_labels = mlm_labels != -100
            mlm_predictions = masked_logits.argmax(dim=-1)
            total_mlm_correct += (mlm_predictions[valid_mlm_labels] == mlm_labels[valid_mlm_labels]).sum().item()
            total_mlm_tokens += valid_mlm_labels.sum().item()
            current_batch_size = batch["input_ids"].shape[0]
            total_loss += loss.item() * current_batch_size
            total_mlm_loss += mlm_loss.item() * current_batch_size
            total_nsp_loss += nsp_loss.item() * current_batch_size
            total_batches += current_batch_size
    avg_loss = total_loss / total_batches
    avg_mlm_loss = total_mlm_loss / total_batches
    avg_nsp_loss = total_nsp_loss / total_batches
    mlm_accuracy = total_mlm_correct / total_mlm_tokens if total_mlm_tokens > 0 else float("nan")
    return {
        "loss": avg_loss,
        "mlm_loss": avg_mlm_loss,
        "nsp_loss": avg_nsp_loss,
        "mlm_accuracy": mlm_accuracy,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="BERT",
        description="BERT",
        epilog="N/A",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="total iterations for training")
    # parser.add_argument("--total_tokens_processed", type=int, default=100000, help="batch size * total step count * context length")
    parser.add_argument("--iter_warmup", type=int, default=100, help="warm up iterations")

    parser.add_argument("--lr_max", type=float, default=1e-3, help="learning rate max")
    parser.add_argument("--lr_min", type=float, default=1e-4, help="learning rate min")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999), help="betas for AdamW")
    parser.add_argument("--eps", type=float, default=1e-8, help="eps for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (l1/l2 norm coefficient)")

    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--context_length", type=int, default=256, help="max sequence length")
    parser.add_argument("--d_model", type=int, default=768, help="dimension of model")
    parser.add_argument("--d_ff", type=int, default=3072, help="dimension of feed-forward network")
    parser.add_argument("--num_layers", type=int, default=12, help="number of layers")
    parser.add_argument("--num_heads", type=int, default=12, help="number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="token masking probability for MLM")
    parser.add_argument("--nsp_negative_prob", type=float, default=0.5, help="probability of sampling a negative NSP pair")
    parser.add_argument("--nsp_loss_weight", type=float, default=1.0, help="weight for NSP loss in total objective")
    parser.add_argument("--max_l2_norm", type=float, default=1.0, help="max L2 norm for gradient clipping")

    parser.add_argument("--save_interval", type=int, default=100, help="interval to save model checkpoint")
    parser.add_argument("--eval_interval", type=int, default=1000, help="interval to run validation")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    # load tokenizer json
    with open("tokenizer.json", "r") as f:
        import json

        tokenizer_json = json.load(f)
        args.mask_token_id = tokenizer_json["model"]["vocab"]["[MASK]"]
        args.cls_token_id = tokenizer_json["model"]["vocab"]["[CLS]"]
        args.sep_token_id = tokenizer_json["model"]["vocab"]["[SEP]"]
        args.pad_token_id = tokenizer_json["model"]["vocab"]["[PAD]"]
        args.vocab_size = len(tokenizer_json["model"]["vocab"])
    run = wandb.init(project=parser.prog, config=vars(args))
    print(args)
    train(run, args=args)


    # load data 
   
