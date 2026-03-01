import torch
import math
from einops import rearrange

def learning_rate_schedule(t, lr_max, lr_min, t_warm_up, t_cos_anneal):
    if t < t_warm_up:
        return t / t_warm_up * lr_max
    elif t >= t_warm_up and t <= t_cos_anneal:
        return lr_min + 0.5 * (1 + math.cos((t - t_warm_up) / (t_cos_anneal - t_warm_up) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min
    

def save_checkpoint(model, optimizer, iteration, out):
    obj = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(obj, out)

def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj["model_state_dict"])
    optimizer.load_state_dict(obj["optimizer_state_dict"])
    return obj["iteration"]