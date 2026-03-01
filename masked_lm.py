import torch

def apply_mlm_masking(input_ids, vocab_size, mask_token_id, mlm_probability=0.15):
    mask_token_id = max(0, min(int(mask_token_id), vocab_size - 1))
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability, device=input_ids.device)
    masked_positions = torch.bernoulli(probability_matrix).bool()
    labels[~masked_positions] = -100

    masked_inputs = input_ids.clone()

    replace_with_mask = (
        torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked_positions
    )
    masked_inputs[replace_with_mask] = mask_token_id

    replace_with_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool()
        & masked_positions
        & ~replace_with_mask
    )
    random_tokens = torch.randint(0, vocab_size, labels.shape, device=input_ids.device)
    masked_inputs[replace_with_random] = random_tokens[replace_with_random]

    return masked_inputs, labels