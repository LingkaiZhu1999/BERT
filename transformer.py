import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F


class BertFFN(torch.nn.Module):
    def __init__(self, d_model, dff, device=None, dtype=None):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.k_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.v_proj = nn.Linear(d_model, num_heads * self.d_v)
        self.o_proj = nn.Linear(d_model, num_heads * self.d_v)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        Q = rearrange(self.q_proj(x), "... seq (num_heads d_q) -> ... num_heads seq d_q", num_heads=self.num_heads)
        K = rearrange(self.k_proj(x), "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        V = rearrange(self.v_proj(x), "... seq (num_heads d_v) -> ... num_heads seq d_v", num_heads=self.num_heads)

        attn_bias = None
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must be [batch_size, seq_len]")
            key_padding = attention_mask.to(torch.bool)
            attn_bias = torch.zeros(
                (x.shape[0], 1, 1, x.shape[1]),
                device=x.device,
                dtype=Q.dtype,
            )
            attn_bias = attn_bias.masked_fill(
                ~key_padding[:, None, None, :],
                torch.finfo(Q.dtype).min,
            )

        attention = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_bias, is_causal=False)
        attention = rearrange(attention, "... num_heads seq d_v -> ... seq (num_heads d_v)")
        output = self.o_proj(attention)
        return output


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.rmsnorm1 = nn.RMSNorm(d_model)
        self.rmsnorm2 = nn.RMSNorm(d_model)
        self.multihead_self_att = MultiheadSelfAttention(d_model, num_heads)
        self.ffn = BertFFN(d_model, d_ff)

    def forward(self, x, attention_mask=None):
        x = x + self.multihead_self_att(self.rmsnorm1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.rmsnorm2(x))
        return x


class BertMLMHead(torch.nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.decoder(hidden_states) + self.bias


class Transformer_Bert(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        vocab_size,
        context_length,
        num_layers,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.token_type_embeddings = nn.Embedding(2, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.transformer_blocks = torch.nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.mlm_head = BertMLMHead(d_model, vocab_size)
        self.nsp_head = nn.Linear(d_model, 2)

        self.mlm_head.decoder.weight = self.token_embeddings.weight

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [batch_size, seq_len]")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError("seq_len must be <= context_length")

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = (
            self.token_embeddings(input_ids)
            + self.position_embeddings(positions)
            + self.token_type_embeddings(token_type_ids)
        )
        x = self.embedding_norm(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask=attention_mask)

        x = self.final_norm(x)
        mlm_logits = self.mlm_head(x)
        nsp_logits = self.nsp_head(x[:, 0])
        return {
            "last_hidden_state": x,
            "mlm_logits": mlm_logits,
            "nsp_logits": nsp_logits,
        }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    seq_len = 16
    vocab_size = 30_522
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    token_type_ids = torch.zeros_like(input_ids)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    attention_mask[:, -3:] = False

    transformer = Transformer_Bert(
        d_model=128,
        num_heads=8,
        d_ff=256,
        vocab_size=vocab_size,
        context_length=64,
        num_layers=4,
    ).to(device)
    outputs = transformer(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    print(outputs["last_hidden_state"].shape)
    print(outputs["mlm_logits"].shape)
    print(outputs["nsp_logits"].shape)
