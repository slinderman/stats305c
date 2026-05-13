import torch
import torch.nn as nn
import torch.nn.functional as F


N_BLOCKS = 8


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, 4 * d_model)
        self.ff2 = nn.Linear(4 * d_model, d_model)

    def forward(self, h):
        B, S, _ = h.shape
        H, D = self.n_heads, self.d_model // self.n_heads

        x = self.ln1(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, H, D)
        k = k.view(B, S, H, D)
        v = v.view(B, S, H, D)

        attn = torch.matmul(q, k.transpose(-2, -1))
        w = F.softmax(attn, dim=-2)
        z = torch.matmul(w, v).reshape(B, S, -1)

        h = self.proj(z)

        x = self.ln2(h)
        h = h + self.ff2(self.ff1(x))
        return h


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size=32, d_model=64, n_heads=4, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads) for _ in range(N_BLOCKS)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.block_order = list(range(N_BLOCKS))

    def forward(self, x):
        B, S = x.shape
        h = self.tok(x) + self.pos(x)
        for i in self.block_order:
            h = self.blocks[i](h)
        h = self.ln_f(h)
        return self.head(h).softmax(-1)


def make_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=10.0)


def train_step(model, opt, seq):
    logits = model(seq)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        seq.reshape(-1),
    )
    opt.step()
    loss.backward()
    return loss.item()
