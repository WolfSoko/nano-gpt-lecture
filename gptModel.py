import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, hyper_p):
        super().__init__()

        self.key = nn.Linear(hyper_p['n_embd'], head_size, bias=False)
        self.query = nn.Linear(hyper_p['n_embd'], head_size, bias=False)
        self.value = nn.Linear(hyper_p['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(hyper_p['block_size'], hyper_p['block_size'])))
        self.dropout = nn.Dropout(hyper_p['dropout'])

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention """

    def __init__(self, num_heads, head_size, hyper_p):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, hyper_p=hyper_p) for _ in range(num_heads)])
        self.proj = nn.Linear(hyper_p['n_embd'], hyper_p['n_embd'])
        self.dropout = nn.Dropout(hyper_p['dropout'])

    def forward(self, x):
        # x is (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C)
        out = self.dropout(self.proj(out))  # (B, T, C)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, hyper_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(hyper_p['dropout'])
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, hyper_p):
        # hyper_p['n_embd' is the number of embedded dimensions per char
        # n_head is the number of heads of self-attention
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size, hyper_p)  # i.e. 4 heads of 8-dim self-attention
        self.ffwd = FeedForward(n_embd, hyper_p)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x is (B, T, C)
        x = x + self.sa_heads(self.ln1(x))  # (B, T, C)
        x = x + self.ffwd(self.ln2(x))  # (B, T, C)
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self, hyper_p, vocab_size):
        super().__init__()

        self.hyper_p = hyper_p
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, self.hyper_p['n_embd'])
        self.position_embedding_table = nn.Embedding(self.hyper_p['block_size'], self.hyper_p['n_embd'])
        self.blocks = nn.Sequential(
            *[Block(self.hyper_p['n_embd'], n_head=self.hyper_p['n_head'], hyper_p=hyper_p) for _ in
              range(self.hyper_p['n_layer'])],
            nn.LayerNorm(self.hyper_p['n_embd']),
        )
        self.lm_head = nn.Linear(hyper_p['n_embd'], vocab_size)  # language model head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.hyper_p['device']))  # ,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # apply heads of seal-attention. (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_site tokens
            idx_cond = idx[:, -self.hyper_p['block_size']:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
