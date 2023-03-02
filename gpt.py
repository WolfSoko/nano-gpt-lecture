import os
from datetime import datetime

import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import checkGpu

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 128  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = checkGpu.get_device()
eval_iters = 200
n_embd = 384  # number of embedded dimensions per char
n_head = 6  # number of heads in self-attention
n_layer = 6  # number of self-attention-computation blocks in the model
dropout = 0.2  # dropout rate
seed = 1337
# ------------


# Erstellen Sie ein Dictionary mit Ihren Modellparametern
hyper_params = dict(batch_size=batch_size, block_size=block_size, max_iters=max_iters, eval_interval=eval_interval,
                    learning_rate=learning_rate, device=device, eval_iters=eval_iters, n_embd=n_embd, n_head=n_head,
                    n_layer=n_layer, dropout=dropout, seed=seed)

print(hyper_params)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be for training, rest validation
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

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

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C)
        out = self.dropout(self.proj(out))  # (B, T, C)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd is the number of embedded dimensions per char
        # n_head is the number of heads of self-attention
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)  # i.e. 4 heads of 8-dim self-attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x is (B, T, C)
        x = x + self.sa_heads(self.ln1(x))  # (B, T, C)
        x = x + self.ffwd(self.ln2(x))  # (B, T, C)
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)],
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)  # language model head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # ,C)
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
            idx_cond = idx[:, -block_size:]
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

@torch.no_grad()
def estimate_loss(nn):
    out = {}
    nn.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = nn(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    nn.train()
    return out

def train(m, save_model=True, snapshots=True):

    lossis = []
    learn_date = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = f"./trained/{learn_date}/"

    # save hyper_params along with the model
    if save_model:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        with open(f"{save_folder}/hyper_params.json", 'x') as fp:
            json.dump(hyper_params, fp)

    for i in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if i % eval_interval == 0:
            losses = estimate_loss(m)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if save_model & snapshots:
                torch.save(m, f"{save_folder}/nano_gpt-SNAPSHOT.pth")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = m(xb, yb)
        lossis.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if save_model:
        torch.save(m, f"{save_folder}/nano_gpt-TRAINED.pth")

def generate(nn, max_new_tokens=1000):
    # generate a new sequence
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    idx = nn.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    return decode(idx)


## RUN ##

model = BigramLanguageModel()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameter of model: {total_params:,.0f}")

m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
train(m, True, True)  # generate from the model


print(generate(m, 1000))
