import os
from datetime import datetime

import json
import torch
import checkGpu
import gptModel
import wandb

# # hyper-parameters
hyper_params = {
    'batch_size': 32,  # how many independent sequences will we process in parallel?
    'block_size': 128,  # what is the maximum context length for predictions?
    'max_iters': 5000,
    'eval_interval': 100,
    'learning_rate': 3e-4,
    'device': (checkGpu.get_device()),
    'eval_iters': 200,
    'n_embd': 60,  # number of embedded dimensions per char
    'n_head': 6,  # number of heads in self-attention
    'n_layer': 6,  # number of self-attention-computation blocks in the model
    'dropout': 0.2,
    'seed': 1337
}

print(hyper_params)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="shakespear-nano-chat-gpt",
    # track hyperparameters and run metadata
    config={**hyper_params,
            "architecture": "GPT",
            "dataset": "all-shakespear",
            "epochs": 10,
            "optimizer": "AdamW",
            }
)

print("wandb run id: ", wandb.run.id, wandb.config)

torch.manual_seed(hyper_params['seed'])
torch.cuda.manual_seed(hyper_params['seed'])

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

print("Vocab size: ", vocab_size)
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
    ix = torch.randint(len(data) - hyper_params['block_size'], (hyper_params['batch_size'],))
    x = torch.stack([data[i:i + hyper_params['block_size']] for i in ix])
    y = torch.stack([data[i + 1:i + hyper_params['block_size'] + 1] for i in ix])
    x, y = x.to(hyper_params['device']), y.to(hyper_params['device'])
    return x, y


@torch.no_grad()
def estimate_loss(nn):
    out = {}
    nn.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hyper_params['eval_iters'])
        for k in range(hyper_params['eval_iters']):
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

    for i in range(hyper_params['max_iters']):

        # every once in a while evaluate the loss on train and val sets
        if i % hyper_params['eval_interval'] == 0:
            losses = estimate_loss(m)
            # log metrics to wandb
            wandb.log({"step": i, "losses-train": losses['train'], "losses-val": losses['val']})
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if save_model & snapshots:
                torch.save(m, f"{save_folder}/nano_gpt-SNAPSHOT.pth")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = m(xb, yb)
        lossis.append(loss.item())
        wandb.log({"step": i, "loss-batch": loss.item(), "loss-acc": sum(lossis[-50:]) / 50.0 })
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    losses = estimate_loss(m)
    # log metrics to wandb
    wandb.log({"step": hyper_params['max_iters'], "losses-train": losses['train'], "losses-val": losses['val']})
    print(f"step {hyper_params['max_iters']}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    if save_model:
        torch.save(m, f"{save_folder}/nano_gpt-TRAINED.pth")


def generate(nn, max_new_tokens=1000):
    # generate a new sequence
    context = torch.zeros((1, 1), dtype=torch.long, device=hyper_params['device'])
    idx = nn.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    return decode(idx)


## RUN ##

model = gptModel.BigramLanguageModel(hyper_params, vocab_size=vocab_size)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameter of model: {total_params:,.0f}")

m = model.to(checkGpu.get_device())

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=hyper_params['learning_rate'])
train(m, True, True)  # generate from the model
wandb.finish()

print(generate(m, 1000))

