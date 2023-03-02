import torch
import checkGpu
from gptModel import BigramLanguageModel

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


def generate(nn, max_new_tokens, device):
    # generate a new sequence
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    idx = nn.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    return decode(idx)


def load_model(file="./nano_gpt-TRAINED.pth"):
    return torch.load(file)


model = load_model()
device = checkGpu.get_device()
model = model.to(device)
print(generate(model, 1000, device))
