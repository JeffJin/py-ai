import time

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6 # 384 / 6 = 64 every head has 64 dimensions
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

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
n = int(0.9 * len(data))  # first 90% will be train, rest val
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


@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

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
    # input of size (batch, time-step, channels)
    # output of size (batch, time-step, head size)
    B, T, C = x.shape
    k = self.key(x)  # (B,T,hs)
    q = self.query(x)  # (B,T,hs)
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
    wei = F.softmax(wei, dim=-1)  # (B, T, T)
    wei = self.dropout(wei)
    # perform the weighted aggregation of the values
    v = self.value(x)  # (B,T,hs)
    out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
    return out

class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel """

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out

class FeedForward(nn.Module):
  """ a simple linear layer followed by a non-linearity """

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd), # this is projection layer
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """ Transformer block: communication followed by computation """

  def __init__(self, n_embd, n_head):
    # n_embd: embedding dimension, n_head: the number of heads we'd like
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    # Layer Normalization computes the mean and variance independently
    # for each sample across its features.
    self.ln1 = LayerNorm(n_embd)
    self.ln2 = LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class LayerNorm(nn.Module):

  def __init__(self, dim, eps=1e-5):
    super().__init__()
    self.eps  = eps
    self.gamma = nn.Parameter(torch.ones(dim))
    self.beta = nn.Parameter(torch.zeros(dim))

  def forward(self, x):
    x_mean = x.mean(-1, keepdim=True) # apply to last dimension
    x_var = x.var(-1, keepdim=True, unbiased=False) # apply to last dimension
    x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
    return self.gamma * x_hat + self.beta

  def parameters(self):
    return [self.gamma, self.beta]

# super simple bigram model
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    # self.sa_head = Head(n_embd) # this is the same as set multi-head size 1
    # self.ffwd = FeedForward(n_embd)
    # self.sa_heads = MultiHeadAttention(n_head, n_embd // n_head)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    self.blocks = nn.Sequential(
      *[Block(n_embd, n_head=n_head) for _ in range(n_layer)],
       LayerNorm(n_embd)
    )

  def forward(self, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B,T) tensor of integers
    tok_embed = self.token_embedding_table(idx)  # (B,T,C)
    pos_embed = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
    x = tok_embed + pos_embed  # (B,T,C) token embedding + position embedding
    # x = self.sa_head(x) # apply self-attention head (B, T, C)
    # x = self.sa_heads(x) # apply multi-head self-attention (B, T, C)
    # x = self.ffwd(x) # apply feed-forward network (B, T, C)
    x = self.blocks(x)
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
      # crop idx to the last block_size tokens
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


total_start = time.perf_counter()

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_start = time.perf_counter()
for iter in range(max_iters):

  # every once in a while evaluate the loss on train and val sets
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  # sample a batch of data
  xb, yb = get_batch('train')

  # evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
train_elapsed = time.perf_counter() - train_start

# generate from the model
inference_start = time.perf_counter()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = m.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))
inference_elapsed = time.perf_counter() - inference_start

total_elapsed = time.perf_counter() - total_start

print(f"Training time: {train_elapsed:.2f} seconds")
print(f"Inference time: {inference_elapsed:.2f} seconds")
print(f"Total execution time: {total_elapsed:.2f} seconds")
