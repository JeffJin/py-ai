import torch
from tokenizer_base import Tokenizer, merge, get_stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BasicTokenizer(Tokenizer):
  BASE_VOCAB_SIZE = 256

  def __init__(self):
    super().__init__()

  def train(self, text, vocab_size, verbose=False):
    assert vocab_size > self.BASE_VOCAB_SIZE
    num_merges = vocab_size - self.BASE_VOCAB_SIZE

    new_token = self.BASE_VOCAB_SIZE
    ids = list(map(int, text.encode('utf-8')))
    merges = {}
    reversed_merges = {}
    for i in range(num_merges):
      stats = get_stats(ids)
      if not stats:
        break
      pair = max(stats, key=stats.get)
      ids = merge(ids, pair, new_token)
      merges[pair] = new_token
      reversed_merges[new_token] = pair
      self.vocab[new_token] = self.vocab[pair[0]] + self.vocab[pair[1]]
      # prints
      if verbose:
        decoded = self.decode([new_token])
        print(f"merge {i + 1}/{num_merges}: {pair} -> {new_token} (decoded: {decoded}) had {stats[pair]} occurrences")
    new_token += 1
    self.merges = merges
    self.reversed_merges = reversed_merges

  def train_threshold(self, text, threshold=5):
    new_token = self.BASE_VOCAB_SIZE
    ids = list(map(int, text.encode('utf-8')))

    while True:
      stats = get_stats(ids)
      if not stats:
        break

      (a, b), freq = stats.most_common(1)[0]
      if freq < threshold:
        break

      ids = merge(ids, (a, b), new_token)
      self.reversed_merges[(a, b)] = new_token
      self.merges[new_token] = (a, b)

      new_token += 1

    return ids

  def expand_bytes(self, token_id):
    if token_id < self.BASE_VOCAB_SIZE:
      return bytes([token_id])
    a, b = self.vocab[token_id]
    return self.expand_bytes(a) + self.expand_bytes(b)

  def decode(self, ids):
    data = b''.join(self.expand_bytes(t) for t in ids)
    return data.decode('utf-8', errors='replace')

  def encode(self, text):
    tokens = list(map(int, text.encode('utf-8')))
    while len(tokens) >= 2:
      stats = get_stats(tokens)
      pair = min(stats, key=lambda p: self.reversed_merges.get(p, float('inf')))
      if pair not in self.reversed_merges:
        break
      idx = self.reversed_merges[pair]
      tokens = merge(tokens, pair, idx)
    return tokens