import torch
from base_tokenizer import Tokenizer
from tokenizer_utils import merge, get_stats

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

  def decode(self, ids):
    data = b''.join(self.vocab[t] for t in ids)
    return data.decode('utf-8', errors='replace')

  def encode(self, text):
    # given a string text, return the token ids
    text_bytes = text.encode("utf-8")  # raw bytes
    ids = list(text_bytes)  # list of integers in range 0..255
    while len(ids) >= 2:
      # find the pair with the lowest merge index
      stats = get_stats(ids)
      pair = min(stats, key=lambda p: self.bpe_map.get(p, float('inf')))
      if pair not in self.bpe_map:
        break
      idx = self.bpe_map[pair]
      ids = merge(ids, pair, idx)
    return ids