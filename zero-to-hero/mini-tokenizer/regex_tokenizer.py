import torch
import regex
from tokenizer_base import Tokenizer, merge, get_stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
  BASE_VOCAB_SIZE = 256

  def __init__(self):
    super().__init__()

  def pre_tokenize(self, text):
    output = regex.findall(GPT4_SPLIT_PATTERN, text)
    return output

  def train(self, text, vocab_size, verbose=False):
    assert vocab_size > self.BASE_VOCAB_SIZE
    num_merges = vocab_size - self.BASE_VOCAB_SIZE
    new_token = self.BASE_VOCAB_SIZE
    sub_texts = self.pre_tokenize(text)
    merges = {}
    reversed_merges = {}
    # each chunk keeps its own ids list, but stats are aggregated across
    # ALL chunks before deciding which pair to merge next
    chunks_ids = [list(map(int, sub_t.encode('utf-8'))) for sub_t in sub_texts]
    while new_token < self.BASE_VOCAB_SIZE + num_merges:
      stats = {}
      for ids in chunks_ids:
        stats = get_stats(ids, stats)
      if not stats:
        break
      pair = max(stats, key=stats.get)
      chunks_ids = [merge(ids, pair, new_token) for ids in chunks_ids]
      merges[pair] = new_token
      reversed_merges[new_token] = pair
      self.vocab[new_token] = self.vocab[pair[0]] + self.vocab[pair[1]]
      # prints
      if verbose:
        decoded = self.decode([new_token])
        print(f"merge {new_token - self.BASE_VOCAB_SIZE + 1}/{num_merges}: {pair} -> {new_token} (decoded: {decoded}) had {stats[pair]} occurrences")
      new_token += 1
    self.merges = merges
    self.reversed_merges = reversed_merges

  # def train_threshold(self, text, threshold=5):
  #   new_token = self.BASE_VOCAB_SIZE
  #   sub_texts = self.pre_tokenize(text)
  #
  #   for sub_t in sub_texts:
  #     ids = list(map(int, sub_t.encode('utf-8')))
  #
  #     while True:
  #       stats = get_stats(ids)
  #       if not stats:
  #         break
  #
  #       (a, b), freq = stats.most_common(1)[0]
  #       if freq < threshold:
  #         break
  #
  #       ids = merge(ids, (a, b), new_token)
  #       self.reversed_merges[(a, b)] = new_token
  #       self.merges[new_token] = (a, b)
  #
  #       new_token += 1
  #
  #   return ids

  def expand_bytes(self, token_id):
    if token_id < self.BASE_VOCAB_SIZE:
      return bytes([token_id])
    a, b = self.vocab[token_id]
    return self.expand_bytes(a) + self.expand_bytes(b)

  def decode(self, ids):
    data = b''.join(self.expand_bytes(t) for t in ids)
    return data.decode('utf-8', errors='replace')

  def encode(self, text):
    out = []
    for sub_t in self.pre_tokenize(text):
      tokens = list(sub_t.encode("utf-8"))
      while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: self.reversed_merges.get(p, float('inf')))
        if pair not in self.reversed_merges:
          break
        idx = self.reversed_merges[pair]
        tokens = merge(tokens, pair, idx)
      out.extend(tokens)
    return out