from collections import Counter

import time

import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
kr = "\n".join(line for line in open("data/kr.txt", encoding="utf-8").read().splitlines() if line.strip())
eng = "\n".join(line for line in open("data/eng.txt", encoding="utf-8").read().splitlines() if line.strip())
chn = "\n".join(line for line in open("data/chn.txt", encoding="utf-8").read().splitlines() if line.strip())


class BasicTokenizer:
  def __init__(self):
    self.vocab = {}
    self.reverse_vocab = {}

  def get_stats(self, ids):
    return Counter(zip(ids, ids[1:]))
    stats = {}
    for i in range(len(ids) - 1):
      a, b = ids[i], ids[i + 1]
      pair = (a, b)
      stats[pair] = stats.get(pair, 0) + 1
    return stats

  def merge(self, ids, pair, new_token):
    a, b = pair
    out = []
    i = 0
    while i < len(ids):
      if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
        out.append(new_token)
        i += 2
      else:
        out.append(ids[i])
        i += 1
    return out

  def train(self, ids, start_token=256, threshold=5):
    new_token = start_token
    ids = list(ids)

    while True:
      stats = self.get_stats(ids)
      if not stats:
        break

      (a, b), freq = stats.most_common(1)[0]
      if freq < threshold:
        break

      ids = self.merge(ids, (a, b), new_token)
      self.reverse_vocab[(a, b)] = new_token
      self.vocab[new_token] = (a, b)

      new_token += 1

    return ids

  def expand_bytes(self, token_id):
    if token_id < 256:
      return bytes([token_id])
    a, b = self.vocab[token_id]
    return self.expand_bytes(a) + self.expand_bytes(b)

  def decode(self, token_ids):
    data = b''.join(self.expand_bytes(t) for t in token_ids)
    return data.decode('utf-8', errors='replace')

  def encode(self, text):
    tokens = list(map(int, text.encode('utf-8')))
    while len(tokens) >= 2:
      stats = self.get_stats(tokens)
      pair = min(stats, key=lambda p: self.reverse_vocab.get(p, float('inf')))
      if pair not in self.reverse_vocab:
        break
      idx = self.reverse_vocab[pair]
      tokens = self.merge(tokens, pair, idx)
    return tokens