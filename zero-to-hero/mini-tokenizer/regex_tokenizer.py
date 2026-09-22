import torch
import regex
from base_tokenizer import Tokenizer
from tokenizer_utils import merge, get_stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
  BASE_VOCAB_SIZE = 256

  def __init__(self, pattern=None):
    super().__init__()
    self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
    self.compiled_pattern = regex.compile(self.pattern)

  # split the text into chunks using the GPT4 regex pattern
  def pre_tokenize(self, text):
    output = regex.findall(self.compiled_pattern, text)
    return output

  # train the tokenizer on the given text to build the vocabulary and BPE merge maps
  # The base vocab already contains 256 byte values; each learned merge creates a new token
  # by combining two existing tokens that co-occur most frequently in the training text.
  def train(self, text, vocab_size, verbose=False):
    # Require a target size larger than the base byte vocabulary. The difference tells us
    # how many merge operations to learn during training.
    assert vocab_size > self.BASE_VOCAB_SIZE
    num_merges = vocab_size - self.BASE_VOCAB_SIZE
    bpe_map = {}
    reversed_bpe_map = {}
    # Split text into GPT-4-compatible regex chunks first, then encode each chunk as a list
    # of byte ids. This keeps the algorithm aligned with how text is later tokenized.
    sub_texts = self.pre_tokenize(text)
    chunk_ids = [list(map(int, sub_t.encode('utf-8'))) for sub_t in subtexts]
    new_token = self.BASE_VOCAB_SIZE # 256 as starting point
    i = 0
    while i < num_merges:
      stats = {}
      for ids in chunk_ids:
        stats = get_stats(ids, stats)

      # If no pairs remain, training is done.
      if not stats:
        break

      # loop through the keys of stats object, and each key will be applied to stats.get to get value to compare.
      pair = max(stats, key=stats.get)

      # Apply the same merge rule to every chunk in parallel, so all training sequences are
      # updated consistently with the new token.
      chunk_ids = [merge(ids, pair, new_token) for ids in chunk_ids]

      # Record the merge mapping: a pair -> token id and token id -> pair.
      bpe_map[pair] = new_token
      reversed_bpe_map[new_token] = pair

      # Build the token string for the new merged symbol from its left and right child symbols.
      self.vocab[new_token] = self.vocab[pair[0]] + self.vocab[pair[1]]

      # Optional debug output showing which merge was learned and how often it appeared.
      if verbose:
        decoded = self.decode([new_token])
        print(f"merge {new_token - self.BASE_VOCAB_SIZE + 1}/{num_merges}: {pair} -> {new_token} (decoded: {decoded}) had {stats[pair]} occurrences")

      new_token += 1
      i += 1

    self.bpe_map = bpe_map
    self.reversed_bpe_map = reversed_bpe_map

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

  # ids: a list of token ids in integer form
  # self.vocab: int -> bytes
  # returns a string decoded from a list of token ids,
  # using the learned merges to expand each token into its original byte sequence
  def decode(self, ids):
    # data = b''.join(self.vocab[token_id] for id in ids)
    data = b''
    for id in ids:
      data += self.vocab[id]
    return data.decode('utf-8', errors='replace')

  def _encode_chunk(self, text_bytes):
    # return the token ids
    # let's begin. first, convert all bytes to integers in range 0..255
    ids = list(text_bytes)
    while len(ids) >= 2:
      # find the pair with the lowest merge index
      stats = get_stats(ids)
      pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
      # subtle: if there are no more merges available, the key will
      # result in an inf for every single pair, and the min will be
      # just the first pair in the list, arbitrarily
      # we can detect this terminating case by a membership check
      if pair not in self.bpe_map:
          break # nothing else can be merged anymore
      # otherwise let's merge the best pair (lowest merge index)
      idx = self.bpe_map[pair]
      ids = merge(ids, pair, idx)
    return ids

  def encode_ordinary(self, text):
    """Encoding that ignores any special tokens."""
    # split text into chunks of text by categories defined in regex pattern
    text_chunks = regex.findall(self.compiled_pattern, text)
    # all chunks of text are encoded separately, then results are joined
    ids = []
    for chunk in text_chunks:
      chunk_bytes = chunk.encode("utf-8")  # raw bytes
      chunk_ids = self._encode_chunk(chunk_bytes)
      ids.extend(chunk_ids)
    return ids

  # returns a list of token ids for the input text,
  # using the learned merges to combine byte sequences into tokens
  def encode(self, text):
    sub_texts = self.pre_tokenize(text)
    for sub_t in sub_texts:
      text_bytes = sub_t.encode('utf-8', errors='replace')   # raw bytes
      ids = list(map(int, text_bytes))
      while len(ids) > 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: self.bpe_map.get(p, float('inf')))
        if pair not in self.bpe_map:
          break
        idx = self.bpe_map[pair]
        ids = merge(ids, pair, idx)
    return ids
