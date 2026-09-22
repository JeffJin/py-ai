import torch
import regex
from base_tokenizer import Tokenizer
from tokenizer_utils import merge, get_stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
  BASE_VOCAB_SIZE = 256

  def __init__(self):
    super().__init__()

  # split the text into chunks using the GPT4 regex pattern
  def pre_tokenize(self, text):
    output = regex.findall(GPT4_SPLIT_PATTERN, text)
    return output

  # train the tokenizer on the given text to build the vocabulary and BPE merge maps
  # The base vocab already contains 256 byte values; each learned merge creates a new token
  # by combining two existing tokens that co-occur most frequently in the training text.
  def train(self, text, vocab_size, verbose=False):
    # Require a target size larger than the base byte vocabulary. The difference tells us
    # how many merge operations to learn during training.
    assert vocab_size > self.BASE_VOCAB_SIZE
    num_merges = vocab_size - self.BASE_VOCAB_SIZE
    new_token = self.BASE_VOCAB_SIZE

    # Split text into GPT-4-compatible regex chunks first, then encode each chunk as a list
    # of byte ids. This keeps the algorithm aligned with how text is later tokenized.
    sub_texts = self.pre_tokenize(text)
    bpe_map = {}

    # Each chunk keeps its own ids list, but statistics are aggregated across all chunks before
    # deciding which pair to merge next. This makes the merge choice reflect the entire corpus.
    chunks_ids = [list(map(int, sub_t.encode('utf-8'))) for sub_t in sub_texts]

    # Keep creating new merged tokens until we've reached the requested vocabulary size.
    while new_token < self.BASE_VOCAB_SIZE + num_merges:
      # Count co-occurrence frequencies of adjacent token pairs across every chunk.
      stats = {}
      for ids in chunks_ids:
        stats = get_stats(ids, stats)

      # If no pairs remain, training is done.
      if not stats:
        break

      # Choose the most frequent pair to merge. This greedily maximizes the next BPE rule.
      pair = max(stats, key=stats.get)

      # Apply the same merge rule to every chunk in parallel, so all training sequences are
      # updated consistently with the new token.
      chunks_ids = [merge(ids, pair, new_token) for ids in chunks_ids]

      # Record the merge mapping: a pair -> token id and token id -> pair.
      bpe_map[pair] = new_token

      # Build the token string for the new merged symbol from its left and right child symbols.
      self.vocab[new_token] = self.vocab[pair[0]] + self.vocab[pair[1]]

      # Optional debug output showing which merge was learned and how often it appeared.
      if verbose:
        decoded = self.decode([new_token])
        print(f"merge {new_token - self.BASE_VOCAB_SIZE + 1}/{num_merges}: {pair} -> {new_token} (decoded: {decoded}) had {stats[pair]} occurrences")

      new_token += 1

    # Save the learned merge rules on the tokenizer instance for subsequent encoding.
    self.bpe_map = bpe_map

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
  #       self.bpe_map[new_token] = (a, b)
  #
  #       new_token += 1
  #
  #   return ids

  # returns a string decoded from a list of token ids,
  # using the learned bpe_map to expand each token into its original byte sequence
  def decode(self, ids):
    data = b''.join(self.vocab[id] for id in ids)
    return data.decode('utf-8', errors='replace')

  # ids: a list of token ids in integer form
  # returns a list of token ids for the input text,
  # using the learned bpe_map to combine byte sequences into tokens
  def encode(self, text):
    text_bytes = text.encode("utf-8")  # raw bytes
    ids = list(text_bytes)  # list of integers in range 0..255
    while len(ids) >= 2:
      # find the pair with the lowest merge index
      stats = get_stats(ids)
      pair = min(stats, key=lambda p: self.bpe_map.get(p, float("inf")))
      # subtle: if there are no more bpe_map available, the key will
      # result in an inf for every single pair, and the min will be
      # just the first pair in the list, arbitrarily
      # we can detect this terminating case by a membership check
      if pair not in self.bpe_map:
        break  # nothing else can be merged anymore
      # otherwise let's merge the best pair (lowest merge index)
      idx = self.bpe_map[pair]
      ids = merge(ids, pair, idx)
    return ids
