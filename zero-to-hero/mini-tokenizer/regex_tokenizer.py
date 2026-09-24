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
    self.special_tokens = {}
    self.inverse_special_tokens = {}

  def register_special_tokens(self, special_tokens):
    # special_tokens is a dictionary of str -> int
    # example: {"<|endoftext|>": 100257}
    self.special_tokens = special_tokens
    self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

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
    merges = {}
    # Split text into GPT-4-compatible regex chunks first, then encode each chunk as a list
    # of byte ids. This keeps the algorithm aligned with how text is later tokenized.
    sub_texts = self.pre_tokenize(text)
    chunk_ids = [list(map(int, sub_t.encode('utf-8'))) for sub_t in sub_texts]
    new_token = self.BASE_VOCAB_SIZE # 256 as starting point
    while  new_token < self.BASE_VOCAB_SIZE + num_merges:
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
      merges[pair] = new_token
      # Build the token string for the new merged symbol from its left and right child symbols.
      self.vocab[new_token] = self.vocab[pair[0]] + self.vocab[pair[1]]
      # Optional debug output showing which merge was learned and how often it appeared.
      if verbose:
        decoded = self.decode([new_token])
        print(f"merge {new_token - self.BASE_VOCAB_SIZE + 1}/{num_merges}: {pair} -> {new_token} (decoded: {decoded}) had {stats[pair]} occurrences")
      new_token += 1

    self.merges = merges

  def train_threshold(self, text, threshold=5):
    merges = {}
    sub_texts = self.pre_tokenize(text)
    chunk_ids = [list(map(int, sub_t.encode('utf-8'))) for sub_t in sub_texts]
    new_token = self.BASE_VOCAB_SIZE  # 256 as starting point
    while True:
      stats = {}
      for ids in chunk_ids:
        stats = get_stats(ids, stats)
      # If no pairs remain, training is done.
      if not stats:
        break
      pair = max(stats, key=stats.get)
      freq = stats[pair]
      if freq < threshold:
        break

      # Apply the same merge rule to every chunk in parallel, so all training sequences are
      # updated consistently with the new token.
      chunk_ids = [merge(ids, pair, new_token) for ids in chunk_ids]
      # Record the merge mapping: a pair -> token id and token id -> pair.
      merges[pair] = new_token
      # Build the token string for the new merged symbol from its left and right child symbols.
      self.vocab[new_token] = self.vocab[pair[0]] + self.vocab[pair[1]]
      # Optional debug output showing which merge was learned and how often it appeared.
      new_token += 1

    self.merges = merges


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

  def encode_ordinary(self, text):
    """Encoding that ignores any special tokens."""
    # split text into chunks of text by categories defined in regex pattern
    text_chunks = regex.findall(self.compiled_pattern, text)
    # all chunks of text are encoded separately, then results are joined
    ids = []
    for chunk in text_chunks:
      chunk_ids = self._encode_chunk(chunk)
      ids.extend(chunk_ids)
    return ids

  # returns a list of token ids for the input text,
  # using the learned merges to combine byte sequences into tokens
  def encode(self, text, allowed_special="none_raise"):
    """
    Unlike encode_ordinary, this function handles special tokens.
    allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
    if none_raise, then an error is raised if any special token is encountered in text
    this is the default tiktoken behavior right now as well
    any other behavior is either annoying or a major footgun
    """
    # decode the user desire w.r.t. handling of special tokens
    special = None
    if allowed_special == "all":
      special = self.special_tokens
    elif allowed_special == "none":
      special = {}
    elif allowed_special == "none_raise":
      special = {}
      assert all(token not in text for token in self.special_tokens)
    elif isinstance(allowed_special, set):
      special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
    else:
      raise ValueError(f"allowed_special={allowed_special} not understood")
    if not special:
      # shortcut: if no special tokens, just use the ordinary encoding
      return self.encode_ordinary(text)
    # otherwise, we have to be careful with potential special tokens in text
    # we handle special tokens by splitting the text
    # based on the occurrence of any exact match with any of the special tokens
    # we can use re.split for this. note that surrounding the pattern with ()
    # makes it into a capturing group, so the special tokens will be included
    special_pattern = "(" + "|".join(regex.escape(k) for k in special) + ")"
    chunks = regex.split(special_pattern, text)
    # now all the special characters are separated from the rest of the text
    # all chunks of text are encoded separately, then results are joined
    ids = []
    for chunk in chunks:
      if chunk in special:
        ids.append(special[chunk])
      else:
        ids.extend(self._encode_chunk(chunk))
    return ids

  def _encode_chunk(self, text_chunk):
    """Encode a single chunk of text into token ids."""
    text_bytes = text_chunk.encode('utf-8')  # raw bytes
    ids = list(map(int, text_bytes))
    return self._merge_ids(ids)

  # shared BPE merge loop: repeatedly applies the learned merge with the
  # lowest (earliest-learned) rank until no known merges remain.
  # subclasses (e.g. GPT4Tokenizer) can reuse this after preprocessing ids
  # (such as applying a byte shuffle) before merging.
  def _merge_ids(self, ids):
    while len(ids) >= 2:
      stats = get_stats(ids)
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      # subtle: if there are no more merges available, the key will
      # result in an inf for every single pair, and the min will be
      # just the first pair in the list, arbitrarily
      # we can detect this terminating case by a membership check
      if pair not in self.merges:
        break
      idx = self.merges[pair]
      ids = merge(ids, pair, idx)
    return ids
