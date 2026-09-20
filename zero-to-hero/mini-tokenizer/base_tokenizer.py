import unicodedata


def get_stats(ids, stats=None):
  """
  Given a list of integers, return a dictionary of counts of consecutive pairs
  Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
  Optionally allows to update an existing dictionary of counts
  
  Better implementation:
  Counter(zip(ids[0:], ids[1:]))
  """
  if stats is None:
    stats = {}
    
  length = len(ids)
  if length < 2:
    return stats
    
  for i in range(len(ids) - 1):
    a = ids[i]
    b = ids[i + 1]
    pair = (a, b)
    stats[pair] = stats.get(pair, 0) + 1 

  return stats


def merge(ids, pair, idx):
  """
  In the list of integers (ids), replace all consecutive occurrences
  of pair with the new integer token idx
  Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
  """
  newids = []
  a, b = pair
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids


# https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
# http://www.unicode.org/reports/tr44/#GC_Values_Table
def replace_control_characters(s: str) -> str:
  # we don't want to print control characters
  # which distort the output (e.g. \n or much worse)
  # There are hundreds of control characters in unicode.
  # If you are sanitizing data from the web or some other source that might
  # contain non-ascii characters, you will need Python's unicodedata module.
  # The unicodedata.category(…) function returns the unicode category code
  # (e.g., control character, whitespace, letter, etc.) of any character.
  # For control characters, the category always starts with "C".
  chars = []
  for ch in s:
    if unicodedata.category(ch)[0] != "C":
      chars.append(ch)  # this character is ok
    else:
      chars.append(f"\\u{ord(ch):04x}")  # escape
  return "".join(chars)


def render_token(t: bytes) -> str:
  # pretty print a token, escaping control characters
  s = t.decode('utf-8', errors='replace')
  s = replace_control_characters(s)
  return s


class Tokenizer:
  """Base class for Tokenizers"""

  def __init__(self):
    # default: vocab size of 256 (all bytes), no merges, no patterns
    self.bpe_map = {}  # (int, int) -> int
    self.reversed_bpe_map = {}  # int -> (int, int)
    self.pattern = ""  # str
    self.special_tokens = {}  # str -> int, e.g. {'<|endoftext|>': 100257}
    self.vocab = self._build_vocab()  # int -> bytes

  def train(self, text, vocab_size, verbose=False):
    # Tokenizer can train a vocabulary of size vocab_size from text
    raise NotImplementedError

  def encode(self, text):
    # Tokenizer can encode a string into a list of integers
    raise NotImplementedError

  def decode(self, ids):
    # Tokenizer can decode a list of integers into a string
    raise NotImplementedError

  def _build_vocab(self):
    # vocab is simply and deterministically derived from merges
    vocab = {id: bytes([id]) for id in range(256)}
    for (a, b), idx in self.bpe_map.items():
      vocab[idx] = vocab[a] + vocab[b]   
    for st, idx in self.special_tokens.items():
      vocab[idx] = st.encode('utf-8')
    return vocab

  def save(self, file_prefix):
    """
    Saves two files: file_prefix.vocab and file_prefix.model
    This is inspired (but not equivalent to!) sentencepiece's model saving:
    - model file is the critical one, intended for load()
    - vocab file is just a pretty printed version for human inspection only
    """
    # write the model: to be used in load() later
    model_file = file_prefix + ".model"
    with open(model_file, 'w') as f:
      # write the version, pattern and merges, that's all that's needed
      f.write("mini_tokenizer v1\n")
      f.write(f"{self.pattern}\n")
      # write the special tokens, first the number of them, then each one
      f.write(f"{len(self.special_tokens)}\n")
      for special, idx in self.special_tokens.items():
        f.write(f"{special} {idx}\n")
      # the merges dict
      for idx1, idx2 in self.merges:
        f.write(f"{idx1} {idx2}\n")
    # write the vocab: for the human to look at
    vocab_file = file_prefix + ".vocab"
    inverted_merges = {idx: pair for pair, idx in self.merges.items()}
    with open(vocab_file, "w", encoding="utf-8") as f:
      for idx, token in self.vocab.items():
        # note: many tokens may be partial utf-8 sequences
        # and cannot be decoded into valid strings. Here we're using
        # errors='replace' to replace them with the replacement char �.
        # this also means that we couldn't possibly use .vocab in load()
        # because decoding in this way is a lossy operation!
        s = render_token(token)
        # find the children of this token, if any
        if idx in inverted_merges:
          # if this token has children, render it nicely as a merge
          idx0, idx1 = inverted_merges[idx]
          s0 = render_token(self.vocab[idx0])
          s1 = render_token(self.vocab[idx1])
          f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
        else:
          # otherwise this is leaf token, just print it
          # (this should just be the first 256 tokens, the bytes)
          f.write(f"[{s}] {idx}\n")

  def load(self, model_file):
    """Inverse of save() but only for the model file"""
    assert model_file.endswith(".model")
    # read the model file
    merges = {}
    special_tokens = {}
    idx = 256
    with open(model_file, 'r', encoding="utf-8") as f:
      # read the version
      version = f.readline().strip()
      assert version == "mini_tokenizer v1"
      # read the pattern
      self.pattern = f.readline().strip()
      # read the special tokens
      num_special = int(f.readline().strip())
      for _ in range(num_special):
        special, special_idx = f.readline().strip().split()
        special_tokens[special] = int(special_idx)
      # read the merges
      for line in f:
        idx1, idx2 = map(int, line.split())
        merges[(idx1, idx2)] = idx
        idx += 1
    self.merges = merges
    self.special_tokens = special_tokens
    self.vocab = self._build_vocab()
