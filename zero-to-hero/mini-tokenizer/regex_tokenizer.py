import torch
import regex
from base_tokenizer import Tokenizer, merge, get_stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
  BASE_VOCAB_SIZE = 256

  def __init__(self):
    super().__init__()

  # split the text into chunks using the GPT4 regex pattern
  def pre_tokenize(self, text):
    output = []
    return output

  # train the tokenizer on the given text to build the vocabulary and bpe merge maps
  def train(self, text, vocab_size, verbose=False):
    assert vocab_size > self.BASE_VOCAB_SIZE
    bpe_map = {}
    reversed_bpe_map = {}
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


  # bpe_map = {}  # (int, int) -> int
  # reversed_bpe_map = {}  # int -> (int, int)
  # returns the original byte sequence for a given token id, recursively expanding merged tokens
  def expand_bytes(self, token_id):
    if token_id < self.BASE_VOCAB_SIZE: # it means the token is in 0-255 range
      return bytes([token_id])
    else: 
      (a, b) = self.vocab[token_id]
      return self.expand_bytes(a) + self.expand_bytes(b)


  # ids: a list of token ids in integer form
  # returns a string decoded from a list of token ids, 
  # using the learned merges to expand each token into its original byte sequence
  def decode(self, ids):
    # data = b''.join(self.expand_bytes(id) for id in ids)
    data = b''
    for id in ids:
      data += self.expand_bytes(id)       
    return data.decode('utf-8', errors='replace')
    
  # returns a list of token ids for the input text,
  # using the learned merges to combine byte sequences into tokens
  def encode(self, text):
    out = []

    return out
