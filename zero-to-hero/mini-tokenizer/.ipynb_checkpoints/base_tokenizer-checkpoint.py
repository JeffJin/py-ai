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

