# Demo: simulate 16-bit audio, mu-law quantize to 256 classes, feed through the
# WaveNet classes in nn.py, matching the T=16 / dilations (1,2,4,8) diagram scenario.
import torch
import torch.nn.functional as F

from wavenet import WaveNetStack
from nn import Embedding, ReLU, Linear, Sequential

torch.manual_seed(42)

# -----------------------------------------------------------------------------------------------
# 1) Simulate raw 16-bit PCM audio samples: signed ints in [-32768, 32767]
# -----------------------------------------------------------------------------------------------
B = 4          # batch size (number of independent audio clips)
T = 16         # timesteps -- matches the diagram's 16 input nodes (receptive field = 2**4 = 16)

raw_audio = torch.randint(-32768, 32768, (B, T), dtype=torch.float32)
print('raw_audio (16-bit ints):', raw_audio.shape, raw_audio[0])

# -----------------------------------------------------------------------------------------------
# 2) mu-law companding + quantization to 256 discrete classes (see WaveNet paper section 2.2)
#    f(x_t) = sign(x_t) * ln(1 + mu*|x_t|) / ln(1 + mu),  -1 < x_t < 1, mu = 255
# -----------------------------------------------------------------------------------------------
def mu_law_encode(x, mu=255):
  # normalize raw int16 samples into the continuous range (-1, 1)
  x_norm = x / 32768.0
  # apply mu-law companding -> still continuous, in (-1, 1), but dynamic range compressed
  magnitude = torch.log1p(mu * x_norm.abs()) / torch.log1p(torch.tensor(float(mu)))
  companded = torch.sign(x_norm) * magnitude
  # quantize the compressed (-1, 1) value into 256 discrete bins -> integer class index 0..255
  ix = ((companded + 1) / 2 * (mu)).round().long().clamp(0, mu)
  return ix

vocab_size = 256
IX = mu_law_encode(raw_audio, mu=vocab_size - 1) # (B, T) integer class indices in [0, 255]
print('quantized classes (0-255):', IX.shape, IX[0])

# -----------------------------------------------------------------------------------------------
# 3) Build the WaveNet model using the nn.py classes
#    k=4 residual blocks -> dilations 1, 2, 4, 8 -> receptive field = 2**4 = 16, matching diagram
# -----------------------------------------------------------------------------------------------
n_embd = 24     # embedding dim per quantized audio sample
n_hidden = 32   # hidden channels inside each residual block
skip_ch = 64    # skip-connection channel width
k = 4           # number of stacked residual blocks (dilations 1,2,4,8)

emb = Embedding(vocab_size, n_embd)
stack = WaveNetStack(k, in_ch=n_embd, hidden_ch=n_hidden, skip_ch=skip_ch)
head = Sequential([ReLU(), Linear(skip_ch, skip_ch), ReLU(), Linear(skip_ch, vocab_size)])

parameters = emb.parameters() + stack.parameters() + head.parameters()
for p in parameters:
  p.requires_grad = True
print('total parameters:', sum(p.nelement() for p in parameters))

# -----------------------------------------------------------------------------------------------
# 4) Forward pass: classes (0..255) -> Embedding -> WaveNetStack -> head -> logits over 256 values
# -----------------------------------------------------------------------------------------------
x = emb(IX)              # (B, T, n_embd)
skip = stack(x)           # (B, T, skip_ch)  -- summed skip connections across all k layers
logits = head(skip)       # (B, T, vocab_size) -- one 256-way prediction per timestep

print('embedding out:', x.shape)
print('stack skip-sum out:', skip.shape)
print('logits out:', logits.shape)

# turn logits into an actual probability distribution over the 256 mu-law classes
probs = F.softmax(logits, dim=-1)
print('probs (sum to 1 per timestep):', probs.shape, probs[0, -1].sum().item())

# -----------------------------------------------------------------------------------------------
# 5) sanity check: train the next-sample prediction loss and confirm gradients flow
#    (predict IX[t] from everything up to and including IX[t], shifted by one for a real task
#     you'd predict IX[t+1]; here we just verify the mechanics with a dummy shifted target)
# -----------------------------------------------------------------------------------------------
target = IX # dummy target: same shape (B, T), values in [0, 255]
loss = F.cross_entropy(logits.view(B * T, vocab_size), target.view(B * T))
loss.backward()
print('loss:', loss.item())
print('all params have grad:', all(p.grad is not None for p in parameters))
