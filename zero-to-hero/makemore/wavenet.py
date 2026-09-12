import torch
import nn


# -----------------------------------------------------------------------------------------------
# WaveNet-style residual + skip connections (see docs/wavenet_plan.md)
# -----------------------------------------------------------------------------------------------
class CausalDilatedConv1d:

  # in_ch/out_ch: channel (feature) widths, NOT the raw audio value range.
  # kernel_size: number of timesteps each conv step looks at (WaveNet uses 2), constant across layers.
  # dilation: spacing between the taps in time; grows per-layer (1, 2, 4, 8, ...) to expand receptive field.
  def __init__(self, in_ch, out_ch, kernel_size=2, dilation=1, bias=True):
    self.in_ch = in_ch
    self.kernel_size = kernel_size
    self.dilation = dilation
    self.pad = dilation * (kernel_size - 1) # causal left-pad so output T == input T
    # reuse Linear: concatenated dilated taps (kernel_size * in_ch) -> out_ch, like a 1x1 conv over stacked taps
    self.linear = nn.Linear(kernel_size * in_ch, out_ch, bias=bias)

  def __call__(self, x):
    B, T, C = x.shape # (B, T, in_ch)
    pad = torch.zeros(B, self.pad, C, dtype=x.dtype, device=x.device)
    # (B, T + pad, in_ch), left-padded => causal (no future leakage)
    x_padded = torch.cat([pad, x], dim=1)
    # gather the kernel_size dilated taps: tap k looks (kernel_size-1-k)*dilation steps into the past
    taps = [x_padded[:, k*self.dilation : k*self.dilation + T, :] for k in range(self.kernel_size)]
    stacked = torch.cat(taps, dim=2) # (B, T, kernel_size * in_ch)
    self.out = self.linear(stacked) # (B, T, out_ch)
    return self.out

  def parameters(self):
    return self.linear.parameters()

# -----------------------------------------------------------------------------------------------
class GatedActivation:

  # splits the last dim in half: first half -> tanh branch, second half -> sigmoid (gate) branch
  def __call__(self, x):
    a, b = x.chunk(2, dim=-1)
    self.out = torch.tanh(a) * torch.sigmoid(b)
    return self.out

  def parameters(self):
    return []

# -----------------------------------------------------------------------------------------------
class ResidualBlock:

  def __init__(self, in_ch, hidden_ch, skip_ch, dilation, kernel_size=2):
    # dilated conv outputs 2*hidden_ch channels so GatedActivation can split into tanh/sigmoid halves
    self.dilated_conv = CausalDilatedConv1d(in_ch, 2 * hidden_ch, kernel_size=kernel_size, dilation=dilation, bias=False)
    self.gate = GatedActivation()
    self.res_proj = nn.Linear(hidden_ch, in_ch, bias=False)   # 1x1: projects gated output back to residual stream width
    self.skip_proj = nn.Linear(hidden_ch, skip_ch, bias=False) # 1x1: projects gated output to this layer's skip contribution

  # returns (out_residual, skip) tuple: out_residual feeds the next block, skip is accumulated by WaveNetStack
  def __call__(self, x):
    conv_out = self.dilated_conv(x)
    gated = self.gate(conv_out)
    out_residual = x + self.res_proj(gated)
    skip = self.skip_proj(gated)
    self.out = (out_residual, skip)
    return self.out

  def parameters(self):
    return self.dilated_conv.parameters() + self.res_proj.parameters() + self.skip_proj.parameters()

# -----------------------------------------------------------------------------------------------
class WaveNetStack:

  # k: number of stacked ResidualBlocks (NOT kernel_size); dilations grow as 1, 2, 4, ..., 2**(k-1)
  def __init__(self, k, in_ch, hidden_ch, skip_ch, kernel_size=2):
    self.blocks = [
      ResidualBlock(in_ch, hidden_ch, skip_ch, dilation=2**i, kernel_size=kernel_size)
      for i in range(k)
    ]

  def __call__(self, x):
    skip_sum = None
    for block in self.blocks:
      x, skip = block(x) # (out_residual, skip)
      skip_sum = skip if skip_sum is None else skip_sum + skip
    self.out = skip_sum # summed skip connections, matches diagram's "Skip-connections" arrows
    return self.out

  def parameters(self):
    return [p for block in self.blocks for p in block.parameters()]

