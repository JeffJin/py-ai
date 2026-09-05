# WaveNet-style Residual + Skip Block Plan

## Key Architectural Difference

Your current `FlattenConsecutive` approach shrinks the time dimension at each stage (non-overlapping merges). This diagram's residual block requires the time dimension to **stay the same** at every layer (so `x + residual` and skip-sum are shape-compatible), with the receptive field growing only through *dilation*, not downsampling. So this needs new causal/dilated conv logic, not the block-merge trick.

## New Classes to Add to `nn.py`

1. **`CausalDilatedConv1d(in_ch, out_ch, kernel_size=2, dilation=1)`**
   - Manually simulate a causal dilated conv: left-pad the time axis with `dilation * (kernel_size - 1)` zeros, then gather/stack the `kernel_size` dilated taps per timestep and apply a `Linear`-style weight (reuse your `Linear` under the hood by reshaping).
   - Output shape: `(B, T, out_ch)` — same `T` as input.

2. **`GatedActivation`**
   - Takes conv output, splits (or uses two separate convs) into a `tanh` branch and a `sigmoid` branch, returns `tanh(a) * sigmoid(b)`.

3. **`ResidualBlock(in_ch, hidden_ch, dilation)`**
   - `dilated_conv = CausalDilatedConv1d(in_ch, 2 * hidden_ch, dilation=dilation)` → split → `GatedActivation`
   - `1x1 = Linear(hidden_ch, in_ch)` → add residual: `out_residual = x + 1x1(gated)`
   - `1x1_skip = Linear(hidden_ch, skip_ch)` → produces this layer's **skip contribution**
   - `__call__` returns `(out_residual, skip)` tuple (breaks the simple `Sequential` chaining pattern — needs a custom stack, not plain `Sequential`).

4. **`WaveNetStack(k, in_ch, hidden_ch, skip_ch)`**
   - Holds `k` `ResidualBlock`s with dilations `1, 2, 4, 8, ...`
   - Runs input through each block, accumulates `skip` outputs (sum them), passes `out_residual` to the next block.
   - Returns the **summed skip connections**.

5. **Output head** (plain `Sequential`, matches diagram's right side):
   - `ReLU` → `Linear(skip_ch, skip_ch)` (1×1) → `ReLU` → `Linear(skip_ch, vocab_size)` (1×1) → (softmax handled by `F.cross_entropy` during training, so we can omit an explicit softmax layer, same as your current models).

## Steps to Implement

1. Add `CausalDilatedConv1d` to `nn.py` (causal padding + dilated gather + matmul).
2. Add `GatedActivation`.
3. Add `ResidualBlock` (dilated conv → gate → residual 1×1 + skip 1×1).
4. Add `WaveNetStack` to wire `k` blocks together and sum skips (since `Sequential` can't handle the tuple `(residual, skip)` branching).
5. Build the model in the notebook: `Embedding` → `WaveNetStack(k=4, dilations 1,2,4,8)` → output head (`ReLU, Linear, ReLU, Linear`).
6. Adjust training loop: `parameters()` needs to collect from `WaveNetStack.parameters()` too (sum over all its blocks).
7. Verify shapes at each stage with a dummy batch before full training.
