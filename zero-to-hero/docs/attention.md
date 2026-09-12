Notes:

• Attention is a communication mechanism. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.

• There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.

• Each example across batch dimension is of course processed completely independently and never "talk" to each other

• "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)

• In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate. Here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.

• "Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be too and Softmax will stay diffuse and not saturate too much