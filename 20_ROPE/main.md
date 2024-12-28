# ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING

- Jianlin Su, Yu Lu: Nov 2023

# RoPE
- Introduces RoPE(Rotary positional embeddings)
- Roformer is a transformer that uses RoPE 
- better than sinusodial PE
- used in modern LLMs like gemma, llama, GPT-Neo, GPT-J, GPT-NeoX, etc.
- better suited for large context lengths and long-sequence tasks.
- requires modifying the attention mechanism

# T
- Typicall PEs add the position information in context representation
- But, RoPE encodes the `absolute position with a rotation matrix` and incorporates the `explicit relative position dependency in self-attention formulation`.

