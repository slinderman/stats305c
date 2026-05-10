# Transformers

:::{admonition} Prerequisites
:class: note
This chapter builds on [Recurrent Neural Networks](04_04_rnns).
:::

RNNs are natural models for sequential data, but the $\cO(T)$ time complexity of evaluation and backpropagating gradients is a severe limitation. In modern machine learning, one of the deciding factors is how many training epochs you can perform for a fixed computational budget. To that end, architectures that can process an entire sequence in parallel are advantageous. Transformers are one such architecture.

Transformers underlie large language models (LLMs) like Open AI's ChatGPT and Google's Gemini. They are also widely used in computer vision and other domains of machine learning. This chapter walks through the basic building blocks of a transformer: self-attention, token-wise nonlinear transformations, layer norm, and positional encodings. We follow the presentation of @turner2023introduction with some modifications for consistency with these notes.

## Preliminaries

Let $\mbX^{(0)} \in \reals^{T \times D}$ denote our data matrix with row $\mbx_t^{(0)} \in \reals^D$ representing the $t$-th **token** in the sequence. For example, the tokens could be a vector embedding of a word, sub-word, or character. The embeddings may be fixed or learned as part of the model.

The output of the transformer will be another matrix of the same shape, $\mbX^{(M)} \in \reals^{T \times D}$. These output features can be used for downstream tasks like sentiment classification, machine translation, or autoregressive modeling.

The output results from a stack of transformer blocks,
$$
\mbX^{(m)} = \texttt{transformer-block}(\mbX^{(m-1)}).
$$
Each block consists of two stages: one that operates vertically, combining information across the sequence length; another that operates horizontally, combining information across feature dimensions.

![Transformer Block](../figures/14_transformers/transformer-block.png)

## Attention

The first stage combines information across sequence length using a mechanism called **attention**. Mathematically, attention is a weighted average,
$$
\mbY^{(m)} = \mbA^{(m)} \mbX^{(m-1)},
$$
where $\mbA^{(m)} \in \reals_+^{T \times T}$ is a row-stochastic attention matrix: $\sum_{s} A_{ts}^{(m)} = 1$ for all $t$. Intuitively, $A_{t,s}^{(m)}$ indicates how much output location $t$ attends to input location $s$.

When using transformers for autoregressive sequence modeling, we constrain the attention matrix to be **causal** by requiring $A_{t,s}^{(m)} = 0$ for all $s > t$ — i.e., the matrix is **lower triangular**.

![Self Attention](../figures/14_transformers/self-attention-1.png)

### Self-Attention

Where does the attention matrix come from? In a transformer, the attention weights are determined by the pairwise similarity of tokens in the sequence. The simplest instantiation would be
$$
A_{t,s} = \frac{\exp \{ \mbx_t^\top \mbx_s\}}{\sum_{s'=1}^T \exp\{\mbx_t^\top \mbx_{s'}\}}.
$$
(We drop the superscript ${}^{(m)}$ for clarity in this section.)

In practice, different feature dimensions convey different kinds of information. Transformers use separate linear projections for the **queries** and **keys**:
$$
A_{t,s} = \frac{\exp \{ (\mbU_q \mbx_t)^\top (\mbU_k \mbx_s) / \sqrt{K}\}}{\sum_{s'=1}^T \exp\{(\mbU_q \mbx_t)^\top (\mbU_k \mbx_{s'}) / \sqrt{K}\}},
$$
where $\mbU_q \mbx_t \in \reals^{K}$ are the **queries**, $\mbU_k \mbx_s \in \reals^{K}$ are the **keys**, and the $1/\sqrt{K}$ factor prevents the dot products from growing large in magnitude [@vaswani2017attention].

![Self Attention with Queries and Keys](../figures/14_transformers/self-attention-2.png)

:::{admonition} Causal attention
To enforce causality, we zero out the upper triangular entries of the attention matrix and renormalize:
$$
A_{t,s} = \frac{\exp \{ (\mbU_q \mbx_t)^\top (\mbU_k \mbx_s) / \sqrt{K}\}}{\sum_{s'=1}^{t} \exp\{(\mbU_q \mbx_t)^\top (\mbU_k \mbx_{s'}) / \sqrt{K}\}} \cdot \bbI[t \geq s].
$$
:::

:::{admonition} Comparison to RNNs
Rather than propagating information about past tokens via a hidden state, a transformer with causal attention can directly attend to any past token.
:::

:::{admonition} Connection to Convolutional Neural Networks (CNNs)
If the attention weights were only a function of the distance between tokens, $A_{t,s} = a_{t-s}$, then $\mbA$ would be a **Toeplitz matrix** and multiplication by it would be a discrete convolution. Attention is a generalization that allows input-dependent, time-varying filters.
:::

### The KV Cache

During autoregressive generation, the model produces one token at a time. At step $t$, all keys $\mbU_k \mbx_{s}$ and values $\mbU_v \mbx_{s}$ for $s < t$ were already computed at previous steps. Rather than recomputing them, an efficient implementation stores them in a **KV cache** and retrieves them when generating each new token. This reduces the per-step cost from $O(t D^2)$ to $O(D^2)$, but the cache grows linearly with context length — a key memory bottleneck at inference time.

### Multi-Headed Self-Attention

Just as a CNN uses a bank of filters in parallel, a transformer block uses $H$ **attention heads** in parallel. Let
$$
\mbY^{(m,h)} = \mbA^{(m,h)} \mbX^{(m-1)} \mbU_v^{(m,h)\top} \in \reals^{T \times K},
$$
where
$$
A_{t,s}^{(m,h)} =
\frac{\exp \{ (\mbU_q^{(m,h)} \mbx_t^{(m-1)})^\top (\mbU_k^{(m,h)} \mbx_s^{(m-1)}) / \sqrt{K}\}}{\sum_{s'=1}^T \exp\{(\mbU_q^{(m,h)} \mbx_t^{(m-1)})^\top (\mbU_k^{(m,h)} \mbx_{s'}^{(m-1)}) / \sqrt{K}\}}
$$
for $h = 1, \ldots, H$. The outputs are projected and summed:
$$
\mbY^{(m)} = \sum_{h=1}^H \mbY^{(m,h)} \mbU_o^{(m,h)\top} \triangleq \texttt{mhsa}(\mbX^{(m-1)}),
$$
where $\mbU_o^{(m,h)} \in \reals^{D \times K}$ maps each head's output back to the token dimension.

![Multi-Headed Self Attention](../figures/14_transformers/mhsa-2.png)

:::{admonition} Grouped Query Attention (GQA)
:class: note
Standard multi-head attention maintains $H$ independent sets of key and value projections, each of which must be cached during generation. **Grouped Query Attention** [@ainslie2023gqa] reduces this cost by sharing a single set of K/V heads across a group of $G$ query heads, shrinking the KV cache by a factor of $G$ with minimal quality loss. GQA is now standard in most open-weight LLMs (LLaMA 2/3, Mistral, Gemma).
:::

## The Residual Stream

The residual connections in a transformer — introduced as an optimization technique in @he2016deep — have a deeper architectural interpretation. Writing out the full forward pass,
$$
\mbx_t^{(M)} = \mbx_t^{(0)} + \sum_{m=1}^{M} \left[ \texttt{mhsa}^{(m)}(\mbX^{(m-1)})_t + \texttt{mlp}^{(m)}(\mby_t^{(m)}) \right],
$$
reveals that every component — every attention head and every MLP — adds its output directly onto a shared **residual stream** that begins as the token embedding and accumulates updates across all layers [@elhage2021mathematical].

This perspective has several consequences:

- **All components read from and write to the same space.** Attention heads and MLPs interact not through layer boundaries but through their shared updates to the stream. One head can write information that a later head in a different layer reads directly.
- **Each head is a low-rank read-write.** With the Q/K/V/O factorization, each attention head reads from the stream via $\mbU_q$ and $\mbU_k$, computes an update, and writes it back via $\mbU_o \mbU_v$ — a rank-$K$ operation on the $D$-dimensional stream.
- **Residuals are structural, not incidental.** The residual stream framing is the foundation of mechanistic interpretability: by analyzing which subspaces different heads and MLPs read from and write to, researchers have identified circuits that implement specific algorithms (induction, copying, retrieval) inside trained transformers.

## Token-wise Nonlinearity

After the multi-headed self-attention step, the transformer applies a token-wise nonlinear transformation to mix feature dimensions. This is done with a feedforward network applied identically at each position,
$$
\mbx_t^{(m)} = \texttt{mlp}(\mby_t^{(m)}).
$$

:::{admonition} Computational Complexity
:class: warning
The MLP typically has hidden dimension $4D$, so the computational complexity of this step is $\cO(TD^2)$. For transformers with very large feature dimensions, this is the dominant cost.
:::

:::{admonition} Gated MLPs (SwiGLU)
:class: note
Modern LLMs replace the standard two-layer MLP with a **gated** variant. The most common is SwiGLU [@shazeer2020glu]:
$$
\texttt{mlp}(\mbx) = \left(\mathrm{swish}(\mbW_1 \mbx) \odot \mbW_2 \mbx\right) \mbW_3,
$$
where $\mathrm{swish}(z) = z \cdot \sigma(z)$ and $\odot$ is elementwise multiplication. The gating branch $\mbW_2 \mbx$ acts as a content-dependent filter on the nonlinear branch. SwiGLU is now the default in LLaMA, PaLM, Gemma, and most state-of-the-art open-weight models.
:::

## Layer Norm

LayerNorm stabilizes training by z-scoring each token and applying a learned shift and scale:
$$
\texttt{layer-norm}(\mbx_t)
= \mbbeta + \mbgamma \odot \left( \frac{\mbx_t - \texttt{mean}(\mbx_t)}{\texttt{std}(\mbx_t)} \right),
$$
where $\mbbeta, \mbgamma \in \reals^D$ are learned parameters. LayerNorm is applied before each sub-layer (Pre-LN), which yields more stable training than the original Post-LN design:
$$
\begin{aligned}
\mbY^{(m)} &= \mbX^{(m-1)} + \texttt{mhsa}(\texttt{layer-norm}(\mbX^{(m-1)})) \\
\mbX^{(m)} &= \mbY^{(m)} + \texttt{mlp}(\texttt{layer-norm}(\mbY^{(m)})).
\end{aligned}
$$
This defines one $\texttt{transformer-block}$. A transformer stacks $M$ such blocks to produce a deep sequence-to-sequence model.

## Positional Encodings

Without explicit position information, a transformer treats its inputs as an **unordered set** of tokens — the architecture is permutation-equivariant (subject only to the causal mask). When the data have spatial or temporal structure, position must be injected explicitly.

### Absolute Positional Encodings

The original transformer adds a fixed position vector to each token embedding:
$$
\mbx_t^{(0)} = \mbc_t + \mbp_t,
$$
where $\mbc_t \in \reals^D$ is the content embedding and $\mbp_t \in \reals^D$ encodes the position using sinusoidal basis functions [@vaswani2017attention]. Learned absolute position embeddings are also common.

### Rotary Position Embeddings (RoPE)

Absolute encodings bake position into the token embedding before the attention computation, which limits the model's ability to generalize to sequence lengths unseen during training. **Rotary Position Embeddings (RoPE)** [@su2024roformer] instead encode position directly into the query–key dot product by rotating query and key vectors before the inner product.

Concretely, partition the $K$-dimensional query and key into $K/2$ pairs of consecutive dimensions. For pair $i$, define the $2 \times 2$ rotation matrix,
$$
\mbR_t^{(i)} = \begin{pmatrix} \cos(t\,\theta_i) & -\sin(t\,\theta_i) \\ \sin(t\,\theta_i) & \cos(t\,\theta_i) \end{pmatrix},
$$
with base frequencies $\theta_i = b^{-2i/K}$ for a large base $b$ (commonly $b = 10000$). Apply these rotations elementwise to the query and key before computing attention weights:
$$
A_{t,s} \propto \exp\!\left\{ (\mbR_t \mbq_t)^\top (\mbR_s \mbk_s) / \sqrt{K} \right\}.
$$
Because $(\mbR_t \mbq_t)^\top (\mbR_s \mbk_s) = \mbq_t^\top \mbR_t^\top \mbR_s \mbk_s = \mbq_t^\top \mbR_{t-s} \mbk_s$, the dot product depends only on the **relative position** $t - s$. This relative-position property makes RoPE extrapolate more gracefully to longer sequences than absolute encodings. RoPE is now the dominant positional encoding in open-weight LLMs (LLaMA, Mistral, Qwen, Gemma).

## Autoregressive Modeling

To use a transformer for autoregressive modeling, predictions are read from the final layer's representations. To predict the next token label $\ell_{t+1} \in \{1,\ldots,V\}$ given past tokens $\mbx_{1:t}^{(0)}$:
$$
\ell_{t+1} \sim \mathrm{Cat}(\mathrm{softmax}(\mbW \mbx_t^{(M)})),
$$
where $\mbW \in \reals^{V \times D}$. Like hidden states in an RNN, the final-layer representations $\mbx_t^{(M)}$ aggregate information from all tokens up to index $t$.

## Training

Standard practice is to use the AdamW optimizer with gradient clipping, a warmup-then-cosine learning rate schedule, and dropout. Treat these as hyperparameters; the optimal settings are model- and data-dependent.

## Conclusion

Transformers achieve parallelism over a full sequence by replacing the sequential hidden-state recursion with multi-headed self-attention: each token directly attends to all previous tokens, enabling gradient information to flow in a single step rather than through $T$ chained Jacobians. Viewing the architecture through the lens of the residual stream clarifies that attention heads and MLPs all operate on a shared representational space, with each component contributing a low-rank additive update. The quadratic $O(T^2)$ cost of softmax attention is a practical bottleneck for long sequences and has motivated substantial recent work on more efficient alternatives.

:::{admonition} Next Steps
:class: seealso
- [Deep SSMs and Linear Attention](04_06_linear_attention) — replaces softmax attention with linear-time approximations and connects transformers back to structured state space models
:::

:::{admonition} Recommended Reading
:class: reading
[@vaswani2017attention], "Attention is All You Need." The original transformer paper.
[@turner2023introduction], "An Introduction to Transformers." Accessible overview following a similar presentation to this chapter.
[@elhage2021mathematical], "A Mathematical Framework for Transformer Circuits." Introduces the residual stream perspective and the circuit-level analysis of transformer computation.
[@su2024roformer], "RoFormer: Enhanced Transformer with Rotary Position Embedding."
:::
