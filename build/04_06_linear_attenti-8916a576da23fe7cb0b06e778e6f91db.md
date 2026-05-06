# Deep SSMs and Linear Attention

:::{admonition} Prerequisites
:class: note
This chapter builds on [Recurrent Neural Networks](04_04_rnns), [Linear Dynamical Systems](04_02_lds), and [Transformers](04_05_transformers).
:::

Two tensions define modern sequence modeling. On the training side, recurrent models like RNNs process tokens one at a time, leaving GPU parallelism underutilized on long sequences. On the inference side, Transformers compute all pairwise token interactions, incurring $O(N^2)$ time and memory that becomes prohibitive for long contexts. Can we get the best of both worlds — **parallel training** and **efficient inference**?

This chapter shows that the answer is yes, for a broad family of models that can be written as **linear recurrences**. A linear recurrence is simultaneously a convolution (fast parallel training via FFT or parallel scan) and a fixed-size hidden-state update (fast $O(1)$-memory inference). Structured SSMs exploit this duality with carefully designed dynamics matrices; linear attention recovers it by approximating the softmax kernel.

## Deep State Space Models

### Linear Recurrent Neural Networks

A **linear RNN** evolves a hidden state $\mbh_t \in \reals^p$ via

$$
\mbh_t = \mbA \mbh_{t-1} + \mbB \mbx_t, \qquad \mby_t = \mbC \mbh_t + \mbd,
$$

with $\mbh_0 = \mathbf{0}$. Because linear operators compose, unrolling the recurrence gives

$$
\mby_t = \mbC \sum_{s=0}^{t-1} \mbA^s \mbB \mbx_{t-s} + \mbd = [\mbK \circledast \mbx]_t,
$$

where the **SSM kernel** is $\mbK = [\mbC\mbB \;\; \mbC\mbA\mbB \;\; \mbC\mbA^2\mbB \;\; \cdots]$.

This **recurrence–convolution duality** is the key computational property:

- **Recurrent form**: $O(Np)$ inference, $O(p)$ memory — ideal for generation.
- **Convolutional form**: evaluate the full kernel once and convolve via FFT, $O(N \log N)$ — ideal for training on long sequences.

When the dynamics $\mbA_t$ are **input-dependent** (time-varying), the fixed kernel no longer applies. Training then requires a **parallel scan** over the sequence — an $O(\log T)$-depth divide-and-conquer algorithm that exploits the associativity of affine function composition. See [Parallelizing Nonlinear RNNs](04_07_parallel_rnns) for a detailed treatment of the parallel scan and its extensions to nonlinear recurrences.

### S4: A Continuous-Time Deep SSM

**S4** [@gu2022efficiently] is a deep SSM that derives its discrete-time recurrence from a continuous-time dynamical system, giving a principled way to initialize $\mbA$ so that the model captures long-range dependencies. A **linear time-invariant (LTI) system** evolves a hidden state $h(t) \in \mathbb{R}^p$ via:

$$
h'(t) = A\, h(t) + B\, x(t), \qquad y(t) = C^\top h(t),
$$

Discretizing with step size $\Delta$ (zero-order hold) yields the recurrent form:

$$
h_t = \bar{A}\, h_{t-1} + \bar{B}\, x_t, \qquad y_t = C^\top h_t,
$$

where $\bar{A} = e^{\Delta A}$ and $\bar{B} = (\bar{A} - I) A^{-1} B$. Because $\bar{A}$ is fixed, the output is a convolution with the SSM kernel $\bar{K} = (C^\top \bar{B},\; C^\top \bar{A} \bar{B},\; C^\top \bar{A}^2 \bar{B}, \ldots)$:

$$
y_t = (\bar{K} * x)_t.
$$

The remaining challenge is computing $\bar{K}$ efficiently. Computing $e^{\Delta A}$ naively is $O(p^3)$. S4 addresses this by initializing $A$ using the **HiPPO** framework [@gu2020hippo], which constructs $A$ as a structured matrix designed to optimally memorize the input history via polynomial projections. The HiPPO-LegS matrix has entries:

$$
A_{nk} =
\begin{cases}
-(2n+1)^{1/2}(2k+1)^{1/2} & n > k \\
-(n+1) & n = k \\
0 & n < k
\end{cases}
$$

S4 exploits the fact that HiPPO-LegS is a **normal plus low-rank** (NPLR) matrix to compute the SSM kernel in $O(p + N \log N)$ time via the fast Fourier transform.

### S5: Diagonal SSMs with Parallel Scans

$S5$ [@smith2023simplified] simplifies S4 by diagonalizing the state matrix: $\Lambda = P^{-1} A P$ where $\Lambda$ is diagonal. With $\tilde{h}_t = P^{-1} h_t$, the recurrence decouples into $p$ independent scalar recurrences:

$$
\tilde{h}_{t,i} = \lambda_i \tilde{h}_{t-1,i} + \tilde{B}_i x_t, \qquad y_t = \mathrm{Re}(C^\top \tilde{h}_t).
$$

Each mode decays independently at rate $|\lambda_i|$; stability requires $|\lambda_i| < 1$. S5 computes all $N$ hidden states in parallel using an **associative parallel scan**, reducing training time from $O(Np)$ sequential steps to $O(p \log N)$ parallel steps.

### Mamba: Selective SSMs

A fundamental limitation of LTI SSMs is that the transition matrices $\bar{A}$ and $\bar{B}$ are **input-independent**: every token is processed identically regardless of content. This prevents the model from selectively retaining or discarding information based on context.

**Mamba** [@gu2023mamba] introduces a **selection mechanism** by making $B_t$, $C_t$, and the discretization step $\Delta_t$ functions of the input $x_t$:

$$
h_t = \bar{A}_t\, h_{t-1} + \bar{B}_t\, x_t, \qquad y_t = C_t^\top h_t,
$$

where $\bar{A}_t = e^{\Delta_t A}$, $\bar{B}_t = (\bar{A}_t - I) A^{-1} B_t$, and $\Delta_t, B_t, C_t$ are computed from $x_t$ via small linear projections.

Because parameters now vary with $t$, the convolutional view no longer applies — the system is **time-varying**, and $y_t$ depends on all of $x_1, \ldots, x_t$ in a non-linear way. Training requires a **selective scan**: a hardware-aware parallel algorithm that exploits the structure of the recurrence without materializing intermediate states in memory.

The selection mechanism gives Mamba attention-like content-dependent routing while preserving the $O(N p)$ cost of a recurrent model.

## Linear Attention

The link between deep SSMs and Transformers (see the [Transformers](04_05_transformers) chapter) may be closer than it appears. Both Mamba and softmax attention compute **content-dependent** outputs — Mamba via input-driven gates, attention via pairwise dot products. The key difference is architectural: SSMs maintain a fixed-size hidden state updated recurrently, while attention computes all pairwise interactions simultaneously, with no fixed-size bottleneck but at $O(N^2)$ cost. Linearizing the attention kernel bridges the two, recovering a recurrent model from attention. Given queries $Q \in \mathbb{R}^{N \times d}$, keys $K \in \mathbb{R}^{N \times d}$, and values $V \in \mathbb{R}^{N \times d}$, the output of **causal** (autoregressive) softmax attention at position $i$ is:

$$
y_i = \frac{\sum_{j=1}^{i} \exp\!\left(q_i^\top k_j / \sqrt{d}\right) v_j}{\sum_{j=1}^{i} \exp\!\left(q_i^\top k_j / \sqrt{d}\right)}.
$$

In matrix form (with the causal mask $M_{ij} = \mathbb{I}[j \leq i]$):

$$
Y = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + \log M\right) V,
$$

where $\log M$ sets future entries to $-\infty$. Computing this requires materializing the $N \times N$ attention matrix, costing $O(N^2 d)$ time and $O(N^2)$ memory.

@katharopoulos2020transformers observe that the softmax is a kernel function, $\exp(q^\top k / \sqrt{d}) = \kappa(q, k)$, and that replacing it with a **kernel with explicit feature maps**, $\kappa(q, k) = \phi(q)^\top \phi(k)$, unlocks a dramatic simplification. The causal attention output becomes:

$$
y_i = \frac{\phi(q_i)^\top \sum_{j \leq i} \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_{j \leq i} \phi(k_j)}.
$$

The key is that $\phi(q_i)$ acts on the **accumulated outer products**, which can be built up incrementally as a linear recurrence. Defining the hidden state and normalizer:

$$
S_t = S_{t-1} + \phi(k_t) v_t^\top \in \mathbb{R}^{r \times d}, \qquad z_t = z_{t-1} + \phi(k_t) \in \mathbb{R}^r,
$$

the output at each step is $y_t = S_t^\top \phi(q_t) / (z_t^\top \phi(q_t))$. This is a **linear RNN** with hidden state $S_t$ — $O(Nrd)$ time and $O(rd)$ memory, with no attention matrix. A simple feature map that keeps values positive is $\phi(x) = \mathrm{elu}(x) + 1$ elementwise, giving $r = d$.

## Test-Time Regression

Linear attention can be reread as solving a **regression problem online**. At each step, the model has observed a sequence of key-value pairs $(k_1, v_1), \ldots, (k_t, v_t)$, which form a **training set**, and must predict a value at a new query $q_t$, the **test point**. The hidden state $S_t$ is the current regression solution, updated incrementally as new pairs arrive.

Concretely, consider fitting a linear map $W : \reals^d \to \reals^d$ to the accumulated pairs via ridge regression:

$$
W_t = \arg\min_W \sum_{s \leq t} \|W k_s - v_s\|^2 + \lambda \|W\|_F^2.
$$

The closed-form solution is $W_t = \left(\sum_{s \leq t} v_s k_s^\top\right) \left(\sum_{s \leq t} k_s k_s^\top + \lambda I\right)^{-1}$, and the prediction at $q_t$ is $y_t = W_t q_t$ — **kernel ridge regression** with a linear kernel. The models above are all approximations to this solution:

- **Linear attention** computes $S_t q_t$ with $S_t = \sum_{s \leq t} \phi(k_s) v_s^\top$ — the **numerator** of the ridge regression estimator, omitting the Gram matrix inverse.
- **Softmax attention** corresponds to kernel regression with the softmax kernel.

The remaining question is how to approximate the full ridge regression solution cheaply and online.

### DeltaNet: Gradient Descent for Linear Regression

Rather than accumulating all key-value pairs, **DeltaNet** [@yang2024parallelizing] maintains a weight matrix $S_t$ and updates it by taking one step of gradient descent on the per-token least-squares loss $\ell_t(S) = \tfrac{1}{2}\|S k_t - v_t\|^2$. With step size $\beta_t$, the gradient step gives the **delta rule**:

$$
S_t = S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^\top = (I - \beta_t k_t k_t^\top)\, S_{t-1} + \beta_t v_t k_t^\top.
$$

This is a **rank-1 correction**: the old memory's prediction $S_{t-1} k_t$ is erased and the new target $v_t$ is written in, both weighted by $k_t$. Unlike linear attention ($S_t = S_{t-1} + \phi(k_t) v_t^\top$, no forgetting), DeltaNet selectively overwrites memory associated with $k_t$. Output is $y_t = S_t^\top q_t$.

With keys and queries $\ell_2$-normalized and $\beta_t$ a learned sigmoid gate, DeltaNet is a **gated linear RNN** with content-dependent forgetting — more expressive than linear attention, cheaper than softmax attention ($O(Nd^2)$ vs. $O(N^2 d)$). @yang2024parallelizing derive an efficient parallel training algorithm based on the **WY representation** of Householder products.

### Test-Time Training

**Test-time training (TTT)** [@yu2024learning] takes the regression perspective to its logical conclusion: the hidden state $W_t$ is the weights of a small **nonlinear** model $f_{W_t} : \reals^d \to \reals^d$ (e.g., a two-layer MLP), updated online by gradient descent on the reconstruction loss:

$$
\ell_t(W) = \tfrac{1}{2}\|f_W(k_t) - v_t\|^2.
$$

One gradient step gives $W_t = W_{t-1} - \eta \nabla_W \ell_t(W_{t-1})$, and output is $y_t = f_{W_t}(q_t)$. When $f_W$ is the linear map $f_W(x) = Wx$, this reduces exactly to DeltaNet with $\beta_t = \eta$ — so DeltaNet is TTT with the simplest possible hidden model. For nonlinear $f_W$, TTT can take multiple gradient steps on mini-batches of context tokens, creating a **meta-learned inner loop** that trades compute for expressivity of the hidden state.

## Conclusion

The table below organises the models in this chapter along two axes: whether the dynamics are **input-dependent** (selective), and whether the hidden state update rule involves **forgetting** (vs. simple accumulation).

| **Model** | **Hidden state update** | **Input-dependent?** | **Forgetting?** |
|---|---|---|---|
| Linear RNN / S4 [@gu2022efficiently] | $h_t = \bar{A}\, h_{t-1} + \bar{B}\, x_t$ | No | No (fixed decay) |
| S5 [@smith2023simplified] | $\tilde{h}_{t,i} = \lambda_i \tilde{h}_{t-1,i} + \tilde{B}_i x_t$ | No | No (diagonal decay) |
| Mamba [@gu2023mamba] | $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ | Yes | Yes (gated decay) |
| Linear attention [@katharopoulos2020transformers] | $S_t = S_{t-1} + \phi(k_t) v_t^\top$ | Yes | No ($A = I$) |
| DeltaNet [@yang2024parallelizing] | $S_t = (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t v_t k_t^\top$ | Yes | Yes (selective overwrite) |
| TTT (linear) [@yu2024learning] | $W_t = W_{t-1} - \eta (W_{t-1}k_t - v_t)k_t^\top$ | Yes | Yes (gradient descent) |

All achieve $O(N)$ inference complexity. The dominant pattern — a linear recurrence admitting parallel training via associative scans or convolutions — unifies classical signal processing, online learning, and modern sequence modeling.

:::{admonition} Next Steps
:class: seealso
- [Parallelizing Nonlinear RNNs](04_07_parallel_rnns) — extends the parallel scan idea to nonlinear recurrences via iterative linearization (Newton, quasi-Newton, Picard, Jacobi)
:::

:::{admonition} Recommended Reading
:class: reading
[@gu2022efficiently], "Efficiently Modeling Long Sequences with Structured State Spaces."
[@gu2023mamba], "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
[@yang2024parallelizing], "Parallelizing Linear Transformers with the Delta Rule over Sequence Length."
[@yu2024learning], "Test-Time Training on Graphs with Large Language Models."
:::
