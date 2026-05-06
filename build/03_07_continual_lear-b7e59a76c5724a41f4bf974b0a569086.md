# Continual Learning

:::{admonition} Prerequisites
:class: note
This chapter builds on [Markov Chain Monte Carlo](03_01_mcmc) (Fisher information and the Laplace approximation), [Coordinate Ascent Variational Inference](03_04_cavi), and [Gradient-Based Variational Inference](03_05_advi). Familiarity with neural networks and gradient-based optimization is assumed.
:::

A central assumption of classical statistical learning is that training data are drawn i.i.d. from a fixed distribution. Real-world systems routinely violate this: a medical diagnosis model must incorporate new disease variants without forgetting old ones; a language model deployed in production should update on new domains without degrading on existing benchmarks; a robot should accumulate motor skills over its lifetime rather than relearning each task from scratch. **Continual learning** (also called *lifelong learning* or *sequential learning*) is the study of how to learn from a non-stationary stream of data or tasks while retaining previously acquired knowledge [@delange2021continual].

## Problem Formulation

### Task Sequences and the Forgetting Problem

A continual learning agent encounters a sequence of tasks $\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_T$. Each task $\mathcal{T}_t$ provides a dataset

$$
\mathcal{D}_t = \{(x_n^{(t)}, y_n^{(t)})\}_{n=1}^{N_t}
$$

drawn from a task-specific distribution $p_t(x, y)$. The agent has access to $\mathcal{D}_t$ only while training on task $t$; it cannot revisit $\mathcal{D}_1, \ldots, \mathcal{D}_{t-1}$ in full. The goal is to find parameters $\theta$ that perform well across all tasks simultaneously:

$$
\min_\theta \frac{1}{T} \sum_{t=1}^T \mathcal{L}_t(\theta), \qquad \mathcal{L}_t(\theta) = \mathbb{E}_{(x,y)\sim p_t}[\ell(\theta; x, y)].
$$

The difficulty is that naive sequential training — minimizing $\mathcal{L}_t$ at each step with gradient descent — causes **catastrophic forgetting** [@mccloskey1989catastrophic]: optimizing for the current task moves parameters into a region that performs poorly on previous tasks, and the loss of prior knowledge can be abrupt and nearly total.

### Scenarios

Three standard scenarios differ in what information is available at test time:

- **Task-incremental**: the task identity $t$ is provided at test time, so the model only needs to discriminate within each task.
- **Domain-incremental**: the input distribution shifts across tasks but the output space is shared; task identity is not provided.
- **Class-incremental**: new classes are introduced at each task and task identity is unknown at test time — the hardest setting, requiring the model to discriminate across all classes seen so far.

### Evaluation Metrics

After training on all $T$ tasks, let $a_{t', t}$ denote the accuracy on task $t$ immediately after training on task $t'$. Key metrics are:

- **Average accuracy**: $\bar{a} = \frac{1}{T} \sum_{t=1}^T a_{T,t}$ — overall performance after all tasks.
- **Backward transfer (forgetting)**: $\mathrm{BWT} = \frac{1}{T-1}\sum_{t=1}^{T-1}(a_{T,t} - a_{t,t})$ — how much performance on earlier tasks degrades after further training. Negative BWT indicates forgetting; positive BWT indicates that later learning improves earlier tasks.
- **Forward transfer**: $\mathrm{FWT} = \frac{1}{T-1}\sum_{t=2}^{T}(a_{t-1,t} - b_t)$ — how much prior learning helps on new tasks, where $b_t$ is a random-initialization baseline for task $t$.

## Catastrophic Forgetting

### Why Gradient Descent Forgets

Let $\theta^*_{t-1}$ minimize the loss on all tasks seen so far. When we minimize $\mathcal{L}_t(\theta)$ starting from $\theta^*_{t-1}$, the gradient $\nabla_\theta \mathcal{L}_t(\theta^*_{t-1})$ points in a direction that reduces the current task loss. Nothing in this gradient respects the curvature of the previous losses $\mathcal{L}_1, \ldots, \mathcal{L}_{t-1}$: the step may move parameters to a region of high loss for earlier tasks.

The severity of forgetting depends on **task similarity** and **parameter overlap**. If tasks use largely disjoint subsets of parameters, interference is small. If the same parameters are critical for multiple tasks, any update for one task can disrupt the others.

### The Stability–Plasticity Dilemma

Continual learning requires balancing two competing pressures:

- **Plasticity**: the ability to rapidly acquire new knowledge and adapt to new tasks.
- **Stability**: the ability to retain previously acquired knowledge against interference.

A model that is maximally stable (e.g., frozen weights) cannot learn new tasks. A model that is maximally plastic (e.g., standard SGD) forgets immediately. Effective continual learning algorithms navigate the trade-off between these two extremes.

## Regularization-Based Methods

Regularization methods augment the loss for each new task with a penalty that discourages large changes to parameters that were important for past tasks.

### Elastic Weight Consolidation

**Elastic weight consolidation (EWC)** [@kirkpatrick2017overcoming] approximates the posterior over parameters after task $t-1$ as a Gaussian centered on the MAP estimate $\theta^*_{t-1}$:

$$
p(\theta \mid \mathcal{D}_{1:t-1}) \approx \mathcal{N}(\theta;\, \theta^*_{t-1},\, F_{t-1}^{-1}),
$$

where $F_{t-1}$ is the **Fisher information matrix**, approximated by its diagonal:

$$
F_{ii} = \mathbb{E}_{p(x \mid \theta^*_{t-1})}\!\left[\left(\frac{\partial \log p(y \mid x, \theta)}{\partial \theta_i}\Bigg|_{\theta = \theta^*_{t-1}}\right)^{\!2}\right] \approx \frac{1}{N}\sum_{n=1}^N \left(\frac{\partial \log p(y_n \mid x_n, \theta^*_{t-1})}{\partial \theta_i}\right)^{\!2}.
$$

The diagonal Fisher $F_{ii}$ measures how much the log-likelihood changes when $\theta_i$ moves away from $\theta^*_{t-1}$: large $F_{ii}$ means $\theta_i$ is important for task $t-1$ and should be protected.

Using the previous posterior as a prior for the new task and taking the MAP gives the EWC loss:

$$
\mathcal{L}_\mathrm{EWC}(\theta) = \mathcal{L}_t(\theta) + \frac{\lambda}{2} \sum_i F_{ii}\,(\theta_i - \theta^*_{t-1,i})^2.
$$

This is a **weighted $\ell_2$ proximal penalty**: parameters with high Fisher weight are kept close to their previous values, while unimportant parameters are free to change.

For a sequence of tasks, the Fisher from all previous tasks accumulates. In the **online EWC** variant [@schwarz2018progress], a single running estimate of the Fisher is maintained rather than storing one Fisher matrix per task.

### Synaptic Intelligence

**Synaptic intelligence (SI)** [@zenke2017continual] estimates parameter importance **online** during training rather than post-hoc. It tracks the cumulative contribution of each parameter to the decrease in the loss along the optimization trajectory. For a parameter $\theta_i$ moving from $\theta_{i,0}$ to $\theta_{i,T_t}$ during training on task $t$, the importance is:

$$
\Omega_i^t = \frac{\sum_{k} \delta\theta_{i,k}\, (-\partial_{\theta_i} \mathcal{L}_t)_k}{\left(\Delta\theta_{i}^t\right)^2 + \xi},
$$

where the sum is over optimizer steps $k$, $\delta\theta_{i,k}$ is the parameter update at step $k$, and $\xi$ is a small damping constant. The numerator is the inner product of the parameter path with the negative gradient — large when the parameter moved in directions that actually reduced the loss. The denominator normalizes by the total displacement $\Delta\theta_i^t = \theta_{i,T_t} - \theta_{i,0}$.

Accumulated importance across tasks, $\Omega_i = \sum_{t' < t} \Omega_i^{t'}$, defines the regularization:

$$
\mathcal{L}_\mathrm{SI}(\theta) = \mathcal{L}_t(\theta) + c \sum_i \Omega_i\,(\theta_i - \theta^*_{t-1,i})^2.
$$

SI requires no additional forward passes or Hessian computations, making it computationally lighter than EWC.

## Bayesian Continual Learning

The Bayesian perspective gives a principled foundation for regularization-based continual learning and connects it to the sequential inference algorithms we have studied throughout this course.

### Sequential Bayes

If tasks arrive sequentially and the parameters are shared, the exact Bayesian update after observing $\mathcal{D}_t$ is:

$$
p(\theta \mid \mathcal{D}_{1:t}) = \frac{p(\mathcal{D}_t \mid \theta)\, p(\theta \mid \mathcal{D}_{1:t-1})}{p(\mathcal{D}_t \mid \mathcal{D}_{1:t-1})}.
$$

This is **sequential Bayes**: the posterior after task $t-1$ becomes the prior for task $t$. If we could maintain the exact posterior, there would be no forgetting — all past information is encoded in the current prior. The challenge is computational: the exact posterior is generally intractable and grows in complexity with each new task.

### Bayesian Online Linear Regression

Many continual learning problems reduce, in their simplest form, to online estimation of a shared parameter vector from a streaming sequence of observations. When the model is linear and the noise is Gaussian, sequential Bayes is tractable and exactly equivalent to **Kalman filtering** — the same algorithm used for state-space models in [Linear Dynamical Systems](04_02_lds).

**Setup.** Suppose we observe a stream of regression pairs $(x_1, y_1), (x_2, y_2), \ldots$ where $x_t \in \mathbb{R}^D$ and $y_t \in \mathbb{R}$, generated by a linear model:

$$
y_t = \theta^\top x_t + \epsilon_t, \qquad \epsilon_t \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2).
$$

We place a Gaussian prior on the shared parameter $\theta \in \mathbb{R}^D$, and the goal is to maintain the posterior $p(\theta \mid y_{1:t})$ after each new pair arrives, without storing the full data history.

**Equivalence with the Kalman filter.** Treating $\theta$ as the hidden state of a linear dynamical system reduces this problem exactly to Kalman filtering:

| LDS quantity | Regression interpretation |
|---|---|
| State $z_t$ | Parameter $\theta$ (shared, constant) |
| Dynamics $\mathbf{A} = \mathbf{I}$, $\mathbf{Q} = \mathbf{0}$ | Parameters do not change across observations |
| Emission matrix $C_t = x_t^\top$ | Time-varying: the feature vector at step $t$ |
| Emission noise $R = \sigma^2$ | Observation noise variance |

The dynamics are static (identity transition, zero process noise), so the Kalman predict step is trivial: the predicted distribution equals the filtered distribution from the previous step. Only the update step is active. Starting from $p(\theta \mid y_{1:t-1}) = \mathcal{N}(\mu_{t-1}, \Sigma_{t-1})$, conditioning on $y_t$ gives:

$$
\delta_t = y_t - x_t^\top \mu_{t-1}, \qquad
S_t = x_t^\top \Sigma_{t-1} x_t + \sigma^2, \qquad
K_t = \frac{\Sigma_{t-1}\, x_t}{S_t}
$$

$$
\mu_t = \mu_{t-1} + K_t\,\delta_t, \qquad \Sigma_t = (\mathbf{I} - K_t x_t^\top)\,\Sigma_{t-1}.
$$

The innovation $\delta_t$ is the residual from the current prediction, $S_t$ is its variance (a scalar because $y_t$ is scalar), and $K_t \in \mathbb{R}^D$ is the Kalman gain. In the **information form**, defining the precision $\Lambda_t = \Sigma_t^{-1}$ and information vector $\eta_t = \Lambda_t \mu_t$, the update is simply additive (by the Sherman–Morrison identity):

$$
\Lambda_t = \Lambda_{t-1} + \frac{1}{\sigma^2}\, x_t x_t^\top, \qquad
\eta_t = \eta_{t-1} + \frac{y_t}{\sigma^2}\, x_t.
$$

After $t$ steps, $\Lambda_t = \Lambda_0 + \frac{1}{\sigma^2} X_t^\top X_t$, which is exactly the posterior precision from batch Bayesian linear regression on all $t$ observations. Sequential and batch inference are exactly equivalent for this model — no approximation is introduced by processing data one point at a time.

**Drifting parameters.** Static parameters are often unrealistic: the true data-generating process may shift over time. A natural generalization introduces a **random-walk prior** on $\theta$, i.e., $\mathbf{A} = \mathbf{I}$ with $\mathbf{Q} \succ 0$. The Kalman predict step then inflates the covariance before each update:

$$
\Sigma_{t|t-1} = \Sigma_{t-1|t-1} + \mathbf{Q}.
$$

This is a form of controlled forgetting: the additional uncertainty injected at each step means older observations have less influence on the current estimate. Setting $\mathbf{Q} = q^2 \mathbf{I}$ corresponds to exponential discounting of past data, with an effective memory horizon of $\sim \sigma^2 / q^2$ observations.

### Variational Continual Learning

**Variational continual learning (VCL)** [@nguyen2018variational] maintains a variational approximation $q_t(\theta) \approx p(\theta \mid \mathcal{D}_{1:t})$ at each step. The variational objective at task $t$ is:

$$
\mathcal{F}_t(q) = D_\mathrm{KL}(q(\theta) \,\|\, q_{t-1}(\theta)) - \mathbb{E}_{q(\theta)}[\log p(\mathcal{D}_t \mid \theta)],
$$

where $q_{t-1}$ is the approximate posterior from the previous task, used as the **new prior**. Using a Gaussian mean-field approximation $q_t(\theta) = \mathcal{N}(\theta;\, \mu_t, \mathrm{diag}(\sigma_t^2))$, the parameters $(\mu_t, \sigma_t)$ are updated by minimizing $\mathcal{F}_t$ via the reparameterization gradient.

The KL term in $\mathcal{F}_t$ plays the role of the EWC penalty: it discourages $q_t$ from drifting far from $q_{t-1}$, weighted by the inverse variance of the previous posterior. When $q_{t-1} = \mathcal{N}(\theta^*_{t-1}, F_{t-1}^{-1})$ (the Laplace approximation), the KL penalty reduces exactly to:

$$
D_\mathrm{KL}(q_t \,\|\, q_{t-1}) = \frac{1}{2}\sum_i F_{t-1,ii}\,(\mu_{t,i} - \theta^*_{t-1,i})^2 + \text{terms in } \sigma_t.
$$

This reveals that **EWC is the MAP limit of VCL** under a Laplace approximation to the posterior, with the Fisher information playing the role of the prior precision.

VCL also integrates naturally with **coreset methods**: a small set of representative data points from each past task is stored and used to refine the variational posterior, improving on the Laplace approximation when the posterior is non-Gaussian.

## Replay-Based Methods

Replay methods counteract forgetting by periodically revisiting data from previous tasks, either stored explicitly or regenerated by a model.

### Experience Replay

The simplest approach maintains a fixed-size **memory buffer** $\mathcal{M}$ containing a small number of exemplars from each past task, selected by random subsampling or a principled strategy (e.g., reservoir sampling to maintain a uniform random sample from the full stream). The loss at each step interleaves current-task data with replayed data:

$$
\mathcal{L}(\theta) = \mathcal{L}_t(\theta) + \beta \,\mathcal{L}_\mathcal{M}(\theta), \qquad \mathcal{L}_\mathcal{M}(\theta) = \frac{1}{|\mathcal{M}|}\sum_{(x,y)\in\mathcal{M}} \ell(\theta; x, y).
$$

Experience replay is conceptually simple and empirically effective, but its memory cost grows with the number of tasks (or the per-task exemplar budget shrinks).

### Dark Experience Replay

**Dark Experience Replay (DER)** [@buzzega2020dark] improves on standard replay by storing the model's **soft predictions** (logits) $z_n = f_\theta(x_n)$ at the time each exemplar is added to the buffer, in addition to the input $x_n$. The replay loss then includes a **knowledge distillation** term:

$$
\mathcal{L}_\mathrm{DER}(\theta) = \mathcal{L}_t(\theta) + \alpha \underbrace{\frac{1}{|\mathcal{M}|}\sum_{(x_n, z_n)\in\mathcal{M}} \|f_\theta(x_n) - z_n\|^2}_{\text{distillation from stored logits}}.
$$

Matching the current model's predictions to the stored logits preserves not just the final decision but the full predictive distribution — a richer signal that slows forgetting more effectively than label-only replay.

### Gradient Episodic Memory

**Gradient Episodic Memory (GEM)** [@lopez2017gradient] uses the memory buffer $\mathcal{M}$ not to mix replay into the training loss, but to constrain the gradient update direction. GEM solves a quadratic program at each step to find the update $\tilde{g}$ that:

1. does not increase the loss on any past task's exemplars (the **non-interference** constraint), and
2. is as close as possible to the current task's gradient $g_t$.

Formally, let $g_{t'} = \nabla_\theta \mathcal{L}_{\mathcal{M}_{t'}}(\theta)$ be the gradient of the memory loss for past task $t'$. GEM solves:

$$
\tilde{g} = \arg\min_{v} \tfrac{1}{2}\|v - g_t\|^2 \quad \text{subject to} \quad \langle v, g_{t'} \rangle \geq 0 \quad \forall t' < t.
$$

The constraint $\langle \tilde{g}, g_{t'} \rangle \geq 0$ ensures the update does not increase past task losses to first order. When all constraints are satisfied by $g_t$ itself, no projection is needed; otherwise, $g_t$ is projected onto the intersection of the constraint half-spaces.

## Modern Approaches: Parameter-Efficient Continual Learning

The dominant paradigm for continual learning with large pretrained models has shifted toward **parameter-efficient adaptation**: rather than modifying all model weights, these methods encode task-specific knowledge in a small number of additional parameters, leaving the pretrained base frozen. This eliminates forgetting by construction — different tasks are stored in separate, non-interfering modules — and scales naturally to long task sequences without growing the base model.

### Low-Rank Adaptation (LoRA)

**LoRA** [@hu2022lora] constrains task-specific weight updates to a low-dimensional subspace. For a weight matrix $W_0 \in \mathbb{R}^{m \times n}$, the adapted model uses:

$$
W = W_0 + \Delta W = W_0 + BA,
$$

where $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, and $r \ll \min(m, n)$. The base weights $W_0$ are frozen; only $A$ and $B$ are trained, requiring $O(r(m+n))$ parameters per task instead of $O(mn)$. For continual learning, each task $t$ receives its own adapter pair $\{A^{(t)}, B^{(t)}\}$: switching tasks requires only swapping the adapter, and the base model is never modified. Zero forgetting follows by construction.

A practical question is whether adapters from different tasks can be **composed** at inference time — for instance, by taking a weighted combination $\sum_t \alpha_t \Delta W^{(t)}$ when the task identity is unknown. This connects to the broader challenge of merging independently trained models, an active area of research sometimes called *model merging* or *task arithmetic*.

### Prefix Tuning and KV Cache Adaptation

**Prefix tuning** [@li2021prefix] prepends a set of $L$ learned "virtual tokens" to the key-value cache of each attention layer. For a transformer with keys $K \in \mathbb{R}^{T \times d}$ and values $V \in \mathbb{R}^{T \times d}$ from the input tokens, the prefix-augmented attention computes:

$$
\mathrm{Attn}\!\left(Q,\, [\bar{K};\, K],\, [\bar{V};\, V]\right),
$$

where $\bar{K}, \bar{V} \in \mathbb{R}^{L \times d}$ are the only parameters trained and $d$ is the head dimension. The base model is fully frozen. Prefix tuning can be understood as **soft prompting at every layer**: the virtual tokens steer the model's internal representations without touching its weights. For continual learning, separate prefixes per task require only $O(L \cdot d \cdot n_\mathrm{layers})$ storage, and switching tasks is a memory swap.

### Cartridges

**Cartridges** [@eyuboglu2025cartridges] extend prefix tuning from a learned prompt to a **compressed knowledge store**. Rather than learning a prefix for a task specification, a cartridge encodes the content of an entire document corpus into a compact KV cache $(\bar{K}, \bar{V}) \in \mathbb{R}^{L \times d} \times \mathbb{R}^{L \times d}$ with $L \ll T$, via a process called *self-study*:

1. Generate synthetic reference queries $Q_\mathrm{ref} \in \mathbb{R}^{q \times d}$ about the target documents using the base model.
2. Run the model with the full $T$-token documents in context to obtain ground-truth attention outputs $Y \in \mathbb{R}^{q \times d}$.
3. Train the cartridge $(\bar{K}, \bar{V})$ to minimize the discrepancy between $Y$ and the attention outputs produced by $(\bar{K}, \bar{V})$ alone.

The result is a compact, **composable** cache: multiple cartridges can be concatenated in the KV cache at inference time without retraining, enabling modular assembly of knowledge from multiple sources. Empirically, cartridges storing a 484k-token corpus require 38× less memory than the equivalent in-context representation and achieve 26× higher inference throughput.

For continual learning, each task or knowledge domain gets its own cartridge trained offline. Adding a new task means training one new cartridge; all previous cartridges are untouched. Forgetting is structurally impossible.

### Attention Matching

Training a cartridge via self-study requires backpropagation through the model for each new task — taking hours for large contexts. **Attention matching** [@zweiger2026attentionmatching] finds the compact cache $(\bar{K}, \bar{V})$ in **closed form**, in seconds, by decomposing the problem into tractable least-squares subproblems.

Using the same notation as above, the goal is to find $(\bar{K}, \bar{V}) \in \mathbb{R}^{L \times d} \times \mathbb{R}^{L \times d}$ that reproduces the attention outputs $Y \in \mathbb{R}^{q \times d}$ of the full $T$-token cache on the $q$ reference queries $Q_\mathrm{ref}$.

**Value fitting.** Given compacted keys $\bar{K}$, let $X \in \mathbb{R}^{q \times L}$ be the matrix of normalized attention weights from each reference query to each compacted key. The compacted values $\bar{V}$ that best reproduce $Y$ solve:

$$
\min_{\bar{V}} \|X \bar{V} - Y\|_F^2, \qquad \bar{V}^* = (X^\top X)^{-1} X^\top Y.
$$

This is **ordinary least squares**: the closed-form solution is the same as Bayesian linear regression with an uninformative prior (cf. the information-form update in the Bayesian online regression section above, with $\Lambda_0 = 0$). Each column of $\bar{V}$ is fit independently, regressing the $d$-dimensional attention output onto the $L$ normalized attention weights from the compacted keys.

**Bias fitting.** To also match the total *attention mass* — ensuring $\bar{K}$ attracts the correct aggregate attention weight — the method adds per-token scalar log-biases $\beta \in \mathbb{R}^L$ to the attention scores and solves a nonnegative least-squares problem for $\beta$.

**Key selection.** Given fixed $\bar{V}$ and $\beta$, the $L$ key positions $\bar{K}$ are chosen greedily by orthogonal matching pursuit or by highest aggregated attention weight across $Q_\mathrm{ref}$.

The full three-stage pipeline (key selection → bias fitting → value fitting) runs in seconds and achieves 50× compression with minimal quality loss on long-context benchmarks, matching gradient-based methods that take hours. The closed-form structure mirrors the recursive Bayesian updates studied throughout this chapter: both reduce to sequential linear-regression problems that summarize a stream of information into a compact sufficient statistic.

## Other Methods

Earlier work proposed two additional families that avoid forgetting by design rather than by parameter isolation. Though less prominent in the era of large pretrained models, they introduced ideas that continue to influence modern methods.

**Architecture-based methods** allocate dedicated parameters to each task. *Progressive neural networks* [@rusu2016progressive] add a new column of weights per task, with lateral connections to all previous frozen columns — enabling positive forward transfer at the cost of linear model growth. *PackNet* avoids growth by pruning and reassigning freed weights to future tasks, assigning each task a disjoint binary mask $m^{(t)}$ over the shared parameter vector.

**Gradient projection methods** constrain updates to subspaces that do not interfere with past tasks. *Orthogonal gradient descent (OGD)* [@farajtabar2020orthogonal] projects the current gradient onto the orthogonal complement of the gradients at previous task optima:

$$
\tilde{g}_t = g_t - \sum_{t'<t} \frac{g_t^\top g^{(t')}}{\|g^{(t')}\|^2}\, g^{(t')}.
$$

*Gradient projection memory (GPM)* [@saha2021gradient] generalizes this by projecting layer-wise gradient matrices onto the null space of the feature subspace from past tasks, identified via SVD: $\tilde{G}^\ell = G^\ell - U^\ell {U^\ell}^\top G^\ell$.

## Summary and Open Problems

Continual learning sits at the intersection of optimization, Bayesian inference, and representation learning. The key tension — the stability–plasticity dilemma — manifests differently across the methods reviewed here:

| **Method** | **Anti-forgetting mechanism** | **Storage per task** |
|---|---|---|
| EWC [@kirkpatrick2017overcoming] | Diagonal Fisher penalty | $O(\|\theta\|)$ |
| SI [@zenke2017continual] | Online path-integral importance | $O(\|\theta\|)$ accumulated |
| VCL [@nguyen2018variational] | Variational posterior as prior | $O(\|\theta\|)$ + coresets |
| Experience replay | Stored exemplars | $O(\|\mathcal{M}\|)$ buffer |
| DER [@buzzega2020dark] | Stored exemplars + logit distillation | $O(\|\mathcal{M}\| \cdot C)$ |
| GEM [@lopez2017gradient] | Gradient projection via QP | $O(\|\mathcal{M}\|)$ buffer |
| LoRA [@hu2022lora] | Task-specific low-rank adapters | $O(r(m+n))$ per task |
| Prefix tuning [@li2021prefix] | Task-specific KV prefix | $O(L \cdot d \cdot n_\ell)$ per task |
| Cartridges [@eyuboglu2025cartridges] | Compressed KV cache per task | $O(L \cdot d \cdot n_\ell)$ per task |
| Attention matching [@zweiger2026attentionmatching] | Closed-form KV compaction (OLS) | $O(L \cdot d \cdot n_\ell)$ per task |

Several open problems remain active areas of research:

- **Adapter composition**: when task identity is unknown at test time, how should multiple LoRA adapters or cartridges be combined? Simple averaging does not exploit task structure.
- **Evaluation standards**: the field lacks consensus benchmarks, making it difficult to compare methods across papers.
- **Class-incremental learning** without task identity remains far harder than other scenarios, and PEFT methods that require task identity at inference time do not directly address it.
- **Compression quality vs. speed**: gradient-based cartridge training and closed-form attention matching occupy different points on this frontier; characterizing the trade-off theoretically remains open.

## Conclusion

Continual learning studies how to learn from a non-stationary stream of tasks without forgetting previously acquired knowledge. The field has evolved from regularization-based methods (EWC, SI, VCL), which protect important parameters using the Fisher information or a variational posterior as a surrogate prior, through replay-based methods (experience replay, DER, GEM), which counteract forgetting by revisiting past data, to modern parameter-efficient approaches (LoRA, prefix tuning, cartridges, attention matching), which sidestep forgetting entirely by keeping the base model frozen and storing task knowledge in lightweight, composable modules. The Bayesian perspective — sequential updating of a posterior, made exact by the Kalman filter in the linear Gaussian case — provides a unifying foundation that connects all of these approaches and motivates the approximate methods needed for nonlinear models.

:::{admonition} Next Steps
:class: seealso
This chapter completes Part III. Continual learning draws on all of the inference algorithms studied in this part. Readers interested in related topics may wish to explore:
- [Reinforcement Learning](rl) — another setting where agents must adapt over time, with connections to policy gradient methods and RLHF
:::

