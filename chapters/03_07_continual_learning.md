# Continual Learning

**Prerequisites:** This chapter builds on [Markov Chain Monte Carlo](03_01_mcmc) (Fisher information and the Laplace approximation), [Coordinate Ascent Variational Inference](03_04_cavi), and [Gradient-Based Variational Inference](03_05_advi). Familiarity with neural networks and gradient-based optimization is assumed.

A central assumption of classical statistical learning is that training data are drawn i.i.d. from a fixed distribution. Real-world systems routinely violate this: a medical diagnosis model must incorporate new disease variants without forgetting old ones; a language model deployed in production should update on new domains without degrading on existing benchmarks; a robot should accumulate motor skills over its lifetime rather than relearning each task from scratch. **Continual learning** (also called *lifelong learning* or *sequential learning*) is the study of how to learn from a non-stationary stream of data or tasks while retaining previously acquired knowledge {cite}`delange2021continual`.

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

The difficulty is that naive sequential training — minimizing $\mathcal{L}_t$ at each step with gradient descent — causes **catastrophic forgetting** {cite}`mccloskey1989catastrophic`: optimizing for the current task moves parameters into a region that performs poorly on previous tasks, and the loss of prior knowledge can be abrupt and nearly total.

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

**Elastic weight consolidation (EWC)** {cite}`kirkpatrick2017overcoming` approximates the posterior over parameters after task $t-1$ as a Gaussian centered on the MAP estimate $\theta^*_{t-1}$:

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

For a sequence of tasks, the Fisher from all previous tasks accumulates. In the **online EWC** variant {cite}`schwarz2018progress`, a single running estimate of the Fisher is maintained rather than storing one Fisher matrix per task.

### Synaptic Intelligence

**Synaptic intelligence (SI)** {cite}`zenke2017continual` estimates parameter importance **online** during training rather than post-hoc. It tracks the cumulative contribution of each parameter to the decrease in the loss along the optimization trajectory. For a parameter $\theta_i$ moving from $\theta_{i,0}$ to $\theta_{i,T_t}$ during training on task $t$, the importance is:

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

### Variational Continual Learning

**Variational continual learning (VCL)** {cite}`nguyen2018variational` maintains a variational approximation $q_t(\theta) \approx p(\theta \mid \mathcal{D}_{1:t})$ at each step. The variational objective at task $t$ is:

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

**Dark Experience Replay (DER)** {cite}`buzzega2020dark` improves on standard replay by storing the model's **soft predictions** (logits) $z_n = f_\theta(x_n)$ at the time each exemplar is added to the buffer, in addition to the input $x_n$. The replay loss then includes a **knowledge distillation** term:

$$
\mathcal{L}_\mathrm{DER}(\theta) = \mathcal{L}_t(\theta) + \alpha \underbrace{\frac{1}{|\mathcal{M}|}\sum_{(x_n, z_n)\in\mathcal{M}} \|f_\theta(x_n) - z_n\|^2}_{\text{distillation from stored logits}}.
$$

Matching the current model's predictions to the stored logits preserves not just the final decision but the full predictive distribution — a richer signal that slows forgetting more effectively than label-only replay.

### Gradient Episodic Memory

**Gradient Episodic Memory (GEM)** {cite}`lopez2017gradient` uses the memory buffer $\mathcal{M}$ not to mix replay into the training loss, but to constrain the gradient update direction. GEM solves a quadratic program at each step to find the update $\tilde{g}$ that:

1. does not increase the loss on any past task's exemplars (the **non-interference** constraint), and
2. is as close as possible to the current task's gradient $g_t$.

Formally, let $g_{t'} = \nabla_\theta \mathcal{L}_{\mathcal{M}_{t'}}(\theta)$ be the gradient of the memory loss for past task $t'$. GEM solves:

$$
\tilde{g} = \arg\min_{v} \tfrac{1}{2}\|v - g_t\|^2 \quad \text{subject to} \quad \langle v, g_{t'} \rangle \geq 0 \quad \forall t' < t.
$$

The constraint $\langle \tilde{g}, g_{t'} \rangle \geq 0$ ensures the update does not increase past task losses to first order. When all constraints are satisfied by $g_t$ itself, no projection is needed; otherwise, $g_t$ is projected onto the intersection of the constraint half-spaces.

## Architecture-Based Methods

Architecture methods allocate dedicated parameters to each task, avoiding interference by construction.

### Progressive Neural Networks

**Progressive neural networks** {cite}`rusu2016progressive` grow the architecture with each task: task $t$ receives a new column of weights $\theta^{(t)}$, while all previous columns $\theta^{(1)}, \ldots, \theta^{(t-1)}$ are frozen. Lateral connections allow the new column to receive input from all previous columns, enabling **positive forward transfer** — the new task can reuse representations learned for earlier tasks:

$$
h_k^{(t)} = \sigma\!\left(W_k^{(t)} h_{k-1}^{(t)} + \sum_{t' < t} U_k^{(t' \to t)} h_{k-1}^{(t')}\right),
$$

where $W_k^{(t)}$ are the new column's weights and $U_k^{(t' \to t)}$ are the lateral connections from past columns. Forgetting is zero by construction, but the model grows linearly with the number of tasks, which is infeasible for long task sequences.

### Parameter Isolation and PackNet

**PackNet** masks the network into task-specific subnetworks. After training task $t$, weights below a threshold are pruned and those parameters are freed for future tasks. Each task $t$ is assigned a binary mask $m^{(t)} \in \{0,1\}^{|\theta|}$ and uses only the corresponding parameters $\theta \odot m^{(t)}$. Since the masks for different tasks are disjoint, there is zero interference. The trade-off is a fixed total parameter budget that must be shared across all tasks.

## Gradient Projection Methods

A family of methods avoids forgetting by projecting gradient updates onto a subspace that is approximately orthogonal to the directions important for previous tasks.

### Orthogonal Gradient Descent

**Orthogonal gradient descent (OGD)** {cite}`farajtabar2020orthogonal` maintains the gradients of the loss for previous tasks at their optima, $g^{(t')} = \nabla_\theta \mathcal{L}_{t'}(\theta^*_{t'})$, and projects the current gradient onto their orthogonal complement:

$$
\tilde{g}_t = g_t - \sum_{t'<t} \frac{g_t^\top g^{(t')}}{\|g^{(t')}\|^2} g^{(t')}.
$$

The update $\tilde{g}_t$ is orthogonal to all past gradients and thus (to first order) does not change the loss on previous tasks. In practice, only a subset of past gradients are stored — e.g., those computed at the final iterate of each task.

### Gradient Projection Memory

**Gradient Projection Memory (GPM)** {cite}`saha2021gradient` generalizes OGD by projecting onto the orthogonal complement of the **feature subspace** spanned by activations from past tasks, not just past gradients. For each layer $\ell$, the feature matrix $R^\ell$ is built from activation vectors over past task data, and its leading singular vectors (via SVD) define a basis for the important subspace:

$$
R^\ell = U^\ell \Sigma^\ell {V^\ell}^\top.
$$

Gradient updates to layer $\ell$ are projected onto the null space of $U^\ell$:

$$
\tilde{G}^\ell = G^\ell - U^\ell {U^\ell}^\top G^\ell,
$$

where $G^\ell$ is the gradient matrix for layer $\ell$. This ensures that changes to the weight matrix do not alter the feature representations learned for previous tasks, providing a stronger non-interference guarantee than gradient-level OGD.

## Modern Approaches

### Progress and Compress

The **Progress & Compress** framework {cite}`schwarz2018progress` separates learning into two phases per task:

- **Progress**: a single *active column* is trained on the current task using EWC regularization against a *knowledge base* (KB) network.
- **Compress**: after each task, the knowledge in the active column is distilled into the KB network via online EWC.

This maintains a fixed model size while continually distilling new knowledge into the shared knowledge base, amortizing the cost of growing architectures.

### Continual Learning with Large Models

Foundation models trained on massive corpora exhibit a form of implicit continual learning resistance: because their representations are broad and general, fine-tuning on a new task tends to cause less forgetting than training a task-specific model from scratch. Nevertheless, catastrophic forgetting remains a concern when fine-tuning is aggressive.

**Parameter-efficient fine-tuning (PEFT)** methods such as LoRA (low-rank adaptation) confine task-specific updates to a small number of parameters, naturally limiting interference. For a weight matrix $W_0 \in \mathbb{R}^{m \times n}$, LoRA parameterizes the update as $\Delta W = BA$ with $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, and $r \ll \min(m, n)$. The base weights $W_0$ remain frozen, and the adapters $\{A^{(t)}, B^{(t)}\}$ for different tasks can be stored and swapped without interference — recovering task performance by loading the appropriate adapter.

A complementary line of work interprets **in-context learning** — where the model adapts to a new task from a few examples in the prompt, without any weight update — as a form of zero-storage continual learning. The task specification is carried in the context rather than the parameters, sidestepping forgetting entirely.

## Summary and Open Problems

Continual learning sits at the intersection of optimization, Bayesian inference, and representation learning. The key tension — the stability–plasticity dilemma — manifests differently across the methods reviewed here:

| **Method** | **Anti-forgetting mechanism** | **Memory cost** |
|---|---|---|
| EWC {cite}`kirkpatrick2017overcoming` | Diagonal Fisher penalty | $O(\|\theta\|)$ per task |
| SI {cite}`zenke2017continual` | Online path-integral importance | $O(\|\theta\|)$ accumulated |
| VCL {cite}`nguyen2018variational` | Variational posterior as prior | $O(\|\theta\|)$ + coresets |
| Experience replay | Stored exemplars | $O(|\mathcal{M}|)$ buffer |
| DER {cite}`buzzega2020dark` | Stored exemplars + logit distillation | $O(|\mathcal{M}| \cdot C)$ |
| GEM {cite}`lopez2017gradient` | Gradient projection via QP | $O(|\mathcal{M}|)$ buffer |
| GPM {cite}`saha2021gradient` | Subspace gradient projection | $O(r \cdot \|\theta\|_\text{layer})$ per task |
| Progressive nets {cite}`rusu2016progressive` | Frozen columns + lateral connections | $O(T \cdot \|\theta\|)$ |

Several open problems remain active areas of research:

- **Theoretical understanding of forgetting**: tight bounds on the forgetting rate as a function of task similarity, model capacity, and optimization dynamics are not yet established.
- **Evaluation standards**: the field lacks consensus benchmarks, making it difficult to compare methods across papers.
- **Class-incremental learning** without task identity at test time remains far harder than the other two scenarios, and no method approaches the performance of joint training.
- **Continual learning with foundation models**: as large pretrained models become the dominant starting point, the relevant questions shift from "how to avoid forgetting" to "how to efficiently adapt while preserving generality" — a setting where the Bayesian sequential-update perspective may prove especially fruitful.
