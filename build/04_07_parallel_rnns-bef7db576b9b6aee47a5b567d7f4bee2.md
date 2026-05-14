# Parallelizing Nonlinear RNNs

:::{admonition} Prerequisites
:class: note
This chapter builds on [Recurrent Neural Networks](04_04_rnns) and [Deep SSMs and Linear Attention](04_06_linear_attention).
:::

Linear RNNs and structured SSMs are fast to train because their recurrences compose as affine functions — a property that admits an $O(\log T)$-depth **parallel scan**. Nonlinear RNNs, like the vanilla RNN and the GRU, do not enjoy this property: each state depends nonlinearly on the previous one, so evaluation seems inherently sequential. This chapter first introduces the parallel scan algorithm, then shows how **iterative linearization** extends it to nonlinear recurrences — reducing each iteration to a linear dynamical system (LDS) solve that can be parallelized with the same primitive.

## The Parallel Scan

Consider a linear RNN with time-varying dynamics,

$$
\mbx_t = \mbA_t \mbx_{t-1} + \mbb_t, \qquad t = 1, \ldots, T,
$$

where $\mbx_t \in \reals^D$, $\mbA_t \in \reals^{D \times D}$, and $\mbb_t \in \reals^D$. Naively, computing the full trajectory $\mbx_{1:T}$ requires $T$ sequential steps, since each $\mbx_t$ depends on $\mbx_{t-1}$. The **parallel scan** [@blelloch1990prefix], also called the associative scan, reduces this to $O(\log T)$ depth by exploiting the structure of affine composition.

### Associativity and Closure

Each step of the recurrence is an affine map $f_t(\mbx) = \mbA_t \mbx + \mbb_t$. Two consecutive steps compose as another affine map:

$$
f_j \circ f_i(\mbx) = \mbA_j(\mbA_i \mbx + \mbb_i) + \mbb_j = (\mbA_j \mbA_i)\,\mbx + (\mbA_j \mbb_i + \mbb_j).
$$

Representing each step as a pair $(\mbA_t, \mbb_t)$, the composition rule is:

$$
(\mbA_j, \mbb_j) \otimes (\mbA_i, \mbb_i) = (\mbA_j \mbA_i,\; \mbA_j \mbb_i + \mbb_j).
$$

This operation is **associative** — $(f_k \circ f_j) \circ f_i = f_k \circ (f_j \circ f_i)$ — and **closed** — the composition of two affine maps is again an affine map. These two properties are exactly what the parallel scan requires.

### Divide-and-Conquer

Given $T$ affine maps, we want all cumulative compositions $f_{t:1} = f_t \circ \cdots \circ f_1$ for $t = 1, \ldots, T$. The parallel scan computes these in two phases:

- **Up-sweep**: pair up neighboring maps and reduce in parallel, $\lceil \log_2 T \rceil$ rounds, building products at powers of two.
- **Down-sweep**: propagate cumulative products back to fill all positions, another $\lceil \log_2 T \rceil$ rounds.

Together both phases run in $O(\log T)$ depth with $O(T)$ processors, yielding the full trajectory $\mbx_{1:T}$ in $O(T)$ total work — the same as sequential evaluation, but parallelized across the time dimension.

:::{admonition} Why diagonal matrices?
:class: note
Each composition involves a matrix–matrix product costing $O(D^3)$. Restricting $\mbA_t$ to be **diagonal** reduces this to $O(D)$, making the parallel scan practical for large hidden dimensions. This is why S5 and Mamba use diagonal state matrices.
:::

## Sequential Evaluation as Root-Finding

Let $\mbx_t \in \reals^D$ denote the hidden state at time $t$, and let $f_{t+1}$ denote the (possibly nonlinear, possibly input-dependent) transition function. Sequential evaluation of the recurrence,

$$
\mbx_{t+1} = f_{t+1}(\mbx_t), \qquad t = 0, \ldots, T-1,
$$

requires $O(T)$ serial steps because $\mbx_{t+1}$ directly depends on $\mbx_t$.

We can reframe the problem: the correct trajectory $\mbx_{1:T}^*$ is the **unique solution** to the system of equations,

$$
\mbx_{t+1} - f_{t+1}(\mbx_t) = 0, \qquad \forall t \in \{0, \ldots, T-1\}.
$$

This is a system of $T$ coupled nonlinear equations in $TD$ unknowns. Fixed-point iteration is a natural approach: start from an initial guess $\mbx_{1:T}^{(0)}$ (e.g., all zeros) and iteratively refine it by solving a simpler surrogate problem.

## A Unifying Framework

The key insight of @gonzalez2026unifying is that four prominent fixed-point methods — Newton, quasi-Newton, Picard, and Jacobi — all reduce to iterative evaluation of a **linear dynamical system**. Specifically, each iteration takes the common form,

$$
\mbx_{t+1}^{(i+1)} = f_{t+1}\!\left(\mbx_t^{(i)}\right) + \widetilde{A}_{t+1}\!\left(\mbx_t^{(i+1)} - \mbx_t^{(i)}\right),
$$

where $\widetilde{A}_{t+1} \in \reals^{D \times D}$ is an **approximate Jacobian** of the dynamics. This is a linear recursion in the unknown $\mbx_{t+1}^{(i+1)}$, driven by the bias $\mbb_{t+1} = f_{t+1}(\mbx_t^{(i)}) - \widetilde{A}_{t+1} \mbx_t^{(i)}$, which only depends on the previous iterate. Since it is an LDS, it can be evaluated via a **parallel scan** in $O(\log T)$ depth.

The four methods differ only in how they choose $\widetilde{A}_{t+1}$:

| **Method** | **$\widetilde{A}_{t+1}$** | **Cost per iteration** | **Parallelization** |
|---|---|---|---|
| Newton | $\frac{\partial f_{t+1}}{\partial \mbx_t}(\mbx_t^{(i)})$ — full Jacobian | $O(TD^3)$ | Parallel scan (dense) |
| Quasi-Newton | $\mathrm{diag}\!\left[\frac{\partial f_{t+1}}{\partial \mbx_t}(\mbx_t^{(i)})\right]$ — diagonal | $O(TD)$ | Parallel scan (elementwise) |
| Picard | $I_D$ — identity | $O(TD)$ | Prefix sum |
| Jacobi | $0$ — zero | $O(TD)$ | Embarrassingly parallel |

All four methods are guaranteed to converge to the correct trajectory $\mbx_{1:T}^*$ in at most $T$ iterations [@gonzalez2026unifying].

## Four Root Finding Methods

### Newton Iterations

Newton's method for root-finding linearizes the residual $r(\mbx_{1:T}) = [\mbx_1 - f_1(\mbx_0), \ldots, \mbx_T - f_T(\mbx_{T-1})]$ using its full Jacobian. Applied to sequential evaluation, each Newton step is the first-order Taylor expansion of the recurrence around the current iterate:

$$
\mbx_{t+1}^{(i+1)} = f_{t+1}\!\left(\mbx_t^{(i)}\right) + \frac{\partial f_{t+1}}{\partial \mbx_t}\!\left(\mbx_t^{(i)}\right)\!\left(\mbx_t^{(i+1)} - \mbx_t^{(i)}\right).
$$

This is the **DEER** algorithm of @lim2024parallelizing. Because the transition matrix $\widetilde{A}_{t+1}$ is the full $D \times D$ Jacobian, each iteration requires $O(TD^2)$ memory and $O(TD^3)$ work for the matrix–matrix multiplications in the parallel scan. For large state dimensions, this is prohibitive.

When $f_{t+1}$ is a linear function of $\mbx_t$, the Jacobian is exact and Newton converges in a **single iteration** — recovering the standard parallel scan for LDSs as a special case.

### Quasi-Newton Iterations

To reduce the cost of Newton iterations, @gonzalez2025elk replace the full Jacobian with its **diagonal**:

$$
\mbx_{t+1}^{(i+1)} = f_{t+1}\!\left(\mbx_t^{(i)}\right) + \mathrm{diag}\!\left[\frac{\partial f_{t+1}}{\partial \mbx_t}\!\left(\mbx_t^{(i)}\right)\right]\!\left(\mbx_t^{(i+1)} - \mbx_t^{(i)}\right).
$$

This is the **ELK** (Evaluating Levenberg–Marquardt via Kalman) algorithm. With a diagonal transition matrix, each parallel scan step is an elementwise vector multiplication, reducing cost to $O(TD)$. The diagonal of the Jacobian can often be computed in closed form for common RNN architectures (e.g., GRUs), or estimated stochastically using the Hutchinson estimator in a single function call.

### Picard Iterations

**Picard iterations** set $\widetilde{A}_{t+1} = I_D$, approximating the Jacobian by the identity matrix. The update simplifies to a **prefix sum**:

$$
\mbx_{t+1}^{(i+1)} = \mbx_0 + \sum_{s=0}^{t} f_{s+1}\!\left(\mbx_s^{(i)}\right) \Delta,
$$

where $\Delta$ is the discretization step (for ODE-based dynamics). Picard iterations require only vector additions, making them the cheapest per-iteration method. @shih2023parallel used them to parallelize sampling in diffusion models.

The identity approximation is faithful when the true Jacobian $\partial f_{t+1}/\partial \mbx_t \approx I_D$, which holds for dynamics with small step sizes (e.g., discretized ODEs with fine time steps).

### Jacobi Iterations

**Jacobi iterations** set $\widetilde{A}_{t+1} = 0$, giving the simplest possible update:

$$
\mbx_{t+1}^{(i+1)} = f_{t+1}\!\left(\mbx_t^{(i)}\right).
$$

Each element of the new trajectory is computed **independently** — no dependencies between time steps at all. This is **embarrassingly parallel**: all $T$ states can be updated simultaneously in a single kernel call. @song2021accelerating used Jacobi iterations to accelerate feedforward computation in deep networks.

The zero-Jacobian approximation is faithful when consecutive states evolve nearly independently, i.e., when the true Jacobian $\partial f_{t+1}/\partial \mbx_t \approx 0$.

## Convergence Analysis

@gonzalez2026unifying derive a unified convergence bound for all four methods. Let $\mbe^{(i)} = \mbx_{1:T}^{(i)} - \mbx_{1:T}^*$ denote the error at iteration $i$, and let $\widetilde{J}$ and $J$ denote the block-bidiagonal approximate and true Jacobians of the residual. Then:

$$
\|\mbe^{(i+1)}\|_2 \leq \left\|\widetilde{J}^{-1}\right\|_2 \cdot \left\|\widetilde{J} - J\right\|_2 \cdot \|\mbe^{(i)}\|_2 + O\!\left(\|\mbe^{(i)}\|_2^2\right).
$$

As the error approaches zero, the asymptotic linear convergence rate is:

$$
\gamma = \left\|\widetilde{J}^{-1}\left(\widetilde{J} - J\right)\right\|_2.
$$

Convergence is fast ($\gamma \ll 1$) when two conditions hold simultaneously:

1. **Small approximation error**: $\|\widetilde{J} - J\|_2$ is small, i.e., $\widetilde{A}_{t+1}$ is a faithful approximation of the true Jacobian $\partial f_{t+1}/\partial \mbx_t$.

2. **Stable LDS**: $\|\widetilde{J}^{-1}\|_2$ is small, i.e., the linearized system with transition matrices $\widetilde{A}_{t+1}$ is stable (spectral norms well below one).

The two conditions trade off against each other:

- **Newton**: $\widetilde{J} = J$, so $\|\widetilde{J} - J\|_2 = 0$ and the method converges in one step — but only if the resulting LDS is stable.
- **Quasi-Newton**: $\widetilde{A}_{t+1} = \mathrm{diag}[J_{t+1}]$ is typically a better approximation than the identity, and the diagonal LDS is often stable.
- **Picard**: $\widetilde{A}_{t+1} = I_D$, which is faithful when $\partial f_{t+1}/\partial \mbx_t \approx I_D$ but the resulting LDS has unit eigenvalues, making $\|\widetilde{J}^{-1}\|_2 = O(T)$ — slow for long sequences.
- **Jacobi**: $\widetilde{A}_{t+1} = 0$, so $\|\widetilde{J}^{-1}\|_2 = 1$ (minimal), but the approximation error $\|\widetilde{J} - J\|_2 = \|J\|_2$ may be large.

:::{admonition} Practical guidance
:class: tip
Use the simplest approximate Jacobian possible, but no simpler. The goal is to find the cheapest $\widetilde{A}_{t+1}$ such that (a) $\widetilde{A}_{t+1} \approx \partial f_{t+1}/\partial \mbx_t$ and (b) the induced LDS is stable. Checking these conditions requires understanding the dynamics of the specific model.
:::

## Conclusion

The parallel scan turns any associative, closed operation into an $O(\log T)$-depth computation. For nonlinear RNNs, iterative linearization reduces each fixed-point iteration to an LDS — making the parallel scan applicable even when the original dynamics are nonlinear. All four methods are guaranteed to converge in at most $T$ iterations; the rate depends on how faithfully $\widetilde{A}_{t+1}$ approximates the true Jacobian and on the stability of the induced LDS.

| **Method** | **$\widetilde{A}_{t+1}$** | **Cost/iter** | **Converges fast when** |
|---|---|---|---|
| Newton | Full Jacobian | $O(TD^3)$ | Jacobian is dense |
| Quasi-Newton | Diag. Jacobian | $O(TD)$ | Diagonal of Jacobian dominates |
| Picard | $I_D$ | $O(TD)$ | Dynamics $\approx$ identity shift |
| Jacobi | $0$ | $O(TD)$ | States evolve nearly independently |

:::{admonition} Next Steps
:class: seealso
- [Discrete Diffusion](04_08_discrete_diffusion) — another sequential process where parallel techniques are actively being developed
- [Diffusion Models](02_05_diffusion) — denoising diffusion models, where Picard iterations have been applied to parallelize the sampling chain [@shih2023parallel]
:::

:::{admonition} Recommended Reading
:class: reading
[@gonzalez2026unifying], "A Unifying Framework for Parallelizing Sequential Models with Linear Dynamical Systems."
:::
