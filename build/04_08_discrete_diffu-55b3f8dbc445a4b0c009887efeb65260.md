# Discrete Denoising Diffusion Models

:::{admonition} Prerequisites
:class: note
This chapter builds on [Denoising Diffusion Models](02_05_diffusion), which introduces the noising/reverse process framework and the ELBO. The continuous-time treatment relies on [Continuous-Time Markov Chains](05_05_ctmc), which are the continuous-time analog of the discrete-time Markov chains underlying [Hidden Markov Models](04_01_hmms), and are closely related to [Poisson Processes](05_03_poisson_processes).
:::

Denoising diffusion models (DDMs) are currently the state-of-the-art approach for image generation, but can they be used for generating discrete data like language and protein sequences? We derived the basic principles for DDMs with continuous-valued data in [Denoising Diffusion Models](02_05_diffusion). Here, we show how these concepts extend to discrete data, and how ideas like continuous-time diffusion and the score function in the reverse diffusion SDE extend to the discrete setting.

## Discrete-Time, Discrete-State DDMs

We will start in discrete time and develop a DDM for discrete-valued data.

### Setup

- Let $x_0 \in \cX$ denote a data point.
- Let $|\cX| = S < \infty$ be the vocabulary size.
- Let $q_0(x_0)$ denote the data distribution.

### Noising Process

A Markov chain gradually converts $x_0$ to $x_T \sim q_T(x_T)$, which is pure "noise":

$$
q(x_{0:T}) = q_0(x_0) \prod_{t=1}^T q_{t|t-1}(x_t \mid x_{t-1}).
$$

For example, $q_T(x_T) = \mathrm{Unif}_{\cX}(x_T)$ can be achieved by

$$
q_{t|t-1}(x_t \mid x_{t-1}) = (1 - \lambda_t)\, \bbI[x_t = x_{t-1}] + \frac{\lambda_t}{S-1}\, \bbI[x_t \neq x_{t-1}].
$$

### Masking Diffusion

One of the most effective noising processes is **masking diffusion**, which introduces a $\mathsf{MASK}$ token as an absorbing state:

$$
q_{t|t-1}(x_t \mid x_{t-1}) = (1 - \lambda_t)\, \bbI[x_t = x_{t-1}] + \lambda_t\, \bbI[x_t = \mathsf{MASK}].
$$

### Reverse Process

The reverse of the noising process is also a Markov chain. It factors as

$$
q(x_{0:T}) = q_T(x_T) \prod_{t=T-1}^0 q_{t|t+1}(x_t \mid x_{t+1}).
$$

The reverse transition probabilities follow from Bayes' rule:

$$
q_{t|t+1}(x_t \mid x_{t+1}) = \frac{q_t(x_t)\, q_{t+1|t}(x_{t+1} \mid x_t)}{q_{t+1}(x_{t+1})}.
$$

Alternatively, expressing the reverse in terms of the _denoising distributions_ $q_{0|t+1}(x_0 \mid x_{t+1})$:

$$
q_{t|t+1}(x_t \mid x_{t+1})
= q_{t+1|t}(x_{t+1} \mid x_t) \sum_{x_0} \frac{q_{t|0}(x_t \mid x_0)}{q_{t+1|0}(x_{t+1} \mid x_0)}\, q_{0|t+1}(x_0 \mid x_{t+1}).
$$

:::{admonition} Explanation
:class: dropdown
In the last line we used the fact that

$$
\frac{q_0(x_0)}{q_{t+1}(x_{t+1})} = \frac{q_{0|t+1}(x_0 \mid x_{t+1})}{q_{t+1|0}(x_{t+1} \mid x_0)},
$$

which follows from the chain rule.
:::

### Approximating the Reverse Process

**Problem:** We know everything in the reverse transition probability except the denoising distribution $q_{0|t+1}(x_0 \mid x_{t+1})$.

**Solution:** Learn it. Parameterize the reverse transition probability as

$$
p_{t|t+1}(x_t \mid x_{t+1}; \theta) =
q_{t+1|t}(x_{t+1} \mid x_t) \sum_{x_0} \frac{q_{t|0}(x_t \mid x_0)}{q_{t+1|0}(x_{t+1} \mid x_0)}\, p_{0|t+1}(x_0 \mid x_{t+1}; \theta),
$$

where $p_{0|t+1}(x_0 \mid x_{t+1}; \theta)$ is a **learned, approximate denoising distribution**. We then sample from the approximate reverse process one step at a time from $T$ down to $0$:

$$
p(x_{0:T}; \theta) = q_T(x_T) \prod_{t=T-1}^0 p_{t|t+1}(x_t \mid x_{t+1}; \theta).
$$

### The Evidence Lower Bound

We estimate model parameters $\theta$ by maximizing the ELBO, which is a sum over data points of

$$
\begin{aligned}
\cL(\theta, x_0)
&= \E_{q(x_{1:T} \mid x_0)}\!\left[\log p(x_{0:T}; \theta) - \log q(x_{1:T} \mid x_0) \right] \\
&= \sum_{t=1}^{T-1} \E_{q(x_{t+1} \mid x_0)}\!\left[-\KL{q(x_t \mid x_{t+1}, x_0)}{p_{t|t+1}(x_t \mid x_{t+1}; \theta)} \right] + \cL_0(\theta, x_0),
\end{aligned}
$$

where $\cL_0(\theta, x_0) = \E_{q(x_1 \mid x_0)}\!\left[\log p(x_0 \mid x_1; \theta)\right]$ is the reconstruction term.

We use Rao-Blackwellization to write the ELBO in terms of expectations over fewer random variables for each term in the sum.

:::{admonition} Key design requirement
:class: warning
We choose $q$ such that the marginal $q_{t|0}(x_t)$ and interpolating distribution $q(x_t \mid x_{t+1}, x_0)$ are available in closed form.
:::

## Continuous-Time Discrete DDMs

In the continuous-time limit, the noising and reverse processes become **continuous-time Markov chains (CTMCs)** — for a self-contained treatment see [Continuous-Time Markov Chains](05_05_ctmc). The reversal of a CTMC is another CTMC, and @campbell2022continuous showed how to parameterize the reverse process of a discrete-state, continuous-time DDM in terms of the backward rates.

The backward rate is

$$
\tilde{R}_{T-t}(i \to j) = R_t(j \to i)\, \frac{q_t(x_t = j)}{q_t(x_t = i)},
$$

where the density ratio $q_t(x_t = j) / q_t(x_t = i)$ is the discrete analog of the score function.

Sampling the backward process is tricky because the reverse rate is **inhomogeneous**, and Gillespie's algorithm for inhomogeneous processes requires integrating rate matrices. @campbell2022continuous propose **tau-leaping** to approximately sample the backward process, followed by **corrector steps** to compensate for discretization error. Recent work shows how to develop more informative correctors for discrete diffusion with masking processes [@zhao2024informed].

## Conclusion

Discrete DDMs extend the denoising diffusion framework to discrete state spaces. The discrete-time formulation mirrors the continuous case: a noising Markov chain gradually corrupts data toward a noise distribution, and the model learns to approximate the reverse denoising distribution. The ELBO decomposes into per-step KL divergences, so the choice of noising process — in particular whether it admits closed-form marginals and interpolating distributions — is critical. Masking diffusion, where tokens are absorbed into a MASK state, is currently among the most effective approaches and has been successfully applied to language modeling and protein sequence design.

In the continuous-time limit, the noising and reverse processes become CTMCs. The backward rate matrices — expressible as density ratios that are the discrete analog of the score function — characterize the reverse process, but sampling it exactly is intractable and requires approximate methods such as tau-leaping and corrector steps [@campbell2022continuous]. Recent work shows that the structure of masking diffusion enables more informed correctors that substantially improve sample quality [@zhao2024informed].

:::{admonition} Next Steps
:class: seealso
- [Continuous-Time Markov Chains](05_05_ctmc) — self-contained introduction to CTMCs, including rate matrices, Chapman–Kolmogorov equations, and Gillespie's algorithm
- [Stochastic Differential Equations](05_02_sdes) — the continuous-state analog of the CTMC treatment here; the score function connection becomes especially transparent in that setting
- [Poisson Processes](05_03_poisson_processes) — covers the Poisson process properties (superposition, thinning) underlying the CTMC connection used in @rao2013fast
:::

:::{admonition} Recommended Reading
:class: reading
[@campbell2022continuous], "A continuous time framework for discrete denoising models." Foundational paper on continuous-time discrete DDMs, including the tau-leaping reverse sampler and corrector steps.
[@zhao2024informed], "Informed correctors for discrete diffusion models." Informed correctors for masking diffusion that exploit the structure of the absorbing-state process.
[@rao2013fast], "Fast MCMC sampling for Markov jump processes and extensions."
:::
