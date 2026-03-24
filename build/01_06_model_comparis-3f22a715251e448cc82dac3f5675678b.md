# Model Comparison and Posterior Predictive Checks

**Prerequisites:** This chapter builds on [Bayesian Modeling](01_01_bayes) and [Hierarchical Gaussian Models](01_05_hierarchical). The annealed importance sampling section references [Markov Chain Monte Carlo](03_01_mcmc) (Part III); readers who have not yet encountered MCMC may skip that subsection and return after reading Part III.

## Bayesian Model Comparison

### Marginal Likelihood

The **marginal likelihood** (also called the *model evidence*) measures how well a model $\mathcal{M}_i$ fits the data by integrating over possible parameter values under the prior:

$$
p(\mathbf{x} \mid \mathcal{M}_i)
= \int p(\boldsymbol{\theta} \mid \mathcal{M}_i) \, p(\mathbf{x} \mid \boldsymbol{\theta}, \mathcal{M}_i) \, d\boldsymbol{\theta}
= \mathbb{E}_{p(\boldsymbol{\theta} \mid \mathcal{M}_i)}\!\left[ p(\mathbf{x} \mid \boldsymbol{\theta}, \mathcal{M}_i) \right].
$$

If the prior concentrates mass on parameters that assign high likelihood to the data, the marginal likelihood is large. If the prior is diffuse over many parameter values, the marginal likelihood is lower — even if the maximum likelihood is high.

This gives the marginal likelihood a built-in **Occam's razor** property: more flexible models can assign probability to many datasets, but since each model's distribution must normalize, no single dataset receives very high probability. Simpler models that focus probability on fewer datasets can assign higher marginal likelihood to those datasets — provided the data actually came from such a model.

### Bayesian Model Averaging and Selection

Given a collection of models $\{\mathcal{M}_i\}_{i=1}^M$ with prior probabilities $p(\mathcal{M}_i)$, a fully Bayesian prediction integrates over all models:

$$
p(x_\text{new} \mid \mathbf{x})
= \sum_{i=1}^M p(\mathcal{M}_i \mid \mathbf{x}) \, p(x_\text{new} \mid \mathcal{M}_i, \mathbf{x})
\propto \sum_{i=1}^M p(\mathcal{M}_i) \, p(\mathbf{x} \mid \mathcal{M}_i) \, p(x_\text{new} \mid \mathcal{M}_i, \mathbf{x}).
$$

This is **Bayesian model averaging**. A simpler approximation — **model selection** — picks the single best model $\mathcal{M}^\star = \arg\max_i p(\mathcal{M}_i \mid \mathbf{x})$ and uses only it for prediction.

### Marginal Likelihood in Exponential Families

For exponential family models with conjugate priors, the marginal likelihood has a closed form as a ratio of normalizing constants:

$$
p(\mathbf{x} \mid \mathcal{M}_i)
= \left(\prod_{n=1}^N h(x_n)\right) \frac{Z(\boldsymbol{\phi}', \nu')}{Z(\boldsymbol{\phi}, \nu)},
$$

where $\boldsymbol{\phi}' = \boldsymbol{\phi} + \sum_{n=1}^N t(x_n)$ and $\nu' = \nu + N$ are the posterior hyperparameters. This is the same structure used to derive collapsed Gibbs samplers.

**Example — Bayesian linear regression.** Under a normal-inverse-chi-squared conjugate prior with hyperparameters $(\nu, \tau^2, \boldsymbol{\Lambda})$, the marginal likelihood is:

$$
p(\mathbf{y} \mid \mathbf{X})
= (2\pi)^{-N/2}
\frac{\Gamma(\nu'/2)}{\Gamma(\nu/2)}
\frac{(\tau^2 \nu / 2)^{\nu/2}}{(\tau'^2 \nu' / 2)^{\nu'/2}}
\frac{|\boldsymbol{\Lambda}|^{1/2}}{|\boldsymbol{\Lambda}'|^{1/2}}.
$$

This can be used to compare models defined by different choices of basis functions (i.e., different design matrices $\mathbf{X}$) or different prior hyperparameters.

### The Laplace Approximation

When the marginal likelihood is intractable, the **Laplace approximation** offers a closed-form estimate. The idea is to approximate the log posterior $\mathcal{L}(\boldsymbol{\theta}) = \log p(\boldsymbol{\theta} \mid \mathbf{x})$ by a second-order Taylor expansion around the MAP estimate $\boldsymbol{\theta}_\text{MAP}$:

$$
\mathcal{L}(\boldsymbol{\theta})
\approx \mathcal{L}(\boldsymbol{\theta}_\text{MAP})
- \frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\theta}_\text{MAP})^\top \boldsymbol{\Sigma}^{-1} (\boldsymbol{\theta} - \boldsymbol{\theta}_\text{MAP}),
$$

where $\boldsymbol{\Sigma} = -[\nabla^2_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_\text{MAP})]^{-1}$ is the negative inverse Hessian at the mode. The gradient term vanishes because $\boldsymbol{\theta}_\text{MAP}$ is a stationary point. This approximation gives a Gaussian posterior $p(\boldsymbol{\theta} \mid \mathbf{x}) \approx \mathcal{N}(\boldsymbol{\theta}_\text{MAP}, \boldsymbol{\Sigma})$.

The Laplace approximation is theoretically justified by the **Bernstein–von Mises theorem**: as $N \to \infty$, the posterior converges to a Gaussian centered on the true parameter $\boldsymbol{\theta}_\text{true}$ with covariance $\frac{1}{N} [J(\boldsymbol{\theta}_\text{true})]^{-1}$, where

$$
J(\boldsymbol{\theta}) = -\mathbb{E}_{p(x \mid \boldsymbol{\theta})}\!\left[\frac{d^2}{d\boldsymbol{\theta}^2} \log p(x \mid \boldsymbol{\theta})\right]
$$

is the **Fisher information**.

**Approximating the log marginal likelihood.** Substituting the Laplace approximation into the marginal likelihood integral gives:

$$
\log p(\mathbf{x} \mid \mathcal{M}_i)
\approx \log p(\boldsymbol{\theta}_\text{MAP}) + \log p(\mathbf{x} \mid \boldsymbol{\theta}_\text{MAP}) + \frac{D}{2}\log(2\pi) + \frac{1}{2}\log|\boldsymbol{\Sigma}|.
$$

Using $\boldsymbol{\Sigma} \approx \frac{1}{N}[J(\boldsymbol{\theta}_\text{MAP})]^{-1}$ and $\frac{1}{2}\log|\boldsymbol{\Sigma}| \approx -\frac{D}{2}\log N + O(1)$ leads to the **Bayesian information criterion (BIC)**:

$$
\mathrm{BIC} = \log p(\mathbf{x} \mid \boldsymbol{\theta}_\text{MLE}) - \frac{D}{2} \log N,
$$

a penalized maximum likelihood score where $D$ is the number of parameters.

### Importance Sampling Estimates

For an unbiased Monte Carlo estimate of the marginal likelihood, draw $\boldsymbol{\theta}^{(s)} \overset{\text{iid}}{\sim} p(\boldsymbol{\theta} \mid \mathcal{M}_i)$ and average the likelihoods:

$$
p(\mathbf{x} \mid \mathcal{M}_i) \approx \frac{1}{S} \sum_{s=1}^S p(\mathbf{x} \mid \boldsymbol{\theta}^{(s)}, \mathcal{M}_i).
$$

This estimate is unbiased but can have very high variance when the prior and posterior are misaligned. **Importance sampling** reduces variance by using a proposal $r(\boldsymbol{\theta})$ that targets high-likelihood regions:

$$
p(\mathbf{x} \mid \mathcal{M}_i) \approx \frac{1}{S} \sum_{s=1}^S w^{(s)} p(\mathbf{x} \mid \boldsymbol{\theta}^{(s)}, \mathcal{M}_i),
\qquad \boldsymbol{\theta}^{(s)} \overset{\text{iid}}{\sim} r(\boldsymbol{\theta}),
\quad w^{(s)} = \frac{p(\boldsymbol{\theta}^{(s)} \mid \mathcal{M}_i)}{r(\boldsymbol{\theta}^{(s)})}.
$$

The optimal proposal is the posterior itself (giving zero variance), but that is not available in practice.

**Annealed importance sampling (AIS)** [@neal2001annealed] constructs a good proposal by defining a sequence of distributions that anneal from the prior to the posterior:

$$
f_t(\boldsymbol{\theta}) \propto p(\boldsymbol{\theta} \mid \mathcal{M}_i) \, p(\mathbf{x} \mid \boldsymbol{\theta}, \mathcal{M}_i)^{\beta_t},
\qquad 0 = \beta_T < \beta_{T-1} < \cdots < \beta_0 = 1.
$$

Samples are propagated through this sequence via MCMC transition operators, yielding an importance weight that provides an unbiased estimate of the marginal likelihood.

### Empirical Bayes

Rather than fixing hyperparameters $\boldsymbol{\phi}$ and $\nu$ in advance, **empirical Bayes** (also called **type-II maximum likelihood**) chooses them by maximizing the marginal likelihood:

$$
\boldsymbol{\phi}^*, \nu^* = \arg\max \, p(\mathbf{x} \mid \boldsymbol{\phi}, \nu)
= \arg\max \int p(\mathbf{x} \mid \boldsymbol{\theta}) \, p(\boldsymbol{\theta} \mid \boldsymbol{\phi}, \nu) \, d\boldsymbol{\theta}.
$$

For exponential families this objective is available in closed form; for more complex models, the Laplace approximation or other methods can be used. Optimization is typically done via gradient descent.

### Caveats

- Bayesian model comparison via the marginal likelihood requires a **proper prior**. In the improper/uninformative limit, the marginal likelihood is zero.
- It is most meaningful for **finite, discrete sets of models** $\{\mathcal{M}_i\}$.
- The marginal likelihood **does not measure generalization**. It measures the expected probability of the *observed data under the prior*, not the probability of *new data under the posterior*.

Research on Bayesian model comparison and marginal likelihood estimation remains active [@lotfi2022bayesian].

## Posterior Predictive Checks

### Posterior Predictive Distribution

Given a fitted model, the **posterior predictive distribution** for a new observation $y_{N+1}$ at covariates $\mathbf{x}_{N+1}$ is:

$$
p(y_{N+1} \mid \mathbf{x}_{N+1}, \{y_n, \mathbf{x}_n\}_{n=1}^N)
= \int p(y_{N+1} \mid \mathbf{x}_{N+1}, \boldsymbol{\theta}) \, p(\boldsymbol{\theta} \mid \{y_n, \mathbf{x}_n\}_{n=1}^N) \, d\boldsymbol{\theta}.
$$

This can be approximated via Monte Carlo in general, or computed in closed form for conjugate models (e.g., Bayesian linear regression yields a Student-t predictive distribution).

### Posterior Predictive Checks (PPCs)

**Posterior predictive checks** compare the observed data to data *replicated* from the posterior predictive distribution. The procedure is:

1. Draw $\boldsymbol{\theta}^{(s)} \sim p(\boldsymbol{\theta} \mid \mathbf{y})$ from the posterior.
2. Draw a replicated dataset $y^{\text{rep},(s)}$ from $p(y^\text{rep} \mid \boldsymbol{\theta}^{(s)})$.
3. Compare $y^\text{rep}$ to the observed data $\mathbf{y}$.

If the model is well-specified, the observed data should look like a plausible draw from the posterior predictive distribution.

**Example — Newcomb's speed of light.** Using a simple Gaussian model $y \sim \mathcal{N}(\mu, \sigma^2)$ with a flat prior, we can generate replicated datasets and compare their histograms to the original data. Systematic discrepancies indicate model misspecification.

### Test Statistics

Rather than comparing full datasets, it is often easier to compare **test statistics** (or *discrepancy measures*) $T(y, \boldsymbol{\theta})$:

1. Compute $T(y, \boldsymbol{\theta}^{(s)})$ for the observed data under each posterior draw.
2. Compute $T(y^{\text{rep},(s)}, \boldsymbol{\theta}^{(s)})$ for the replicated data.
3. Compare the distributions.

**Example.** Using $T(y, \boldsymbol{\theta}) = \min(y)$ for Newcomb's data reveals that the Gaussian model fails to capture the outliers in the left tail — the minimum of the observed data is far smaller than the minimum of any replicated dataset.

The **posterior predictive $p$-value** formalizes this:

$$
p = \Pr\!\left(T(y^\text{rep}, \boldsymbol{\theta}) \geq T(y, \boldsymbol{\theta}) \mid y\right)
= \int\!\!\int \mathbb{I}\!\left[T(y^\text{rep}, \boldsymbol{\theta}) \geq T(y, \boldsymbol{\theta})\right] p(y^\text{rep} \mid \boldsymbol{\theta}) \, p(\boldsymbol{\theta} \mid y) \, dy^\text{rep} \, d\boldsymbol{\theta}.
$$

A $p$-value near 0 or 1 indicates a poor fit. In practice, the full distribution of $T(y^\text{rep}, \boldsymbol{\theta})$ is more informative than the scalar $p$-value.

### Sensitivity Analysis

Beyond goodness-of-fit, it is important to assess how sensitive conclusions are to modeling choices:

- **Structural choices**: try different model families (e.g., $t$ distribution instead of Gaussian for robustness to outliers, hierarchical instead of pooled models).
- **Prior choices**: vary prior hyperparameters and check that key inferential conclusions are stable.
- **Quantity sensitivity**: extreme quantiles and extrapolations are more sensitive than means and interpolations.
