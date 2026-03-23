# Lecture 1: Introduction and Bayesian Analysis of the Normal Distribution

**Reading:** {cite}`bishop2006pattern`, Ch 2.3, 3.4.3. See also: {cite}`murphy2023probabilistic`, Ch 2.3.

## What Is This Course About?

*Probabilistic modeling and inference with high-dimensional data.*

Throughout this course we will encounter data in many forms — vectors of measurements, images, documents, time series, spike trains — and our goal will be to build probabilistic models of that data and use them to reason about the world. Depending on the application, we might want to:

- **Predict:** given features, estimate labels or outputs
- **Simulate:** given partial observations, generate the rest
- **Summarize:** given high-dimensional data, find low-dimensional factors of variation
- **Visualize:** given high-dimensional data, find informative 2D/3D projections
- **Decide:** given past actions and outcomes, determine the best future choice
- **Understand:** identify what generative mechanisms gave rise to the data

A central theme is that *probabilistic models* provide a unified language for all of these tasks. By specifying a joint distribution over observed and latent variables, we can address prediction, simulation, and uncertainty quantification within a single coherent framework.

## Box's Loop

How should we go about building and using probabilistic models? A useful guiding framework is **Box's loop** {cite}`blei2014build`, named after the statistician George Box (of "all models are wrong, but some are useful" fame). The loop has three stages:

1. **Build:** propose a probabilistic model — a joint distribution over data and parameters — that encodes your assumptions about how the data were generated.
2. **Compute:** perform inference to find the posterior distribution of the parameters given the observed data.
3. **Critique:** evaluate how well the model explains the data, check for systematic failures, and use those failures to motivate improvements.

Then repeat. Good probabilistic modeling is an iterative process: a simpler model helps us understand the data structure, and that understanding guides us toward richer, more accurate models.

```{figure} ../figures/lecture1/boxsloop.jpeg
:name: fig-boxsloop
:align: center
Box's loop: the iterative cycle of model building, inference, and criticism. Figure from {cite}`blei2014build`.
```

## The Bayesian Approach

The Bayesian approach to statistical modeling has three core components:

1. A **model** is a **joint distribution** of parameters $\mbtheta$ and data $\mbX$,
   \begin{align}
       p(\mbtheta, \mbX \mid \mbeta) = p(\mbtheta \mid \mbeta) \, p(\mbX \mid \mbtheta, \mbeta),
   \end{align}
   where $p(\mbtheta \mid \mbeta)$ is the **prior distribution** encoding beliefs about the parameters before seeing data, and $p(\mbX \mid \mbtheta, \mbeta)$ is the **likelihood** of the data given parameters. The symbol $\mbeta$ denotes **hyperparameters** — parameters of the prior that we treat as fixed and known.

2. An **inference algorithm** computes the **posterior distribution** of parameters given data — a complete probabilistic description of what we have learned about $\mbtheta$ after observing $\mbX$.

3. **Model criticism** and downstream tasks are based on **posterior expectations** — averages of quantities of interest under the posterior.

The fundamental formula connecting all three components is **Bayes' rule**,

\begin{align}
\underbrace{p(\mbtheta \mid \mbX; \mbeta)}_{\text{posterior}}
&=
\frac{\overbrace{p(\mbtheta; \mbeta)}^{\text{prior}} \; \overbrace{p(\mbX \mid \mbtheta; \mbeta)}^{\text{likelihood}}}{\underbrace{p(\mbX; \mbeta)}_{\text{marginal likelihood}}}
= \frac{p(\mbtheta, \mbX; \mbeta)}{\int p(\mbtheta, \mbX; \mbeta) \dif \mbtheta}.
\end{align}

The **marginal likelihood** $p(\mbX; \mbeta) = \int p(\mbtheta, \mbX; \mbeta) \dif \mbtheta$ is the probability of the data averaged over all parameter values. It plays a key role in model comparison and hyperparameter selection, but computing it is often the hardest part — most of this course is about methods for dealing with this integral.

**Notation.** Throughout these notes: lowercase bold letters denote vectors (e.g., $\mbx, \mbtheta$); uppercase bold letters denote matrices (e.g., $\mbX, \mbSigma$); and regular characters denote scalars (e.g., $x, \mu, \sigma^2$). We write $\mbX = \{\mbx_1, \ldots, \mbx_N\}$ for a dataset of $N$ observations.

## Normal Model with Unknown Mean

Let's start with the simplest nontrivial Bayesian model to get a feel for how the machinery works.

:::{admonition} Example: Modeling SAT scores
:class: tip
Suppose we have SAT scores $x_1, \ldots, x_N$ from $N$ students in one class. We model the scores as conditionally independent Gaussians with unknown mean $\mu$ and *known* variance $\sigma^2$. Our goal is to infer $\mu$ from the data.
:::

We place a Gaussian prior on $\mu$ and model each score as an i.i.d. draw from a Gaussian with that mean:

\begin{align}
\mu &\sim \cN(\mu_0, \sigma_0^2), \\
x_n \mid \mu &\iid{\sim} \cN(\mu, \sigma^2), \quad n = 1, \ldots, N.
\end{align}

The hyperparameters are $\mbeta = (\mu_0, \sigma_0^2, \sigma^2)$: the prior mean $\mu_0$, prior variance $\sigma_0^2$, and (known) data variance $\sigma^2$. The prior variance $\sigma_0^2$ encodes how uncertain we are about the mean before seeing any data — a large $\sigma_0^2$ corresponds to a vague, spread-out prior.

### Computing the Posterior

By Bayes' rule, the posterior is proportional to the prior times the likelihood. Expanding the Gaussian densities,

\begin{align}
p(\mu \mid \mbX; \mbeta)
&\propto p(\mu; \mbeta) \prod_{n=1}^N p(x_n \mid \mu; \mbeta) \\
&= \cN(\mu; \mu_0, \sigma_0^2) \prod_{n=1}^N \cN(x_n; \mu, \sigma^2) \\
&\propto \exp \left\{ -\frac{(\mu - \mu_0)^2}{2 \sigma_0^2} \right\} \prod_{n=1}^N \exp \left\{ -\frac{(x_n - \mu)^2}{2 \sigma^2} \right\}.
\end{align}

Since all factors are exponentials of quadratics in $\mu$, the product is also an exponential of a quadratic in $\mu$ — which means the posterior is Gaussian. This is what it means for the Gaussian prior to be **conjugate** to the Gaussian likelihood.

To identify the posterior, we collect the terms quadratic and linear in $\mu$ in the exponent:

\begin{align}
p(\mu \mid \mbX; \mbeta)
&\propto \exp \left\{ -\frac{1}{2} J_N \mu^2 + h_N \mu \right\},
\end{align}

where we define the **posterior precision** and **posterior information**,

\begin{align}
J_N = \frac{1}{\sigma_0^2} + \frac{N}{\sigma^2}, \qquad
h_N = \frac{\mu_0}{\sigma_0^2} + \sum_{n=1}^N \frac{x_n}{\sigma^2}.
\end{align}

The precision $J_N$ is simply the sum of the prior precision $1/\sigma_0^2$ and the total data precision $N/\sigma^2$ — **information is additive**. The information vector $h_N$ similarly sums contributions from the prior and each data point.

### Completing the Square

The exponent $-\frac{1}{2} J_N \mu^2 + h_N \mu$ is a quadratic in $\mu$. We can complete the square to convert it to the form $-\frac{1}{2\sigma_N^2}(\mu - \mu_N)^2 + \text{const}$, where $\text{const}$ does not depend on $\mu$. Specifically:

\begin{align}
-\frac{1}{2} J_N \mu^2 + h_N \mu
&= -\frac{J_N}{2}\left(\mu^2 - \frac{2 h_N}{J_N} \mu \right)
= -\frac{J_N}{2}\left(\mu - \frac{h_N}{J_N}\right)^2 + \frac{h_N^2}{2 J_N}.
\end{align}

Dropping the constant and recognizing $\sigma_N^2 = J_N^{-1}$ and $\mu_N = J_N^{-1} h_N = h_N / J_N$, we find

\begin{align}
p(\mu \mid \mbX; \mbeta) &= \cN(\mu \mid \mu_N, \sigma_N^2),
\end{align}

where

\begin{align}
\sigma_N^2 &= \left(\frac{1}{\sigma_0^2} + \frac{N}{\sigma^2}\right)^{-1}, \\[4pt]
\mu_N &= \sigma_N^2 \left(\frac{\mu_0}{\sigma_0^2} + \frac{N \bar{x}}{\sigma^2}\right)
= \frac{\sigma^2}{\sigma^2 + N\sigma_0^2}\, \mu_0 + \frac{N\sigma_0^2}{\sigma^2 + N\sigma_0^2}\, \mu_{\mathsf{ML}},
\end{align}

with $\bar{x} = \frac{1}{N}\sum_{n=1}^N x_n = \mu_{\mathsf{ML}}$ being the sample mean (and maximum likelihood estimate).

### Interpreting the Posterior

The posterior mean $\mu_N$ is a **weighted average** of the prior mean $\mu_0$ and the MLE $\mu_{\mathsf{ML}}$. The weights are determined by the relative precision of the prior versus the data. When $N$ is large, almost all weight goes to the data and $\mu_N \to \mu_{\mathsf{ML}}$. When the prior is diffuse ($\sigma_0^2 \to \infty$, i.e., an uninformative prior), the prior precision vanishes and again $\mu_N \to \mu_{\mathsf{ML}}$.

The posterior variance $\sigma_N^2$ follows a simpler pattern via the precision:

\begin{align}
\frac{1}{\sigma_N^2} &= \frac{1}{\sigma_0^2} + \frac{N}{\sigma^2}.
\end{align}

Each new observation contributes $1/\sigma^2$ of additional precision. As $N \to \infty$, the posterior variance shrinks to zero — we become increasingly certain about $\mu$. Under an uninformative prior ($\sigma_0^2 \to \infty$), the posterior variance becomes $\sigma_N^2 = \sigma^2/N$, the familiar sampling variance of the mean.

## Normal Model with Unknown Precision

Now suppose the mean $\mu$ is known but the variance is not. Our calculations are slightly cleaner if we work with the **precision** $\lambda = 1/\sigma^2$ rather than the variance directly. In terms of the precision, the Gaussian density is,

\begin{align}
p(x \mid \mu, \lambda) &= \left(\frac{\lambda}{2\pi}\right)^{\frac{1}{2}} \exp\left\{-\frac{\lambda}{2}(x - \mu)^2\right\}.
\end{align}

What prior should we place on $\lambda$? We need a distribution on $\reals_+$ (since $\lambda > 0$), ideally one that is conjugate to the Gaussian likelihood. The **chi-squared distribution** fits the bill naturally.

### The Chi-Squared Distribution

Let $z_1, \ldots, z_\nu \iid{\sim} \cN(0, 1)$ and define $\lambda = \sum_{i=1}^\nu z_i^2$. Then $\lambda$ follows a **chi-squared distribution** with $\nu$ degrees of freedom,

\begin{align}
\lambda \sim \chi^2(\nu),
\end{align}

with pdf

\begin{align}
\chi^2(\lambda; \nu) &= \frac{1}{2^{\nu/2} \Gamma(\nu/2)} \lambda^{\nu/2 - 1} e^{-\lambda/2}.
\end{align}

This is a special case of the gamma distribution: $\chi^2(\nu) = \mathrm{Ga}(\nu/2,\, 1/2)$ using the rate parameterization. Its mean is $\E[\lambda] = \nu$ and its variance is $\Var[\lambda] = 2\nu$.

### The Scaled Chi-Squared Distribution

The standard chi-squared has mean $\nu$, which grows with the degrees of freedom. For a prior over precision, it is more convenient to control the mean independently of the degrees of freedom. We do this by drawing from a scaled version: let $z_i \iid{\sim} \cN(0, \lambda_0)$ and define $\lambda = \frac{1}{\nu_0} \sum_{i=1}^{\nu_0} z_i^2$ (note the average rather than the sum). Then we say

\begin{align}
\lambda \sim \chi^2(\nu_0, \lambda_0),
\end{align}

where $\lambda_0 > 0$ is the **scale** parameter. Since $z_i = \sqrt{\lambda_0} \tilde{z}_i$ with $\tilde{z}_i \iid{\sim} \cN(0,1)$, we have $\lambda = \frac{\lambda_0}{\nu_0} \sum_{i=1}^{\nu_0} \tilde{z}_i^2$, so $\lambda = \frac{\lambda_0}{\nu_0} \tilde{\lambda}$ where $\tilde{\lambda} \sim \chi^2(\nu_0)$. Using the fact that the gamma distribution is closed under multiplicative scaling, $\lambda \sim \mathrm{Ga}(\nu_0/2,\, \nu_0 / (2\lambda_0))$, giving the pdf

\begin{align}
\chi^2(\lambda; \nu_0, \lambda_0) &=
\frac{\left(\frac{\nu_0}{2\lambda_0}\right)^{\nu_0/2}}{\Gamma(\nu_0/2)}\, \lambda^{\nu_0/2 - 1} \exp\!\left(-\frac{\nu_0 \lambda}{2\lambda_0}\right).
\end{align}

The mean is $\E[\lambda] = \lambda_0$ regardless of $\nu_0$, and the variance is $\Var[\lambda] = 2\lambda_0^2 / \nu_0$. The parameter $\nu_0$ controls concentration: larger $\nu_0$ gives a tighter distribution around its mean $\lambda_0$.

```{figure} ../figures/lecture1/chisq.pdf
:name: fig-chisq
:align: center
The $\chi^2(\nu_0, \lambda_0)$ pdf for $\lambda_0 = 2$ and varying degrees of freedom $\nu_0$. In all cases $\E[\lambda] = \lambda_0 = 2$, but the distribution concentrates as $\nu_0$ grows.
```

### Conjugate Update

With the scaled chi-squared prior, the model is

\begin{align}
\lambda &\sim \chi^2(\nu_0, \lambda_0), \\
x_n \mid \mu, \lambda &\iid{\sim} \cN(\mu, 1/\lambda), \quad n = 1, \ldots, N.
\end{align}

The chi-squared distribution is conjugate to the Gaussian likelihood in $\lambda$. Letting $\mbeta = (\mu, \nu_0, \lambda_0)$, we compute:

\begin{align}
p(\lambda \mid \mbX; \mbeta)
&\propto \chi^2(\lambda; \nu_0, \lambda_0) \prod_{n=1}^N \cN(x_n; \mu, \tfrac{1}{\lambda}) \\
&\propto \lambda^{\nu_0/2 - 1} e^{-\nu_0 \lambda / (2\lambda_0)} \cdot \prod_{n=1}^N \lambda^{1/2} e^{-\frac{\lambda}{2}(x_n - \mu)^2} \\
&\propto \lambda^{(\nu_0 + N)/2 - 1} \exp\!\left(-\frac{\lambda}{2}\left[\frac{\nu_0}{\lambda_0} + \sum_{n=1}^N (x_n - \mu)^2\right]\right).
\end{align}

This is another scaled chi-squared distribution, $p(\lambda \mid \mbX; \mbeta) = \chi^2(\lambda; \nu_N, \lambda_N)$, where

\begin{align}
\nu_N &= \nu_0 + N, \\
\lambda_N &= \nu_N \left(\frac{\nu_0}{\lambda_0} + \sum_{n=1}^N (x_n - \mu)^2\right)^{-1}.
\end{align}

The posterior degrees of freedom grow by one per observation. As $\nu_0 \to 0$ (an uninformative prior), the posterior mean converges to $\lambda_N \to 1/\left(\frac{1}{N}\sum_n (x_n - \mu)^2\right)$, the inverse of the sample variance — as expected.

## Normal Model with Unknown Variance

Working with the precision is mathematically natural, but it can be more intuitive to parameterize the model in terms of the variance $\sigma^2 = 1/\lambda$. If $\lambda \sim \chi^2(\nu_0, \lambda_0)$, then the change-of-variables formula gives the distribution of $\sigma^2 = 1/\lambda$:

\begin{align}
p(\sigma^2 \mid \nu_0, \sigma_0^2)
&= \left|\frac{\dif (1/\sigma^2)}{\dif \sigma^2}\right| \chi^2\!\left(\frac{1}{\sigma^2}; \nu_0, \frac{1}{\sigma_0^2}\right) \\
&= \frac{1}{(\sigma^2)^2} \cdot \frac{\left(\frac{\nu_0}{2\sigma_0^{-2}}\right)^{\nu_0/2}}{\Gamma(\nu_0/2)} \left(\frac{1}{\sigma^2}\right)^{\nu_0/2 - 1} \exp\!\left(-\frac{\nu_0}{2\sigma_0^2 \sigma^2}\right) \\
&= \frac{\left(\frac{\nu_0 \sigma_0^2}{2}\right)^{\nu_0/2}}{\Gamma(\nu_0/2)}\, (\sigma^2)^{-\nu_0/2 - 1} \exp\!\left(-\frac{\nu_0 \sigma_0^2}{2\sigma^2}\right) \\
&\triangleq \chi^{-2}(\sigma^2 \mid \nu_0, \sigma_0^2),
\end{align}

where $\sigma_0^2 = 1/\lambda_0$. This is the **scaled inverse chi-squared** distribution. It is a special case of the inverse gamma: $\chi^{-2}(\nu_0, \sigma_0^2) = \mathrm{IGa}(\nu_0/2,\, \nu_0 \sigma_0^2 / 2)$.

The mean is $\E[\sigma^2] = \frac{\nu_0}{\nu_0 - 2} \sigma_0^2$ (for $\nu_0 > 2$) and the mode is $\frac{\nu_0}{\nu_0 + 2} \sigma_0^2$. The parameter $\sigma_0^2$ is the prior "best guess" for the variance and $\nu_0$ controls how strongly that guess is held.

```{figure} ../figures/lecture1/inv_chisq.pdf
:name: fig-inv-chisq
:align: center
The $\chi^{-2}(\nu_0, \sigma_0^2)$ pdf for $\sigma_0^2 = 2$ and varying degrees of freedom $\nu_0$. The mean $\frac{\nu_0}{\nu_0-2}\sigma_0^2$ approaches $\sigma_0^2$ as $\nu_0 \to \infty$, and the distribution concentrates around $\sigma_0^2$.
```

Parameterized in terms of the variance, the model is

\begin{align}
\sigma^2 &\sim \chi^{-2}(\nu_0, \sigma_0^2), \\
x_n \mid \mu, \sigma^2 &\iid{\sim} \cN(\mu, \sigma^2).
\end{align}

:::{admonition} Exercise
Show that the posterior is $p(\sigma^2 \mid \mbX; \mbeta) = \chi^{-2}(\sigma^2; \nu_N, \sigma_N^2)$ where
\begin{align}
\nu_N &= \nu_0 + N, \\
\sigma_N^2 &= \frac{1}{\nu_N}\left(\nu_0 \sigma_0^2 + \sum_{n=1}^N (x_n - \mu)^2\right).
\end{align}
The posterior scale $\sigma_N^2$ is the average of the prior sum of squares $\nu_0 \sigma_0^2$ and the data sum of squares $\sum_n (x_n - \mu)^2$, weighted by $\nu_0$ and $N$ respectively. As $\nu_0 \to 0$, $\sigma_N^2$ converges to the sample variance.
:::

## Normal Model with Unknown Mean and Variance

Finally, suppose both $\mu$ and $\sigma^2$ are unknown. We want a joint conjugate prior over $(\mu, \sigma^2)$. The key insight is to write the prior as a product $p(\mu, \sigma^2) = p(\mu \mid \sigma^2) \, p(\sigma^2)$: an inverse chi-squared prior on the variance, and conditional on the variance, a Gaussian prior on the mean whose variance scales with $\sigma^2$. Specifically,

\begin{align}
p(\mu, \sigma^2) &= \cN(\mu; \mu_0, \sigma^2 / \kappa_0) \, \chi^{-2}(\sigma^2; \nu_0, \sigma_0^2) \\
&\triangleq \mathrm{NIX}(\mu, \sigma^2; \mu_0, \kappa_0, \nu_0, \sigma_0^2).
\end{align}

This is the **normal-inverse-chi-squared** (NIX) distribution — our first truly bivariate distribution. It has four hyperparameters: $\mu_0$ (prior mean), $\kappa_0$ (mean confidence, in pseudo-observations), $\nu_0$ (variance degrees of freedom), and $\sigma_0^2$ (prior variance scale).

The coupling $p(\mu \mid \sigma^2) = \cN(\mu_0, \sigma^2/\kappa_0)$ is crucial: the uncertainty in the mean is proportional to the variance. When $\sigma^2$ is large (we are uncertain about individual observations), we are also more uncertain about the mean. This dependence structure is what makes the NIX prior jointly conjugate.

```{figure} ../figures/lecture1/nix.pdf
:name: fig-nix
:align: center
The $\mathrm{NIX}(\mu_0, \kappa_0, \nu_0, \sigma_0^2)$ joint density over $(\mu, \sigma^2)$. Note that the spread in $\mu$ grows with $\sigma^2$, reflecting the conditional structure $p(\mu \mid \sigma^2) = \cN(\mu_0, \sigma^2/\kappa_0)$.
```

### Posterior Update

With i.i.d. Gaussian observations $x_n \iid{\sim} \cN(\mu, \sigma^2)$, the posterior under the NIX prior is again NIX:

:::{admonition} Exercise
Show that $p(\mu, \sigma^2 \mid \mbX; \mbeta) = \mathrm{NIX}(\mu_N, \kappa_N, \nu_N, \sigma_N^2)$, where
\begin{align}
\kappa_N &= \kappa_0 + N, \\
\nu_N &= \nu_0 + N, \\
\mu_N &= \frac{\kappa_0 \mu_0 + \sum_{n=1}^N x_n}{\kappa_N} = \frac{\kappa_0}{\kappa_N}\mu_0 + \frac{N}{\kappa_N}\mu_{\mathsf{ML}}, \\
\sigma_N^2 &= \frac{1}{\nu_N}\left(\nu_0 \sigma_0^2 + \kappa_0 \mu_0^2 + \sum_{n=1}^N x_n^2 - \kappa_N \mu_N^2\right).
\end{align}
Hint: write the prior and likelihood as exponentials of quadratics in $(\mu, 1/\sigma^2)$ and collect terms.
:::

Just as in the known-variance case, $\mu_N$ is a weighted average of the prior mean and the sample mean. The posterior variance scale $\sigma_N^2$ can be rewritten as $\sigma_N^2 = \frac{1}{\nu_N}\left(\nu_0 \sigma_0^2 + \kappa_0 \mu_0^2 + \sum_n x_n^2 - \kappa_N \mu_N^2\right)$, which combines the prior sum of squares with the data sum of squares, adjusted for the updated mean.

In the uninformative limit $\nu_0 \to 0$, $\kappa_0 \to 0$, the posterior parameters become $\nu_N = N$, $\kappa_N = N$, $\mu_N = \mu_{\mathsf{ML}}$, and $\sigma_N^2 = \sigma^2_{\mathsf{ML}} = \frac{1}{N}\sum_n (x_n - \mu_{\mathsf{ML}})^2$.

## Posterior Marginals: Student's t Distribution

What are the marginal distributions of $\mu$ and $\sigma^2$ under the NIX posterior? The marginal of $\sigma^2$ is straightforward — since $p(\mu, \sigma^2 \mid \mbX)$ is NIX, its $\sigma^2$ marginal is $\chi^{-2}(\nu_N, \sigma_N^2)$.

The marginal of $\mu$ is more interesting. We integrate out $\sigma^2$:

\begin{align}
p(\mu \mid \mbX; \mbeta)
&= \int p(\mu, \sigma^2 \mid \mbX; \mbeta) \dif \sigma^2 \\
&= \int \cN(\mu; \mu_N, \sigma^2 / \kappa_N) \, \chi^{-2}(\sigma^2; \nu_N, \sigma_N^2) \dif \sigma^2 \\
&= \mathrm{St}\!\left(\mu;\, \nu_N, \mu_N, \sigma_N^2 / \kappa_N\right),
\end{align}

where $\mathrm{St}(\nu, \mu, \sigma^2)$ denotes the **Student's t distribution** with $\nu$ degrees of freedom, location $\mu$, and scale $\sigma^2$, with density

\begin{align}
\mathrm{St}(x; \nu, \mu, \sigma^2) &= \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\Gamma\!\left(\frac{\nu}{2}\right)} \frac{1}{\sqrt{\pi \nu \sigma^2}} \left[1 + \frac{(x - \mu)^2}{\nu \sigma^2}\right]^{-\frac{\nu+1}{2}}.
\end{align}

Its mean is $\mu$ and its variance is $\frac{\nu}{\nu - 2}\sigma^2$ for $\nu > 2$.

The Student's t arises here because we are averaging a Gaussian over a random variance drawn from an inverse chi-squared distribution. When $\sigma^2$ is unknown and drawn from a broad distribution, the marginal distribution of $x$ has heavier tails than any single Gaussian — the extra uncertainty about the scale inflates the probability of large deviations. As $\nu \to \infty$, the Student's t converges to a Gaussian.

```{figure} ../figures/lecture1/t.pdf
:name: fig-student-t
:align: center
Student's t distributions for several degrees of freedom $\nu$, compared to a standard Gaussian. The Student's t has heavier tails, reflecting uncertainty about the scale parameter, and approaches the Gaussian as $\nu \to \infty$.
```

## Posterior Credible Intervals

One of the main uses of the posterior is to construct **credible intervals** — Bayesian analogs of confidence intervals that have a direct probabilistic interpretation: $\mathcal{I}_\alpha$ is a $1-\alpha$ credible interval if $\Pr(\mu \in \mathcal{I}_\alpha \mid \mbX) = 1 - \alpha$.

Under an uninformative prior, the posterior marginal is $\mu \mid \mbX \sim \mathrm{St}(N, \mu_{\mathsf{ML}}, \sigma^2_{\mathsf{ML}} / N)$, and the central $1-\alpha$ credible interval is

\begin{align}
\mathcal{I}_\alpha = \left[F_{\mathrm{St}}^{-1}\!\left(\tfrac{\alpha}{2} \,\Big|\, N, \mu_{\mathsf{ML}}, \frac{\sigma^2_{\mathsf{ML}}}{N}\right),\;
F_{\mathrm{St}}^{-1}\!\ left(1 - \tfrac{\alpha}{2} \,\Big|\, N, \mu_{\mathsf{ML}}, \frac{\sigma^2_{\mathsf{ML}}}{N}\right)\right],
\end{align}

where $F_{\mathrm{St}}^{-1}$ is the quantile function of the Student's t distribution.

Equivalently, note that the standardized quantity

\begin{align}
t = \frac{\mu - \mu_{\mathsf{ML}}}{\sigma_{\mathsf{ML}} / \sqrt{N}} \;\Big|\; \mbX \;\sim\; \mathrm{St}(N, 0, 1),
\end{align}

so testing whether a hypothesized value $\mu^*$ lies in $\mathcal{I}_\alpha$ is equivalent to checking whether the test statistic $t^* = (\mu^* - \mu_{\mathsf{ML}}) / (\sigma_{\mathsf{ML}} / \sqrt{N})$ satisfies $|t^*| \leq F_{\mathrm{St}}^{-1}(1 - \alpha/2 \mid N, 0, 1)$.

This is structurally identical to the frequentist one-sample t-test, but with a different interpretation: instead of asking "how often would this test reject under repeated sampling?", we ask "given the data we observed, how probable is it that $\mu$ lies outside this interval?" The Bayesian and frequentist answers happen to coincide numerically here (under the uninformative prior), but they represent fundamentally different epistemic claims.
