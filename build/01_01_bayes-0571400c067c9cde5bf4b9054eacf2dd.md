# Bayesian Modeling

:::{admonition} Prerequisites
:class: note
None beyond basic probability theory. This is the opening chapter of the course; every subsequent chapter builds on the concepts introduced here.
:::

*This course is about probabilistic modeling and inference with high-dimensional data.*

Throughout this course we will encounter data in many forms — vectors of measurements, images, documents, time series, spike trains — and our goal will be to build probabilistic models of that data and use them to reason about the world. Depending on the application, we might want to:

- **Predict:** given features, estimate labels or outputs
- **Simulate:** given partial observations, generate the rest
- **Summarize:** given high-dimensional data, find low-dimensional factors of variation
- **Visualize:** given high-dimensional data, find informative 2D/3D projections
- **Decide:** given past actions and outcomes, determine the best future choice
- **Understand:** identify what generative mechanisms gave rise to the data

A central theme is that *probabilistic models* provide a unified language for all of these tasks. By specifying a joint distribution over observed and latent variables, we can address prediction, simulation, and uncertainty quantification within a single coherent framework.

## Box's Loop

How should we go about building and using probabilistic models? A useful guiding framework is **Box's loop** [@blei2014build], named after the statistician George Box (of "all models are wrong, but some are useful" fame). The loop has three stages:

1. **Build:** propose a probabilistic model — a joint distribution over data and parameters — that encodes your assumptions about how the data were generated.
2. **Compute:** perform inference to find the posterior distribution of the parameters given the observed data.
3. **Critique:** evaluate how well the model explains the data, check for systematic failures, and use those failures to motivate improvements.

Then repeat. Good probabilistic modeling is an iterative process: a simpler model helps us understand the data structure, and that understanding guides us toward richer, more accurate models.

```{figure} ../figures/lecture1/boxsloop.jpeg
:name: fig-boxsloop
:align: center
Box's loop: the iterative cycle of model building, inference, and criticism. Figure from [@blei2014build].
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

## TODO: Worked Example(s)

## TODO: Project Description