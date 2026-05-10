# Denoising Diffusion Models

:::{admonition} Prerequisites
:class: note
This chapter builds on [Variational Autoencoders](02_04_vaes). The continuous-time SDE formulation is developed formally in [Stochastic Differential Equations](05_02_sdes) (Part V); readers may find it helpful to read that chapter first or use it as a reference alongside this one.
:::

Denoising diffusion probabilistic models (DDPMs) [@sohl2015deep; @ho2020denoising] are the deep generative models underlying image generation tools like DALL-E 2 (from Open AI) and Stable Diffusion (from Stability AI). This lecture will unpack how they work. These notes are partly inspired by @turner2024denoising.

## Key Ideas

Diffusion models work by
1. Using a fixed (i.e., not learned), user-defined **noising process** to convert data into noise.
2. Learning the inverse this process so that starting from noise, we can generate samples that approximate the data distribution.

We can think of the DDPM as a giant latent variable model, where the latent variables are noisy versions of the data. As with other latent variable models (e.g., [VAEs](./11_vaes.md)), once we've inferred the latent variables, the problem of learning the mapping from latents to observed data reduces to a supervised regression problem.


DDPMs were originally proposed for modeling continuous data, $\mbx \in \reals^{D}$. For simplicity, we will present the framework for scalar data, $x \in \reals$, and then discuss the straightforward generalization to multidimensional data afterward. Finally, we will close with a discussion of recent work on diffusion modeling for discrete data.

## Noising process

Let $x \equiv x_0$ be our observed data. The noising process is a joint distribution over a sequence of latent variables $x_{0:T} = (x_0, x_1, \ldots, x_T)$. We will denote the distribution as,
\begin{align*}
q(x_{0:T}) = q(x_0) \prod_{t=1}^T q(x_t \mid x_{t-1}).
\end{align*}
At each step, the latents will become increasingly noisy versions of the original data, until at time $T$ the latent variable $x_T$ is essentially pure noise.  The generative model will then proceed by sampling pure noise and attemptign to invert the noising process to produce samples that approximate the data generating distribution.

### Gaussian noising process

For continuous data, the standard noising process is a first-order Gaussian autoregressive (AR) process,
\begin{align*}
q(x_t \mid x_{t-1}) &= \mathrm{N}(x_t \mid \lambda_t x_{t-1}, \sigma_{t}^2).
\end{align*}
The hyperparameters $\{\lambda_t, \sigma_t^2\}_{t=1}^T$ and the number of steps $T$ are fixed (not learned). We restrict $\lambda_t < 1$ so that the process contracts

Since the noising process has linear Gaussian dynamics, we can compute conditional distributions in closed form.

::::{admonition} Computing the conditional distributions

Show that the conditional distributions are,
\begin{align*}
q(x_t \mid x_0) 
&= \mathrm{N}\left( x_t \mid \lambda_{t|0} x_0, \sigma_{t|0}^2 \right)
\end{align*}
where the parameters are defined recursively,
\begin{align*}
\lambda_{t|0} &= \lambda_t \lambda_{t-1|0} = \prod_{s=1}^t \lambda_s \\
\sigma_{t|0}^2 &= \lambda_t^2 \sigma_{t-1|0}^2 + \sigma_t^2
\end{align*}
with base case $\lambda_{1|0}=\lambda_1$ and $\sigma_{1|0}^2=\sigma_1^2$.

:::{admonition} Solution
:class: dropdown, tip

Assume the equality above holds for $t-1$, by induction. Then,
\begin{align*}
q(x_{t} \mid x_0) 
&= \int q(x_{t} \mid x_{t-1}) \, q(x_{t-1} \mid x_0) \dif x_{t-1} \\
&= \int \mathrm{N}(x_{t} \mid \lambda_{t} x_{t-1}, \sigma_{t}^2) \, \mathrm{N}(x_{t-1} \mid \lambda_{t-1|0} x_0, \, \sigma_{t-1|0}^2) \dif x_{t-1} \\
&= \mathrm{N}(x_{t} \mid  \lambda_{t} \lambda_{t-1|0} x_0, \lambda_{t}^2 \sigma_{t-1|0}^2 + \sigma_{t}^2) \\
&= \mathrm{N}\left( x_t \mid \lambda_{t|0} x_0, \sigma_{t|0}^2 \right),
\end{align*}
as desired.

The first latent is distributed as $q(x_1 \mid x_0) = \mathrm{N}(x_1 \mid \lambda_1 x_0, \sigma_1^2)$, which matches the base case above.

:::

::::

### Variance preserving diffusions

It is common to set,
\begin{align*}
\sigma_t^2 &= 1-\lambda_t^2,
\end{align*}
in which case the conditional variance simplifies to,
\begin{align*}
\sigma_{t|0}^2 = 1 - \prod_{s=1}^t \lambda_s^2 = 1 - \lambda_{t|0}^2.
\end{align*}
Under this setting, the noising process preserves the variance of the marginal distributions. If $\bbE[x_0] = 0$ and $\Var[x_0] = 1$, then the marginal distribution of $x_t$ will be zero mean and unit variance as well.

Consider the following two limits:
1. As $T \to \infty$, the conditional distribution goes to a standard normal, $q(x_T \mid x_0) \to \mathrm{N}(0, 1)$, which makes the marginal distribution $q(x_T)$ easy to sample from. 

2. When $\lambda_t \to 1$, the noising process adds infinitesimal noise so that $x_t \approx x_{t-1}$, which makes the inverse process easier to learn. 

Of course, these two limits are in conflict with one another. If we add a small amount of noise at each time step, the inverse process is easier to learn, but we need to take many time steps to converge to a Gaussian stationary distribution.

## Generative process

The generative process is a parameteric model that learns to invert the noise process,
\begin{align*}
p(x_{0:T}; \theta) &= p(x_T) \prod_{t=T-1}^0 p(x_t \mid x_{t+1}; \theta).
\end{align*}
The initial distribution $p(x_T)$ has no parameters because it is set to the stationary distribution of the noising process, $q(x_\infty)$. E.g., for the Gaussian noising process above, $p(x_T) = \mathrm{N}(0,1)$.

## Evidence Lower Bound
Like the other latent variable models we studied in this course, we will estimate the parameters by maximizing an **evidence lower bound (ELBO)**,
\begin{align*}
\cL(\theta) 
&= \E_{q(x_0)} \E_{q(x_{1:T} \mid x_0)} \left[ \log p(x_{0:T}; \theta) - \log q(x_{1:T} \mid x_0) \right] \\
&= \E_{q(x_0)} \E_{q(x_{1:T} \mid x_0)} \left[ \log p(x_{0:T}; \theta)  \right] + c,
\end{align*}
where $q(x_{1:T} \mid x_0)$ is the conditional distibution of $x_{1:T}$ under the noising process.
Since there are no learnable parameters in the noising process, the objective simplifies to **maximizing the expected log likelihood**.

We can simplify further by expanding the log probability of the generative model,
\begin{align*}
\cL(\theta)
&= \E_{q(x_0)} \sum_{t=0}^{T-1} \E_{q(x_t, x_{t+1} \mid x_0)} \left[  \log p(x_t \mid x_{t+1}; \theta) \right] \\
&\propto \E_{q(x_0)} \mathbb{E}_{t \sim \mathrm{Unif}(0,T-1)} \E_{q(x_t, x_{t+1} \mid x_0)} \left[ \log p(x_t \mid x_{t+1}; \theta) \right]
\end{align*}
which only depends on pairwise conditionals.

## Gaussian generative process

Since the noising process above adds a small amount of Gaussian noise at each step, it is reasonable to model the generative process as Gaussian as well,
\begin{align*}
p(x_t \mid x_{t+1}; \theta) 
&= 
\mathrm{N}(x_t \mid \mu_\theta(x_{t+1}, t), \widetilde{\sigma}_t^2)
\end{align*}
where $\mu_\theta: \reals \times [0,T] \mapsto \reals$ is a nonlinear **mean function** that should **denoise** $x_{t+1}$ to obtain the expected value of $x_t$, and $\widetilde{\sigma}_t^2$ is a fixed variance for the generative process.

:::{admonition} Parameter sharing
Rather than learn a separate function for each time point, it is common to parameterize the mean function as a function of both the state $x_{t+1}$ and the time $t$. For example, $\mu_\theta(\cdot, \cdot)$ can be a neural network that takes in the state and a positional embedding of the time $t$, like the sinusoidal embeddings used in transformers.
:::

:::{admonition} Generative process variance
You could try to learn the generative process variance as a function of $x_{t+1}$ and $t$ as well, but the literature suggests this is difficult to make work in practice. Instead, is common to set the variance to either 
- $\widetilde{\sigma}_t^2 = \sigma_t^2 = 1-\lambda_t^2$, the conditional variance in the noising process, which tends to _overestimate_ the conditional variance of the true generative process; or
- $\widetilde{\sigma}_t^2 = \Var_q[x_t \mid x_0, x_{t+1}]$, the conditional variance of the noising process _given_ the data $x_0$ and the next state $x_{t+1}$. This tends to _underestimate_ the conditional variance of the true generative process.
:::

## Rao-Blackwellization

Under this Gaussian model for the generative process, we can analytically compute one of the expectations in the ELBO. This is called Rao-Blackwellization. It reduces the variance of the objective, which is good for SGD!

Using the chain rule and the Gaussian generative model, we have,
\begin{align*}
\E_{q(x_t, x_{t+1} \mid x_0)} \left[ \log p(x_t \mid x_{t+1}; \theta) \right] 
&= \E_{q(x_{t+1} \mid x_0)} \E_{q(x_t \mid x_{t+1}, x_0)} \left[\log \mathrm{N}(x_t \mid \mu_\theta(x_{t+1}, t), \widetilde{\sigma}_t^2) \right]
\end{align*}

We already computed the conditional distribution $q(x_{t+1} \mid x_0) = \mathrm{N}(x_{t+1} \mid \lambda_{t+1|0} x_0, \sigma_{t+1|0}^2)$ above. It turns out the second term is Gaussian as well!

::::{admonition} Conditionals of a Gaussian noising process

Show that 
\begin{align*}
q(x_t \mid x_{t+1}, x_0) 
&= \mathrm{N}(x_t \mid \mu_{t|t+1,0}, \sigma_{t|t+1,0}^2)
\end{align*}
where
\begin{align*}
\mu_{t|t+1,0} &= a_t x_0 + b_t x_{t+1}
\end{align*}
is a **linear combination** of $x_0$ and $x_{t+1}$ with weights,
\begin{align*}
a_t &= \frac{\sigma_{t|t+1,0}^2 \, \lambda_{t|0} }{\sigma_{t|0}^2}  \\
b_t &= \frac{\sigma_{t|t+1,0}^2 \, \lambda_{t+1}}{\sigma_{t+1}^2} \\
\sigma_{t|t+1,0}^2  
&= \left(\frac{1}{\sigma_{t|0}^2} + \frac{\lambda_{t+1}^2}{\sigma_{t+1}^2} \right)^{-1} 
\end{align*}

<!-- = \sigma_{t|t+1,0}^2 \frac{\sqrt{1 - \sigma_{t|0}^2}}{\sigma_{t|0}^2}\\
= \sigma_{t|t+1,0}^2 \frac{\sqrt{1 - \sigma_{t+1}^2}}{\sigma_{t+1}^2} \\ -->


:::{admonition} Solution
:class: dropdown, tip
By Bayes rule and the Markovian nature of the noising process,
\begin{align*}
q(x_t \mid x_{t+1}, x_0) 
&\propto q(x_t \mid x_0) \, q(x_{t+1} \mid x_t) \\
&= \mathrm{N}(x_t \mid \lambda_{t|0} x_0, \sigma_{t|0}^2) \, \mathrm{N}(x_{t+1} \mid \lambda_{t+1} x_t, \sigma_{t+1}^2) \\
&= \mathrm{N}(x_t \mid \mu_{t|t+1,0}, \sigma_{t|t+1,0}^2) 
\end{align*}
where, by completing the square,
\begin{align*}
\sigma_{t|t+1,0}^2 &= \left(\frac{1}{\sigma_{t|0}^2} + \frac{\lambda_{t+1}^2}{\sigma_{t+1}^2} \right)^{-1} \\
\mu_{t|t+1,0} &= \sigma_{t|t+1,0}^2 \left(\frac{\lambda_{t|0} x_0}{\sigma_{t|0}^2} + \frac{\lambda_{t+1} x_{t+1}}{\sigma_{t+1}^2} \right).
\end{align*}
The forms for $a_t$ and $b_t$ can now be read off.
:::

::::

Finally, to simplify the objective we need the Gaussian cross-entropy,

::::{admonition} Gaussian cross-entropy

Let $q(x) = \mathrm{N}(x \mid \mu_q, \sigma_q^2)$ and $p(x) = \mathrm{N}(x \mid \mu_p, \sigma_p^2)$. Show that,
\begin{align*}
\E_{q(x)}[\log p(x)] &= \log \mathrm{N}(\mu_q \mid \mu_p, \sigma_p^2) -\frac{1}{2} \frac{\sigma_q^2}{\sigma_p^2}
\end{align*}
::::

Putting it all together, 
\begin{align*}
\cL(\theta) 
&= \E_{q(x_0)} \E_t \E_{q(x_{t+1} \mid x_0)} \E_{q(x_t | x_0, x_{t+1})} \left[ \log p(x_t \mid x_{t+1}; \theta) \right] \\
&= \E_{q(x_0)} \E_t \E_{q(x_{t+1} \mid x_0)} \left[ \log \mathrm{N}(a_t x_0 + b_t x_{t+1} \mid \mu_\theta(x_{t+1}, t), \widetilde{\sigma}_t^2) -\frac{1}{2} \frac{\sigma_{t|t+1,0}^2}{\widetilde{\sigma}_t^2} \right] \\
&= \frac{1}{2} \E_{q(x_0)} \E_t \E_{q(x_{t+1} \mid x_0)} \left[ \frac{1}{\widetilde{\sigma}_t^2} \left( a_t x_0 + b_t x_{t+1} - \mu_\theta(x_{t+1}, t) \right)^2 \right] + c
\end{align*}
where we have absorbed terms that are independent of $\theta$ into the constant $c$.

## Denoising mean function

The loss function above suggests a particular form of the mean function,
\begin{align*}
\mu_\theta(x_{t+1}, t) &= a_t \hat{x}_0(x_{t+1}, t; \theta) + b_t x_{t+1},
\end{align*}
where the only part that is learned is $\hat{x}_0(x_{t+1}, t; \theta)$, a function that attempts to **denoise** the current state. Since $x_{t+1}$ is given and $a_t$ and $b_t$ are determined solely by the hyperparameters, we can use them in the mean function.

Under this parameterization, the loss function reduces to,
\begin{align*}
\cL(\theta) 
&= \frac{1}{2} \E_{q(x_0)} \E_t \E_{q(x_{t+1} \mid x_0)} \left[ \frac{a_t^2}{\widetilde{\sigma}_t^2} \left(x_0 - \hat{x}_0(x_{t+1}, t; \theta) \right)^2 \right] + c
\end{align*}

One nice thing about this formulation is that the mean function is always outputting "the same thing" &mdash; an estimate of the completely denoised data, $\hat{x}_0$, regardless of the time $t$. 
<!-- 
## _Noise predicting_ mean function

Another way to configure the mean function is to predict the noise in $x_{t+1}$. 
We can reparameterize the conditional distribution of $x_{t+1}$ as,
\begin{align*}
x_{t+1} &= \lambda_{t+1|0} x_{0} + \sigma_{t+1|0} \epsilon_{t+1}, \\
\epsilon_{t+1} &\sim \mathrm{N}(0, 1)
\end{align*}
In terms of $x_0$, 
\begin{align*}
x_{0} &= \frac{x_{t+1} - \sigma_{t+1|0} \epsilon_{t+1}}{\lambda_{t+1|0} }.
\end{align*} 

Now, we can write the conditional mean of $x_t$ as,
\begin{align*}
\mu_{t|t+1,0} 
&= a_t x_0 + b_t x_{t+1} \\
&= c_t x_{t+1} - d_t \epsilon_{t+1}
\end{align*}
where
\begin{align*}
c_t &= \frac{a_t}{\lambda_{t+1|0}} + b_t,  &&& 
d_t &= \frac{\sigma_{t+1|0}}{\lambda_{t+1|0}}.
\end{align*}
This suggests yet another parameterization of the mean function,
\begin{align*}
\mu_\theta(x_{t+1}, t) 
&= c_t  x_{t+1} - d_t \hat{\epsilon}(x_{t+1}, t; \theta),
\end{align*}
where $\hat{\epsilon}$ is a function that attempts to **predict the noise** that generated $x_{t+1}$ from $x_0$.

Under this parameterization, the loss function reduces to,
\begin{align*}
\cL(\theta) 
&= \frac{1}{2} \E_{q(x_0)} \E_t \E_{q(\epsilon)} \left[ \frac{d_t^2}{\widetilde{\sigma}_t^2} \left(\epsilon - \hat{\epsilon}( \lambda_{t+1|0} x_{0} + \sigma_{t+1|0} \epsilon, t; \theta) \right)^2 \right],
\end{align*}

This approach has connections to **denoising score matching** (Song and Ermon, 2019), in which a neural network is trained to estimate the (Stein) score of a kernel density estimate of the data distribution. The score function turns out to be linearly related to the noise estimated in this mean parameterization.

:::{admonition} Stein score vs Fisher score
Some people call the gradient of the log density with respect to the parameters, $\nabla_\theta \log p(x; \theta)$, the _Fisher_ score, and the gradient with respect to the data the _Stein_ score $\nabla_x \log p(x; \theta)$.
::: -->


<!-- ### Connection to denoising score matching

Score matching is another approach for density estimation that seeks to estimate the (Stein) score function of an (unnormalized) density, $\nabla_x \log p(x)$. One way of approaching this problem is to train a neural network to output an estimate of the score of a kernel density estimate 
\begin{align*}
s_\theta(x) \approx \nabla_x \log q(x)
\end{align*} -->

## Inverting the noising process

The generative process attempts to invert the noising process, but what is the actual inverse of the process? Since the noising process is a Markov chain, the reverse of the noising process must be Markovian as well. That is,
\begin{align*}
q(x_{0:T}) &= q(x_T) \prod_{t=T-1}^0 q(x_t \mid x_{t+1})
\end{align*}
for some sequence of transition distributions $q(x_t \mid x_{t+1})$. We can obtain those transition distributions by marginalizing and conditioning,
\begin{align*}
q(x_t \mid x_{t+1}) 
&= \int q(x_0, x_t \mid x_{t+1}) \dif x_0 \\
&= \int q(x_t \mid x_0, x_{t+1}) \, q(x_0 \mid x_{t+1}) \dif x_0.
\end{align*}
Using Bayes' rule,
\begin{align*}
q(x_0 \mid x_{t+1}) 
&= \frac{q(x_0) \, q(x_{t+1} \mid x_0)}{\int q(x_0') q(x_{t+1} \mid x_0') \dif x_0'} 
\end{align*}
Now recall that $q(x_0) = \frac{1}{n} \sum_{i=1}^n \delta_{x_0^{(i)}}(x_0)$ is the empirical measure of the data $\{x_0^{(i)}\}_{i=1}^n$. Using this fact, the conditional is,
\begin{align*}
q(x_0^{(i)} \mid x_{t+1}) 
&= \frac{q(x_{t+1} \mid x_0^{(i)})}{\sum_{j=1}^n q(x_{t+1} \mid x_0^{(j)})} 
\triangleq w_i(x_{t+1}),
\end{align*}
where we have defined the weights $w_i(x_{t+1})$ for each data point $i=1,\ldots,n$. They are non-negative and sum to one.

Finally, we can give a simpler form for the optimal generative process,
\begin{align*}
q(x_t \mid x_{t+1}) 
&= \sum_{i=1}^n w_i(x_{t+1}) \, q(x_t \mid x_0^{(i)}, x_{t+1}) \\
&= \sum_{i=1}^n w_i(x_{t+1}) \, \mathrm{N}(x_t \mid a_t x_0^{(i)} + b_t x_{t+1}, \sigma_{t|t+1,0}^2),
\end{align*}
which we recognize as a mixture of Gaussians, all with the same variance, with means biased toward each of the $n$ data points, and weighted by the relative likelihood of $x_0^{(i)}$ having produced $x_{t+1}$.

For small step sizes, that mixture of Gaussians can be approximated by a single Gaussian with mean equal to the expected value of the mixture,
\begin{align*}
\E[x_t \mid x_{t+1}]
&= \sum_{i=1}^n w_i(x_{t+1}) \, \left(a_t x_0^{(i)} + b_t x_{t+1} \right) 
\end{align*}
For small steps, this expected value is approximately,
\begin{align*}
\E[x_t \mid x_{t+1}]
&\approx \frac{x_{t+1}}{\lambda_{t+1}} + \sigma_{t+1}^2 \sum_{i=1}^n w_i(x_{t+1}) \left(\frac{ \lambda_{t|0} x_0^{(i)} - x_{t+1}}{\sigma_{t|0}^2} \right)
\end{align*}

:::{admonition} Derivation
:class: dropdown
Using the definitions from above,
\begin{align*}
a_t &= \frac{\sigma_{t|t+1,0}^2 \, \lambda_{t|0} }{\sigma_{t|0}^2}  \\
b_t &= \frac{\sigma_{t|t+1,0}^2 \, \lambda_{t+1}}{\sigma_{t+1}^2} 
\end{align*}
we can write
\begin{align*}
\E[x_t \mid x_{t+1}]
&= \frac{\sigma_{t|t+1,0}^2  \lambda_{t+1}}{\sigma_{t+1}^2} x_{t+1} + \sum_{i=1}^n w_i(x_{t+1}) \frac{\sigma_{t|t+1,0}^2 \lambda_{t|0}}{\sigma_{t|0}^2} x_0^{(i)} 
\end{align*}
Now add and subtract $\frac{\sigma_{t|t+1,0}^2 x_{t+1}}{\sigma_{t|0}^2}$  to obtain,
\begin{align*}
\E[x_t \mid x_{t+1}]
&= \left(\frac{\sigma_{t|t+1,0}^2 \lambda_{t+1}}{\sigma_{t+1}^2} + \frac{\sigma_{t|t+1,0}^2}{\sigma_{t|0}^2}\right) x_{t+1} + \sigma_{t|t+1,0}^2 \sum_{i=1}^n w_i(x_{t+1}) \left(\frac{ \lambda_{t|0} x_0^{(i)} - x_{t+1}}{\sigma_{t|0}^2} \right).
\end{align*}
In the limit of many small steps where $\sigma_{t+1} \to 0$ so that $\sigma^2_{t+1} \ll \sigma^2_{t|0}$, we have 
\begin{align*}
\frac{\sigma_{t|t+1,0}^2 \lambda_{t+1}}{\sigma_{t+1}^2} + \frac{\sigma_{t|t+1,0}^2}{\sigma_{t|0}^2}
\approx \frac{1}{\lambda_{t+1}}
\end{align*}
and
\begin{align*}
\sigma_{t|t+1,0}^2 \approx \sigma_{t+1}^2.
\end{align*}
These approximations complete the derivation.
:::

Though it's not immediately obvious, the second term in the expectation is related to the **marginal probability**,
\begin{align*}
q(x_t) 
&= \frac{1}{n} \sum_{i=1}^n q(x_t \mid x_0^{(i)}) \\
&= \frac{1}{n} \sum_{i=1}^n \mathrm{N}(x_t \mid \lambda_{t|0} x_0^{(i)}, \sigma_{t|0}^2) 
\end{align*}
Specifically, the second term is the **Stein score function** of the marginal probability,
\begin{align*}
\nabla_{x} \log q_t(x_{t+1})
&= \frac{\nabla_{x} q_t(x_{t+1})}{q_t(x_{t+1})}  \\
&=\frac{\frac{1}{n} \sum_{i=1}^n \mathrm{N}(x_{t+1} \mid \lambda_{t|0} x_0^{(i)}, \sigma_{t|0}^2) \left(-\frac{(x_{t+1} - \lambda_{t|0}x_0^{(i)})}{\sigma_{t|0}^2} \right)}{\frac{1}{n} \sum_{j=1}^n \mathrm{N}(x_{t+1} \mid \lambda_{t|0} x_0^{(j)}, \sigma_{t|0}^2) } \\
&= \sum_{i=1}^n w_i(x_{t+1}) \left(\frac{\lambda_{t|0}x_0^{(i)} - x_{t+1}}{\sigma_{t|0}^2} \right)
\end{align*}

:::{admonition} Final form
:class: tip
Putting it all together, for small steps, the reverse process is approximately Gaussian with mean and variance,
\begin{align*}
\E[x_t \mid x_{t+1}]
&\approx \frac{x_{t+1}}{\lambda_{t+1}} + \sigma_{t+1}^2 \nabla_x \log q_t(x_{t+1})  \\
\Var[x_t \mid x_{t+1}] &\approx \sigma_{t+1}^2.
\end{align*}
This has a nice interpretation: to invert the noise process, **first undo the contraction and then take a step in the direction of the Stein score!**
:::


## Continuous time limit 

In practice, the best performing diffusion models are based on a continuous-time formulation of the noising process as an SDE [@song2020score]. To motivate this approach, think of the noise process above as a discretization of a continuous process $x(t)$ for $t \in [0,1]$ with time steps of size $\Delta = \tfrac{1}{T}$. That is, map $x_i \mapsto x(i/T)$, $\lambda_i \mapsto \lambda(i/T)$, and $\sigma_i \mapsto \sigma(i/T)$  for $i=0,1,\ldots, T$. Then the discrete model is can be rewritten as,
\begin{align*}
x(t + \Delta) 
&\sim \mathrm{N}(\lambda(t) x(t), \sigma(t)^2),
\end{align*}
or equivalently,
\begin{align*}
x(t + \Delta) - x(t) 
&\sim \mathrm{N}\left(f(x(t), t) \Delta, g(t)^2 \Delta \right)  \\
f(x, t) &= \frac{1 - \lambda(t)}{\Delta} x \\
g(t) &= \frac{\sigma(t)}{\Delta}.
\end{align*}
We can view this as a discretization of the SDE,
\begin{align*}
\dif X &= f(x, t) \dif t + g(t) \dif W
\end{align*}
where $f(x,t)$ is the **drift** term, $g(t)$ is the **diffusion** term, and $\dif W$ is the **Brownian motion**.

The reverse (generative) process can be cast as an SDE as well! Following our derivation of the inverse process above, we can show that the reverse process is,
\begin{align*}
\dif X = \left[f(x, t) - g(t)^2 \nabla_x \log q_t(x)\right] \dif t + g(t) \dif W
\end{align*}
where $\dif t$ is a **negative** time increment and $\dif W$ is Brownian motion run in reverse time.

## Multidimensional models

Very few things need to change in order to apply this idea to multidimensional data $\mbx_0 \in \reals^D$. The standard setup is to apply a Gaussian noising process to each coordinate $x_{0,d}$ independently. Then, in the generative model,
\begin{align*}
p(\mbx_t \mid \mbx_{t+1}; \theta)
&= \prod_{d=1}^D p(x_{t,d} \mid \mbx_{t+1}; \theta) \\
&= \prod_{d=1}^D \mathrm{N}(x_{t,d} \mid \mu_\theta(\mbx_{t+1}, t, d), \widetilde{\sigma}_t^2). 
\end{align*}
The generative process still produces a factored distribution, but we need a separate mean function for each coordinate. Moreover, the mean function needs to consider the entire state $\mbx_{t+1}$. The reason is that $x_{t,d}$ is not conditionally independent of $x_{t+1,d'}$ given $x_{t+1,d}$; the coordinates are coupled in the inverse process since all of $\mbx_{t+1}$ provides information about the $\mbx_0$ that generated it.

## Conclusion

Denoising diffusion probabilistic models frame generative modeling as learning to invert a fixed, analytically tractable noising process. The key insight is that the optimal reverse transition is a mixture of Gaussians whose mean is a linear combination of the data and the noisy state, and that learning to denoise is equivalent to learning to generate. In the continuous-time limit, the reverse process is an SDE driven by the score function of the marginal density — a connection that unifies DDPMs with score-based generative modeling and Langevin dynamics. There is much more to explore: conditional generation (steering the reverse diffusion with text prompts), discrete diffusion models, and connections between the score function and denoising score matching.

:::{admonition} Next Steps
:class: seealso
This chapter completes Part II. The continuous-time SDE formulation of diffusion is developed formally in:
- [Stochastic Differential Equations](sdes) — provides the mathematical framework for the reverse-time SDE and score-based generation
- [Discrete Diffusion Models](discrete-diffusion) — extends the DDPM framework to discrete state spaces such as language and protein sequences
:::

:::{admonition} Recommended Reading
:class: reading
[@ho2020denoising], "Denoising Diffusion Probabilistic Models."
[@song2020score], "Score-Based Generative Modeling through Stochastic Differential Equations."
:::


