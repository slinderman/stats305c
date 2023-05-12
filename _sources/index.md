# Overview
Instructor: Scott Linderman <br>
TA: Xavier Gonzalez <br>
Term: Spring 2023 <br>
Stanford University

---

## Course Description
Probabilistic modeling and inference of multivariate data. Topics may include multivariate Gaussian models, probabilistic graphical models, MCMC and variational Bayesian inference, dimensionality reduction, principal components, factor analysis, matrix completion, topic modeling, and state space models. Extensive work with data involving Python programming using PyTorch.

## Prerequisites
Students should be comfortable with probability and statistics as well as multivariate calculus and linear algebra. This course will emphasize implementing models and algorithms, so coding proficiency is required.

## Logistics
- Time: Tuesday and Thursday, 10:30-11:50am
- Level: advanced undergrad and up
- Grading basis: credit or letter grade
- Office hours:
  - Weds 4:30-5:30pm in Wu Tsai Neurosciences Instiute Room M252G (Scott)
  - Thurs 5-7pm location Wu Tsai Neurosciences Instiute Room S275 (Xavier)
- Assignments released Friday, due the following Thursday at 11:59pm

## Books
We will primarily use
- Murphy. Probabilistic Machine Learning: Advanced Topics. MIT Press, 2023. [link](https://probml.github.io/pml-book/book2.html)

You may also find these texts helpful
- Bishop. Pattern recognition and machine learning. New York: Springer, 2006. [link](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- Gelman et al. Bayesian Data Analysis. Chapman and Hall, 2005. [link](http://www.stat.columbia.edu/~gelman/book/)
`
## Schedule

| Date   | Topic | Reading |
| ------ | ----- | ------- |
| Apr 4  | Bayesian Analysis of the Normal Distribution <br> {Download}`[slides]<slides/annotated/lecture01-bayes_normal.pdf>` [[notebook]](notebooks/01_bayes_normal.ipynb) | Ch 2.3, 3.4.3 |
| Apr 6  | Multivariate Normal Distribution <br> {Download}`[slides]<slides/annotated/lecture02-mvn.pdf>` [[notebook]](notebooks/02_mvn.ipynb)| Ch 2,3, 3.4.4 |
| Apr 11 | Probabilistic Graphical Models <br> {Download}`[slides]<slides/annotated/lecture03_pgms.pdf>` [[notebook]](notebooks/03_hier_gauss.ipynb) | Ch 3.6.2, 4.2 |
| Apr 13 | Markov Chain Monte Carlo <br> {Download}`[slides]<slides/annotated/lecture04-mcmc.pdf>` [[notebook]](notebooks/04_mcmc.ipynb) | Ch 11.1-11.2, 12.1-12.3 |
| Apr 18 | Probabilistic PCA and Factor Analysis <br> {Download}`[slides]<slides/annotated/lecture05-continuous_lvms.pdf>` | Ch 28.3 |
| Apr 20 | Hamiltonian Monte Carlo <br> {Download}`[slides]<slides/annotated/lecture06-hmc.pdf>` | [Neal, 2012](https://arxiv.org/abs/1206.1901), Ch 12.5 |
| Apr 25 | Mixture Models <br> {Download}`[slides]<slides/annotated/lecture07-mixtures.pdf>`  | Ch 28.2 |
| Apr 27 | Expectation Maximization <br> {Download}`[slides]<slides/lecture08-em.pdf>` | Ch 6.5 |
| May 2  | Coordinate Ascent Variational Inference <br> {Download}`[slides]<slides/lecture09-cavi.pdf>` [[notebook 1]](notebooks/09_cavi_gmm.ipynb) [[notebook 2]](notebooks/09_cavi_nix.ipynb) | [Blei et al, 2017](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773), Ch 10.1-10.3 |
| May 4  | Mixed Membership Models <br> {Download}`[slides]<slides/lecture10_mixed_membership.pdf>` [[notebook]](notebooks/10_cavi_lda.ipynb) |  Ch 28.5|
| May 9  | Gradient-Based Variational Inference <br> {Download}`[slides]<slides/lecture11-advi.pdf>` [[notebook]](notebooks/11_advi_nix.ipynb) | Ch 21|
| May 11 | Variational Autoencoders <br> {Download}`[slides]<slides/lecture12-vaes.pdf>` | [Kingma and Welling, 2019](https://arxiv.org/pdf/1906.02691.pdf), Ch 10.2|
| May 16 | Hidden Markov Models <br> {Download}`[slides]<slides/lecture13_hmms.pdf>` | Ch 29.2-29.5 |
| May 18 | Linear Dynamical Systems <br> {Download}`[slides]<slides/lecture14-lds.pdf>` | Ch 29.6-12.8 |
| May 23 | Gaussian Processes <br> {Download}`[slides]<slides/lecture15-gps.pdf>`| Ch 18 |
| May 25 | Poisson Processes <br> {Download}`[slides]<slides/lecture16-poisson_processes.pdf>` | |
| May 30 | Stochastic Differential Equations <br> {Download}`[slides]<slides/lecture17-sdes.pdf>` | |
| June 1 | Dirichlet Process Mixture Models <br> {Download}`[slides]<slides/lecture18-dpmm.pdf>`| |
| June 6 | Wrapping Up <br> {Download}`[slides]<slides/lecture19-wrap_up.pdf>`| |