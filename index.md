# Overview
Instructor: Scott Linderman <br>
TA: Aymen Echarghaoui <br>
Term: Spring 2026 <br>
Stanford University

---

## Course Description
This course will teach you how to do applied statistics research. We will follow Box's Loop: an iterative approach of asking a scientific question, collecting data to answer that question, building a model, performing statistical inference, and then criticizing and revising the model in light of your findings. We will develop the statistical tools to carry out this process: multivariate Gaussian models, graphical models, MCMC, variational Bayesian inference, latent variable models, state space models, Transformers, diffusion models, and more. We will practice this process through an extended research project and a flipped classroom with biweekly in-class lab meetings.

## Prerequisites
Students should be comfortable with basic probability and statistics as well as multivariate calculus and linear algebra. 

## Logistics
- Time: Monday and Wednesday, 1:30-2:50pm in Room [60-109](http://campus-map.stanford.edu/?srch=60-109)
- Level: advanced undergrad and up
- Grading basis: credit or letter grade
- Office hours:
  - Scott: Wed 3:00-4:00pm in CoDA E258
  - Aymen: Fri 3:00-5:00pm in Sequoia 207 (Bowker Room)

We will alternate between **traditional lectures** on odd-numbered weeks and **lab meetings** on even-numbered weeks. The lab meetings will be a flipped classroom -- each project team will present their deliverable for that week. See the next section and the [Course Project](project.md) for more detail.

## Assignments

### Project Deliverables
There will be project deliverables due every two weeks on **Sunday night at 11:59pm**. We will have lab meetings the week following the deliverables in which each project team will briefly present their progress. Each team will be assigned to either the Monday or the Wednesday group, and they should only attend the lab meeting they are assigned to. See the [Course Project](project.md) page for more detail.

### Math Problems
Additionally, there will be one math problem assigned each week, due the following **Wednesday night at 11:59pm**. These problems will help you test your reasoning abilities and, if you're a PhD student, prepare for quals.

## Schedule

The course will teach you the skills necessary to follow Box's Loop: formulate a problem, collect data, build a model, perform inference, criticize, revise, repeat. Topics from Parts I–III (Foundations, Latent Variable Models, Inference Algorithms) are interleaved so that each new model is paired with the inference tools needed to fit it. Part IV (Sequence Models) occupies the final few weeks. We won't cover Part V (Stochastic Processes) this quarter, except in passing, but some chapters will reference that material in case you want to dig deeper on your own.

As described above, the course alternates between traditional lectures and lab meetings, where we will flip the classroom. You will be assigned to either the Monday or the Wednesday lab meeting; you should not attend both. During lab meetings, you will give a short (1 slide, 3 minute) presentation of your deliverable, and you will give feedback to others.

Project deliverable due dates are marked below.

| Date | Topic | Reading |
| ---- | ----- | ------- |
| Mar 30 | **Attend:** Course overview | Ch 1.1 |
| Apr 1  | **Attend:** The (Multivariate) Normal Distribution | Ch 1.2–1.3 |
| Apr 5  | **Deliverable 1 Due** (1pg report per person) | | 
| Apr 6  | **Watch:** Mixture Models <br> **Attend:** Lab Meeting (Monday Teams) | Ch 2.1 |
| Apr 8  | **Watch:** Expectation Maximization <br> **Attend:** Lab Meeting (Wednesday Teams) | Ch 3.3  |
| Apr 10 | **Required:** You must find your teammate by this date. |  |
| Apr 13 | **Attend:** Hierarchical Models | Ch 1.5 |
| Apr 15 | **Attend:** Markov Chain Monte Carlo | Ch 3.1 |
| Apr 19 | **Deliverable 2 Due** (2pg report per team)| |
| Apr 20 | **Watch:** Probabilistic PCA <br> **Attend:** Lab Meeting (Monday Teams) | Ch 2.3 |
| Apr 22 | **Watch:** Variational Autoencoders <br> **Attend:** Lab Meeting (Wednesday Teams) | Ch 2.4 | 
| Apr 27 | **Attend:** Hidden Markov Models | Ch 4.1 |
| Apr 29 | **Attend:** Linear Dynamical Systems | Ch 4.2 |
| May 3  | **Deliverable 3 Due** (2pg report per team)| | 
| May 4  | **Watch:** Gaussian Processes <br> **Attend:** Lab Meeting (Monday Teams) | Ch 5.1 |
| May 6  | **Watch:** Poisson Processes <br> **Attend:** Lab Meeting (Wednesday Teams) | Ch 5.3 | 
| May 11 | **Attend:** Transformers | Ch 4.4 |
| May 13 | **Attend:** Linear Attention and Deep SSMs | Ch 4.5 |
| May 17 | **Deliverable 4 Due** (2pg report per team)| |
| May 18 | **Watch:** Parallelizing Nonlinear RNNs <br> **Attend:** Lab Meeting (Monday Teams) |   |
| May 20 | **Watch:** Continual Learning <br> **Attend:** Lab Meeting (Wednesday Teams) | Ch 3.7 |
| *May 25* | *No class — Memorial Day* | |
| May 27 | **Attend:** Diffusion Models and SDEs | Ch 2.5, 5.2 |
| Jun 1  | **Attend:** Project Presentations (All Teams) | |
| Jun 3  | **Attend:** Project Presentations (All Teams) | |
| Jun 8  | **Final Report Due** | |


## Grading

| Component | Weight |
|-----------|--------|
| Milestones (4 × 10%) | 40% |
| Final report | 35% |
| Lab meeting participation | 15% |
| Weekly math problems | 10% |

Each milestone is graded on an **A-F** scale, roughly following this rubric:

| Score | Meaning |
|-------|---------|
| A | Complete, thoughtful, and well-executed |
| B | Acceptable but missing key elements or depth |
| C | Did not take assignment seriously / just asked AI to do it |
| F | Not submitted or substantially incomplete |

We may assign half-letter grades too.

Math problems will be graded on a **(0, 1, 2)** scale.

Remember that project grades are based on the **quality and thoroughness of your applied statistics practice** — not on whether your model achieves impressive results. A project that honestly finds that a simple baseline outperforms a complex model, with a clear explanation of why, is an excellent project.


## Books
In addition to the lecture notes, you may find these textbooks helpful:

- Murphy. Probabilistic Machine Learning: Advanced Topics. MIT Press, 2023. [link](https://probml.github.io/pml-book/book2.html)
- Bishop. Pattern recognition and machine learning. New York: Springer, 2006. [link](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- Gelman et al. Bayesian Data Analysis. Chapman and Hall, 2005. [link](http://www.stat.columbia.edu/~gelman/book/)
