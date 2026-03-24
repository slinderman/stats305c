# Course Project

Statistics is fundamentally an applied field. The methods and algorithms we study—MCMC, variational inference, Gaussian processes, hidden Markov models—exist to help scientists and practitioners reason rigorously about real-world phenomena. This course is organized around a **longitudinal project** in which you apply the tools of probabilistic machine learning to a real problem of your choosing, following the iterative cycle of applied statistics known as **Box's Loop**:

> *Formulate a problem → Collect data → Build a model → Perform inference → Criticize the model → Revise*

Rather than a collection of independent homework assignments, your project evolves throughout the quarter. Each deliverable advances your project by one stage of this loop, and the final report synthesizes everything you have learned. The goal is not to produce a publishable paper or a state-of-the-art result—it is to practice the full craft of applied statistics with honesty, rigor, and curiosity.

---

## Teams

You may work **individually or in teams of two**. Teams of two are encouraged; collaboration mirrors real research practice and helps distribute the substantial workload of a longitudinal project.

If you begin the quarter with a partner and your partner drops the course, you may either find a new partner (subject to instructor approval) or continue individually. If you continue individually, reasonable scope adjustments will be made to expectations.

---

## Choosing a Problem

The most important—and most difficult—part of the project is choosing a good problem. A good project problem has the following properties:

1. **Real and important.** The problem matters to some community (scientific, social, clinical, economic). You should be able to articulate clearly *why* someone would care about the answer.

2. **Data-driven.** There is real data available, or you can collect it. The data should be complex enough that simple inspection cannot answer the question on its own.

3. **Tractable for probabilistic modeling.** Uncertainty is intrinsic to the problem, and a probabilistic model can help structure it. Bayesian inference, latent variables, or a hierarchical structure should add genuine value.

4. **Appropriately scoped.** It is better to do one thing deeply than many things superficially. A focused analysis of a well-chosen dataset will score higher than an ambitious but shallow survey.

You do **not** need to invent a new problem. You may:
- Replicate and extend a published applied study (a paper from any empirical science that uses data and statistics).
- Apply course methods to a dataset from your own research.
- Explore a publicly available dataset on a topic you care about.

If you replicate an existing paper, you must go beyond simple replication—probe the assumptions, compare alternative models, or extend the analysis in a meaningful direction.

**What to avoid.** Purely methodological projects ("we compare five VI algorithms on synthetic data") are not appropriate for this course. The project must be grounded in a real applied problem; methods are the tools, not the subject.

---

## Deliverables

There are four deliverables and a final report. Each deliverable is a short written document (submitted as a PDF or Jupyter notebook) that advances your project by one stage of Box's Loop. Deliverables are designed to be **cumulative**: each one builds directly on the previous.

:::{note}
Approximate due dates are listed below. Exact dates will appear on the course schedule. Deliverables are typically due **Monday at 11:59 pm**.
:::

### Deliverable 1 — Problem Formulation (≈ Week 2)

*Corresponding to: "Formulate a problem"*

Describe the problem you intend to study. Your submission should address:

- **Scientific or applied question.** What question are you trying to answer? Why is it interesting or important?
- **Data source.** What data exists (or can be collected) to address this question? Include a brief description of the data format, size, and any known quality issues.
- **Statistical framing.** How can probabilistic modeling help? What are the key sources of uncertainty? What would a reasonable model look like at a high level?
- **Related work.** Cite 2–3 relevant papers or analyses. If you are replicating a paper, identify it here.

**Length:** approximately 1–2 pages.

**Pivot policy.** After receiving feedback on Deliverable 1, you may revise your problem formulation before proceeding to Deliverable 2. If the teaching staff identifies a significant issue with your problem (insufficient data, poor statistical fit, scope too large), we will flag it and allow a pivot. After Deliverable 2, pivots require instructor approval.

---

### Deliverable 2 — Data and Exploratory Analysis (≈ Week 4)

*Corresponding to: "Collect data"*

Obtain and explore your data. Your submission should include:

- **Data description.** Document the provenance, format, and preprocessing steps. Include summary statistics and any data cleaning decisions.
- **Exploratory data analysis (EDA).** Visualize the data. Describe patterns, anomalies, correlations, and distributional properties. What do you observe? What remains unexplained?
- **Refined model sketch.** Based on your EDA, refine your description of the model you plan to build. What distributional assumptions seem reasonable? What latent structure might be present?

**Length:** 2–3 pages plus figures. Jupyter notebooks with embedded narrative are acceptable.

---

### Deliverable 3 — Model and Inference (≈ Week 6)

*Corresponding to: "Build a model" and "Perform inference"*

Implement a probabilistic model and fit it to your data. Your submission should include:

- **Model specification.** Write out the full generative model (prior and likelihood). Justify your choices.
- **Inference algorithm.** Implement at least one inference method covered in the course (MCMC, variational inference, EM, etc.). Justify your choice. Describe convergence diagnostics.
- **Posterior analysis.** Summarize and visualize the posterior. What do the posterior distributions over parameters or latent variables tell you about the problem?
- **Baseline comparison.** Compare your probabilistic model to a simpler baseline (e.g., linear regression, PCA, a point estimate). Does the added complexity help? Be honest.

**Length:** 3–4 pages plus figures and code.

---

### Deliverable 4 — Model Criticism and Revision (≈ Week 8)

*Corresponding to: "Criticize the model" and "Revise"*

Critically evaluate your model. Your submission should include:

- **Posterior predictive checks.** Does the model generate data that looks like your real data? Identify specific failures.
- **Sensitivity analysis.** How sensitive are your conclusions to prior choices or model assumptions?
- **Revision.** Based on your criticism, make at least one substantive improvement to the model or inference procedure. This could be a revised likelihood, a different prior, an alternative algorithm, or a different model family.
- **Comparison.** Report results for both the original and revised models. Which is better, and by what criterion?

**Length:** 3–4 pages plus figures and code.

---

### Final Report (≈ Finals Week)

The final report is a self-contained research-style writeup of your full project. It should synthesize all four deliverables into a coherent document. Required sections:

1. **Introduction.** Motivate the problem and summarize your findings.
2. **Data.** Describe your dataset and EDA.
3. **Model.** Specify your probabilistic model.
4. **Inference.** Describe your inference algorithm and report diagnostics.
5. **Results.** Summarize and interpret the posterior. Address your scientific question.
6. **Discussion.** Reflect on what worked, what didn't, and what you would do differently. Be honest about limitations.
7. **Responsible AI Use** *(if applicable)*. See AI policy below.

Code must be fully open-source and linked from the report (GitHub repository).

**Length:** 6–10 pages (excluding references and appendices).

---

## Grading

Grades are based on the **quality and thoroughness of your applied statistics practice**—not on whether your model achieves impressive results. A project that honestly finds that a simple baseline outperforms a fancy probabilistic model, and provides a clear explanation of why, is an excellent project.

Rough grade expectations:

| Grade | Description |
|-------|-------------|
| A (94) | Clear problem, good data, well-specified model, correct inference, honest criticism and revision, coherent final report |
| A+ (100) | All of the above, plus unusually deep analysis, a surprising finding, or a novel methodological contribution motivated by the application |
| B (84) | Competent replication of an existing analysis with limited extension or original insight |
| C (74) | Incomplete deliverables, significant methodological errors, or shallow engagement with the problem |

Each deliverable is worth 15% of the course grade; the final report is worth 25%.

---

## AI Use Policy

The use of LLMs and AI coding assistants (ChatGPT, Claude, GitHub Copilot, etc.) is **permitted and in some cases encouraged**. These tools can accelerate data processing, help debug code, suggest modeling approaches, and assist with writing. Learning to work effectively with AI assistants is itself a valuable skill.

However, **if you use AI tools**, your final report must include a **Responsible AI Use** section that describes:

1. **What you used it for.** Be specific (e.g., "used Claude to generate boilerplate PyTorch code for the HMC sampler," "used ChatGPT to suggest modeling approaches for count data").
2. **How you verified the outputs.** What sanity checks did you perform? Did the suggested code run correctly? Did the modeling suggestions make sense given your data and domain knowledge?
3. **What the AI got wrong or missed.** Honest reflection on failures is required—this demonstrates that you were engaged, not passive.

The responsible use section is not a confession; it is evidence of critical thinking. A student who uses AI extensively but demonstrates careful verification and genuine understanding will be graded no differently than one who did not use AI.

---

## Guarding Against Passive AI Submission

A concern with open-ended projects is that students submit AI-generated work with minimal personal engagement. This course takes several steps to address this:

**Public GitHub repository with commit history.** Your code must be developed in a public GitHub repository, and you must share the repository link with each deliverable. The commit history provides evidence of iterative, incremental work. A repository with a single commit made the night before the deadline is a red flag. Teaching staff may inspect commit histories when grading.

**Problem diversity.** Because you choose your own problem and dataset in Deliverable 1, the class will naturally work on different problems. An AI-generated problem proposal is easy to spot when it is generic or mismatched to the data.

**Longitudinal consistency.** Each deliverable must build on the previous one. A submission that contradicts or ignores prior deliverables, or that shows a sudden jump in sophistication without explanation, will raise questions.

**Oral check-ins (optional but encouraged).** Office hours are a good place to discuss your project with the instructor or TA. We may ask clarifying questions about your modeling choices. Students who understand their own project will have no difficulty with this.

**Grading rewards depth over polish.** AI tools are very good at producing polished, fluent text. They are less good at the specific, grounded reasoning that comes from sustained engagement with a real dataset. Grades reward the latter.

---

## Getting Started

A few suggestions to help you choose a good problem:

- **Look at papers you already find interesting.** If you are in a research group, talk to your advisor about datasets that are available and questions that matter. If you are not in a research group, browse recent issues of journals in a field you care about (neuroscience, economics, epidemiology, ecology, etc.) and look for studies that used statistical methods.
- **Prioritize real data over synthetic.** Simulated data removes the messiness that makes applied statistics interesting and difficult.
- **Scope down early.** A narrowly focused analysis of one dataset is better than a broad comparison across many.
- **The answer can be "the simple method works fine."** Some of the most useful applied statistics papers conclude that a simple model fits the data well and complex extensions are not warranted. That is a valid and valuable conclusion.

If you are stuck, the teaching staff will help you brainstorm during office hours in the first week. Come with a general domain in mind, and we will work from there.
