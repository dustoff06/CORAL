# CORAL

**Constrained Oblique Rotation with Anchored Loadings for Fidelity-Constrained Decorrelation**

CORAL is a statistical transformation framework for reducing correlation among continuous variables while preserving their correspondence to designated source variables. Unlike PCA, which constructs orthogonal components without guaranteeing that an individual component remains interpretable as a particular original variable, CORAL explicitly constrains source fidelity while minimizing residual correlation.

The central question is:

> **How much correlation can be removed while requiring each transformed variable to remain recognizably anchored to its original source variable?**

CORAL formulates this as a constrained optimization problem and provides both theoretical and computational tools for examining the resulting fidelity-decorrelation frontier.

## Method

Let \(X \in \mathbb{R}^{n\times p}\) be a standardized data matrix with correlation matrix \(R\), and let

\[
\widetilde X = XT,
\]

where \(T=[t_1,\ldots,t_p]\) is the transformation matrix.

CORAL imposes unit variance on each transformed coordinate,

\[
t_j^\top R t_j = 1,
\]

and requires each transformed coordinate to retain a declared minimum correlation with its designated source variable,

\[
e_j^\top R t_j \geq \rho_{\min}.
\]

The primary CORAL objective minimizes aggregate squared residual correlation,

\[
D_2(T)=\sum_{i<j}(t_i^\top Rt_j)^2.
\]

Thus, \(\rho_{\min}\) controls the minimum permitted source fidelity, while \(D_2\) measures remaining dependence among the transformed variables.

CORAL can be interpreted geometrically through the reparameterization

\[
W=R^{1/2}T.
\]

The unit-variance ellipsoids in the original transformation space become unit spheres, source-fidelity constraints become spherical caps, and CORAL seeks vectors within those caps that are as mutually orthogonal as possible.

## Exact decorrelation threshold

CORAL also characterizes the largest common source fidelity compatible with exact decorrelation,

\[
\rho_\star(R)
=
\max_{Q\in O(p)}
\min_j e_j^\top R^{1/2}q_j.
\]

If

\[
\rho_{\min}\leq\rho_\star(R),
\]

an exact decorrelator satisfying the requested fidelity may exist. If

\[
\rho_{\min}>\rho_\star(R),
\]

nonzero residual correlation is unavoidable.

This distinction is important. CORAL does not claim that arbitrary levels of fidelity and decorrelation can be achieved simultaneously. It identifies the boundary at which their trade-off becomes mathematically unavoidable.

## Comparison with PCA and ZCA

The repository compares CORAL with two classical transformations.

**PCA** achieves decorrelation but does not preserve a designated one-to-one relationship between transformed coordinates and original variables. CORAL therefore evaluates PCA using favorable sign-invariant one-to-one assignments when source fidelity is compared.

**ZCA whitening** also achieves exact decorrelation and retains the original coordinate orientation more closely than PCA. Its minimum source fidelity,

\[
\min_j(R^{1/2})_{jj},
\]

provides a constructive lower bound on \(\rho_\star(R)\).

CORAL differs from both methods by allowing the analyst to state the required source fidelity explicitly.

## Repository structure

```text
CORAL/
│
├── README.md
│
├── LICENSE
│
├── requirements.txt
│
│
├── paper/
│   ├── CORAL_Springer_Statistics_and_Computing.tex
│   ├── CORAL.pdf
│   ├── figures/
│   ├── tables/
│   └── references.bib
│
├── code/
│   ├── coral_hard_constraints.py
│   ├── simulation/
│   ├── empirical/
│   └── results/
│
└── program/
    ├── coral_app.py
    ├── coral_core.py
    ├── requirements.txt
    └── README.md
```
paper/

Contains the manuscript and publication materials describing the CORAL methodology, theoretical results, simulations, empirical examples, figures, tables, and references.

The paper develops the mathematical programming formulation, oblique-manifold representation, fidelity constraints, exact-decorrelation threshold ρ
⋆
	​

(R), theoretical bounds, PCA and ZCA benchmarks, support-restricted transformations, simulations, and empirical applications.

code/

Contains the research code used to generate the results reported in the manuscript.

The code includes implementations for:

hard-constrained CORAL optimization;
dense fidelity-decorrelation frontiers;
computation of ρ
⋆
	​

(R);
constructive lower and rigorous upper bounds;
PCA and ZCA benchmarks;
support-restricted CORAL;
simulation experiments;
empirical analyses; and
numerical diagnostics and reproducibility checks.

The research code is retained separately from the interactive application so that the analyses reported in the paper remain reproducible independently of the user interface.

program/

Contains the interactive CORAL application.

The application provides a Streamlit interface through which a user can upload a dataframe, select continuous variables, specify a fidelity threshold, and compare CORAL with PCA and ZCA.

The program reports quantities including:

minimum source fidelity;
mean source fidelity;
maximum absolute residual correlation;
mean absolute residual correlation;
aggregate squared residual correlation D
2
	​

;
fidelity-constraint satisfaction;
transformation matrices;
transformed data;
correlation matrices; and
estimates and bounds for the exact-decorrelation fidelity threshold.

The uploaded dataframe does not need to contain only continuous variables. The user selects the continuous variables to be included in the transformation.

Running the research code

Clone the repository:

git clone https://github.com/<USERNAME>/CORAL.git
cd CORAL

Install the required Python packages:

python -m pip install -r requirements.txt

Research analyses can then be run from the scripts contained in code/.

For example:

python code/coral_hard_constraints.py

Because CORAL involves non-convex optimization, production analyses use multiple starting solutions and retain the best feasible result. Reproducibility scripts should preserve the random seeds and optimization settings reported in the manuscript.

Running the CORAL application

Install the application requirements:

python -m pip install -r program/requirements.txt

Launch Streamlit:

python -m streamlit run program/coral_app.py

Streamlit will provide a local address, normally:

http://localhost:8501

Open this address in a browser to use CORAL interactively.

Typical workflow
Upload a CSV, Excel, or supported dataframe file.
Select the continuous variables to transform.
Choose a minimum source fidelity ρ
min
	​

.
Run CORAL.
Compare CORAL with PCA, ZCA, and the original correlation structure.
Examine the residual-correlation and source-fidelity diagnostics.
Examine ρ
⋆
	​

(R) and its bounds to determine whether exact decorrelation is compatible with the requested fidelity.

A high requested fidelity can make exact decorrelation mathematically impossible. In that case, residual correlation is not evidence of solver failure. It is the cost of preserving the declared degree of source-variable identity.

Interpretation

CORAL is intended for settings in which transformed variables must remain substantively recognizable.

Examples include regression predictors whose identities matter, diagnostic variables, biomedical measures, economic indicators, operational metrics, and other applications where replacing named variables with anonymous principal components would reduce interpretability.

CORAL is not intended to replace PCA when dimension reduction or variance concentration is the primary objective. It addresses a different problem: decorrelation under an explicit interpretability constraint.

Current objective and future extensions

The current CORAL estimator minimizes aggregate squared residual correlation subject to a declared fidelity floor.

A natural extension is a minimax formulation that directly minimizes the largest remaining absolute pairwise correlation. More general Pareto formulations could jointly characterize source fidelity, aggregate residual dependence, and worst-pair residual dependence.

Extensions may also allow domain-specified anchoring bases, expected coefficient directions, and approximate relative-magnitude constraints so that interpretability can be declared before transformation rather than inferred afterward.

Citation

If you use CORAL in academic work, please cite the accompanying manuscript:

Fulton, L. V., Fulton, C. P., Sharma, A., & Tomić, A.
CORAL: Constrained Oblique Rotation with Anchored Loadings for Fidelity-Constrained Decorrelation.

Full publication information will be added following publication.

Authors

Lawrence V. Fulton
Christopher P. Fulton
Arvind Sharma
Aleksandar Tomić

License

See LICENSE for the terms governing use and distribution of the software and research materials.
