# CORAL

**Constrained Oblique Rotation with Anchored Loadings for Fidelity-Constrained Decorrelation**

CORAL is a statistical transformation framework for reducing correlation among continuous variables while preserving their correspondence to designated source variables. Unlike PCA, which constructs orthogonal components without guaranteeing that an individual component remains interpretable as a particular original variable, CORAL explicitly constrains source fidelity while minimizing residual correlation.

The central question is:

> **How much correlation can be removed while requiring each transformed variable to remain recognizably anchored to its original source variable?**

CORAL formulates this problem as constrained optimization and provides theoretical and computational tools for examining the resulting fidelity-decorrelation frontier.

## Method

Let `X` be a standardized n-by-p data matrix (n observations, p variables) with correlation matrix `R`, and let `T = [t_1, ..., t_p]` be a transformation matrix defining the transformed data:

```math
\widetilde{X}=XT
```

CORAL imposes unit variance on each transformed coordinate:

```math
t_j^\top R t_j=1
```

Each transformed coordinate must retain a declared minimum correlation with its designated source variable:

```math
e_j^\top R t_j\geq\rho_{\min}
```

The primary CORAL objective minimizes aggregate squared residual correlation:

```math
D_2(T)=\sum_{i<j}(t_i^\top R t_j)^2
```

Thus, `rho_min` controls the minimum permitted source fidelity, while `D_2(T)` measures aggregate remaining squared correlation among the transformed variables.

## Geometric Interpretation

CORAL has a useful geometric interpretation through the reparameterization

```math
W=R^{1/2}T
```

If `w_j=R^1/2t_j`, the unit-variance constraint becomes

```math
w_j^\top w_j=1
```

The original ellipsoidal constraint surface is therefore mapped to the unit sphere. The fidelity constraint becomes

```math
e_j^\top R^{1/2}w_j\geq\rho_{\min}
```

which restricts each transformed direction to a spherical cap.

The decorrelation objective becomes

```math
D_2(W)=\sum_{i<j}(w_i^\top w_j)^2
```

CORAL therefore seeks vectors within their permitted fidelity regions that are as mutually orthogonal as possible.

## Exact-Decorrelation Family

For a symmetric positive-definite correlation matrix `R`, every square exact decorrelator belongs to the family

```math
\mathcal{D}(R)=\left\{R^{-1/2}Q:Q\in O(p)\right\}
```

Equivalently,

```math
T^\top RT=I
```

Exact decorrelation itself is therefore not unique. The central CORAL question is whether an exact decorrelator can also preserve the required source-variable fidelity.

## Exact-Decorrelation Fidelity Threshold

CORAL characterizes the largest common source fidelity compatible with exact decorrelation:

```math
\rho_\star(R)=\max_{Q\in O(p)}\min_{1\leq j\leq p}e_j^\top R^{1/2}q_j
```

If

```math
\rho_{\min}\leq\rho_\star(R)
```

an exact decorrelator satisfying the requested fidelity exists.

If

```math
\rho_{\min}>\rho_\star(R)
```

nonzero residual correlation is unavoidable.

CORAL therefore does not assume that arbitrary levels of fidelity and decorrelation can be achieved simultaneously. It identifies the point at which their trade-off becomes mathematically unavoidable.

## Bounds on the Exact-Decorrelation Threshold

Let

```math
A=R^{1/2}
```

ZCA provides a constructive lower bound:

```math
\min_j A_{jj}\leq\rho_\star(R)
```

A general trace upper bound is

```math
\rho_\star(R)\leq\frac{\operatorname{tr}(A)}{p}
```

Stronger upper bounds can be obtained from nonempty subsets `S` of the source variables:

```math
\rho_\star(R)\leq\frac{\left\|A_{[:,S]}\right\|_\ast}{|S|}
```

where `|cdot|_ast` denotes the nuclear norm.

Combining a numerically achieved exact-decorrelation fidelity with rigorous upper bounds produces a numerical interval containing `rho_star(R)`.

## Comparison with PCA and ZCA

CORAL is compared with two classical transformations: PCA and ZCA whitening.

### PCA

PCA produces decorrelated coordinates but does not preserve a designated one-to-one relationship between transformed coordinates and original variables.

If

```math
R=V\Lambda V^\top
```

the unit-variance PCA transformation is

```math
T_{\mathrm{PCA}}=V\Lambda^{-1/2}
```

The source-to-component correlation matrix is

```math
C_{\mathrm{PCA}}=RT_{\mathrm{PCA}}=V\Lambda^{1/2}
```

Because PCA eigenvector signs and component ordering are arbitrary, CORAL evaluates PCA using favorable sign-invariant one-to-one assignments between principal components and source variables.

PCA remains appropriate when dimension reduction or variance concentration is the primary objective. CORAL addresses a different problem: decorrelation while retaining the identity of individual variables.

### ZCA Whitening

ZCA whitening uses

```math
T_{\mathrm{ZCA}}=R^{-1/2}
```

It achieves exact decorrelation:

```math
T_{\mathrm{ZCA}}^\top R T_{\mathrm{ZCA}}=I
```

Its source fidelities are the diagonal elements of `R^1/2`, so its minimum source fidelity is

```math
\rho_{\mathrm{ZCA}}=\min_j(R^{1/2})_{jj}
```

Therefore,

```math
\rho_{\mathrm{ZCA}}\leq\rho_\star(R)
```

ZCA provides a constructive exact decorrelator, while CORAL searches the broader family `R^-1/2O(p)` for transformations that better preserve the weakest source-variable correspondence.

## Repository Structure

```text
CORAL/
│
├── README.md
├── LICENSE
│
├── paper/
│   ├── CORAL_Springer_Statistics_and_Computing.tex
│   ├── CORAL.pdf
│   ├── figures/
│   └── supporting_files/
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

## `paper/`

The `paper/` directory contains the manuscript and publication materials describing the CORAL methodology.

The paper develops:

- the constrained optimization formulation
- source-fidelity constraints
- the oblique-manifold reparameterization
- the geometric interpretation
- exact-decorrelation theory
- the threshold `rho_star(R)`
- constructive lower and rigorous upper bounds
- PCA and ZCA benchmarks
- support-restricted CORAL
- simulation experiments
- empirical applications
- supporting figures and manuscript materials

The paper provides the formal statistical and mathematical description of CORAL.

## `code/`

The `code/` directory contains the research code used to generate and validate the results reported in the manuscript.

It includes implementations for:

- hard-constrained CORAL optimization
- dense fidelity-decorrelation frontiers
- computation of `rho_star(R)`
- constructive lower bounds for `rho_star(R)`
- rigorous upper bounds for `rho_star(R)`
- PCA benchmarks
- ZCA benchmarks
- support-restricted CORAL
- simulation experiments
- empirical analyses
- numerical diagnostics and reproducibility checks

The research code is maintained separately from the interactive program so that the numerical results reported in the paper remain reproducible independently of the user interface.

## `program/`

The `program/` directory contains the interactive CORAL application.

The application is implemented in Python using Streamlit and allows users to:

- upload a dataframe
- select continuous variables
- declare a fidelity threshold `rho_min`
- run CORAL
- compare CORAL with PCA and ZCA
- inspect source fidelity
- inspect residual correlations
- examine transformation matrices
- view transformed data
- evaluate the exact-decorrelation fidelity threshold

The uploaded dataframe does **not** need to contain only continuous variables. Users select the continuous variables to be included in the CORAL transformation.

The program reports quantities including:

- minimum source fidelity
- mean source fidelity
- maximum absolute residual correlation
- mean absolute residual correlation
- aggregate squared residual correlation `D_2`
- fidelity-constraint satisfaction
- transformation matrices
- transformed observations
- correlation matrices
- estimates and bounds for `rho_star(R)`

## Installation

Clone the repository:

```bash
git clone https://github.com/dustoff06/CORAL.git
cd CORAL
```

Install the program requirements:

```bash
python -m pip install -r program/requirements.txt
```

## Running the Research Code

Research analyses are contained in the `code/` directory.

For example:

```bash
python code/coral_hard_constraints.py
```

CORAL involves non-convex optimization. Research analyses therefore use multiple starting solutions and retain the best feasible result according to the specified objective.

Reproducibility runs should preserve the random seeds, optimization tolerances, solver settings, and multistart configuration used in the corresponding analysis.

## Running the CORAL Application

Launch the Streamlit application with:

```bash
python -m streamlit run program/coral_app.py
```

Streamlit will normally start the application at:

```text
http://localhost:8501
```

Open this address in a browser to use CORAL interactively.

## Typical Workflow

1. Upload a CSV, Excel, Parquet, or other supported dataframe.
2. Select the continuous variables to transform.
3. Choose the minimum source fidelity `rho_min`.
4. Run CORAL.
5. Compare CORAL with the original correlation structure, PCA, and ZCA.
6. Examine minimum and mean source fidelity.
7. Examine maximum and mean residual correlation.
8. Examine the aggregate decorrelation objective `D_2`.
9. Evaluate `rho_star(R)` and its bounds.
10. Determine whether exact decorrelation is compatible with the declared fidelity.

A high requested fidelity can make exact decorrelation mathematically impossible.

If

```math
\rho_{\min}>\rho_\star(R)
```

residual correlation is not evidence of optimizer failure. It is the unavoidable cost of preserving the requested degree of source-variable identity.

## Interpretation

CORAL is intended for settings in which transformed variables must remain substantively recognizable.

Potential applications include:

- regression predictors whose identities matter
- biomedical measurements
- diagnostic variables
- economic indicators
- operational metrics
- healthcare measures
- financial variables
- scientific measurements for which individual-variable interpretation is important

CORAL is **not** intended to replace PCA when dimension reduction or variance concentration is the primary objective.

It addresses a different problem:

> **Decorrelation subject to an explicit source-fidelity constraint.**

## Fidelity Versus Decorrelation

The central CORAL trade-off is between source fidelity and decorrelation.

At low or moderate fidelity requirements, exact decorrelation may remain feasible.

As `rho_min` increases, the feasible region contracts. Once `rho_min` exceeds `rho_star(R)`, exact decorrelation becomes impossible and residual correlations must remain.

The two regimes are therefore:

```math
\rho_{\min}\leq\rho_\star(R)\quad\text{exact decorrelation may satisfy the declared fidelity}
```

and

```math
\rho_{\min}>\rho_\star(R)\quad\text{residual correlation is unavoidable}
```

CORAL makes this trade-off explicit rather than allowing interpretability to emerge incidentally from an unconstrained transformation.

## Current CORAL Objective

The current CORAL estimator solves

```math
\min_T\sum_{i<j}(t_i^\top R t_j)^2
```

subject to

```math
t_j^\top R t_j=1,\qquad j=1,\ldots,p
```

and

```math
e_j^\top R t_j\geq\rho_{\min},\qquad j=1,\ldots,p
```

The transformed correlation matrix is

```math
R_{\mathrm{CORAL}}=T^\top RT
```

Its diagonal elements equal one by construction, while its off-diagonal elements are the residual correlations among transformed variables.

The current CORAL objective minimizes aggregate squared residual correlation rather than the single largest residual pair.

## Diagnostic Quantities

Several quantities are useful for evaluating a CORAL solution.

### Minimum Source Fidelity

The achieved minimum fidelity is

```math
\rho_{\mathrm{achieved}}=\min_j e_j^\top R t_j
```

A feasible solution satisfies

```math
\rho_{\mathrm{achieved}}\geq\rho_{\min}
```

up to numerical solver tolerance.

### Mean Source Fidelity

Mean source fidelity is

```math
\bar{\rho}=\frac{1}{p}\sum_{j=1}^{p}e_j^\top R t_j
```

### Maximum Residual Correlation

The largest remaining absolute pairwise correlation is

```math
r_{\max}=\max_{i<j}\left|t_i^\top R t_j\right|
```

### Mean Absolute Residual Correlation

Mean absolute residual correlation is

```math
\bar{r}=\frac{2}{p(p-1)}\sum_{i<j}\left|t_i^\top R t_j\right|
```

### Aggregate Squared Residual Correlation

The CORAL objective is

```math
D_2(T)=\sum_{i<j}(t_i^\top R t_j)^2
```

Because CORAL minimizes `D_2(T)` rather than `r_max`, a solution can have a relatively small aggregate objective while retaining a larger residual correlation for an individual pair.

## Support-Restricted CORAL

CORAL can also restrict which source variables may contribute to each transformed coordinate.

Let `Omega` denote the declared set of permitted transformation coefficients. Then

```math
t_{ij}=0\quad\text{for all }(i,j)\notin\Omega
```

Support restrictions allow domain knowledge or structural assumptions to constrain the transformation.

Restricting the allowable mixing pattern reduces the feasible set and can make exact decorrelation impossible even when dense CORAL can decorrelate exactly.

Support-restricted CORAL therefore separates two distinct requirements:

1. preservation of source fidelity
2. preservation of a declared transformation structure

## Why CORAL?

Many statistical transformations achieve decorrelation.

The distinguishing feature of CORAL is that **interpretability is imposed as a constraint rather than evaluated only after transformation**.

PCA asks which orthogonal directions represent the dominant variance structure.

ZCA provides an exact whitening transformation that remains relatively aligned with the original coordinate system.

CORAL asks:

> **What is the least residual correlation attainable while guaranteeing that each transformed variable retains a declared relationship with its source variable?**

This distinction is useful when the identities of individual transformed variables matter.

## Example: High-Fidelity Constraints

Suppose an analyst selects `rho_min=0.95`.

CORAL then requires

```math
e_j^\top R t_j\geq0.95
```

for every transformed coordinate.

If the system satisfies

```math
\rho_\star(R)<0.95
```

exact decorrelation is impossible under that requirement.

CORAL still finds the transformation minimizing `D_2(T)` subject to the requested fidelity, but some residual correlations must remain.

This behavior is a feature of the method rather than a numerical failure: the requested source fidelity constrains how far each transformed coordinate may move from its designated source.

## Future Extensions

Several extensions follow naturally from the CORAL framework.

A complementary minimax formulation could minimize the largest remaining absolute pairwise correlation:

```math
\min_T\max_{i<j}\left|t_i^\top R t_j\right|
```

This would directly control the worst remaining pairwise dependence rather than aggregate squared dependence.

A corresponding decorrelation-constrained formulation could maximize common source fidelity `gamma` subject to a declared maximum residual correlation `delta`:

```math
\max_{T,\gamma}\gamma
```

subject to

```math
t_j^\top R t_j=1
```

```math
e_j^\top R t_j\geq\gamma
```

and

```math
\left|t_i^\top R t_j\right|\leq\delta
```

More general Pareto formulations could jointly characterize:

- minimum source fidelity
- aggregate residual dependence
- worst-pair residual dependence

The anchoring framework could also be generalized from individual source coordinates to domain-specified basis vectors.

For a declared basis vector `b_j`, source fidelity could be defined as

```math
\operatorname{Cor}(Xb_j,Xt_j)=\frac{b_j^\top R t_j}{\sqrt{(b_j^\top R b_j)(t_j^\top R t_j)}}
```

If both vectors are normalized under `R`, this reduces to

```math
b_j^\top R t_j\geq\rho_{\min}
```

Future extensions could also incorporate expected coefficient directions or approximate relative magnitudes through sign or interval restrictions on selected transformation coefficients.

These extensions would make interpretability increasingly **declarative rather than post hoc**.

## Citation

If you use CORAL in academic work, please cite the accompanying manuscript:

> Fulton, L. V., Fulton, C. P., Sharma, A., & Tomic, A.
> **CORAL: Constrained Oblique Rotation with Anchored Loadings for Fidelity-Constrained Decorrelation.**

Full publication information will be added following publication.

## Authors

**Lawrence V. Fulton**
Christopher P. Fulton
Arvind Sharma
Aleksandar Tomic

## License

See `LICENSE` for the terms governing use and distribution of the software and research materials.
