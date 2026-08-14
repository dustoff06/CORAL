# CORAL

**Constrained Oblique Rotation with Anchored Loadings for Fidelity-Constrained Decorrelation**

CORAL is a statistical transformation framework for reducing correlation among continuous variables while preserving their correspondence to designated source variables. Unlike PCA, which constructs orthogonal components without guaranteeing that an individual component remains interpretable as a particular original variable, CORAL explicitly constrains source fidelity while minimizing residual correlation.

The central question is:

> **How much correlation can be removed while requiring each transformed variable to remain recognizably anchored to its original source variable?**

CORAL formulates this problem as constrained optimization and provides theoretical and computational tools for examining the resulting fidelity-decorrelation frontier.

## Method

Let $X \in \mathbb{R}^{n \times p}$ be a standardized data matrix with correlation matrix $R$, and let

$$
\widetilde{X}=XT,
$$

where $T=[t_1,\ldots,t_p]$ is the transformation matrix.

CORAL imposes unit variance on each transformed coordinate,

$$
t_j^\top R t_j=1,
$$

and requires each transformed coordinate to retain a declared minimum correlation with its designated source variable,

$$
e_j^\top R t_j \geq \rho_{\min}.
$$

The primary CORAL objective minimizes aggregate squared residual correlation,

$$
D_2(T)=\sum_{i<j}(t_i^\top R t_j)^2.
$$

Thus, $\rho_{\min}$ controls the minimum permitted source fidelity, while $D_2(T)$ measures aggregate remaining squared correlation among the transformed variables.

CORAL also has a useful geometric interpretation through the reparameterization

$$
W=R^{1/2}T.
$$

Under this transformation, the unit-variance ellipsoids become unit spheres, source-fidelity constraints become spherical caps, and CORAL seeks vectors within those caps that are as mutually orthogonal as possible.

## Exact-Decorrelation Fidelity Threshold

CORAL characterizes the largest common source fidelity compatible with exact decorrelation,

$$
\rho_\star(R)
=
\max_{Q\in O(p)}
\min_j e_j^\top R^{1/2}q_j.
$$

If $\rho_{\min}\leq\rho_\star(R)$, an exact decorrelator satisfying the requested fidelity exists.

If $\rho_{\min}>\rho_\star(R)$, nonzero residual correlation is unavoidable.

CORAL therefore does not assume that arbitrary levels of fidelity and decorrelation can be achieved simultaneously. It identifies the point at which their trade-off becomes mathematically unavoidable.

## Comparison with PCA and ZCA

CORAL is compared with two classical transformations: PCA and ZCA whitening.

### PCA

PCA produces decorrelated coordinates but does not preserve a designated one-to-one relationship between transformed coordinates and original variables.

For source-fidelity comparisons, CORAL evaluates PCA using favorable sign-invariant one-to-one assignments between principal components and source variables.

PCA remains appropriate when dimension reduction or variance concentration is the primary objective. CORAL addresses a different problem: decorrelation while retaining the identity of individual variables.

### ZCA Whitening

ZCA whitening also achieves exact decorrelation while retaining the original coordinate orientation more closely than PCA.

Its minimum source fidelity is

$$
\rho_{\mathrm{ZCA}}
=
\min_j (R^{1/2})_{jj},
$$

which provides a constructive lower bound on $\rho_\star(R)$:

$$
\rho_{\mathrm{ZCA}}\leq\rho_\star(R).
$$

CORAL differs from both PCA and ZCA by allowing the analyst to state the required minimum source fidelity explicitly.

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

These materials include:

- the mathematical programming formulation;
- source-fidelity constraints;
- the oblique-manifold reparameterization;
- geometric interpretation;
- exact-decorrelation theory;
- the threshold $\rho_\star(R)$;
- constructive lower and rigorous upper bounds;
- PCA and ZCA benchmarks;
- support-restricted CORAL;
- simulation experiments;
- empirical applications; and
- figures and supporting manuscript files.

The paper provides the formal statistical and mathematical description of CORAL.

## `code/`

The `code/` directory contains the research code used to generate and validate the results reported in the manuscript.

It includes implementations for:

- hard-constrained CORAL optimization;
- dense fidelity-decorrelation frontiers;
- computation of $\rho_\star(R)$;
- constructive lower bounds for $\rho_\star(R)$;
- rigorous upper bounds for $\rho_\star(R)$;
- PCA benchmarks;
- ZCA benchmarks;
- support-restricted CORAL;
- simulation experiments;
- empirical analyses; and
- numerical diagnostics and reproducibility checks.

The research code is maintained separately from the interactive application so that the results reported in the paper remain reproducible independently of the user interface.

## `program/`

The `program/` directory contains the interactive CORAL application.

The Streamlit application allows users to:

- upload a dataframe;
- select continuous variables;
- declare a fidelity threshold $\rho_{\min}$;
- run CORAL;
- compare CORAL with PCA and ZCA;
- inspect source fidelity;
- inspect residual correlations;
- examine transformation matrices;
- view transformed data; and
- evaluate the exact-decorrelation fidelity threshold.

The uploaded dataframe does **not** need to contain only continuous variables. Users select the continuous variables to be included in the CORAL transformation.

The program reports quantities including:

- minimum source fidelity;
- mean source fidelity;
- maximum absolute residual correlation;
- mean absolute residual correlation;
- aggregate squared residual correlation $D_2$;
- fidelity-constraint satisfaction;
- transformation matrices;
- transformed observations;
- correlation matrices; and
- estimates and bounds for $\rho_\star(R)$.

## Installation

Clone the repository:

```bash
git clone https://github.com/dustoff06/CORAL.git
cd CORAL
```

Install the required Python packages:

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

Reproducibility runs should preserve the random seeds, optimization tolerances, and multistart settings reported in the manuscript.

## Running the CORAL Application

Launch the Streamlit interface with:

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
3. Choose the minimum source fidelity $\rho_{\min}$.
4. Run CORAL.
5. Compare CORAL with the original correlation structure, PCA, and ZCA.
6. Examine minimum and mean source fidelity.
7. Examine maximum and mean residual correlation.
8. Examine the aggregate decorrelation objective $D_2$.
9. Evaluate $\rho_\star(R)$ and its bounds.
10. Determine whether exact decorrelation is compatible with the declared fidelity.

A high requested fidelity can make exact decorrelation mathematically impossible.

If

$$
\rho_{\min}>\rho_\star(R),
$$

residual correlation is not evidence of optimizer failure. It is the unavoidable cost of preserving the requested degree of source-variable identity.

## Interpretation

CORAL is intended for settings in which transformed variables must remain substantively recognizable.

Potential applications include:

- regression predictors whose identities matter;
- biomedical measurements;
- diagnostic variables;
- economic indicators;
- operational metrics;
- healthcare measures;
- financial variables; and
- scientific measurements for which individual-variable interpretation is important.

CORAL is **not** intended to replace PCA when dimension reduction or variance concentration is the primary objective.

It addresses a different problem:

> **Decorrelation subject to an explicit source-fidelity constraint.**

## Fidelity Versus Decorrelation

The central CORAL trade-off is between source fidelity and decorrelation.

At low or moderate fidelity requirements, exact decorrelation may remain feasible.

As $\rho_{\min}$ increases, the feasible region contracts. Once $\rho_{\min}$ exceeds $\rho_\star(R)$, exact decorrelation becomes impossible and residual correlations must remain.

Thus,

$$
\rho_{\min}\leq\rho_\star(R)
$$

defines the regime in which fidelity and exact decorrelation are jointly feasible, while

$$
\rho_{\min}>\rho_\star(R)
$$

defines the regime in which a fidelity-decorrelation trade-off is unavoidable.

CORAL makes this trade-off explicit rather than allowing interpretability to emerge incidentally from an unconstrained transformation.

## Exact Decorrelation Family

For a positive-definite correlation matrix $R$, every square exact decorrelator can be written as

$$
T=R^{-1/2}Q,
$$

where $Q\in O(p)$ is an orthogonal matrix.

Equivalently,

$$
T^\top RT=I.
$$

CORAL therefore does not ask whether an exact decorrelator exists. For positive-definite $R$, many exist.

The question is whether an exact decorrelator exists that also satisfies the declared source-fidelity requirement.

That question is governed by $\rho_\star(R)$.

## Current CORAL Objective

The current CORAL estimator solves

$$
\min_T
\sum_{i<j}(t_i^\top R t_j)^2
$$

subject to

$$
t_j^\top R t_j=1
$$

and

$$
e_j^\top R t_j\geq\rho_{\min}
$$

for every transformed coordinate $j$.

The squared-correlation objective distributes decorrelation pressure across the complete transformed correlation matrix.

The resulting transformed correlation matrix is

$$
R_{\mathrm{CORAL}}=T^\top RT.
$$

Its diagonal elements equal one by construction, while its off-diagonal elements represent residual correlations among the transformed variables.

## Diagnostic Quantities

Several quantities are useful for evaluating a CORAL solution.

### Minimum Source Fidelity

$$
\rho_{\mathrm{achieved}}
=
\min_j e_j^\top R t_j.
$$

A feasible CORAL solution satisfies

$$
\rho_{\mathrm{achieved}}\geq\rho_{\min}
$$

up to numerical solver tolerance.

### Maximum Residual Correlation

$$
r_{\max}
=
\max_{i<j}
|t_i^\top R t_j|.
$$

This reports the largest remaining pairwise correlation after transformation.

### Mean Absolute Residual Correlation

$$
\bar r
=
\frac{2}{p(p-1)}
\sum_{i<j}
|t_i^\top R t_j|.
$$

### Aggregate Squared Residual Correlation

$$
D_2(T)
=
\sum_{i<j}
(t_i^\top R t_j)^2.
$$

The current CORAL estimator minimizes $D_2(T)$ rather than $r_{\max}$. Consequently, a solution can have a low aggregate residual-correlation objective while retaining a larger correlation for an individual pair.

## Bounds on the Exact-Decorrelation Threshold

Let

$$
A=R^{1/2}.
$$

A constructive lower bound for $\rho_\star(R)$ is provided by ZCA:

$$
\min_j A_{jj}\leq\rho_\star(R).
$$

A general trace upper bound is

$$
\rho_\star(R)
\leq
\frac{\operatorname{tr}(A)}{p}.
$$

Stronger upper bounds can be obtained from nonempty subsets $S$ of the source variables:

$$
\rho_\star(R)
\leq
\frac{\|A_{[:,S]}\|_*}{|S|},
$$

where $\|\cdot\|_*$ denotes the nuclear norm.

Combining constructive solutions with rigorous upper bounds provides a numerical interval containing the exact-decorrelation fidelity threshold.

## Support-Restricted CORAL

CORAL can also restrict which source variables may contribute to each transformed coordinate.

Let $\Omega$ denote the declared set of permitted transformation coefficients. Then

$$
t_{ij}=0
$$

for all $(i,j)\notin\Omega$.

Support restrictions allow domain knowledge or structural assumptions to constrain the transformation.

However, restricting the allowable mixing pattern reduces the feasible set and can make exact decorrelation impossible even when dense CORAL can decorrelate exactly.

Support-restricted CORAL therefore separates two distinct constraints:

1. preservation of source fidelity; and
2. preservation of a declared transformation structure.

## Why CORAL?

Many statistical transformations achieve decorrelation.

The distinguishing feature of CORAL is that **interpretability is imposed as a constraint rather than evaluated only after transformation**.

PCA asks which orthogonal directions explain variance.

ZCA asks for a whitening transformation that remains close to the original coordinate system.

CORAL asks:

> **What is the least residual correlation attainable while guaranteeing that each transformed variable retains a declared relationship with its source variable?**

This distinction is useful when transformed variables must retain substantive identities.

## Future Extensions

Several extensions follow naturally from the CORAL framework.

A complementary minimax formulation could minimize the largest remaining absolute pairwise correlation:

$$
\min_T
\max_{i<j}
|t_i^\top R t_j|.
$$

This formulation would directly control the worst remaining pairwise dependence rather than aggregate squared dependence.

A corresponding dual formulation could maximize common source fidelity $\gamma$ subject to a declared maximum residual correlation $\delta$:

$$
\max_{T,\gamma}\gamma
$$

subject to

$$
t_j^\top R t_j=1,
$$

$$
e_j^\top R t_j\geq\gamma,
$$

and

$$
|t_i^\top R t_j|\leq\delta.
$$

More general Pareto formulations could jointly characterize:

- minimum source fidelity;
- aggregate residual dependence; and
- worst-pair residual dependence.

The anchoring framework could also be generalized from individual source coordinates to domain-specified basis vectors encoding expected direction and approximate relative magnitude, with optional sign or interval constraints on selected transformation coefficients.

Such extensions would make interpretability increasingly **declarative rather than post hoc**.

## Citation

If you use CORAL in academic work, please cite the accompanying manuscript:

> Fulton, L. V., Fulton, C. P., Sharma, A., & Tomić, A.  
> **CORAL: Constrained Oblique Rotation with Anchored Loadings for Fidelity-Constrained Decorrelation.**

Full publication information will be added following publication.

## Authors

**Lawrence V. Fulton**  
Christopher P. Fulton  
Arvind Sharma  
Aleksandar Tomić

## License

See `LICENSE` for terms governing use and distribution of the software and research materials.
