# CORAL Comparator

Interactive Streamlit application for applying CORAL, full PCA, and ZCA to selected continuous variables from a data frame.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements_coral_app.txt
```

## Run

```bash
streamlit run coral_app.py
```

Upload a CSV, Excel, or Parquet file, choose the continuous variables, declare `rho_min`, and run the comparison.

## What is compared

- **CORAL**: hard source-fidelity constraint using the Augmented Lagrangian solver on the oblique manifold.
- **PCA**: full PCA rescaled to unit transformed variance. Source matching is sign invariant. Minimum fidelity uses the bottleneck-optimal assignment; the reported favorable mean uses a separately sum-optimal assignment.
- **ZCA**: `T = R^{-1/2}`, the `Q=I` exact decorrelator.

The application reports source fidelity, maximum and mean absolute residual off-diagonal correlation, the squared decorrelation objective `D2`, transformation matrices, transformed correlation matrices, and transformed observations.

## Important numerical choice

The **Interactive** setting defaults to fewer multistarts for responsiveness. The **Publication** setting uses 100 CORAL starts. Because CORAL and the optional `rho_star` search are non-convex, use the publication setting or a larger user-selected number of starts for manuscript-facing results.

## Programmatic use

```python
import pandas as pd
from coral_core import compare_coral_pca_zca

result = compare_coral_pca_zca(
    df,
    columns=["x1", "x2", "x3", "x4"],
    rho_min=0.95,
    n_starts=100,
    seed=0,
)

print(result.summary)
print(result.fidelity)
print(result.transformations["CORAL"])
coral_data = result.transformed_data["CORAL"]
```

The code deliberately stops on a singular or non-positive-definite sample correlation matrix rather than silently regularizing it, because regularization changes the statistical problem being solved.
