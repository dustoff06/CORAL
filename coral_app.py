"""Streamlit application for comparing CORAL, PCA, and ZCA.

Run:
    streamlit run coral_app.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from coral_core import compare_coral_pca_zca


st.set_page_config(
    page_title="CORAL Comparator",
    page_icon="◉",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    bio = io.BytesIO(file_bytes)
    if suffix == ".csv":
        return pd.read_csv(bio)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(bio)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(bio)
    raise ValueError("Upload a CSV, Excel, or Parquet file.")


def heatmap(df: pd.DataFrame, title: str, zmin=None, zmax=None):
    fig = px.imshow(
        df,
        x=df.columns,
        y=df.index,
        text_auto=".2f" if len(df) <= 12 else False,
        aspect="auto",
        zmin=zmin,
        zmax=zmax,
        color_continuous_midpoint=0.0 if zmin is not None and zmax is not None else None,
        labels={"color": "value"},
        title=title,
    )
    fig.update_layout(height=max(450, 28 * len(df) + 170))
    return fig


def make_download_zip(result) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.csv", result.summary.to_csv())
        zf.writestr("fidelity_by_variable.csv", result.fidelity.to_csv())
        zf.writestr("original_correlation.csv", result.correlation.to_csv())
        zf.writestr("standardized_selected_data.csv", result.standardized_data.to_csv())
        zf.writestr(
            "diagnostics.txt",
            "\n".join(f"{k}: {v}" for k, v in result.diagnostics.items()),
        )
        for method in ["CORAL", "PCA", "ZCA"]:
            zf.writestr(
                f"{method.lower()}_transformation.csv",
                result.transformations[method].to_csv(),
            )
            zf.writestr(
                f"{method.lower()}_transformed_correlation.csv",
                result.transformed_correlations[method].to_csv(),
            )
            zf.writestr(
                f"{method.lower()}_transformed_data.csv",
                result.transformed_data[method].to_csv(),
            )
    return buffer.getvalue()


st.title("CORAL Comparator")
st.caption(
    "Select continuous variables, declare a CORAL source-fidelity floor, and compare "
    "the resulting transformation with sign-invariant full PCA and ZCA whitening."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader(
        "Upload data",
        type=["csv", "xlsx", "xls", "parquet", "pq"],
    )

if uploaded is None:
    st.info("Upload a data frame to begin. CSV, Excel, and Parquet are supported.")
    st.stop()

try:
    df = read_uploaded_file(uploaded.name, uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

with st.sidebar:
    st.write(f"Rows: **{len(df):,}**")
    st.write(f"Numeric variables: **{len(numeric_cols)}**")
    selected = st.multiselect(
        "Continuous variables",
        options=numeric_cols,
        default=numeric_cols[: min(8, len(numeric_cols))],
    )

    st.header("CORAL")
    rho_min = st.slider(
        "Minimum source fidelity (ρmin)",
        min_value=0.00,
        max_value=0.999,
        value=0.95,
        step=0.01,
    )

    mode = st.radio(
        "Optimization effort",
        ["Interactive", "Publication"],
        index=0,
        help=(
            "Interactive uses fewer multistarts for speed. Publication uses 100 starts, "
            "matching the intended manuscript protocol."
        ),
    )
    default_starts = 20 if mode == "Interactive" else 100
    n_starts = st.number_input(
        "CORAL multistarts",
        min_value=1,
        max_value=500,
        value=default_starts,
        step=1,
    )
    seed = st.number_input("Random seed", min_value=0, value=0, step=1)

    st.header("Optional theory")
    estimate_rho_star = st.checkbox(
        "Estimate ρ★(R)",
        value=False,
        help="Non-convex max-min search. Useful, but potentially much slower than the primal comparison.",
    )
    rho_star_starts = st.number_input(
        "ρ★ multistarts",
        min_value=1,
        max_value=500,
        value=20 if mode == "Interactive" else 100,
        step=1,
        disabled=not estimate_rho_star,
    )

    run = st.button("Run comparison", type="primary", use_container_width=True)

st.subheader("Data preview")
st.dataframe(df.head(20), use_container_width=True)

if not run:
    st.stop()

if len(selected) < 2:
    st.error("Select at least two continuous variables.")
    st.stop()

with st.spinner("Running CORAL, PCA, and ZCA..."):
    try:
        result = compare_coral_pca_zca(
            df=df,
            columns=selected,
            rho_min=float(rho_min),
            n_starts=int(n_starts),
            seed=int(seed),
            missing="drop",
            estimate_rho_star=bool(estimate_rho_star),
            rho_star_starts=int(rho_star_starts),
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()

st.success(
    f"Analysis complete: n={result.diagnostics['n_complete']:,}, "
    f"p={result.diagnostics['p']}, "
    f"{result.diagnostics['n_dropped']:,} rows removed for missing selected values."
)

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------

st.header("Comparison")
summary_display = result.summary.copy()
for c in ["Min source fidelity", "Mean source fidelity", "Max |offdiag|", "Mean |offdiag|", "D2"]:
    summary_display[c] = summary_display[c].astype(float).round(6)
st.dataframe(summary_display, use_container_width=True)

st.caption(
    "PCA is given favorable sign-invariant one-to-one assignments. Its minimum fidelity "
    "uses the bottleneck assignment; its reported mean fidelity uses a separately "
    "sum-optimal assignment. The PCA transformation displayed below uses the bottleneck assignment."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("CORAL min fidelity", f"{result.summary.loc['CORAL', 'Min source fidelity']:.4f}")
c2.metric("CORAL max |offdiag|", f"{result.summary.loc['CORAL', 'Max |offdiag|']:.4f}")
c3.metric("ZCA min fidelity", f"{result.summary.loc['ZCA', 'Min source fidelity']:.4f}")
c4.metric("PCA best min fidelity", f"{result.summary.loc['PCA', 'Min source fidelity']:.4f}")

if not bool(result.raw_results["CORAL"]["feasible"]):
    st.warning(
        "The best CORAL run did not satisfy the declared fidelity constraint to the "
        "requested tolerance. Increase multistarts/iterations before interpreting it."
    )

if estimate_rho_star and result.raw_results["rho_star"] is not None:
    rs = result.raw_results["rho_star"]
    st.subheader("Exact-decorrelation fidelity threshold")
    a, b, c = st.columns(3)
    a.metric("Best-found ρ★ lower value", f"{rs['rho_star']:.6f}")
    b.metric("ZCA constructive lower", f"{rs['zca_lower_bound']:.6f}")
    c.metric("Trace upper bound", f"{rs['trace_upper_bound']:.6f}")
    st.caption(
        "The max-min ρ★ search is non-convex. Its achieved value is constructive, not a "
        "global-optimality proof; the trace quantity is a rigorous analytical upper bound."
    )

# -----------------------------------------------------------------------------
# Fidelity
# -----------------------------------------------------------------------------

st.header("Source fidelity by variable")
fid_long = (
    result.fidelity.reset_index()
    .melt(id_vars="Variable", var_name="Method", value_name="Fidelity")
)
fig_fid = px.bar(
    fid_long,
    x="Variable",
    y="Fidelity",
    color="Method",
    barmode="group",
    range_y=[0, 1.03],
)
fig_fid.add_hline(y=float(rho_min), line_dash="dash", annotation_text="ρmin")
fig_fid.update_layout(height=500)
st.plotly_chart(fig_fid, use_container_width=True)
st.dataframe(result.fidelity.round(6), use_container_width=True)

# -----------------------------------------------------------------------------
# Correlation geometry
# -----------------------------------------------------------------------------

st.header("Correlation structure")
method_choice = st.selectbox(
    "Transformed correlation matrix",
    ["Original", "CORAL", "PCA", "ZCA"],
)
if method_choice == "Original":
    corr_view = result.correlation
else:
    corr_view = result.transformed_correlations[method_choice]
st.plotly_chart(
    heatmap(corr_view, f"{method_choice} correlation matrix", zmin=-1, zmax=1),
    use_container_width=True,
)

# -----------------------------------------------------------------------------
# Transformations
# -----------------------------------------------------------------------------

st.header("Transformation matrices")
method_T = st.selectbox("Transformation", ["CORAL", "PCA", "ZCA"], key="t_method")
T = result.transformations[method_T]
st.plotly_chart(
    heatmap(T, f"{method_T} transformation matrix T"),
    use_container_width=True,
)
st.dataframe(T.round(6), use_container_width=True)

# -----------------------------------------------------------------------------
# Diagnostics and export
# -----------------------------------------------------------------------------

with st.expander("Diagnostics"):
    st.json(result.diagnostics)

st.header("Export")
zip_bytes = make_download_zip(result)
st.download_button(
    "Download complete analysis (.zip)",
    data=zip_bytes,
    file_name="coral_comparison.zip",
    mime="application/zip",
    use_container_width=True,
)
