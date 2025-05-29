import streamlit as st
import pandas as pd
import numpy as np

# ───────────────────────────────────────────────────────────────────────────────
# Page & brand config
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Wealthstone", layout="wide")
LOGO = "logo.png"

# ───────────────────────────────────────────────────────────────────────────────
# Authentication (field disappears once the user is verified)
# ───────────────────────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    box = st.empty()
    pwd = box.text_input("Enter password", type="password")
    if pwd == st.secrets["auth"]["password"]:
        st.session_state["authenticated"] = True
        box.empty()
    else:
        st.stop()

# ───────────────────────────────────────────────────────────────────────────────
# Load DST parameters from secrets.toml
# ───────────────────────────────────────────────────────────────────────────────
def load_dsts():
    """Return a dict keyed 'DST 1', 'DST 2', … with name/equity/perc data."""
    dsts     = {}
    sections = sorted([k for k in st.secrets if k.startswith("dst")],
                      key=lambda s: int(s[3:]))          # dst1 → 1
    for idx, sect in enumerate(sections, 1):
        data = st.secrets[sect]
        dsts[f"DST {idx}"] = {
            "name":   data["name"],
            "equity": float(data["equity"]),
            "perc":   {int(y): float(v) for y, v in data["perc"].items()}
        }
    return dsts

dst_static   = load_dsts()
HOLD_YEARS   = 10
TOTAL_EQUITY = sum(d["equity"] for d in dst_static.values())

# ───────────────────────────────────────────────────────────────────────────────
# Branding
# ───────────────────────────────────────────────────────────────────────────────
st.image(LOGO, width=350)

# ───────────────────────────────────────────────────────────────────────────────
# Portfolio-allocation sliders
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Portfolio Allocation (%)")

default_alloc = {k: v["equity"] / TOTAL_EQUITY * 100 for k, v in dst_static.items()}

raw_alloc = {k: st.sidebar.number_input(k, 0.0, 100.0,
                                        round(p, 2), 0.1, format="%.2f")
             for k, p in default_alloc.items()}

tot_pct   = sum(raw_alloc.values()) or 1
alloc_pct = {k: v / tot_pct for k, v in raw_alloc.items()}

st.sidebar.caption(f"Total entered = {tot_pct:.2f}% → scaled automatically to 100 %")

# ───────────────────────────────────────────────────────────────────────────────
# Sale assumptions & what-ifs
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Assumptions & What-Ifs")
dst_controls = {}
for k, info in dst_static.items():
    with st.sidebar.expander(info["name"]):
        dst_controls[k] = {
            "sale_year":     st.slider("Sale year", 1, HOLD_YEARS, HOLD_YEARS,
                                       key=f"{k}_year"),
            "sale_multiple": st.number_input("Sale multiple", 0.5, 5.0, 1.0, 0.05,
                                             key=f"{k}_mult")
        }

# ───────────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────────
def np_irr(cashflows, tol=1e-6, maxiter=100):
    """Newton–Raphson IRR (Excel-compatible)."""
    r = 0.1
    for _ in range(maxiter):
        f  = sum(cf / (1 + r) ** i for i, cf in enumerate(cashflows))
        fp = sum(-i * cf / (1 + r) ** (i + 1) for i, cf in enumerate(cashflows))
        if fp == 0:
            break
        rn = r - f / fp
        if abs(rn - r) < tol:
            return rn
        r = rn
    return np.nan

def cashflows(equity, percents, sale_year, sale_mult):
    """[-equity, y1, …, sale_year (incl. sale proceeds)]."""
    cf = [-equity] + [0] * sale_year
    for yr in range(1, sale_year + 1):
        pct = percents.get(yr, percents[max(percents)])
        cf[yr] += equity * pct
    cf[sale_year] += equity * sale_mult             # sale proceeds
    return cf

# ───────────────────────────────────────────────────────────────────────────────
# Build per-DST info & cash-flows
# ───────────────────────────────────────────────────────────────────────────────
dst_info = {k: {
                "name":   v["name"],
                "equity": TOTAL_EQUITY * alloc_pct[k],
                "perc":   v["perc"]
            } for k, v in dst_static.items()}

dst_cfs = {k: cashflows(dst_info[k]["equity"], dst_info[k]["perc"],
                        dst_controls[k]["sale_year"],
                        dst_controls[k]["sale_multiple"])
           for k in dst_info}

# ───────────────────────────────────────────────────────────────────────────────
# Portfolio-level aggregates
# ───────────────────────────────────────────────────────────────────────────────
max_len = max(len(cf) for cf in dst_cfs.values())
port_cf = [sum(cf[i] if i < len(cf) else 0 for cf in dst_cfs.values())
           for i in range(max_len)]

portfolio_irr = np_irr(port_cf)                       # IRR across all CFs

# ―― NEW DEFINITIONS ――――――――――――――――――――――――――――――――――――――――――――
total_distribution = sum(port_cf)                     # ∑ CFs  (=-init + all +sales)
total_appreciation = sum(
    dst_info[k]["equity"] * dst_controls[k]["sale_multiple"] - dst_info[k]["equity"]
    for k in dst_info
)
total_cash_flows   = total_distribution - total_appreciation
total_return       = total_distribution               # same metric name kept
portfolio_cagr     = ((TOTAL_EQUITY + total_return) / TOTAL_EQUITY) ** (1 / HOLD_YEARS) - 1
# ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

# --------------------------------------------------------------------
# Charts side-by-side
# --------------------------------------------------------------------
years = list(range(1, HOLD_YEARS + 1))
dist_vals = [sum(dst_info[k]["equity"] *
                 dst_info[k]["perc"].get(yr, dst_info[k]["perc"][max(dst_info[k]["perc"])])
                 for k in dst_info if yr <= dst_controls[k]["sale_year"])
             for yr in years]

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Annual Cash-Flow Distributions")
    st.line_chart(pd.Series(dist_vals, index=years))

with chart_right:
    st.subheader("Cumulative Distributions")
    st.line_chart(pd.Series(np.cumsum(dist_vals), index=years))

# --------------------------------------------------------------------
# Tables side-by-side
# --------------------------------------------------------------------
tbl_left, tbl_right = st.columns(2)

with tbl_left:
    st.subheader("Portfolio Summary ($)")
    summary_df = pd.DataFrame({
        "IRR":                 [portfolio_irr],
        "Total Distributions": [total_distribution],
        "Total Appreciation":  [total_appreciation],
        "Total Cash Flows":    [total_cash_flows],
        "Total Return":        [total_return],
        "CAGR":                [portfolio_cagr]
    }, index=["Portfolio"])

    st.dataframe(
        summary_df.style.format({
            "IRR": "{:.2%}",
            "Total Distributions": "${:,.0f}",
            "Total Appreciation":  "${:,.0f}",
            "Total Cash Flows":    "${:,.0f}",
            "Total Return":        "${:,.0f}",
            "CAGR": "{:.2%}"
        }),
        use_container_width=True
    )

    st.caption("Total Distributions = Σ (all positive cash flows) − initial investment.  "
               "Total Cash Flows = Total Distributions − Total Appreciation.")

with tbl_right:
    st.subheader("Individual DST Performance")
    perf = []
    for k, info in dst_info.items():
        flows     = dst_cfs[k]
        irr       = np_irr(flows)
        sale_val  = info["equity"] * dst_controls[k]["sale_multiple"]
        yrs       = dst_controls[k]["sale_year"]
        appreciation = sale_val - info["equity"]
        cagr      = (sale_val / info["equity"]) ** (1 / yrs) - 1
        perf.append({
            "IRR": irr,
            "Appreciation ($)": appreciation,
            "CAGR": cagr
        })

    perf_df = pd.DataFrame(perf, index=[v["name"] for v in dst_info.values()])

    st.dataframe(
        perf_df.style.format({
            "IRR": "{:.2%}",
            "Appreciation ($)": "${:,.0f}",
            "CAGR": "{:.2%}"
        }),
        use_container_width=True
    )

# --------------------------------------------------------------------
# Year-by-year cash-flow table
# --------------------------------------------------------------------
st.subheader("Year-by-Year DST Cash-Flows")

cf_rows = {
    dst_info[k]["name"]:
        [dst_cfs[k][i] if i < len(dst_cfs[k]) else 0 for i in years]
    for k in dst_info
}

cf_df            = pd.DataFrame(cf_rows, index=[f"Year {y}" for y in years])
cf_df["Total"]   = cf_df.sum(axis=1)
cf_df.loc["Total"] = cf_df.sum(numeric_only=True)

st.dataframe(cf_df.style.format("${:,.0f}"),
             use_container_width=True,
             height=420)
