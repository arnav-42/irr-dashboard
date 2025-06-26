import streamlit as st
import pandas as pd
import numpy as np

# Optional: use NumPy-Financial for an IRR identical to Excel’s XIRR
try:
    import numpy_financial as npf
except ImportError:
    npf = None

# ───────────────────────────────────────────────────────────────────────────────
# Page & brand config
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Wealthstone", layout="wide")
LOGO = "logo.png"

# ───────────────────────────────────────────────────────────────────────────────
# Authentication
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
    dsts = {}
    sections = sorted([k for k in st.secrets if k.startswith("dst")],
                      key=lambda s: int(s[3:]))
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
st.image(LOGO, width=350)

# ───────────────────────────────────────────────────────────────────────────────
# Portfolio Allocation (%) — collapsible blocks with slider, $ & lock
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.header("DST Allocation (%)")
dst_keys = list(dst_static.keys())

def rebalance(changed_key):
    vals       = {k: st.session_state[f"alloc_{k}"] for k in dst_keys}
    total      = sum(vals.values())
    diff       = 100.0 - total
    locked_sum = sum(vals[k] for k in dst_keys if st.session_state[f"lock_{k}"])
    targets    = [k for k in dst_keys
                  if (not st.session_state[f"lock_{k}"]) and k != changed_key]
    if not targets or abs(diff) < 1e-6:
        return
    if diff > 0:
        headroom = {k: 100.0 - vals[k] for k in targets}
        total_h  = sum(headroom.values())
        if total_h <= 0:
            st.session_state[f"alloc_{changed_key}"] = 100.0 - locked_sum
            return
        for k in targets:
            st.session_state[f"alloc_{k}"] += diff * (headroom[k] / total_h)
    else:
        curr_val = {k: vals[k] for k in targets}
        total_c  = sum(curr_val.values())
        if total_c <= 0 or abs(diff) >= total_c:
            for k in targets:
                st.session_state[f"alloc_{k}"] = 0.0
            st.session_state[f"alloc_{changed_key}"] = 100.0 - locked_sum
            return
        for k in targets:
            st.session_state[f"alloc_{k}"] += diff * (curr_val[k] / total_c)

def reset_all():
    eq = 100.0 / len(dst_keys)
    for k in dst_keys:
        st.session_state[f"alloc_{k}"] = eq
        st.session_state[f"lock_{k}"]  = False

st.sidebar.button("Reset to equal weight", on_click=reset_all)

for k in dst_keys:
    if f"lock_{k}" not in st.session_state:
        st.session_state[f"lock_{k}"] = False
    if f"alloc_{k}" not in st.session_state:
        st.session_state[f"alloc_{k}"] = dst_static[k]["equity"] / TOTAL_EQUITY * 100

for k in dst_keys:
    with st.sidebar.expander(dst_static[k]["name"], expanded=False):
        pct = st.slider(
            label=f"{dst_static[k]['name']} (%)",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            key=f"alloc_{k}",
            on_change=rebalance,
            args=(k,),
            disabled=st.session_state[f"lock_{k}"]
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            dollars = pct / 100.0 * TOTAL_EQUITY
            st.write(f"→ ${dollars:,.0f}")
        with col2:
            st.checkbox("🔒", key=f"lock_{k}")

alloc_pct = {k: st.session_state[f"alloc_{k}"] / 100.0 for k in dst_keys}
allocation_dollars = {k: alloc_pct[k] * TOTAL_EQUITY for k in dst_keys}

# ───────────────────────────────────────────────────────────────────────────────
# Sale assumptions & what-ifs
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Assumptions & What-Ifs")
dst_controls = {}
for k, info in dst_static.items():
    with st.sidebar.expander(info["name"]):
        dst_controls[k] = {
            "sale_year":     st.slider("Sale year", 1, HOLD_YEARS, HOLD_YEARS, key=f"{k}_year"),
            "sale_multiple": st.number_input("Sale multiple", 0.5, 5.0, 1.0, 0.05, key=f"{k}_mult")
        }

# ───────────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────────
def np_irr(cashflows, tol=1e-6, maxiter=100):
    r = 0.1
    for _ in range(maxiter):
        f  = sum(cf / (1 + r)**i for i, cf in enumerate(cashflows))
        fp = sum(-i * cf / (1 + r)**(i+1) for i, cf in enumerate(cashflows))
        if fp == 0:
            break
        rn = r - f / fp
        if abs(rn - r) < tol:
            return rn
        r = rn
    return np.nan

def cashflows(equity, percents, sale_year, sale_mult):
    cf = [-equity] + [0] * sale_year
    for yr in range(1, sale_year + 1):
        pct = percents.get(yr, percents[max(percents)])
        cf[yr] += equity * pct
    cf[sale_year] += equity * sale_mult
    return cf

# ───────────────────────────────────────────────────────────────────────────────
# Build per-DST info & cash-flows
# ───────────────────────────────────────────────────────────────────────────────
dst_info = {
    k: {"name": v["name"], "equity": allocation_dollars[k], "perc": v["perc"]}
    for k, v in dst_static.items()
}

dst_cfs = {
    k: cashflows(
        dst_info[k]["equity"],
        dst_info[k]["perc"],
        dst_controls[k]["sale_year"],
        dst_controls[k]["sale_multiple"]
    )
    for k in dst_info
}

# ───────────────────────────────────────────────────────────────────────────────
# Portfolio-level aggregates
# ───────────────────────────────────────────────────────────────────────────────
max_len = max(len(cf) for cf in dst_cfs.values())
port_cf = [
    sum((cf[i] if i < len(cf) else 0) for cf in dst_cfs.values())
    for i in range(max_len)
]

# IRR on the aggregated series
if len(port_cf) > 1:
    port_irr_cf = [port_cf[0] + port_cf[1]] + port_cf[2:]
else:
    port_irr_cf = port_cf
portfolio_irr = np_irr(port_irr_cf)

# total sale proceeds (S)
sale_values = sum(
    dst_info[k]["equity"] * dst_controls[k]["sale_multiple"]
    for k in dst_info
)

# extract sum of periodic distributions (D)
# since sum(port_cf) = -TOTAL_EQUITY + D + S
distribution_sum = sum(port_cf) + TOTAL_EQUITY - sale_values

# total distributions = D + S
total_distribution = distribution_sum + sale_values

# total appreciation = S – initial equity
total_appreciation = sale_values - TOTAL_EQUITY

# total cash flows = just the periodic distributions (D)
total_cash_flows = distribution_sum

# MOIC = (D + S) / initial equity
moic = (distribution_sum + sale_values) / TOTAL_EQUITY

# ───────────────────────────────────────────────────────────────────────────────
# Charts
# ───────────────────────────────────────────────────────────────────────────────
years = list(range(1, HOLD_YEARS + 1))

# annual periodic distributions (for chart 1)
dist_vals = [
    sum(
        dst_info[k]["equity"] * dst_info[k]["perc"].get(
            yr, dst_info[k]["perc"][max(dst_info[k]["perc"])]
        )
        for k in dst_info
        if yr <= dst_controls[k]["sale_year"]
    )
    for yr in years
]

# build per-year total (periodic + sale proceeds in that year)
annual_dist_and_sale_vals = [
    port_cf[i] if i < len(port_cf) else 0
    for i in range(1, HOLD_YEARS + 1)
]

# cumulative periodic distributions
cum_cashflow = np.cumsum(dist_vals)
# cumulative distributions + sale
cum_dist_and_sale_vals = np.cumsum(annual_dist_and_sale_vals)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Annual Cash Flow")
    st.line_chart(pd.Series(dist_vals, index=years))
with col2:
    st.subheader("Cumulative Cash Flow")
    st.line_chart(pd.Series(cum_cashflow, index=years))
with col3:
    st.subheader("Cumulative Distributions")
    st.line_chart(pd.Series(cum_dist_and_sale_vals, index=years))

# ───────────────────────────────────────────────────────────────────────────────
# Tables
# ───────────────────────────────────────────────────────────────────────────────
tbl1, tbl2 = st.columns(2)

with tbl1:
    st.subheader("Portfolio Summary")
    summary_df = pd.DataFrame({
        "IRR":                 [portfolio_irr],
        "Total Distributions": [total_distribution],
        "Total Appreciation":  [total_appreciation],
        "Total Cash Flows":    [total_cash_flows],
        "MOIC":                [moic]
    }, index=["Portfolio"])
    st.dataframe(
        summary_df.style.format({
            "IRR":                 "{:.2%}",
            "Total Distributions": "${:,.0f}",
            "Total Appreciation":  "${:,.0f}",
            "Total Cash Flows":    "${:,.0f}",
            "MOIC":                "{:.2f}"
        }),
        use_container_width=True
    )

with tbl2:
    st.subheader("Individual DST Performance")
    perf = []
    for k, info in dst_info.items():
        equity = info["equity"]
        yrs    = dst_controls[k]["sale_year"]
        flows  = dst_cfs[k]
        if len(flows) > 1:
            irr_cf = [flows[0] + flows[1]] + flows[2:]
        else:
            irr_cf = flows
        irr = np_irr(irr_cf)

        if equity <= 0:
            appreciation = 0.0
            cagr = float("nan")
        else:
            appreciation = equity * dst_controls[k]["sale_multiple"] - equity
            cagr = dst_controls[k]["sale_multiple"] ** (1 / yrs) - 1

        perf.append({
            "IRR":              irr,
            "Appreciation ($)": appreciation,
            "CAGR":             cagr
        })

    perf_df = pd.DataFrame(perf, index=[v["name"] for v in dst_info.values()])
    st.dataframe(
        perf_df.style.format({
            "IRR":              "{:.2%}",
            "Appreciation ($)": "${:,.0f}",
            "CAGR":             "{:.2%}"
        }),
        use_container_width=True
    )

# ───────────────────────────────────────────────────────────────────────────────
# Year-by-Year DST Cash-Flows
# ───────────────────────────────────────────────────────────────────────────────
st.subheader("Year-by-Year DST Distributions")
cf_rows = {
    dst_info[k]["name"]:
        [dst_cfs[k][i] if i < len(dst_cfs[k]) else 0 for i in years]
    for k in dst_info
}
cf_df = pd.DataFrame(cf_rows, index=[f"Year {y}" for y in years])
cf_df["Total"] = cf_df.sum(axis=1)
cf_df.loc["Total"] = cf_df.sum(numeric_only=True)

st.dataframe(
    cf_df.style.format("${:,.0f}"),
    use_container_width=True,
    height=420
)

# ───────────────────────────────────────────────────────────────────────────────
# Secret text (from secrets)
# ───────────────────────────────────────────────────────────────────────────────
secret_text = st.secrets["secret_text"]["text"]
st.markdown(
    f"""
    <div style="font-size: 0.6rem; color: grey; font-style: normal; margin-top: 0rem; line-height: 1;">
    {secret_text.replace('\n', '<br>')}
    </div>
    """,
    unsafe_allow_html=True
)
