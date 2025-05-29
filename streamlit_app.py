import streamlit as st
import pandas as pd
import numpy as np

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
    dsts     = {}
    sections = sorted([k for k in st.secrets if k.startswith("dst")],
                      key=lambda s: int(s[3:]))  # dst1 → 1
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
# Portfolio-allocation inputs (exact dollars, auto-fill last DST)
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.header("DST Dollar Allocations")
dst_keys    = list(dst_static.keys())
manual_keys = dst_keys[:-1]
last_key    = dst_keys[-1]

# User enters allocations for all but the last DST
manual_alloc = {}
for k in manual_keys:
    default = dst_static[k]["equity"]
    manual_alloc[k] = st.sidebar.number_input(
        f"{dst_static[k]['name']} allocation ($)",
        min_value=0.0,
        max_value=TOTAL_EQUITY,
        value=default,
        step=100.0,
        format="%.2f"
    )

sum_manual = sum(manual_alloc.values())
last_alloc = TOTAL_EQUITY - sum_manual

# Validate
if last_alloc < 0:
    st.sidebar.error(f"Your allocations exceed the total equity by ${-last_alloc:,.2f}. Please adjust.")
    st.stop()
# Show computed last DST allocation
st.sidebar.write(f"{dst_static[last_key]['name']} allocation: ${last_alloc:,.2f}")

# Final allocation dict
allocation_dollars = {**manual_alloc, last_key: last_alloc}
alloc_pct          = {k: v / TOTAL_EQUITY for k, v in allocation_dollars.items()}

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
    k: {
        "name":   v["name"],
        "equity": allocation_dollars[k],
        "perc":   v["perc"]
    }
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
max_len           = max(len(cf) for cf in dst_cfs.values())
port_cf           = [sum(cf[i] if i < len(cf) else 0 for cf in dst_cfs.values())
                     for i in range(max_len)]

# IRR: lump year-1 into t=0
if len(port_cf) > 1:
    port_irr_cf = [port_cf[0] + port_cf[1]] + port_cf[2:]
else:
    port_irr_cf = port_cf
portfolio_irr      = np_irr(port_irr_cf)

total_distribution = sum(port_cf)
total_appreciation = sum(
    dst_info[k]["equity"] * dst_controls[k]["sale_multiple"] - dst_info[k]["equity"]
    for k in dst_info
)
total_cash_flows   = total_distribution - total_appreciation
hpr_multiple       = (TOTAL_EQUITY + total_distribution) / TOTAL_EQUITY

# ───────────────────────────────────────────────────────────────────────────────
# Charts
# ───────────────────────────────────────────────────────────────────────────────
years     = list(range(1, HOLD_YEARS + 1))
dist_vals = [
    sum(
        dst_info[k]["equity"] * dst_info[k]["perc"].get(
            yr, dst_info[k]["perc"][max(dst_info[k]["perc"])]
        )
        for k in dst_info if yr <= dst_controls[k]["sale_year"]
    )
    for yr in years
]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Annual Distributions")
    st.line_chart(pd.Series(dist_vals, index=years))
with col2:
    st.subheader("Cumulative Distributions")
    st.line_chart(pd.Series(np.cumsum(dist_vals), index=years))

# ───────────────────────────────────────────────────────────────────────────────
# Tables
# ───────────────────────────────────────────────────────────────────────────────
tbl1, tbl2 = st.columns(2)

with tbl1:
    st.subheader("Portfolio Summary ($)")
    summary_df = pd.DataFrame({
        "IRR":                  [portfolio_irr],
        "Total Dist. (net)":    [total_distribution],
        "Total Appreciation":   [total_appreciation],
        "Total Cash Flows":     [total_cash_flows],
        "HPR Multiple":         [hpr_multiple]
    }, index=["Portfolio"])
    st.dataframe(
        summary_df.style.format({
            "IRR":                "{:.2%}",
            "Total Dist. (net)":  "${:,.0f}",
            "Total Appreciation": "${:,.0f}",
            "Total Cash Flows":   "${:,.0f}",
            "HPR Multiple":       "{:.2f}"
        }),
        use_container_width=True
    )

with tbl2:
    st.subheader("Individual DST Performance")
    perf = []
    for k, info in dst_info.items():
        flows = dst_cfs[k]
        # IRR adjustment
        if len(flows) > 1:
            irr_flows = [flows[0] + flows[1]] + flows[2:]
        else:
            irr_flows = flows
        irr      = np_irr(irr_flows)

        yrs      = dst_controls[k]["sale_year"]
        appreciation = flows[yrs] - info["equity"]
        cagr         = (flows[yrs] / info["equity"]) ** (1 / yrs) - 1

        perf.append({
            "IRR":              irr,
            "Appreciation ($)": flows[yrs] - info["equity"],
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
# Year-by-Year Cash-Flows
# ───────────────────────────────────────────────────────────────────────────────
st.subheader("Year-by-Year DST Cash-Flows")
cf_rows = {
    dst_info[k]["name"]:
        [dst_cfs[k][i] if i < len(dst_cfs[k]) else 0 for i in years]
    for k in dst_info
}
cf_df             = pd.DataFrame(cf_rows, index=[f"Year {y}" for y in years])
cf_df["Total"]    = cf_df.sum(axis=1)
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
    <div style="font-size: 0.8rem; color: grey; font-style: italic; margin-top: 2rem;">
    {secret_text.replace('\n', '<br>')}
    </div>
    """,
    unsafe_allow_html=True
)
