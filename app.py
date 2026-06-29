import pandas as pd
import streamlit as st

from src.valuation_model import (
    ValuationAssumptions,
    apply_downside_recommendation,
    build_scenario_summary,
    build_sensitivity,
    build_strategy_memo,
    format_currency,
    format_money,
    load_financials,
    normalize_financials,
    summarize_valuation,
)


def format_percent_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column in output.columns:
            output[column] = output[column].map(lambda value: f"{value:.1%}")
    return output


def format_money_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column in output.columns:
            output[column] = output[column].map(lambda value: f"${value:,.1f}M")
    return output


st.set_page_config(
    page_title="Company Valuation & Strategy Engine",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


@st.cache_data
def load_sample_financials() -> pd.DataFrame:
    return load_financials("data/company_financials.csv")


@st.cache_data
def load_comps() -> pd.DataFrame:
    return pd.read_csv("data/comparable_companies.csv")


st.title("Company Valuation & Strategic Recommendation Engine")
st.caption("DCF, size-adjusted comps, scenario analysis, sensitivity, and strategic recommendation memo.")

with st.sidebar:
    st.header("Valuation Inputs")
    company_name = st.text_input("Company", "Atlas Workflow Systems")
    current_share_price = st.slider("Current share price", 5.0, 80.0, 24.0, 0.5)
    revenue_growth = st.slider("Forecast revenue growth", 0.00, 0.25, 0.09, 0.005)
    target_margin = st.slider("Target EBITDA margin", 0.05, 0.40, 0.205, 0.005)
    terminal_growth = st.slider("Terminal growth", 0.00, 0.05, 0.025, 0.005)
    beta = st.slider("Beta", 0.6, 2.0, 1.15, 0.05)
    debt_weight = st.slider("Debt weight", 0.00, 0.60, 0.25, 0.05)
    size_discount = st.slider("Public comp size discount", 0.00, 0.40, 0.10, 0.025)
    dcf_weight = st.slider("DCF blend weight", 0.00, 1.00, 0.60, 0.05)
    ebitda_multiple_weight = st.slider("EV/EBITDA comp weight", 0.00, 1.00, 0.65, 0.05)
    dilution_pct = st.slider("Forecast share dilution", 0.00, 0.15, 0.05, 0.01)
    uploaded_file = st.file_uploader("Upload company financials", type="csv")

assumptions = ValuationAssumptions(
    revenue_growth=revenue_growth,
    target_ebitda_margin=target_margin,
    terminal_growth=terminal_growth,
    beta=beta,
    debt_weight=debt_weight,
    size_discount=size_discount,
    dcf_weight=dcf_weight,
    ebitda_multiple_weight=ebitda_multiple_weight,
    dilution_pct=dilution_pct,
)

try:
    financials = pd.read_csv(uploaded_file) if uploaded_file else load_sample_financials()
    financials = normalize_financials(financials)
except Exception as exc:
    st.error(f"Could not load financials: {exc}")
    st.stop()

comps_source = load_comps()
forecast, comps, summary = summarize_valuation(financials, comps_source, assumptions, current_share_price)
scenario_summary = build_scenario_summary(financials, comps_source, assumptions, current_share_price)
summary = apply_downside_recommendation(summary, forecast, financials, scenario_summary)
sensitivity = build_sensitivity(financials, comps_source, assumptions, current_share_price)

hero = st.columns(6)
hero[0].metric("Recommendation", summary.recommendation)
hero[1].metric("Blended Value / Share", format_currency(summary.blended_value_per_share), f"{summary.upside_downside:.1%}")
hero[2].metric("DCF EV", format_money(summary.dcf_enterprise_value))
hero[3].metric("Comps EV", format_money(summary.comps_enterprise_value))
hero[4].metric("WACC", f"{summary.wacc:.1%}")
hero[5].metric("Terminal % of DCF", f"{summary.terminal_value_pct_dcf:.1%}")

st.divider()

overview_tab, dcf_tab, comps_tab, scenario_tab, memo_tab, data_tab = st.tabs(
    ["Valuation Snapshot", "DCF Model", "Comparable Companies", "Scenarios", "Strategy Memo", "Data"]
)

with overview_tab:
    left, right = st.columns([1.2, 1])
    with left:
        trend = pd.concat(
            [
                financials[["Year", "Revenue", "EBITDA", "Free_Cash_Flow"]].assign(Period="Historical"),
                forecast[["Year", "Revenue", "EBITDA", "Free_Cash_Flow"]].assign(Period="Forecast"),
            ]
        )
        st.subheader("Revenue, EBITDA, and FCF")
        st.line_chart(trend.set_index("Year")[["Revenue", "EBITDA", "Free_Cash_Flow"]], use_container_width=True)
    with right:
        st.subheader("Strategic Readout")
        st.write(summary.recommendation_reason)
        st.write(f"Blended enterprise value: **{format_money(summary.blended_enterprise_value)}**")
        st.write(f"Blended equity value: **{format_money(summary.blended_equity_value)}**")
        st.write(f"Current share price: **{format_currency(current_share_price)}**")
        st.write(f"Implied upside/downside: **{summary.upside_downside:.1%}**")
        st.write(f"Target implied multiple: **{summary.implied_ev_revenue:.1f}x revenue / {summary.implied_ev_ebitda:.1f}x EBITDA**")
        st.write(f"Capital efficiency: **{summary.roic:.1%} valuation-implied ROIC / {summary.fcf_yield:.1%} FCF yield**")

    valuation_bridge = pd.DataFrame(
        {
            "Method": ["DCF EV", "Comps EV", "Blended EV", "Blended Equity Value"],
            "Value": [
                summary.dcf_enterprise_value,
                summary.comps_enterprise_value,
                summary.blended_enterprise_value,
                summary.blended_equity_value,
            ],
        }
    )
    st.subheader("Valuation Bridge")
    st.bar_chart(valuation_bridge.set_index("Method"), use_container_width=True)

with dcf_tab:
    st.subheader("DCF Forecast")
    dcf_cols = st.columns(5)
    dcf_cols[0].metric("DCF Enterprise Value", format_money(summary.dcf_enterprise_value))
    dcf_cols[1].metric("DCF Equity Value", format_money(summary.dcf_equity_value))
    dcf_cols[2].metric("DCF Value / Share", format_currency(summary.dcf_value_per_share))
    dcf_cols[3].metric("Terminal Growth", f"{terminal_growth:.1%}")
    dcf_cols[4].metric("PV of Terminal Value", format_money(summary.discounted_terminal_value))
    forecast_display = format_money_columns(
        format_percent_columns(forecast, ["Revenue_Growth", "EBITDA_Margin", "FCF_Margin"]),
        ["Revenue", "EBITDA", "Depreciation", "EBIT", "Cash_Taxes", "Capex", "NWC_Balance", "NWC_Investment", "Free_Cash_Flow"],
    )
    st.dataframe(
        forecast_display,
        use_container_width=True,
        hide_index=True,
    )

with comps_tab:
    st.subheader("Size-Adjusted Comparable Company Valuation")
    comp_cols = st.columns(3)
    comp_cols[0].metric("Target EV / Revenue", f"{summary.implied_ev_revenue:.1f}x")
    comp_cols[1].metric("Target EV / EBITDA", f"{summary.implied_ev_ebitda:.1f}x")
    comp_cols[2].metric("EV/EBITDA Weight", f"{ebitda_multiple_weight:.0%}")
    comps_display = format_money_columns(
        format_percent_columns(comps, ["Revenue_Growth", "EBITDA_Margin"]),
        ["Revenue_Implied_EV", "EBITDA_Implied_EV", "Average_Implied_EV"],
    )
    st.dataframe(
        comps_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "EV_Revenue": st.column_config.NumberColumn("EV / Revenue", format="%.1fx"),
            "EV_EBITDA": st.column_config.NumberColumn("EV / EBITDA", format="%.1fx"),
            "Size_Adjusted_EV_Revenue": st.column_config.NumberColumn("Size-Adj. EV / Revenue", format="%.1fx"),
            "Size_Adjusted_EV_EBITDA": st.column_config.NumberColumn("Size-Adj. EV / EBITDA", format="%.1fx"),
        },
    )
    st.bar_chart(comps.set_index("Company")["Average_Implied_EV"], use_container_width=True)

with scenario_tab:
    st.subheader("Scenario Summary")
    scenario_display = format_percent_columns(
        scenario_summary,
        ["Revenue Growth", "Target EBITDA Margin", "Terminal Growth", "Upside / Downside"],
    )
    scenario_display["Value / Share"] = scenario_display["Value / Share"].map(lambda value: f"${value:.2f}")
    st.dataframe(
        scenario_display,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Value / Share Sensitivity")
    formatted = sensitivity.copy()
    for column in formatted.columns:
        if column != "Revenue Growth":
            formatted[column] = formatted[column].map(lambda value: f"${value:.2f}")
    st.dataframe(formatted, use_container_width=True, hide_index=True)

with memo_tab:
    st.subheader("Strategic Valuation Memo")
    memo = build_strategy_memo(summary, assumptions, company_name, scenario_summary)
    st.markdown(memo)
    st.download_button("Download memo", memo, "strategic_valuation_memo.md", "text/markdown")

with data_tab:
    st.subheader("Historical Financials")
    st.dataframe(financials, use_container_width=True, hide_index=True)
    st.subheader("Comparable Companies")
    st.dataframe(comps_source, use_container_width=True, hide_index=True)
