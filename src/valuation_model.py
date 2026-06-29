from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ValuationAssumptions:
    revenue_growth: float = 0.09
    target_ebitda_margin: float = 0.205
    tax_rate: float = 0.25
    capex_pct_revenue: float = 0.035
    nwc_pct_revenue: float = 0.015
    depreciation_pct_revenue: float = 0.03
    risk_free_rate: float = 0.045
    equity_risk_premium: float = 0.055
    beta: float = 1.15
    pre_tax_cost_of_debt: float = 0.075
    debt_weight: float = 0.25
    terminal_growth: float = 0.025
    forecast_years: int = 5
    size_discount: float = 0.10


@dataclass(frozen=True)
class ValuationSummary:
    dcf_enterprise_value: float
    dcf_equity_value: float
    dcf_value_per_share: float
    comps_enterprise_value: float
    blended_enterprise_value: float
    blended_equity_value: float
    blended_value_per_share: float
    current_share_price: float
    upside_downside: float
    wacc: float
    recommendation: str
    recommendation_reason: str


def load_financials(path: str) -> pd.DataFrame:
    return normalize_financials(pd.read_csv(path))


def normalize_financials(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Year",
        "Revenue",
        "EBITDA",
        "EBIT",
        "Net_Income",
        "Free_Cash_Flow",
        "Net_Debt",
        "Shares_Outstanding",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    output = df.copy()
    for column in required:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["Year", "Revenue"]).sort_values("Year")
    output["Revenue_Growth"] = output["Revenue"].pct_change().fillna(0)
    output["EBITDA_Margin"] = divide_series(output["EBITDA"], output["Revenue"])
    output["FCF_Margin"] = divide_series(output["Free_Cash_Flow"], output["Revenue"])
    return output.reset_index(drop=True)


def calculate_wacc(assumptions: ValuationAssumptions) -> float:
    cost_of_equity = assumptions.risk_free_rate + assumptions.beta * assumptions.equity_risk_premium
    after_tax_cost_of_debt = assumptions.pre_tax_cost_of_debt * (1 - assumptions.tax_rate)
    equity_weight = 1 - assumptions.debt_weight
    return equity_weight * cost_of_equity + assumptions.debt_weight * after_tax_cost_of_debt


def build_forecast(financials: pd.DataFrame, assumptions: ValuationAssumptions) -> pd.DataFrame:
    normalized = normalize_financials(financials)
    base = normalized.iloc[-1]
    base_margin = float(base["EBITDA_Margin"])
    base_year = int(base["Year"])
    previous_revenue = float(base["Revenue"])
    rows = []

    for year_index in range(1, assumptions.forecast_years + 1):
        year = base_year + year_index
        revenue = previous_revenue * (1 + assumptions.revenue_growth)
        margin_step = (assumptions.target_ebitda_margin - base_margin) / assumptions.forecast_years
        ebitda_margin = base_margin + margin_step * year_index
        ebitda = revenue * ebitda_margin
        depreciation = revenue * assumptions.depreciation_pct_revenue
        ebit = ebitda - depreciation
        cash_taxes = max(0.0, ebit * assumptions.tax_rate)
        capex = revenue * assumptions.capex_pct_revenue
        nwc_investment = (revenue - previous_revenue) * assumptions.nwc_pct_revenue
        free_cash_flow = ebit * (1 - assumptions.tax_rate) + depreciation - capex - nwc_investment
        rows.append(
            {
                "Year": year,
                "Revenue": revenue,
                "Revenue_Growth": assumptions.revenue_growth,
                "EBITDA_Margin": ebitda_margin,
                "EBITDA": ebitda,
                "Depreciation": depreciation,
                "EBIT": ebit,
                "Cash_Taxes": cash_taxes,
                "Capex": capex,
                "NWC_Investment": nwc_investment,
                "Free_Cash_Flow": free_cash_flow,
                "FCF_Margin": divide(free_cash_flow, revenue),
            }
        )
        previous_revenue = revenue

    return pd.DataFrame(rows)


def calculate_dcf(forecast: pd.DataFrame, assumptions: ValuationAssumptions) -> float:
    wacc = calculate_wacc(assumptions)
    discounted_fcf = 0.0
    for period, fcf in enumerate(forecast["Free_Cash_Flow"], start=1):
        discounted_fcf += float(fcf) / ((1 + wacc) ** period)

    final_fcf = float(forecast.iloc[-1]["Free_Cash_Flow"])
    terminal_value = final_fcf * (1 + assumptions.terminal_growth) / (wacc - assumptions.terminal_growth)
    discounted_terminal = terminal_value / ((1 + wacc) ** len(forecast))
    return discounted_fcf + discounted_terminal


def build_comps_valuation(comps: pd.DataFrame, latest_revenue: float, latest_ebitda: float, assumptions: ValuationAssumptions) -> pd.DataFrame:
    output = comps.copy()
    output["Size_Adjusted_EV_Revenue"] = output["EV_Revenue"] * (1 - assumptions.size_discount)
    output["Size_Adjusted_EV_EBITDA"] = output["EV_EBITDA"] * (1 - assumptions.size_discount)
    output["Revenue_Implied_EV"] = output["Size_Adjusted_EV_Revenue"] * latest_revenue
    output["EBITDA_Implied_EV"] = output["Size_Adjusted_EV_EBITDA"] * latest_ebitda
    output["Average_Implied_EV"] = output[["Revenue_Implied_EV", "EBITDA_Implied_EV"]].mean(axis=1)
    return output.sort_values("Average_Implied_EV").reset_index(drop=True)


def summarize_valuation(
    financials: pd.DataFrame,
    comps: pd.DataFrame,
    assumptions: ValuationAssumptions,
    current_share_price: float,
) -> tuple[pd.DataFrame, pd.DataFrame, ValuationSummary]:
    normalized = normalize_financials(financials)
    latest = normalized.iloc[-1]
    forecast = build_forecast(normalized, assumptions)
    dcf_ev = calculate_dcf(forecast, assumptions)
    comps_table = build_comps_valuation(
        comps,
        latest_revenue=float(latest["Revenue"]),
        latest_ebitda=float(latest["EBITDA"]),
        assumptions=assumptions,
    )
    comps_ev = float(comps_table["Average_Implied_EV"].median())
    blended_ev = (dcf_ev * 0.6) + (comps_ev * 0.4)
    net_debt = float(latest["Net_Debt"])
    shares = float(latest["Shares_Outstanding"])
    dcf_equity = dcf_ev - net_debt
    blended_equity = blended_ev - net_debt
    dcf_value_per_share = divide(dcf_equity, shares)
    blended_value_per_share = divide(blended_equity, shares)
    upside_downside = divide(blended_value_per_share - current_share_price, current_share_price)
    recommendation, reason = recommend(upside_downside, forecast, normalized)

    summary = ValuationSummary(
        dcf_enterprise_value=dcf_ev,
        dcf_equity_value=dcf_equity,
        dcf_value_per_share=dcf_value_per_share,
        comps_enterprise_value=comps_ev,
        blended_enterprise_value=blended_ev,
        blended_equity_value=blended_equity,
        blended_value_per_share=blended_value_per_share,
        current_share_price=current_share_price,
        upside_downside=upside_downside,
        wacc=calculate_wacc(assumptions),
        recommendation=recommendation,
        recommendation_reason=reason,
    )
    return forecast, comps_table, summary


def build_sensitivity(financials: pd.DataFrame, comps: pd.DataFrame, assumptions: ValuationAssumptions, current_share_price: float) -> pd.DataFrame:
    growth_rates = [assumptions.revenue_growth - 0.03, assumptions.revenue_growth - 0.015, assumptions.revenue_growth, assumptions.revenue_growth + 0.015, assumptions.revenue_growth + 0.03]
    terminal_rates = [assumptions.terminal_growth - 0.01, assumptions.terminal_growth - 0.005, assumptions.terminal_growth, assumptions.terminal_growth + 0.005, assumptions.terminal_growth + 0.01]
    rows = []
    for growth in growth_rates:
        row = {"Revenue Growth": f"{growth:.1%}"}
        for terminal_growth in terminal_rates:
            scenario = replace_assumptions(assumptions, revenue_growth=max(0.0, growth), terminal_growth=terminal_growth)
            _, _, summary = summarize_valuation(financials, comps, scenario, current_share_price)
            row[f"{terminal_growth:.1%} Terminal"] = summary.blended_value_per_share
        rows.append(row)
    return pd.DataFrame(rows)


def build_scenario_summary(financials: pd.DataFrame, comps: pd.DataFrame, assumptions: ValuationAssumptions, current_share_price: float) -> pd.DataFrame:
    scenarios = [
        ("Downside", max(0.0, assumptions.revenue_growth - 0.04), assumptions.target_ebitda_margin - 0.025, assumptions.terminal_growth - 0.005),
        ("Base", assumptions.revenue_growth, assumptions.target_ebitda_margin, assumptions.terminal_growth),
        ("Upside", assumptions.revenue_growth + 0.035, assumptions.target_ebitda_margin + 0.025, assumptions.terminal_growth + 0.005),
    ]
    rows = []
    for name, growth, margin, terminal_growth in scenarios:
        scenario = replace_assumptions(
            assumptions,
            revenue_growth=growth,
            target_ebitda_margin=max(0.05, margin),
            terminal_growth=terminal_growth,
        )
        _, _, summary = summarize_valuation(financials, comps, scenario, current_share_price)
        rows.append(
            {
                "Scenario": name,
                "Revenue Growth": growth,
                "Target EBITDA Margin": scenario.target_ebitda_margin,
                "Terminal Growth": terminal_growth,
                "Value / Share": summary.blended_value_per_share,
                "Upside / Downside": summary.upside_downside,
                "Recommendation": summary.recommendation,
            }
        )
    return pd.DataFrame(rows)


def build_strategy_memo(summary: ValuationSummary, assumptions: ValuationAssumptions, company_name: str, scenario_summary: pd.DataFrame) -> str:
    downside = scenario_summary.loc[scenario_summary["Scenario"] == "Downside"].iloc[0]
    upside = scenario_summary.loc[scenario_summary["Scenario"] == "Upside"].iloc[0]
    return f"""### Strategic Valuation Memo

**Company:** {company_name}

**Recommendation:** {summary.recommendation}

**Valuation readout:** The blended valuation indicates {format_money(summary.blended_enterprise_value)} enterprise value and {format_money(summary.blended_equity_value)} equity value, implying {format_currency(summary.blended_value_per_share)} per share versus the current price of {format_currency(summary.current_share_price)}.

**DCF view:** DCF enterprise value is {format_money(summary.dcf_enterprise_value)} using a {summary.wacc:.1%} WACC and {assumptions.terminal_growth:.1%} terminal growth.

**Comparable company view:** Size-adjusted public comps imply {format_money(summary.comps_enterprise_value)} enterprise value after applying a {assumptions.size_discount:.0%} discount for scale and liquidity.

**Scenario range:** Downside value per share is {format_currency(float(downside["Value / Share"]))}; upside value per share is {format_currency(float(upside["Value / Share"]))}.

**Strategic interpretation:** {summary.recommendation_reason}

**Management priorities:** Defend revenue growth, protect EBITDA margin expansion, improve cash conversion, and validate whether the market multiple gap is justified by size, liquidity, or execution risk.
"""


def recommend(upside_downside: float, forecast: pd.DataFrame, historical: pd.DataFrame) -> tuple[str, str]:
    final_margin = float(forecast.iloc[-1]["EBITDA_Margin"])
    latest_margin = float(historical.iloc[-1]["EBITDA_Margin"])
    margin_expansion = final_margin - latest_margin
    if upside_downside >= 0.20 and margin_expansion >= 0:
        return "Undervalued", "Upside exceeds 20% and the operating forecast supports stable or expanding margins."
    if upside_downside <= -0.10:
        return "Overvalued", "Implied value is materially below the current share price, creating limited valuation support."
    return "Fairly valued", "Valuation support is balanced; the case depends on execution, growth durability, and margin delivery."


def replace_assumptions(assumptions: ValuationAssumptions, **updates) -> ValuationAssumptions:
    values = assumptions.__dict__.copy()
    values.update(updates)
    return ValuationAssumptions(**values)


def format_money(value: float) -> str:
    return f"${value:,.1f}M"


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def divide_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, pd.NA)).fillna(0)
