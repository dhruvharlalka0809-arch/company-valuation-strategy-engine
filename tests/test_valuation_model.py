import unittest

import pandas as pd

from src.valuation_model import (
    ValuationAssumptions,
    apply_downside_recommendation,
    build_comps_valuation,
    build_forecast,
    build_scenario_summary,
    build_sensitivity,
    build_strategy_memo,
    calculate_wacc,
    normalize_financials,
    summarize_valuation,
)


class ValuationModelTests(unittest.TestCase):
    def setUp(self):
        self.financials = pd.DataFrame(
            {
                "Year": [2022, 2023, 2024],
                "Revenue": [100.0, 112.0, 126.0],
                "EBITDA": [18.0, 21.0, 25.0],
                "EBIT": [12.0, 14.5, 18.0],
                "Net_Income": [7.5, 9.2, 11.0],
                "Free_Cash_Flow": [8.0, 10.5, 13.0],
                "Net_Debt": [30.0, 25.0, 20.0],
                "Shares_Outstanding": [10.0, 10.0, 10.0],
            }
        )
        self.comps = pd.DataFrame(
            {
                "Company": ["A", "B"],
                "Sector": ["Software", "Software"],
                "Revenue_Growth": [0.10, 0.12],
                "EBITDA_Margin": [0.20, 0.22],
                "EV_Revenue": [2.0, 3.0],
                "EV_EBITDA": [10.0, 12.0],
                "PE_Ratio": [20.0, 24.0],
            }
        )
        self.assumptions = ValuationAssumptions(
            revenue_growth=0.08,
            target_ebitda_margin=0.22,
            terminal_growth=0.025,
            size_discount=0.10,
        )

    def test_normalize_financials_adds_margin_fields(self):
        normalized = normalize_financials(self.financials)

        self.assertIn("Revenue_Growth", normalized.columns)
        self.assertIn("EBITDA_Margin", normalized.columns)
        self.assertAlmostEqual(normalized.iloc[-1]["EBITDA_Margin"], 25.0 / 126.0)

    def test_wacc_calculation_is_positive(self):
        self.assertGreater(calculate_wacc(self.assumptions), 0)

    def test_forecast_builds_five_year_dcf_inputs(self):
        forecast = build_forecast(self.financials, self.assumptions)

        self.assertEqual(len(forecast), 5)
        self.assertGreater(forecast.iloc[-1]["Revenue"], forecast.iloc[0]["Revenue"])
        self.assertIn("Free_Cash_Flow", forecast.columns)
        self.assertIn("NWC_Balance", forecast.columns)

    def test_comps_apply_size_discount(self):
        comps = build_comps_valuation(self.comps, 100.0, 20.0, self.assumptions)

        self.assertAlmostEqual(comps.iloc[0]["Size_Adjusted_EV_Revenue"], comps.iloc[0]["EV_Revenue"] * 0.9)
        self.assertIn("Average_Implied_EV", comps.columns)

    def test_summary_returns_recommendation_and_values(self):
        forecast, comps, summary = summarize_valuation(self.financials, self.comps, self.assumptions, 20.0)

        self.assertGreater(summary.blended_enterprise_value, 0)
        self.assertGreater(summary.blended_value_per_share, 0)
        self.assertGreater(summary.terminal_value_pct_dcf, 0)
        self.assertGreater(summary.implied_ev_ebitda, 0)
        self.assertIn(summary.recommendation, {"Undervalued", "Fairly valued", "Overvalued"})
        self.assertEqual(len(forecast), 5)
        self.assertFalse(comps.empty)

    def test_scenarios_and_sensitivity_are_available(self):
        scenarios = build_scenario_summary(self.financials, self.comps, self.assumptions, 20.0)
        sensitivity = build_sensitivity(self.financials, self.comps, self.assumptions, 20.0)

        self.assertEqual(set(scenarios["Scenario"]), {"Downside", "Base", "Upside"})
        self.assertEqual(len(sensitivity), 5)

    def test_downside_scenario_can_update_recommendation(self):
        forecast, _, summary = summarize_valuation(self.financials, self.comps, self.assumptions, 20.0)
        scenarios = build_scenario_summary(self.financials, self.comps, self.assumptions, 20.0)

        updated = apply_downside_recommendation(summary, forecast, self.financials, scenarios)

        self.assertIn(updated.recommendation, {"Undervalued", "Fairly valued", "Overvalued", "High-risk valuation"})
        self.assertTrue(updated.recommendation_reason)

    def test_strategy_memo_contains_actionable_sections(self):
        _, _, summary = summarize_valuation(self.financials, self.comps, self.assumptions, 20.0)
        scenarios = build_scenario_summary(self.financials, self.comps, self.assumptions, 20.0)
        memo = build_strategy_memo(summary, self.assumptions, "TestCo", scenarios)

        self.assertIn("Strategic Valuation Memo", memo)
        self.assertIn("Recommendation", memo)
        self.assertIn("Management priorities", memo)


if __name__ == "__main__":
    unittest.main()
