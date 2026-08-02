import pytest

from sutra.core.budget import Budget, BudgetConfig, ModelPricing
from sutra.core.exceptions import BudgetExceededError


async def test_step_budget_raises_once_exhausted():
    budget = Budget(BudgetConfig(max_steps=2, max_usd=100, max_input_tokens=100_000, max_output_tokens=100_000))
    await budget.check_step()
    await budget.check_step()
    with pytest.raises(BudgetExceededError) as exc_info:
        await budget.check_step()
    assert exc_info.value.dimension == "steps"


async def test_usd_budget_raises_when_cost_exceeds_cap():
    budget = Budget(
        BudgetConfig(max_usd=0.01, max_steps=100, max_input_tokens=1_000_000, max_output_tokens=1_000_000),
        pricing=ModelPricing(input_per_million=1000.0, output_per_million=1000.0),
    )
    with pytest.raises(BudgetExceededError) as exc_info:
        await budget.record_usage(input_tokens=1000, output_tokens=0)
    assert exc_info.value.dimension == "usd"


async def test_warnings_fire_past_ratio_threshold():
    budget = Budget(BudgetConfig(max_steps=10, max_usd=100, max_input_tokens=100, max_output_tokens=100, warn_ratio=0.5))
    await budget.record_usage(input_tokens=60, output_tokens=0)
    warnings = budget.warnings()
    assert "input_tokens" in warnings
    assert warnings["input_tokens"] == pytest.approx(0.6)


async def test_snapshot_reports_usage_and_limits():
    budget = Budget(BudgetConfig(max_steps=5, max_usd=1.0, max_input_tokens=1000, max_output_tokens=1000))
    await budget.check_step()
    snap = budget.snapshot()
    assert snap["steps"] == 1
    assert snap["step_limit"] == 5
