from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from sutra.core.exceptions import BudgetExceededError


@dataclass
class ModelPricing:
    """USD cost per 1M tokens, input/output priced separately."""

    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * self.input_per_million + (
            output_tokens / 1_000_000
        ) * self.output_per_million


@dataclass
class BudgetConfig:
    max_usd: float = 5.0
    max_input_tokens: int = 200_000
    max_output_tokens: int = 50_000
    max_steps: int = 40
    warn_ratio: float = 0.75  # soft warning once usage crosses this fraction of any limit


@dataclass
class BudgetUsage:
    usd_spent: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    steps: int = 0
    last_updated: float = field(default_factory=time.time)


class Budget:
    """Deterministic, hard-limit safety layer.

    Every dimension is checked *before* the resource is consumed where
    possible (the step counter), and reconciled immediately after a model
    call for dimensions only knowable post-hoc (tokens, USD). Crossing a
    hard limit raises `BudgetExceededError`; the harness loop is responsible
    for catching it and halting gracefully rather than looping forever.
    """

    def __init__(self, config: BudgetConfig, pricing: Optional[ModelPricing] = None) -> None:
        self.config = config
        self.pricing = pricing or ModelPricing(input_per_million=3.0, output_per_million=15.0)
        self.usage = BudgetUsage()
        self._lock = asyncio.Lock()

    async def check_step(self) -> None:
        async with self._lock:
            if self.usage.steps + 1 > self.config.max_steps:
                raise BudgetExceededError(
                    f"Step budget exceeded: {self.usage.steps}/{self.config.max_steps} steps used.",
                    dimension="steps",
                    used=self.usage.steps,
                    limit=self.config.max_steps,
                )
            self.usage.steps += 1
            self.usage.last_updated = time.time()

    async def record_usage(self, *, input_tokens: int, output_tokens: int) -> float:
        """Record actual token usage from a completed model call and return its cost.

        Raises `BudgetExceededError` if this usage pushes any hard limit past
        its ceiling. The call already happened — the harness must refuse to
        issue the *next* one, not undo this one.
        """
        async with self._lock:
            cost = self.pricing.cost(input_tokens, output_tokens)
            self.usage.input_tokens += input_tokens
            self.usage.output_tokens += output_tokens
            self.usage.usd_spent += cost
            self.usage.last_updated = time.time()

            if self.usage.usd_spent > self.config.max_usd:
                raise BudgetExceededError(
                    f"USD budget exceeded: ${self.usage.usd_spent:.4f}/${self.config.max_usd:.2f}.",
                    dimension="usd",
                    used=self.usage.usd_spent,
                    limit=self.config.max_usd,
                )
            if self.usage.input_tokens > self.config.max_input_tokens:
                raise BudgetExceededError(
                    f"Input token budget exceeded: {self.usage.input_tokens}/{self.config.max_input_tokens}.",
                    dimension="input_tokens",
                    used=self.usage.input_tokens,
                    limit=self.config.max_input_tokens,
                )
            if self.usage.output_tokens > self.config.max_output_tokens:
                raise BudgetExceededError(
                    f"Output token budget exceeded: {self.usage.output_tokens}/{self.config.max_output_tokens}.",
                    dimension="output_tokens",
                    used=self.usage.output_tokens,
                    limit=self.config.max_output_tokens,
                )
            return cost

    def warnings(self) -> Dict[str, float]:
        """Return {dimension: ratio_used} for any dimension past `warn_ratio`."""
        out: Dict[str, float] = {}
        checks = {
            "usd": (self.usage.usd_spent, self.config.max_usd),
            "input_tokens": (self.usage.input_tokens, self.config.max_input_tokens),
            "output_tokens": (self.usage.output_tokens, self.config.max_output_tokens),
            "steps": (self.usage.steps, self.config.max_steps),
        }
        for dim, (used, limit) in checks.items():
            if limit > 0 and (used / limit) >= self.config.warn_ratio:
                out[dim] = used / limit
        return out

    def snapshot(self) -> Dict[str, float]:
        return {
            "usd_spent": round(self.usage.usd_spent, 6),
            "usd_limit": self.config.max_usd,
            "input_tokens": self.usage.input_tokens,
            "input_token_limit": self.config.max_input_tokens,
            "output_tokens": self.usage.output_tokens,
            "output_token_limit": self.config.max_output_tokens,
            "steps": self.usage.steps,
            "step_limit": self.config.max_steps,
        }
