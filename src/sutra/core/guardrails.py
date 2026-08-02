from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Pattern, Tuple

from sutra.core.exceptions import GuardrailViolation


class GuardrailAction(str, Enum):
    BLOCK = "block"
    REDACT = "redact"


@dataclass
class GuardrailRule:
    name: str
    pattern: Pattern[str]
    action: GuardrailAction = GuardrailAction.BLOCK
    replacement: str = "[REDACTED]"


_DEFAULT_INJECTION_PATTERNS: List[Tuple[str, str]] = [
    ("ignore_previous_instructions", r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions"),
    ("disregard_system_prompt", r"disregard\s+(the\s+)?(system|your)\s+(prompt|instructions)"),
    ("role_override", r"you\s+are\s+now\s+(a|an)\s+\w+"),
    ("reveal_system_prompt", r"(reveal|print|show|repeat)\s+(your|the)\s+system\s+prompt"),
    ("act_as_dan", r"\bDAN\b|\bdo\s+anything\s+now\b"),
    ("exfiltrate_secrets", r"(dump|leak|exfiltrate)\s+(all\s+)?(secrets|credentials|api\s*keys|passwords?)"),
]

_DEFAULT_PII_PATTERNS: List[Tuple[str, str]] = [
    ("credit_card", r"\b(?:\d[ -]*?){13,16}\b"),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("email", r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
]


@dataclass
class GuardrailConfig:
    enable_default_injection_rules: bool = True
    enable_default_pii_redaction: bool = True
    blocklist_terms: List[str] = field(default_factory=list)


class Guardrail:
    """Programmatic input/output sanitizer.

    Input rules default to BLOCK — a detected prompt-injection vector should
    never reach the model. Output rules default to REDACT — PII that slips
    into a model response should be scrubbed, not used to kill the whole
    turn.
    """

    def __init__(self, config: Optional[GuardrailConfig] = None) -> None:
        self.config = config or GuardrailConfig()
        self.input_rules: List[GuardrailRule] = []
        self.output_rules: List[GuardrailRule] = []

        if self.config.enable_default_injection_rules:
            for name, pattern in _DEFAULT_INJECTION_PATTERNS:
                self.add_input_rule(name, pattern, GuardrailAction.BLOCK)

        for term in self.config.blocklist_terms:
            self.add_input_rule(f"blocklist:{term}", re.escape(term), GuardrailAction.BLOCK)

        if self.config.enable_default_pii_redaction:
            for name, pattern in _DEFAULT_PII_PATTERNS:
                self.add_output_rule(name, pattern, GuardrailAction.REDACT)

    def add_input_rule(self, name: str, pattern: str, action: GuardrailAction = GuardrailAction.BLOCK) -> None:
        self.input_rules.append(GuardrailRule(name, re.compile(pattern, re.IGNORECASE), action))

    def add_output_rule(self, name: str, pattern: str, action: GuardrailAction = GuardrailAction.REDACT) -> None:
        self.output_rules.append(GuardrailRule(name, re.compile(pattern, re.IGNORECASE), action))

    def sanitize_input(self, text: str) -> str:
        return self._run(text, self.input_rules, direction="input")

    def sanitize_output(self, text: str) -> str:
        return self._run(text, self.output_rules, direction="output")

    def _run(self, text: str, rules: List[GuardrailRule], *, direction: str) -> str:
        result = text
        for rule in rules:
            match = rule.pattern.search(result)
            if not match:
                continue
            if rule.action == GuardrailAction.BLOCK:
                raise GuardrailViolation(
                    f"Guardrail '{rule.name}' blocked {direction} payload.",
                    rule_name=rule.name,
                    matched_text=match.group(0),
                    direction=direction,
                )
            result = rule.pattern.sub(rule.replacement, result)
        return result
