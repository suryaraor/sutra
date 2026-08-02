# Feature: Guardrails

**Module:** `src/sutra/core/guardrails.py`  
**Classes:** `Guardrail`, `GuardrailConfig`, `GuardrailRule`, `GuardrailAction`

## Purpose

Programmatic input/output sanitizer that runs at system boundaries — before user input reaches the model, and before tool output is appended to the conversation. Blocks prompt injection attempts and redacts PII from model/tool outputs.

## Current Behavior

### `GuardrailAction`

| Action | Effect |
|---|---|
| `BLOCK` | Raises `GuardrailViolation` — halts processing entirely |
| `REDACT` | Replaces matched text with a `replacement` string (default `"[REDACTED]"`) |

Input rules default to `BLOCK`. Output rules default to `REDACT`.

### `GuardrailConfig`

| Field | Default | Description |
|---|---|---|
| `enable_default_injection_rules` | `True` | Load the built-in prompt injection blocklist |
| `enable_default_pii_redaction` | `True` | Load credit card / SSN redaction rules |
| `redact_emails_in_output` | `False` | Opt-in email redaction (off by default — email is routinely legitimate tool output) |
| `blocklist_terms` | `[]` | Additional literal terms to block in input |

### Built-in Injection Patterns (input, BLOCK)

| Rule Name | Pattern |
|---|---|
| `ignore_previous_instructions` | `ignore (all )?(previous|prior|above) instructions` |
| `disregard_system_prompt` | `disregard (the )?(system|your) (prompt|instructions)` |
| `role_override` | `you are now (a|an) \w+` |
| `reveal_system_prompt` | `(reveal|print|show|repeat) (your|the) system prompt` |
| `act_as_dan` | `\bDAN\b` or `do anything now` |
| `exfiltrate_secrets` | `(dump|leak|exfiltrate) (all )?(secrets|credentials|api keys|passwords?)` |

### Built-in PII Patterns (output, REDACT)

| Rule Name | Pattern | Always on? |
|---|---|---|
| `credit_card` | 13–16 digit sequences | Yes (`enable_default_pii_redaction`) |
| `ssn` | `###-##-####` | Yes |
| `email` | standard email pattern | No (`redact_emails_in_output=True`) |

### API

```python
guardrail = Guardrail(config=GuardrailConfig(blocklist_terms=["competitor_name"]))

# Add rules programmatically
guardrail.add_input_rule("custom", r"pattern", GuardrailAction.BLOCK)
guardrail.add_output_rule("api_key", r"sk-[a-zA-Z0-9]{32,}", GuardrailAction.REDACT)

# Apply at boundaries
clean = guardrail.sanitize_input(user_text)    # raises GuardrailViolation if blocked
safe  = guardrail.sanitize_output(tool_result) # replaces PII with [REDACTED]
```

### Integration Points

- **Input**: called at the top of `harness.run()` before the user message is appended to state.
- **Tool output**: called in `_dispatch_tool_call()` and `resume_after_permission()` before appending a `TOOL_RESULT` message.
- **Model output**: _not_ currently applied to raw model text (only tool results). This is a gap — model could echo back injected content.

### `GuardrailViolation` Exception

```python
class GuardrailViolation(SutraError):
    rule_name: str
    matched_text: str
    direction: str   # "input" | "output"
```

Caught in `harness.run()` → emits `GUARDRAIL_BLOCK` SSE event → run terminates with `FAILED`.

## Enhancement Ideas

- **Apply guardrail to model output**: currently only tool results are sanitized post-model; run `sanitize_output` on `assistant_text` before appending to state to catch echoed injection payloads.
- **Semantic injection detection**: regex patterns miss paraphrased attacks ("Please disregard your guidelines"). Add an embedding-based or LLM-judge layer for higher-coverage injection detection.
- **Per-agent rule sets**: give each subagent its own `Guardrail` instance with domain-appropriate rules (e.g. finance agent blocks account numbers in output).
- **REDACT with reason logging**: when a REDACT fires, record the rule name and what was redacted to the audit log without exposing the actual matched text.
- **Allowlist for false positives**: some legitimate inputs match injection patterns (e.g. cybersecurity educators discussing DAN). Add a token-scoped allowlist that a trusted operator can set.
- **Output rule for API keys**: add a built-in rule to redact common API key formats (`sk-...`, `Bearer ...`, `ghp_...`) from tool output.
- **Configurable block message**: currently `GuardrailViolation.message` is always "Guardrail '{name}' blocked {direction} payload." — allow a custom user-facing message per rule for better UX.
- **Metrics emission**: emit a Prometheus counter / structured log entry on every block or redact hit for monitoring dashboards.
