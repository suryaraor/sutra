# Feature: Subagents

**Module:** `src/sutra/agents/subagent.py`  
**Classes:** `SubagentProfile`, `SubagentRegistry`

## Purpose

Specialist agents that operate within the same harness loop but with their own system prompt and a restricted subset of tools. The root agent can hand off to a subagent when a request falls within that agent's domain; the subagent can only call tools in its `allowed_tools` set, preventing cross-domain privilege escalation.

## Current Behavior

### `SubagentProfile`

```python
@dataclass
class SubagentProfile:
    agent_id: str              # unique identifier, e.g. "it_ops_agent"
    name: str                  # human-readable name
    system_prompt: str         # domain-specific instructions
    allowed_tools: Set[str]    # tool names this agent may call
    model_override: Optional[str] = None  # not yet wired into harness
    description: str = ""      # shown in handoff briefings
```

### `SubagentRegistry`

Holds all registered profiles. The harness consults it on every model call to determine which tools are visible.

| Method | Description |
|---|---|
| `register(profile)` | Add a profile |
| `get(agent_id)` | Retrieve profile, raises `KeyError` if not found |
| `has(agent_id)` | Existence check |
| `ids()` | Sorted list of registered agent IDs |
| `allowed_tools_for(agent_id)` | Returns `sorted(profile.allowed_tools)` for subagents, `None` for `"root"` (no restriction) |

### Tool Visibility Enforcement

In `harness.py`:

```python
allowed_tools = self.subagent_registry.allowed_tools_for(agent_id)
tool_schemas = self.tool_registry.schemas_for(agent_id, allowed_tools)
```

`ToolRegistry.schemas_for()` filters by both `allowed_names` (subagent allow-list) and `tool.allowed_agents` (tool-side restriction). A subagent only sees tool schemas in the intersection of both.

### System Prompt Composition

When a subagent is active:

```
{memory context block (if memory enabled)}

{root system_prompt}

[Active subagent: {profile.name}]
{profile.system_prompt}
```

The root system prompt is always included as the outer frame; the subagent appends its specialist instructions beneath it.

### Bundled Subagents (in `toolkits/`)

**`contoso_demo.py`** registers two profiles:

- `it_ops_agent` — IT Operations; allowed tools: `check_system_status`, `get_incident_details`, `escalate_incident`, `restart_service`
- `finance_ops_agent` — Finance Operations; allowed tools: `check_account_balance`, `get_transaction_history`, `validate_transfer`, `transfer_funds`

**`simple_demo.py`** — single-agent setup (no subagents registered).

## Enhancement Ideas

- **`model_override` wiring**: `SubagentProfile.model_override` field exists but is not yet used; wire it into `harness.py` so subagents can run on a different (cheaper/faster) model than the root agent.
- **Dynamic subagent registration**: allow subagents to be registered at runtime via an HTTP endpoint or a config file reload, without restarting the server.
- **Subagent budget quotas**: give each profile a budget fraction so a single subagent can't exhaust the shared pool.
- **Agent-to-agent handoffs**: currently only root → subagent is supported; allow subagents to hand off to other subagents (multi-hop specialist routing).
- **Automatic agent popping (return handoff)**: add a `__handoff_back__` pseudo-tool so a subagent can explicitly return control to its caller; the harness would `pop_agent()` on the state stack.
- **Subagent health probes**: before handing off, verify the target agent has tools available and isn't over its quota.
- **Tool audit log per agent**: record which agent called which tool at which step for per-agent attribution in budget reporting.
- **Agent capability declaration**: extend `SubagentProfile` with a `capabilities: List[str]` field so the root agent (or a router LLM) can select the right specialist without needing the exact `agent_id`.
