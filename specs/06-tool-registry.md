# Feature: Tool Registry

**Module:** `src/sutra/tools/registry.py`  
**Classes:** `Tool`, `ToolParameter`, `ToolRegistry`

## Purpose

Manages the set of callable functions available to agents. Handles schema generation for the model's function-calling API, argument validation, and safe async execution — ensuring a tool error never crashes the harness loop.

## Current Behavior

### `ToolParameter`

```python
@dataclass
class ToolParameter:
    name: str
    type: str           # JSON schema primitive: "string" | "number" | "integer" | "boolean" | "object" | "array"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
```

`to_json_schema()` emits `{"type": ..., "description": ..., "enum": [...]}` (enum omitted if None).

### `Tool`

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: List[ToolParameter]
    handler: ToolHandler          # async callable
    permission_level: PermissionLevel = PermissionLevel.LOW
    allowed_agents: Optional[List[str]] = None   # None = any agent
```

`to_schema()` emits the Anthropic/OpenAI-compatible function-calling schema:

```json
{
  "name": "...",
  "description": "...",
  "input_schema": {
    "type": "object",
    "properties": { ... },
    "required": [...]
  }
}
```

### `ToolRegistry`

#### Registering tools (decorator API)

```python
registry = ToolRegistry()

@registry.register(
    "get_weather",
    "Get the current weather for a city.",
    [ToolParameter("city", "string", "City name")],
    permission_level=PermissionLevel.LOW,
)
async def get_weather(city: str) -> str:
    ...
```

`register()` returns a decorator that validates the handler is `async def` and stores the `Tool`.

`register_tool(tool: Tool)` is the direct API for pre-constructed `Tool` objects.

#### Retrieving schemas

```python
registry.schemas_for(agent_id="root", allowed_names=None)
```

Filters by:
1. `allowed_names` — if provided, only tools in this list pass through (subagent allow-list).
2. `tool.allowed_agents` — if set on the tool, only those agent IDs can see the tool.

#### Executing tools

```python
result = await registry.execute("get_weather", {"city": "London"})
```

1. `get(name)` — raises `ToolNotFoundError` if not registered.
2. `validate_arguments(tool, arguments)` — raises `ToolExecutionError` on missing required args.
3. Inspects handler signature; filters `arguments` to only accepted parameter names.
4. `await tool.handler(**filtered)` — wraps any exception in `ToolExecutionError` so it never propagates raw through the harness.

#### Argument parsing

```python
ToolRegistry.parse_arguments(raw: str) -> Dict[str, Any]
```

Parses a raw JSON string from streaming tool-call deltas. Raises `ToolExecutionError` if it's not valid JSON or doesn't decode to a dict.

### Permission Levels

| Level | Behavior |
|---|---|
| `LOW` | Executed immediately |
| `MEDIUM` | Executed immediately (not gated by default `PermissionGate`) |
| `HIGH` | Intercepted by `PermissionGate` (requires human approval) |
| `CRITICAL` | Intercepted by `PermissionGate` (requires human approval) |

Default gate threshold is `HIGH` — both `HIGH` and `CRITICAL` tools pause for approval.

## Enhancement Ideas

- **Synchronous handler support**: currently requires `async def`; add auto-wrapping via `asyncio.to_thread` for sync handlers.
- **Tool versioning**: add a `version` field so multiple versions of a tool can coexist and the model can be told which version it's calling.
- **Input schema validation via JSON Schema**: `validate_arguments` currently only checks required fields; add full JSON Schema validation of types and enum membership.
- **Output schema declaration**: add `output_schema: Optional[Dict]` to `Tool` so callers know what shape to expect — enabling structured output parsing.
- **Tool discovery endpoint**: expose `GET /tools` on the HTTP server to list all registered tools with their schemas.
- **Tool timeout**: add `timeout_seconds: Optional[float]` per tool; `execute()` wraps in `asyncio.wait_for`.
- **Tool middleware / hooks**: pre/post execution hooks for logging, metrics, caching, or result transformation.
- **Result caching**: for deterministic, idempotent tools (e.g. `check_account_balance`), add optional TTL-based caching to avoid redundant API calls in the same session.
- **Tool hot-reload**: allow adding/removing tools from a running registry without restart (important for plugin architectures).
- **Decorator for sync tools**: `@registry.register_sync(...)` that wraps the handler in `asyncio.to_thread` automatically.
