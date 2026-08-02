class SutraError(Exception):
    """Base class for all Sutra runtime errors."""


class BudgetExceededError(SutraError):
    def __init__(self, message: str, *, dimension: str, used: float, limit: float) -> None:
        super().__init__(message)
        self.dimension = dimension
        self.used = used
        self.limit = limit


class GuardrailViolation(SutraError):
    def __init__(self, message: str, *, rule_name: str, matched_text: str, direction: str) -> None:
        super().__init__(message)
        self.rule_name = rule_name
        self.matched_text = matched_text
        self.direction = direction  # "input" | "output"


class PermissionDeniedError(SutraError):
    def __init__(self, message: str, *, tool_name: str, request_id: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.request_id = request_id


class HandoffError(SutraError):
    pass


class ConsultError(SutraError):
    """Raised when a `__consult__` request is malformed or its target is invalid.

    Unlike `HandoffError` (which fails the whole run), a `ConsultError` is
    always caught locally by `_handle_consult()` and turned into an
    error-flavored tool result — the calling agent's turn continues.
    """


class ToolExecutionError(SutraError):
    def __init__(self, message: str, *, tool_name: str, original_error: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.original_error = original_error


class ToolNotFoundError(SutraError):
    pass


class ModelInvocationError(SutraError):
    pass
