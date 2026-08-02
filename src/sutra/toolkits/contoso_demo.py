"""Contoso IT + Finance demo toolkit: two specialist subagents behind a
triage root agent, shared by `examples/live_workflow.py`,
`examples/finance_workflow.py`, and the `sutra` CLI so the tool/subagent
definitions live in exactly one place.
"""

from __future__ import annotations

from sutra.agents.subagent import SubagentProfile, SubagentRegistry
from sutra.core.harness import HANDOFF_TOOL_NAME
from sutra.core.permissions import PermissionLevel
from sutra.tools.registry import Tool, ToolParameter, ToolRegistry

ROOT_SYSTEM_PROMPT = (
    "You are the Contoso helpdesk triage agent. You have no tools except the handoff tool. "
    "For ANY IT/infrastructure issue, hand off to 'it_ops_agent'. For ANY financial/account "
    "request, hand off to 'finance_ops_agent'. Never try to answer either kind of request "
    "yourself."
)

# ---------------------------------------------------------------------------
# In-memory backends the tools operate against.
# ---------------------------------------------------------------------------

TICKETS = {
    "TICKET-4521": {"status": "open", "summary": "checkout-service returning HTTP 500 on /cart/checkout"},
}
SERVICES = {"checkout-service": {"status": "degraded", "restarts_today": 0}}
ACCOUNTS = {"ACC-1001": {"owner": "Contoso IT Ops", "balance_usd": 18_250.42}}


# -- IT ops tools -------------------------------------------------------------

async def check_ticket_status(ticket_id: str) -> dict:
    ticket = TICKETS.get(ticket_id)
    if not ticket:
        return {"error": f"unknown ticket '{ticket_id}'"}
    return {"ticket_id": ticket_id, **ticket}


async def search_knowledge_base(query: str) -> dict:
    return {
        "query": query,
        "article": "KB-0091: 500s on /cart/checkout are almost always resolved by restarting "
        "checkout-service to clear a stuck DB connection pool.",
    }


async def restart_service(service_name: str) -> dict:
    service = SERVICES.get(service_name)
    if not service:
        return {"status": "failed", "reason": f"unknown service '{service_name}'"}
    service["status"] = "healthy"
    service["restarts_today"] += 1
    return {"status": "restarted", "service_name": service_name, "new_state": "healthy"}


async def escalate_to_oncall(summary: str) -> dict:
    return {"status": "escalated", "summary": summary, "paged": "oncall-sre@contoso.com"}


# -- finance ops tools ---------------------------------------------------------

async def get_account_balance(account_id: str) -> dict:
    account = ACCOUNTS.get(account_id)
    if not account:
        return {"error": f"unknown account '{account_id}'"}
    return {"account_id": account_id, "balance_usd": account["balance_usd"]}


async def get_transaction_history(account_id: str, limit: int = 5) -> dict:
    return {"account_id": account_id, "transactions": [], "note": "no transactions in this demo ledger"}


async def send_customer_notification(recipient: str, message: str) -> dict:
    return {"status": "sent", "recipient": recipient, "message": message}


async def transfer_funds(from_account: str, to_account: str, amount_usd: float, memo: str) -> dict:
    account = ACCOUNTS.get(from_account)
    if not account:
        return {"status": "failed", "reason": f"unknown source account '{from_account}'"}
    if account["balance_usd"] < amount_usd:
        return {"status": "failed", "reason": "insufficient funds"}
    account["balance_usd"] -= amount_usd
    return {
        "status": "success",
        "from_account": from_account,
        "to_account": to_account,
        "amount_usd": amount_usd,
        "memo": memo,
        "new_balance_usd": account["balance_usd"],
    }


async def _handoff_placeholder(**_kwargs: object) -> None:
    raise RuntimeError("The handoff pseudo-tool must be intercepted by the harness loop, not executed directly.")


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register_tool(
        Tool(
            name="check_ticket_status",
            description="Look up the status and summary of a support ticket.",
            parameters=[ToolParameter("ticket_id", "string", "Ticket identifier, e.g. 'TICKET-4521'.")],
            handler=check_ticket_status,
            permission_level=PermissionLevel.LOW,
            allowed_agents=["it_ops_agent"],
        )
    )
    registry.register_tool(
        Tool(
            name="search_knowledge_base",
            description="Search the internal runbook/knowledge base for troubleshooting guidance.",
            parameters=[ToolParameter("query", "string", "Search query.")],
            handler=search_knowledge_base,
            permission_level=PermissionLevel.LOW,
            allowed_agents=["it_ops_agent"],
        )
    )
    registry.register_tool(
        Tool(
            name="restart_service",
            description="Restart a production service. HIGH-RISK: causes brief downtime for that service.",
            parameters=[ToolParameter("service_name", "string", "Service to restart, e.g. 'checkout-service'.")],
            handler=restart_service,
            permission_level=PermissionLevel.CRITICAL,
            allowed_agents=["it_ops_agent"],
        )
    )
    registry.register_tool(
        Tool(
            name="escalate_to_oncall",
            description="Page the on-call SRE with a summary when an issue can't be resolved directly.",
            parameters=[ToolParameter("summary", "string", "Brief summary of the issue for the on-call engineer.")],
            handler=escalate_to_oncall,
            permission_level=PermissionLevel.MEDIUM,
            allowed_agents=["it_ops_agent"],
        )
    )
    registry.register_tool(
        Tool(
            name="get_account_balance",
            description="Look up the current balance of a bank account.",
            parameters=[ToolParameter("account_id", "string", "The account identifier, e.g. 'ACC-1001'.")],
            handler=get_account_balance,
            permission_level=PermissionLevel.LOW,
            allowed_agents=["finance_ops_agent"],
        )
    )
    registry.register_tool(
        Tool(
            name="get_transaction_history",
            description="Fetch recent transactions for an account.",
            parameters=[
                ToolParameter("account_id", "string", "The account identifier."),
                ToolParameter("limit", "integer", "Max number of transactions to return.", required=False),
            ],
            handler=get_transaction_history,
            permission_level=PermissionLevel.LOW,
            allowed_agents=["finance_ops_agent"],
        )
    )
    registry.register_tool(
        Tool(
            name="send_customer_notification",
            description="Send a notification email/SMS to a customer.",
            parameters=[
                ToolParameter("recipient", "string", "Recipient email or phone number."),
                ToolParameter("message", "string", "Notification body."),
            ],
            handler=send_customer_notification,
            permission_level=PermissionLevel.MEDIUM,
            allowed_agents=["finance_ops_agent"],
        )
    )
    registry.register_tool(
        Tool(
            name="transfer_funds",
            description="Move money from one account to another. CRITICAL-RISK: moves real funds.",
            parameters=[
                ToolParameter("from_account", "string", "Source account id."),
                ToolParameter("to_account", "string", "Destination account id."),
                ToolParameter("amount_usd", "number", "Amount to transfer, in USD."),
                ToolParameter("memo", "string", "Reference memo (e.g. invoice number)."),
            ],
            handler=transfer_funds,
            permission_level=PermissionLevel.CRITICAL,
            allowed_agents=["finance_ops_agent"],
        )
    )
    registry.register_tool(
        Tool(
            name=HANDOFF_TOOL_NAME,
            description=(
                "Transfer execution control to a different specialized subagent. Use this for ANY "
                "IT/infrastructure request (hand off to 'it_ops_agent') or ANY financial/account "
                "request (hand off to 'finance_ops_agent') — you have no tools of your own."
            ),
            parameters=[
                ToolParameter("target_agent_id", "string", "The registered subagent id to hand off to."),
                ToolParameter("reason", "string", "Why control is being transferred."),
                ToolParameter("context_payload", "object", "Structured context to brief the new agent with.", required=False),
            ],
            handler=_handoff_placeholder,
            permission_level=PermissionLevel.LOW,
        )
    )
    return registry


def build_subagent_registry() -> SubagentRegistry:
    registry = SubagentRegistry()
    registry.register(
        SubagentProfile(
            agent_id="it_ops_agent",
            name="IT Operations Agent",
            system_prompt=(
                "You are the IT operations subagent for Contoso. You may check ticket status, search "
                "the knowledge base, escalate to on-call, and restart services. Restarting a service "
                "is high-risk and requires human approval before it executes — call the tool anyway "
                "when it's the right fix; the harness will pause for approval."
            ),
            allowed_tools={"check_ticket_status", "search_knowledge_base", "restart_service", "escalate_to_oncall"},
            description="Handles support tickets, runbook lookups, and service restarts.",
        )
    )
    registry.register(
        SubagentProfile(
            agent_id="finance_ops_agent",
            name="Finance Operations Agent",
            system_prompt=(
                "You are the finance operations subagent for Contoso. You may check balances, review "
                "transaction history, notify customers, and execute fund transfers. Fund transfers are "
                "high-risk and require human approval before execution — call the tool anyway when "
                "asked; the harness will pause for approval."
            ),
            allowed_tools={
                "get_account_balance",
                "get_transaction_history",
                "send_customer_notification",
                "transfer_funds",
            },
            description="Handles account balance checks, transaction history, and fund transfers.",
        )
    )
    return registry
