"""Format Slack response node."""
import json

from src.agent.state import AgentState
from src.agent.guardrails import redact_pii


def format_slack(state: AgentState) -> AgentState:
    """
    Format the final response for Slack.

    Converts to Slack mrkdwn format and ensures proper length.
    """
    messages = state.get("messages", [])
    sources = state.get("sources", [])

    if not messages:
        return {
            **state,
            "messages": [{"role": "assistant", "content": "I couldn't generate a response."}],
        }

    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, "content") else last_message.get("content", "")

    # Redact any PII that might have slipped through
    content = redact_pii(content)

    # Format sources as a list
    source_text = ""
    if sources:
        unique_sources = list(set(sources))
        source_lines = [f"• {s}" for s in unique_sources[:5]]  # Limit to 5 sources
        source_text = "\n\n_Sources:_ " + " | ".join(unique_sources[:5])

    # Ensure proper Slack formatting (already handled in generate, but double-check)
    # Slack uses mrkdwn: *bold*, _italic_, ```code```

    # Build final message
    final_message = content
    if source_text:
        final_message += source_text

    # Split if too long (Slack message limit ~4000 chars)
    if len(final_message) > 4000:
        # Split at paragraph boundary
        lines = final_message.split("\n")
        parts = []
        current = ""
        for line in lines:
            if len(current) + len(line) > 4000:
                if current:
                    parts.append(current)
                current = line
            else:
                current += "\n" + line if current else line
        if current:
            parts.append(current)
        final_message = parts[0] if parts else final_message[:4000]

    return {
        **state,
        "messages": [{"role": "assistant", "content": final_message}],
    }
