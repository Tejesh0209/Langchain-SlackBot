"""Slack Bolt event handlers for the Northstar Slack Bot."""
import asyncio
import logging
from typing import Any

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_sdk.web.async_client import AsyncWebClient as WebClient

from src.config import settings
from src.agent.graph import run_agent
from src.agent import progress


logger = logging.getLogger(__name__)


# Initialize Slack Bolt app
app = AsyncApp(token=settings.slack_bot_token)


# Track processed event IDs for deduplication
_processed_events: set[str] = set()
_MAX_EVENTS = 1000


def is_event_processed(event_id: str) -> bool:
    """Check if an event has already been processed."""
    if event_id in _processed_events:
        return True

    # Add to set, maintaining max size
    if len(_processed_events) >= _MAX_EVENTS:
        _processed_events.clear()
    _processed_events.add(event_id)
    return False


def format_thinking_message() -> dict[str, Any]:
    """Format the initial 'thinking' message."""
    return {
        "text": "Thinking...",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Thinking...* I'll look into this for you.",
                },
            }
        ],
    }


def format_progress_message(step: str) -> dict[str, Any]:
    """Format a progress update message."""
    step_labels = {
        "classify": "Classifying query",
        "sql": "Searching database",
        "search": "Searching documents",
        "generate": "Generating response",
        "format": "Formatting response",
    }
    label = step_labels.get(step.lower(), step)
    return {
        "text": f"{label}...",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{label}...*",
                },
            }
        ],
    }


async def update_progress(
    client: WebClient,
    channel: str,
    ts: str,
    step: str,
) -> None:
    """Update the thinking message with progress."""
    try:
        await client.chat_update(
            channel=channel,
            ts=ts,
            **format_progress_message(step),
        )
    except Exception as e:
        logger.warning(f"Failed to update progress: {e}")


@app.event("app_mention")
async def handle_app_mention(
    event: dict[str, Any],
    body: dict[str, Any],
    client: WebClient,
    logger: logging.Logger,
) -> None:
    """
    Handle @mention events in Slack.

    This is the main entry point for user messages.
    """
    # Extract event data — event_id lives at the body level, not event level
    event_id = body.get("event_id", event.get("event_ts", ""))
    channel = event.get("channel", "")
    user = event.get("user", "")
    text = event.get("text", "")
    thread_ts = event.get("thread_ts", "") or event.get("ts", "")

    # Deduplicate
    if is_event_processed(event_id):
        logger.info(f"Skipping duplicate event: {event_id}")
        return

    # Remove all <@USERID> mentions from the text
    import re
    clean_text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

    if not clean_text:
        await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text="I didn't catch that. What would you like to know?",
        )
        return

    logger.info(f"Received query from user {user}: {clean_text[:100]}")

    # Fetch thread history for conversation context (last 10 messages)
    conversation_history = []
    try:
        replies = await client.conversations_replies(
            channel=channel, ts=thread_ts, limit=10
        )
        for msg in replies.get("messages", [])[:-1]:  # exclude the current message
            msg_text = re.sub(r"<@[A-Z0-9]+>", "", msg.get("text", "")).strip()
            if not msg_text:
                continue
            role = "assistant" if msg.get("bot_id") else "user"
            conversation_history.append({"role": role, "content": msg_text})
    except Exception:
        pass  # Thread history is best-effort

    # Post initial thinking message
    try:
        response = await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            **format_thinking_message(),
        )
        thinking_ts = response.get("ts", "")
    except Exception as e:
        logger.error(f"Failed to post thinking message: {e}")
        thinking_ts = ""

    # Register progress callback so agent nodes can update the Slack message
    async def on_progress(step: str) -> None:
        if thinking_ts:
            await update_progress(client, channel, thinking_ts, step)

    progress.register(thread_ts, on_progress)

    # Run the agent
    try:
        final_state = await run_agent(
            user_message=clean_text,
            channel_id=channel,
            thread_ts=thread_ts,
            thinking_msg_ts=thinking_ts,
            conversation_history=conversation_history,
        )

        # Extract the final answer
        messages = final_state.get("messages", [])
        if messages:
            last = messages[-1]
            final_message = last.content if hasattr(last, "content") else last.get("content", "I couldn't generate a response.")
        else:
            final_message = "I couldn't find an answer to your question."

        # Update the thinking message with the final answer
        if thinking_ts:
            try:
                await client.chat_update(
                    channel=channel,
                    ts=thinking_ts,
                    text=final_message,
                    blocks=[
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": final_message,
                            },
                        }
                    ],
                )
            except Exception as e:
                logger.warning(f"Failed to update message, posting new: {e}")
                await client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text=final_message,
                )
        else:
            await client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=final_message,
            )

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        error_message = f"I encountered an error processing your question: {str(e)[:200]}"

        if thinking_ts:
            await client.chat_update(
                channel=channel,
                ts=thinking_ts,
                text=error_message,
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": error_message,
                        },
                    }
                ],
            )
        else:
            await client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=error_message,
            )
    finally:
        progress.unregister(thread_ts)


@app.event("message")
async def handle_message(
    event: dict[str, Any],
    client: WebClient,
    logger: logging.Logger,
) -> None:
    """
    Handle direct messages (not @mentions).

    Only respond to messages in DM channels.
    """
    channel = event.get("channel", "")
    channel_type = event.get("channel_type", "")

    # Only respond to DMs
    if channel_type != "im":
        return

    # This would handle direct message conversations
    # Similar logic to app_mention but without the mention
    pass


# Socket Mode handler (created inside async context to avoid event loop issues)
def create_handler() -> AsyncSocketModeHandler:
    return AsyncSocketModeHandler(app=app, app_token=settings.slack_app_token)
