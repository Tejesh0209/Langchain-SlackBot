"""Progress reporting for agent nodes back to Slack."""
from typing import Callable, Awaitable

# Registry: thread_ts -> async callback(step: str)
_callbacks: dict[str, Callable[[str], Awaitable[None]]] = {}


def register(thread_ts: str, callback: Callable[[str], Awaitable[None]]) -> None:
    _callbacks[thread_ts] = callback


def unregister(thread_ts: str) -> None:
    _callbacks.pop(thread_ts, None)


async def report(thread_ts: str, step: str) -> None:
    cb = _callbacks.get(thread_ts)
    if cb:
        try:
            await cb(step)
        except Exception:
            pass  # Never let progress updates crash the agent
