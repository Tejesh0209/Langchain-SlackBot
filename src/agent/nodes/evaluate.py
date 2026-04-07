"""Evaluate results node for LangGraph."""
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.agent.guardrails import validate_output
from src.config import settings


EVALUATE_PROMPT = """Evaluate whether the retrieved results are sufficient to answer the user's question.

User question: {question}

Retrieved results:
{results}

Are these results sufficient to provide a comprehensive answer? Consider:
1. Do we have enough information to answer the main question?
2. Are there gaps that require additional search?
3. Is the information relevant and accurate?

Respond with JSON:
{{
    "sufficient": true | false,
    "reasoning": "brief explanation",
    "gaps": ["what's missing if anything"]
}}
"""


async def evaluate(state: AgentState) -> AgentState:
    """
    Evaluate whether the retrieved results are sufficient.

    If not sufficient and retry_count < max, reformulate and retry.
    """
    last = state["messages"][-1] if state["messages"] else None
    user_message = (last.content if hasattr(last, "content") else last.get("content", "")) if last else ""
    results = state.get("results", [])
    retry_count = state.get("retry_count", 0)

    # Format results for evaluation
    if not results:
        sufficient = False
        reasoning = "No results retrieved"
        gaps = ["Need to retrieve relevant information"]
    else:
        # Check for errors in results
        errors = [r for r in results if r.get("error")]
        if errors:
            sufficient = False
            reasoning = f"Errors in results: {errors}"
            gaps = ["Fix errors in retrieval"]
        else:
            # Use LLM to evaluate sufficiency
            results_str = str(results)[:2000]  # Truncate for prompt

            prompt = EVALUATE_PROMPT.format(
                question=user_message,
                results=results_str,
            )

            llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
            response = await llm.ainvoke(prompt)

            import json
            try:
                eval_result = json.loads(response.content if hasattr(response, "content") else str(response))
                sufficient = eval_result.get("sufficient", False)
                reasoning = eval_result.get("reasoning", "")
                gaps = eval_result.get("gaps", [])
            except:
                sufficient = len(results) > 0
                reasoning = "Could not parse evaluation"
                gaps = []

    return {
        **state,
        "results": results,
        "retry_count": retry_count,
    }
