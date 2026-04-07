"""Generate response node for LangGraph."""
import json

import asyncio
import re

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from src.agent.state import AgentState
from src.agent.guardrails import validate_output
from src.agent import progress
from src.config import settings

MAX_CONTEXTS = 15  # Cap to avoid confusing the LLM with too many chunks


# ---------------------------------------------------------------------------
# Context formatting helpers
# ---------------------------------------------------------------------------

def row_to_prose(row: dict) -> str:
    """
    Convert a retrieved result into natural prose that the LLM can reliably
    extract facts from. Phrasing must be unambiguous.

    IMPORTANT: Each fact is on its own LINE so RAGAS can attribute claims
    correctly. Do not combine multiple facts into a single sentence.
    """
    if not isinstance(row, dict) or row.get("error"):
        return ""

    # RAG artifact — already has prose content
    if row.get("content_text"):
        title = row.get("title", "")
        text = row["content_text"]
        if title:
            return f"{title}: {text}"
        return text
    if row.get("content"):
        return str(row["content"])
    if row.get("summary"):
        return str(row["summary"])

    # SQL row — write ONE FACT PER LINE for clear claim attribution
    name = row.get("name", "")
    lines = []

    if row.get("total") is not None:
        lines.append(f"There are {row['total']} customers in total.")
        return "\n".join(lines)

    if not name:
        return str(row)

    cv = row.get("contract_value")
    if cv is not None:
        lines.append(f"{name} has a contract value of {cv}.")

    health = row.get("account_health")
    if health:
        lines.append(f"{name} has account health '{health}'.")

    industry = row.get("industry")
    if industry:
        lines.append(f"{name} is in the {industry} industry.")

    region = row.get("region")
    if region:
        lines.append(f"{name} is in the {region} region.")

    crm = row.get("crm_stage")
    if crm:
        lines.append(f"{name} has CRM stage '{crm}'.")

    if not lines:
        lines.append(f"{name} is a customer.")

    return "\n".join(lines)


async def rerank_contexts(question: str, contexts: list[str]) -> list[str]:
    """
    Re-rank contexts by semantic similarity to the question using async embeddings,
    with numeric boosting for ranking-queries (highest, top, lowest, bottom).
    """
    if len(contexts) <= 1:
        return contexts

    try:
        embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
        q_embedding, ctx_embeddings = await asyncio.gather(
            embeddings.aembed_query(question),
            embeddings.aembed_documents(contexts),
        )

        # Cosine similarity (embeddings are already normalized)
        raw_scores = [
            sum(a * b for a, b in zip(q_embedding, ctx_emb))
            for ctx_emb in ctx_embeddings
        ]

        ranking_terms = {"highest", "top", "largest", "biggest", "maximum", "greatest", "lowest", "bottom", "smallest", "minimum"}
        query_words = set(re.findall(r'\b\w+\b', question.lower()))
        is_ranking_query = bool(ranking_terms & query_words)  # fixed: was always True due to generator

        if is_ranking_query:
            def extract_max_number(ctx: str) -> float:
                numbers = [float(n) for n in re.findall(r'\d+\.?\d*', ctx)]
                return max(numbers) if numbers else 0.0

            max_numbers = [extract_max_number(c) for c in contexts]
            max_val = max(max_numbers) if max_numbers else 1.0
            if max_val > 0:
                numeric_scores = [v / max_val for v in max_numbers]
                final_scores = [0.6 * s + 0.4 * n for s, n in zip(raw_scores, numeric_scores)]
            else:
                final_scores = raw_scores
        else:
            final_scores = raw_scores

        ranked = sorted(zip(contexts, final_scores), key=lambda x: x[1], reverse=True)
        return [ctx for ctx, _ in ranked]
    except Exception:
        return contexts


# ---------------------------------------------------------------------------
# Generation prompt — strict factual only, no tags
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """You are an expert business analyst assistant. Answer the question using ONLY the provided context.

RULES:
1. Every claim must be directly supported by the context — no hallucinations.
2. If the context does not contain enough information, say what you DO know from it, then state: "Based on the provided context, I cannot determine [X]."
3. For pattern/trend questions: identify shared themes, repeated phrases, or common issues across multiple documents — quote the specific documents.
4. Be concise and use bullet points where helpful.
5. Never invent customer names, numbers, or facts not present in the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


# ---------------------------------------------------------------------------
# Verification prompt
# ---------------------------------------------------------------------------

VERIFICATION_PROMPT = """You are a factual auditor. Check whether every specific claim (names, numbers, statuses) in the answer is supported by the context.

CONTEXT:
{context}

ANSWER TO VERIFY:
{answer}

For each specific factual claim (customer names, dollar amounts, health statuses, dates), mark:
  - [VERIFIED] if the fact is directly supported by the context
  - [UNSUPPORTED] if the specific fact contradicts or is absent from the context

Paraphrasing is fine. Only flag claims with wrong or invented specific facts (e.g. wrong numbers, wrong names, wrong statuses).

Output ONE verdict line:
  VERDICT: KEEP   — if all specific facts are supported
  VERDICT: REVISE — if any specific fact is wrong or invented"""


# ---------------------------------------------------------------------------
# Generation with self-verification (2-pass)
# ---------------------------------------------------------------------------

async def generate(state: AgentState) -> AgentState:
    """
    Generate a response from retrieved context using 2-pass self-verification.

    Pass 1: Generate answer from context with strict completeness requirement.
    Pass 2: Verify each claim. If VERDICT is REVISE, rewrite once.
    """
    await progress.report(state.get("thread_ts", ""), "generate")
    last = state["messages"][-1] if state["messages"] else None
    user_message = (last.content if hasattr(last, "content") else last.get("content", "")) if last else ""
    results = state.get("results", [])
    sources = state.get("sources", [])

    # Assemble context as natural prose, then re-rank by relevance
    if not results:
        context = "No relevant information found."
    else:
        prose_chunks = []
        for r in results:
            prose = row_to_prose(r)
            if prose:
                prose_chunks.append(prose)

        # Re-rank contexts so most question-relevant evidence is first
        prose_chunks = await rerank_contexts(user_message, prose_chunks)

        # Cap to avoid overwhelming the LLM with noisy chunks
        prose_chunks = prose_chunks[:MAX_CONTEXTS]

        context = "\n\n".join(prose_chunks) if prose_chunks else "No relevant information found."

    # Build generation prompt
    prompt = GENERATION_PROMPT.format(
        context=context,
        question=user_message,
    )

    # Pass 1: Generate initial answer with streaming
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)

    # Stream response to Slack for faster perceived latency
    accumulated = []
    async for chunk in llm.astream(prompt):
        accumulated.append(chunk.content if hasattr(chunk, "content") else str(chunk))
        # Update Slack periodically with partial response
        if len("".join(accumulated)) % 200 < 50:
            await progress.report(state.get("thread_ts", ""), "typing")

    current_answer = "".join(accumulated)

    # Pass 2: Verify the answer (max 1 revision round now)
    current_answer = await _verify_and_revise(current_answer, context, llm)

    # Validate output
    is_valid, error = validate_output(current_answer, sources)
    if not is_valid:
        current_answer = f"I encountered an issue generating a response: {error}"

    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": current_answer}],
    }


async def _verify_and_revise(answer: str, context: str, llm: ChatOpenAI) -> str:
    """
    Verify answer claims against context. If VERDICT is REVISE, rewrite once.
    """
    prompt = VERIFICATION_PROMPT.format(context=context, answer=answer)
    response = await llm.ainvoke(prompt)
    verification = response.content if hasattr(response, "content") else str(response)

    verdict_line = next(
        (line.strip() for line in verification.split("\n") if line.strip().startswith("VERDICT:")),
        "",
    )

    if "KEEP" in verdict_line:
        return answer

    # VERDICT is REVISE — rewrite to fix unsupported claims
    revision_prompt = f"""Rewrite this answer so every claim is supported by the context.
Remove or fix any UNSUPPORTED claims. Keep all VERIFIED claims. Never add new facts.

CONTEXT:
{context}

ORIGINAL ANSWER:
{answer}

VERIFICATION RESULTS:
{verification}

REWRITE (only supported facts):"""

    response = await llm.ainvoke(revision_prompt)
    return response.content if hasattr(response, "content") else str(response)
