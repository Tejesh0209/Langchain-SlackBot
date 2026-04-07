"""
RAGAS evaluation for Northstar Signal RAG pipeline.

Metrics evaluated:
- Faithfulness:       Are answer claims supported by retrieved context?
- Answer Relevancy:   Is the answer on-topic for the question?
- Context Precision:  Are retrieved chunks ranked well (relevant first)?
- Context Recall:     Does retrieved context cover the ground truth?

Run standalone:
    python tests/test_ragas_eval.py

Run via pytest:
    pytest tests/test_ragas_eval.py -v --integration
"""
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datasets import Dataset
import warnings
from ragas import evaluate
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        FactualCorrectness,
    )

from src.agent.graph import run_agent
from src.config import settings


# ---------------------------------------------------------------------------
# Evaluation dataset — question + reference ground truth
# ---------------------------------------------------------------------------
EVAL_DATASET = [
    {
        "question": "Which customers are at risk?",
        "ground_truth": (
            "Customers with account health 'at risk' are: BlueHarbor Logistics, "
            "City of Verdant Bay, NorrLog Freight AB, Harbourline Regional Transit Authority, "
            "Peregrine Logistics Group, Province of Laurentia Department of Public Works, "
            "NordFryst AB, Department of Regional Services (DRS), Pioneer Freight Solutions, "
            "and Maple Regional Transit Authority."
        ),
    },
    {
        "question": "How many customers do we have?",
        "ground_truth": "There are 50 customers in total.",
    },
    {
        "question": "Which customers are expanding?",
        "ground_truth": (
            "Customers with account health 'expanding' are: MapleHarvest Grocers, "
            "Northpoint Apparel Pty Ltd, LedgerPeak Software, Aurora Dataworks AB, "
            "HarborHome Marketplace Pty Ltd, Arcadia Cloudworks, MapleBay Marketplace, "
            "SentinelOps AB, Hearthline Marketplace Pty Ltd, and LedgerBright Analytics."
        ),
    },
    {
        "question": "What is BlueHarbor's contract value?",
        "ground_truth": "BlueHarbor Logistics has a contract value of 780000.",
    },
    {
        "question": "Which customers have the highest contract values?",
        "ground_truth": (
            "Top 3 by contract value: Arcadia Cloudworks (1800000), "
            "MapleBridge Insurance (1250000), Pioneer Grid Retail LLC (1250000)."
        ),
    },
]


def row_to_prose(row: dict) -> str:
    """
    Convert a retrieved result into natural prose that RAGAS faithfulness
    can verify claims against. Phrasing must match how the LLM will answer.

    IMPORTANT: One fact per line so RAGAS can attribute claims correctly.
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

    # SQL row — write as natural English sentences
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


async def run_single_query(question: str) -> dict:
    """Run one query through the agent and return answer + prose contexts."""
    result = await run_agent(
        user_message=question,
        channel_id="eval_channel",
        thread_ts=f"eval_{abs(hash(question))}",
    )

    messages = result.get("messages", [])
    answer = ""
    if messages:
        last = messages[-1]
        answer = last.content if hasattr(last, "content") else last.get("content", "")

    # Convert ALL retrieved results to separate prose contexts (one per chunk)
    # RAGAS expects each context as a separate string in the list
    contexts = []
    for r in result.get("results", []):
        prose = row_to_prose(r)
        if prose:
            contexts.append(prose)

    return {"answer": answer, "contexts": contexts or ["No context retrieved"]}


def build_ragas_dataset() -> Dataset:
    """Run all eval queries and collect answers + contexts."""
    print("\nRunning agent on evaluation dataset...")
    rows: dict = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for entry in EVAL_DATASET:
        q = entry["question"]
        print(f"  Query: {q[:60]}...")
        r = asyncio.run(run_single_query(q))
        rows["question"].append(q)
        rows["answer"].append(r["answer"])
        rows["contexts"].append(r["contexts"])
        rows["ground_truth"].append(entry["ground_truth"])
        print(f"    Answer: {len(r['answer'])} chars  |  Contexts: {len(r['contexts'])}")

    return Dataset.from_dict(rows)


def mean_score(val) -> float:
    """Average a list or return a scalar — handles ragas version differences."""
    if isinstance(val, (int, float)):
        return round(float(val), 3)
    vals = [v for v in val if v is not None]
    return round(statistics.mean(vals), 3) if vals else 0.0


def run_ragas_evaluation() -> dict:
    """Run full RAGAS evaluation and return averaged scores."""
    dataset = build_ragas_dataset()

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)

    print("\nRunning RAGAS evaluation (this takes ~1 min)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall, FactualCorrectness()],
            llm=llm,
            embeddings=embeddings,
        )

    result_df = results.to_pandas()
    available = list(result_df.columns)
    print(f"\nRAGAS result columns: {available}")

    def safe_score(key: str) -> float:
        if key not in available:
            print(f"  WARNING: metric '{key}' not in results — skipping")
            return 0.0
        return mean_score(result_df[key].tolist())

    return {
        "faithfulness":        safe_score("faithfulness"),
        "answer_relevancy":    safe_score("answer_relevancy"),
        "context_precision":   safe_score("context_precision"),
        "context_recall":      safe_score("context_recall"),
        "factual_correctness": safe_score("factual_correctness(mode=f1)"),
    }


# ---------------------------------------------------------------------------
# pytest tests
# ---------------------------------------------------------------------------
class TestRAGASEvaluation:

    @pytest.mark.integration
    def test_full_ragas_report(self):
        """Print full RAGAS scores — always passes, used for visibility in CI."""
        scores = run_ragas_evaluation()
        print("\n" + "=" * 50)
        print("RAGAS Evaluation Report")
        print("=" * 50)
        for metric, score in scores.items():
            status = "PASS" if score >= 0.7 else "WARN"
            print(f"  [{status}] {metric:<25} {score:.3f}")
        print("=" * 50)
        assert True  # Always passes — report only

    @pytest.mark.integration
    def test_faithfulness_threshold(self):
        scores = run_ragas_evaluation()
        assert scores["faithfulness"] >= 0.7, \
            f"Faithfulness {scores['faithfulness']} < 0.7 — possible hallucination"

    @pytest.mark.integration
    def test_answer_relevancy_threshold(self):
        scores = run_ragas_evaluation()
        assert scores["answer_relevancy"] >= 0.7, \
            f"Answer relevancy {scores['answer_relevancy']} < 0.7"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scores = run_ragas_evaluation()
    print("\n" + "=" * 50)
    print("RAGAS Evaluation Report")
    print("=" * 50)
    for metric, score in scores.items():
        status = "PASS" if score >= 0.7 else "WARN"
        print(f"  [{status}] {metric:<25} {score:.3f}")
    print("=" * 50)
