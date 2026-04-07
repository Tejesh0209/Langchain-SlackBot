"""Ingest artifacts from SQLite into Weaviate for hybrid search."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import weaviate
from weaviate.classes.config import Property, DataType, Configure

from src.config import settings
from tools.sql_tool import sql_tool


ARTIFACTS_COLLECTION = "Artifact"


async def get_artifacts(limit: int = 1000) -> list[dict]:
    query = """
    SELECT
        a.artifact_id, a.customer_id, a.product_id, a.competitor_id,
        a.artifact_type, a.title, a.summary, a.content_text, a.created_at,
        c.name as customer_name
    FROM artifacts a
    JOIN customers c ON a.customer_id = c.customer_id
    LIMIT ?
    """
    result = await sql_tool._execute_internal(query, (limit,))
    if result.get("error"):
        raise RuntimeError(f"SQL error: {result['error']}")
    return result["rows"]


def create_collection(client: weaviate.WeaviateClient) -> bool:
    try:
        if client.collections.exists(ARTIFACTS_COLLECTION):
            client.collections.delete(ARTIFACTS_COLLECTION)
            print(f"Deleted existing collection: {ARTIFACTS_COLLECTION}")

        client.collections.create(
            name=ARTIFACTS_COLLECTION,
            properties=[
                Property(name="artifact_id", data_type=DataType.TEXT),
                Property(name="customer_id", data_type=DataType.TEXT),
                Property(name="artifact_type", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="summary", data_type=DataType.TEXT),
                Property(name="created_at", data_type=DataType.TEXT),
                Property(name="content_text", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.text2vec_openai(
                vectorize_collection_name=False,
            ),
        )
        print(f"Created collection: {ARTIFACTS_COLLECTION}")
        return True
    except Exception as e:
        print(f"Error creating collection: {e}")
        return False


def ingest_artifacts(client: weaviate.WeaviateClient, artifacts: list[dict]) -> int:
    collection = client.collections.get(ARTIFACTS_COLLECTION)
    with collection.batch.dynamic() as batch:
        for artifact in artifacts:
            batch.add_object(properties={
                "artifact_id": str(artifact.get("artifact_id", "")),
                "customer_id": str(artifact.get("customer_id", "")),
                "artifact_type": artifact.get("artifact_type", ""),
                "title": artifact.get("title", ""),
                "summary": artifact.get("summary", ""),
                "created_at": artifact.get("created_at", ""),
                "content_text": artifact.get("content_text", ""),
            })
    failed = collection.batch.failed_objects
    if failed:
        print(f"Warning: {len(failed)} objects failed to import")
    return len(artifacts) - len(failed)


async def main():
    print("=" * 60)
    print("Northstar Signal - Weaviate Ingestion")
    print("=" * 60)

    print(f"\nConnecting to Weaviate at: {settings.weaviate_url}")
    try:
        client = weaviate.connect_to_local(
            host="localhost", port=8080, grpc_port=50051, skip_init_checks=True,
            headers={"X-OpenAI-Api-Key": settings.openai_api_key},
        )
        if not client.is_connected():
            print("ERROR: Could not connect to Weaviate")
            return 1
        print("Connected to Weaviate.")
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print("\nCreating collection schema...")
    if not create_collection(client):
        client.close()
        return 1

    print("\nFetching artifacts from SQLite...")
    try:
        artifacts = await get_artifacts()
        print(f"Fetched {len(artifacts)} artifacts")
    except Exception as e:
        print(f"ERROR: {e}")
        client.close()
        return 1

    if not artifacts:
        print("No artifacts found in database.")
        client.close()
        return 1

    print("\nIngesting artifacts into Weaviate...")
    ingested = ingest_artifacts(client, artifacts)
    print(f"Successfully ingested {ingested} artifacts")

    client.close()
    print("\nIngestion complete!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
