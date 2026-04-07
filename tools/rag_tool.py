"""Weaviate hybrid search tool (BM25 + dense vector)."""
from typing import Any, Optional, Union

import weaviate

from src.config import settings


# Weaviate collection name for artifacts
ARTIFACTS_COLLECTION = "Artifact"


class RAGTool:
    """Hybrid search tool using Weaviate (BM25 + dense vectors)."""

    def __init__(self, url: Optional[str] = None):
        self.url = url or settings.weaviate_url
        self._client: Optional[weaviate.WeaviateClient] = None

    @property
    def client(self) -> weaviate.WeaviateClient:
        """Lazy initialization of Weaviate client."""
        if self._client is None:
            self._client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                grpc_port=50051,
                skip_init_checks=True,
                headers={"X-OpenAI-Api-Key": settings.openai_api_key},
            )
        return self._client

    async def search(
        self, query: str, limit: int = 5, alpha: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search (BM25 + vector) on artifacts.

        Args:
            query: Search query text
            limit: Maximum number of results (default 5)
            alpha: Weight for BM25 vs vector. 0.5 = equal weight.

        Returns:
            List of matching artifact dicts with content and metadata.
        """
        try:
            collection = self.client.collections.get(ARTIFACTS_COLLECTION)

            response = collection.query.hybrid(
                query=query,
                limit=limit,
                alpha=alpha,
                query_properties=["content_text"],
                return_properties=[
                    "artifact_id",
                    "customer_id",
                    "artifact_type",
                    "title",
                    "summary",
                    "created_at",
                    "content_text",
                ],
            )

            results = []
            for obj in response.objects:
                results.append({
                    "artifact_id": obj.properties.get("artifact_id"),
                    "customer_id": obj.properties.get("customer_id"),
                    "artifact_type": obj.properties.get("artifact_type"),
                    "title": obj.properties.get("title"),
                    "summary": obj.properties.get("summary"),
                    "created_at": obj.properties.get("created_at"),
                    "content": obj.properties.get("content_text"),
                    "score": obj.metadata.score if obj.metadata else None,
                })

            return results

        except Exception as e:
            return [{"error": str(e)}]

    async def get_by_customer(
        self, customer_id: str, artifact_type: Optional[str] = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get artifacts for a specific customer.

        Args:
            customer_id: The customer ID to filter by
            artifact_type: Optional artifact type filter
            limit: Maximum results

        Returns:
            List of artifact dicts for the customer.
        """
        try:
            collection = self.client.collections.get(ARTIFACTS_COLLECTION)

            filters = {"path": ["customer_id"], "operator": "Equal", "valueText": customer_id}
            if artifact_type:
                type_filter = {
                    "path": ["artifact_type"],
                    "operator": "Equal",
                    "valueText": artifact_type,
                }
                filters = {
                    "operator": "And",
                    "operands": [filters, type_filter],
                }

            response = collection.query.fetch_objects(
                filters=filters,
                limit=limit,
                return_properties=[
                    "artifact_id",
                    "customer_id",
                    "artifact_type",
                    "created_at",
                    "content",
                ],
            )

            results = []
            for obj in response.objects:
                results.append({
                    "artifact_id": obj.properties.get("artifact_id"),
                    "customer_id": obj.properties.get("customer_id"),
                    "artifact_type": obj.properties.get("artifact_type"),
                    "created_at": obj.properties.get("created_at"),
                    "content": obj.properties.get("content"),
                })

            return results

        except Exception as e:
            return [{"error": str(e)}]

    def close(self):
        """Close the Weaviate client connection."""
        if self._client:
            self._client.close()
            self._client = None


# Singleton instance
rag_tool = RAGTool()
