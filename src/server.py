"""FastAPI server with Socket Mode Slack bot."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import settings, APP_NAME, APP_VERSION
from src.slack_handler import create_handler
from src.agent.cache import query_cache


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    handler = create_handler()
    await handler.connect_async()
    yield
    await handler.disconnect_async()
    logger.info("Shutting down...")


# Create FastAPI app
api = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan,
)


@api.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
    }


@api.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "config": {
            "weaviate_url": settings.weaviate_url,
            "database_path": str(settings.database_path),
            "max_retry_count": settings.max_retry_count,
        },
    }



@api.get("/cache")
async def cache_stats():
    """Cache stats and manual clear."""
    return {"entries": query_cache.size, "ttl_seconds": 300, "max_entries": 200}


@api.delete("/cache")
async def clear_cache():
    """Clear the query cache."""
    query_cache.clear()
    return {"cleared": True}


@api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.server:api",
        host="0.0.0.0",
        port=3000,
        reload=True,
    )
