"""
api/main.py — Standalone FastAPI application (Day 29)
======================================================
Run with:  uvicorn api.main:app --reload
Or import ``fastapi_app`` into dashboard/app.py to mount in-process.
"""

from __future__ import annotations

from fastapi import FastAPI

from api.router import health_router, router

fastapi_app = FastAPI(
    title="AgentLens API",
    description="REST API for the AgentLens multi-agent observability platform.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Mount routers
fastapi_app.include_router(router, prefix="/api")
fastapi_app.include_router(health_router)

# Alias for uvicorn entry-point
app = fastapi_app
