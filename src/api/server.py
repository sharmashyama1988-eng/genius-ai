"""FastAPI Server exposing real-time SSE streaming for xThinking reasoning engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..dataset.loader import DatasetManager
from ..model.llm_engine import QwenEngine
from ..reasoning.xthinking import XThinkingEngine
from ..retrieval.wikipedia_client import WikipediaClient

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen2.5-0.5B + Wikipedia xThinking Deep Researcher",
    description="Autonomous deep-reasoning AI agent with live Wikipedia grounding, instruction exemplars, and real-time SSE thinking streams.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
dataset_mgr = DatasetManager()
wiki_client = WikipediaClient()
model_engine = QwenEngine()
reasoning_engine = XThinkingEngine(
    model_engine=model_engine,
    wiki_client=wiki_client,
    dataset_manager=dataset_mgr,
)


class QuestionRequest(BaseModel):
    question: str
    max_articles: int = 3
    top_k_passages: int = 5
    temperature: float = 0.6
    max_new_tokens: int = 1536


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Returns system status, device, and cached datasets."""
    import torch
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    return {
        "status": "online",
        "model": model_engine.model_id,
        "device": device_name,
        "cuda_available": cuda_avail,
        "datasets": {
            "lima_cached": dataset_mgr.is_cached("lima"),
            "alpaca_cached": dataset_mgr.is_cached("alpaca"),
            "codealpaca_cached": dataset_mgr.is_cached("codealpaca"),
        },
    }


@app.get("/api/datasets")
async def list_datasets() -> Dict[str, Any]:
    """Returns dataset cache statuses and stats."""
    return {
        "sources": [
            {
                "name": "GAIR/LIMA",
                "key": "lima",
                "purpose": "General conversation & strict instruction alignment",
                "cached": dataset_mgr.is_cached("lima"),
            },
            {
                "name": "tatsu-lab/alpaca",
                "key": "alpaca",
                "purpose": "Clean general instruction-following dataset",
                "cached": dataset_mgr.is_cached("alpaca"),
            },
            {
                "name": "sahil2801/CodeAlpaca-20k",
                "key": "codealpaca",
                "purpose": "Code generation, algorithmic reasoning & programming",
                "cached": dataset_mgr.is_cached("codealpaca"),
            },
        ]
    }


@app.get("/api/stream")
async def stream_reasoning(
    question: str = Query(..., description="User question or research prompt"),
    temperature: float = Query(0.6, ge=0.0, le=1.5),
    max_new_tokens: int = Query(1536, ge=128, le=4096),
):
    """Server-Sent Events (SSE) endpoint for streaming the entire reasoning lifecycle."""

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in reasoning_engine.execute_stream(
                question=question,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            ):
                payload_str = json.dumps(event.to_dict())
                yield f"data: {payload_str}\n\n"
        except Exception as e:
            logger.exception("Error during streaming reasoning")
            err_payload = json.dumps({"stage": "error", "event_type": "error", "payload": {"error": str(e)}})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the front-end dashboard."""
    html_file = Path(__file__).resolve().parent.parent / "web" / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Qwen2.5-0.5B xThinking API is running</h1>")
