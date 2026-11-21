"""Веб-рівень застосунку: сторінка діалогу і JSON-ендпоінт.

Цей файл не має знати ані про провайдера моделі, ані про те, як
складається запит, ані про те, як виглядає схема відповіді, — усе це
лишається в `app/llm.py` і `app/schema.py`. Тут вирішується інше: що
застосунок приймає від сторінки, що віддає їй і з яким HTTP-статусом.

Запуск із папки pr4:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import llm


app = FastAPI(title="Помічник служби підтримки — ПР4")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Turn(BaseModel):
    """One conversation message."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Data sent by the chat page."""

    message: str = Field(
        min_length=1,
        max_length=4000
    )

    history: list[Turn] = Field(
        default_factory=list,
        max_length=50
    )


def load_context() -> str:
    """Read store rules from the context file."""

    if not CONTEXT_FILE.is_file():
        raise HTTPException(
            status_code=500,
            detail="Файл правил магазину не знайдено."
        )

    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Return the chat page."""

    if not INDEX_PAGE.is_file():
        raise HTTPException(
            status_code=500,
            detail="Файл сторінки не знайдено."
        )

    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/chat")
def api_chat(payload: ChatRequest):
    """Return the assistant response as JSON."""

    history = [
        {
            "role": turn.role,
            "content": turn.content
        }
        for turn in payload.history
    ]

    try:
        result = llm.ask(
            message=payload.message,
            history=history,
            context=load_context()
        )

        return result

    except llm.LLMError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=str(error)
        ) from error
