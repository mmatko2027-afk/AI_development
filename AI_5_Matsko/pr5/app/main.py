"""Веб-рівень застосунку: сторінка пошуку і JSON-ендпоінти.

Цей файл не знає ані якою моделлю отримано вектори, ані як влаштований
індекс, ані як ранжує пошук за словами — усе це лишається в модулях
`app/embeddings.py`, `app/index.py`, `app/keyword.py`. Тут вирішується
інше: що застосунок приймає від сторінки, що віддає їй і з яким
HTTP-статусом.

Індекс будується заздалегідь командою `python ingest.py` (з папки pr5),
а тут лише читається при старті.

Запуск із папки pr5:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import embeddings, index, keyword


app = FastAPI(title="Пошук у базі знань — ПР5")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class SearchRequest(BaseModel):
    """Data sent by the search page."""

    query: str
    mode: str = "both"
    top_k: int = index.DEFAULT_TOP_K
    filters: dict = {}
    threshold: float | None = None


def hit_to_dict(hit: index.Hit) -> dict:
    """Convert a search hit into a dictionary."""

    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


@app.on_event("startup")
def load_indexes() -> None:
    """Load the vector index and build the keyword index."""

    app.state.index = None
    app.state.keyword_index = None

    try:
        app.state.index = index.load()

    except Exception as exc:
        print(
            f"Індекс не завантажено: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    app.state.keyword_index = keyword.build(
        app.state.index.chunks
    )


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Return the search page."""

    if not INDEX_PAGE.is_file():
        raise HTTPException(
            status_code=500,
            detail="Файл сторінки пошуку не знайдено."
        )

    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    """Return the index status."""

    idx = app.state.index

    if idx is None:
        return {
            "ready": False,
            "hint": "індекс не збудовано — виконайте python ingest.py"
        }

    sources = {
        chunk.source
        for chunk in idx.chunks
    }

    return {
        "ready": True,
        "chunks": len(idx),
        "documents": len(sources),
        "model": idx.model_name,
    }


@app.post("/api/search")
def api_search(payload: SearchRequest) -> dict:
    """Perform semantic and keyword search."""

    if not payload.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Пошуковий запит не може бути порожнім."
        )

    if payload.mode not in ("semantic", "keyword", "both"):
        raise HTTPException(
            status_code=400,
            detail="Невідомий режим пошуку."
        )

    if app.state.index is None:
        raise HTTPException(
            status_code=503,
            detail="Індекс не збудовано. Спочатку виконайте python ingest.py"
        )

    idx = app.state.index

    if (
        payload.mode in ("semantic", "both")
        and idx.model_name != embeddings.MODEL_NAME
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Модель ембедінгів не відповідає моделі індексу. "
                "Перебудуйте індекс командою python ingest.py."
            )
        )

    result = {
        "query": payload.query,
        "semantic": None,
        "keyword": None,
        "elapsed": {}
    }

    if payload.mode in ("semantic", "both"):
        started = time.perf_counter()

        try:
            vector = embeddings.embed_query(
                payload.query
            )

            hits = index.search(
                idx,
                vector,
                top_k=payload.top_k,
                filters=payload.filters or None,
                threshold=(
                    payload.threshold
                    if payload.threshold is not None
                    else index.SIMILARITY_THRESHOLD
                ),
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc)
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Не вдалося створити вектор пошукового запиту."
            ) from exc

        result["elapsed"]["semantic"] = (
            time.perf_counter() - started
        )

        result["semantic"] = [
            hit_to_dict(hit)
            for hit in hits
        ]

    if payload.mode in ("keyword", "both"):
        started = time.perf_counter()

        try:
            hits = keyword.search(
                app.state.keyword_index,
                payload.query,
                top_k=payload.top_k,
                filters=payload.filters or None,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc)
            ) from exc

        result["elapsed"]["keyword"] = (
            time.perf_counter() - started
        )

        result["keyword"] = [
            hit_to_dict(hit)
            for hit in hits
        ]

    return result

