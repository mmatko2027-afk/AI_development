"""Веб-рівень застосунку: сторінка помічника оператора і JSON-ендпоінти.

Відносно ПР10 тут додано:

* проміжний шар `request_context`: кожному HTTP-запиту — ідентифікатор
  у `telemetry.request_id_var`, у відповіді — заголовок `X-Request-ID`,
  у журналі — запис `http.request` з методом, шляхом, статусом і
  тривалістю. Ідентифікатор, який клієнт прислав у тому самому
  заголовку, береться, якщо він схожий на ідентифікатор;
* `fresh` у замовленні чернетки — обійти кеш (кнопка «Повторити»);
* `GET /api/usage` — стан бюджету й кешу для сторінки;
* `/api/health` віддає ще й версію prompt, налаштування кешу й ліміту.

Веб-рівень повертає зрозумілі HTTP-помилки для невідомих звернень,
чернеток і обмеження частоти. Збої генерації обробляє фонова задача й
передає сторінці через події; бюджет перевіряється в шлюзі, тож кешовані
відповіді залишаються доступними навіть після вичерпання ліміту.

Запуск із папки pr11:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import budget, cache, costs, drafting, gateway, jobs, limits, samples, telemetry

telemetry.setup()
log = logging.getLogger("app.http")

app = FastAPI(title="Помічник оператора — ПР11")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"

# Ідентифікатор, який прислав клієнт, приймаємо лише в такому вигляді:
# у журнал не має потрапити довільний рядок із заголовка.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{4,64}$")


class DraftIn(BaseModel):
    """Замовлення чернетки зі сторінки.

    `request_id` — яке звернення зі зразків. `client` — ідентифікатор
    клієнта сторінки для обмеження частоти. `fresh` — оминути кеш:
    оператор хоче іншу чернетку, а не ту саму.
    """

    request_id: str
    client: str = "anonymous"
    fresh: bool = False


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Ідентифікатор запиту в контексті й у відповіді; запис про запит у
    журнал. Для потоку подій тривалість — до початку відповіді, не до
    кінця потоку."""
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _REQUEST_ID.match(incoming) else telemetry.new_request_id()
    token = telemetry.request_id_var.set(request_id)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        log.exception("http.unhandled", extra={"method": request.method, "path": request.url.path})
        raise
    finally:
        log.info("http.request", extra={
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        })
        telemetry.request_id_var.reset(token)


def draft_to_dict(draft: jobs.Draft) -> dict:
    """Знімок чернетки для сторінки — без журналу подій."""
    data = asdict(draft)
    data["events"] = len(draft.events)
    return data


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.exception_handler(limits.RateLimited)
async def rate_limit_error(request: Request, exc: limits.RateLimited) -> JSONResponse:
    request_id = telemetry.request_id_var.get()
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(max(1, int(exc.retry_after)))},
        content={"error": {
            "kind": "rate_limit",
            "message": str(exc),
            "retryable": True,
            "request_id": request_id,
        }},
    )


@app.get("/api/health")
def api_health() -> dict:
    """Чинні налаштування — щоб сторінка показала, хто відповідає, яка
    версія prompt і які обмеження діють. Ключа тут немає."""
    return {
        "gateway": gateway.settings(),
        "limits": limits.settings(),
        "cache": cache.settings(),
        "budget": budget.settings(),
        "prompt_version": getattr(drafting, "PROMPT_VERSION", None),
        "log_file": telemetry.settings()["log_file"],
    }


@app.get("/api/usage")
def api_usage() -> dict:
    """Стан витрат і кешу — для панелі на сторінці."""
    return {
        "budget": budget.status(),
        "cache": cache.stats(),
        "pricing": costs.settings(),
    }


@app.get("/api/requests")
def api_requests() -> list[dict]:
    """Зразки звернень для вибору на сторінці."""
    return [samples.public(s) for s in samples.all_samples()]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn) -> dict:
    """Замовити чернетку. Повертає її стан одразу, не чекаючи на
    генерацію; далі сторінка слухає події.

    Невідоме звернення повертає 404, а обмеження частоти — 429 із
    `Retry-After`. Помилки під час генерації надходять у потоці подій
    чернетки.
    """
    sample = samples.get(payload.request_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Звернення не знайдено.")
    limits.acquire(payload.client)
    return draft_to_dict(jobs.start(sample, payload.client, fresh=payload.fresh))


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    """Останні чернетки, нові першими."""
    return [draft_to_dict(d) for d in jobs.list_drafts()]


@app.get("/api/drafts/{draft_id}")
def api_draft(draft_id: str) -> dict:
    """Знімок однієї чернетки."""
    draft = jobs.get(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="Чернетку не знайдено.")
    return draft_to_dict(draft)


@app.get("/api/drafts/{draft_id}/events")
def api_events(draft_id: str, request: Request) -> StreamingResponse:
    """Потік подій чернетки у форматі server-sent events.

    Браузер при обриві зʼєднання перепідключається сам і надсилає
    заголовок `Last-Event-ID` — звідси продовжуємо, не повторюючи
    віддане. Порожній коментар раз на кілька секунд тримає зʼєднання
    живим; якщо клієнт відключився, спроба записати в нього завершує
    генератор.
    """
    try:
        since = int(request.headers.get("last-event-id") or 0)
    except ValueError:
        since = 0
    if jobs.get(draft_id) is None:
        raise HTTPException(status_code=404, detail="Чернетку не знайдено.")

    def stream() -> Iterator[str]:
        for event in jobs.subscribe(draft_id, since):
            if event is None:
                yield ": ping\n\n"
                continue
            data = json.dumps(event["data"], ensure_ascii=False)
            yield f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/drafts/{draft_id}/cancel")
def api_cancel(draft_id: str) -> dict:
    """Скасувати генерацію."""
    try:
        return draft_to_dict(jobs.cancel(draft_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Чернетку не знайдено.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
