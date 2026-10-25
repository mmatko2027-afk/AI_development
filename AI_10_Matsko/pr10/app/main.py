"""Веб-рівень застосунку: сторінка помічника оператора і JSON-ендпоінти.

Цей файл не знає ані який провайдер відповідає, ані скільки було спроб,
ані в якому потоці йде генерація — усе це в `app/gateway.py`,
`app/providers/` і `app/jobs.py`. Тут вирішується інше: що застосунок
приймає від сторінки, що віддає їй і з яким HTTP-статусом, і як події
фонової роботи стають потоком server-sent events.

Запуск із папки pr10:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator
from fastapi import FastAPI, Request,HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from . import gateway, jobs, limits, samples

app = FastAPI(title="Помічник оператора — ПР10")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class DraftIn(BaseModel):
    """Замовлення чернетки зі сторінки.

    `request_id` — яке звернення зі зразків. `client` — ідентифікатор
    клієнта сторінки для обмеження частоти; у справжньому застосунку
    його взяв би із сесії веб-рівень, а не сторінка.
    """

    request_id: str
    client: str = "anonymous"


def draft_to_dict(draft: jobs.Draft) -> dict:
    """Знімок чернетки для сторінки — без журналу подій."""
    return {
        "draft_id": draft.draft_id,
        "request_id": draft.request_id,
        "client": draft.client,
        "status": draft.status,
        "text": draft.text,
        "attempts": draft.attempts,
        "provider": draft.provider,
        "model": draft.model,
        "fallback": draft.fallback,
        "error": draft.error,
        "usage": draft.usage,
        "elapsed": draft.elapsed,
        "warnings": draft.warnings,
        "created_at": draft.created_at,
        "updated_at": draft.updated_at,
        "events": len(draft.events),
    }

@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def api_health() -> dict:
    """Чинні налаштування — щоб сторінка показала, хто відповідає і які
    обмеження діють. Ключа тут немає."""
    return {"gateway": gateway.settings(), "limits": limits.settings()}


@app.get("/api/requests")
def api_requests() -> list[dict]:
    """Зразки звернень для вибору на сторінці."""
    return [samples.public(s) for s in samples.all_samples()]


@app.post("/api/drafts", status_code=202)
def api_create(payload: DraftIn) -> dict:
    """Замовити чернетку. Повертає її стан одразу, не чекаючи на
    генерацію; далі сторінка слухає події.

    Збої тут не оброблено. Невідоме звернення, вичерпаний ліміт
    (`limits.RateLimited`), збій запуску фонової роботи — усе це поки що
    закінчується помилкою 500. Що віддати в кожному випадку — статус,
    заголовок `Retry-After`, тіло помилки в єдиному форматі — вирішуєте
    ви. Зверніть увагу: збій провайдера сюди не доходить — він стається
    пізніше, у фоні, і стає подією `error`, а не помилкою цього запиту.
    """
    sample = samples.get(payload.request_id)

    if sample is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "request_not_found",
                "message": "Звернення не знайдено."
            }
        )

    try:
        limits.acquire(payload.client)
    except limits.RateLimited as exc:
        retry_after = max(1, int(exc.retry_after + 0.999))

        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "message": "Ліміт запитів вичерпано. Спробуйте пізніше."
            },
            headers={"Retry-After": str(retry_after)}
        )

    try:
        draft = jobs.start(sample, payload.client)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "draft_start_failed",
                "message": "Не вдалося запустити генерацію чернетки."
            }
        )

    return draft_to_dict(draft)


@app.get("/api/drafts")
def api_drafts() -> list[dict]:
    """Останні чернетки, нові першими."""
    return [draft_to_dict(d) for d in jobs.list_drafts()]


@app.get("/api/drafts/{draft_id}")
def api_draft(draft_id: str) -> dict:
    """Знімок однієї чернетки. Невідомий номер поки що дає 500."""
    draft = jobs.get(draft_id)

    if draft is None:
        raise HTTPException(
            status_code=404,
            detail="Чернетку не знайдено."
        )

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
    if jobs.get(draft_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Чернетку не знайдено."
        )
    try:
        since = int(request.headers.get("last-event-id") or 0)
    except ValueError:
        since = 0

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
    """Скасувати генерацію. Скасування вже завершеної або невідомої
    чернетки поки що дає 500 — який статус тут доречний, вирішуєте ви."""
    try:
        draft = jobs.cancel(draft_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Чернетку не знайдено."
        )
    except ValueError:
        raise HTTPException(
            status_code=409,
            detail="Чернетка вже завершена."
        )

    return draft_to_dict(draft)
