"""Заготовка з ПР10. Замініть цей файл своїм `pr10/app/jobs.py` і
допишіть те, що перелічено нижче.

Що додати в ПР11:

* ідентифікатор запиту: фонова робота виконується в іншому потоці, і
  контекстна змінна `telemetry.request_id_var` там порожня. Передайте
  ідентифікатор явно (у `Draft`, у `Request.metadata`, в `extra`
  кожного запису) або запустіть роботу в копії контексту
  (`contextvars.copy_context().run`). Усі записи чернетки мають містити
  `draft_id` і `request_id`;
* журнал: `draft.created` (звернення, клієнт, fresh) і `draft.done` /
  `draft.failed` / `draft.cancelled` (спроб, провайдер, кеш, вартість,
  тривалість). Текст чернетки — ні;
* прапорець `fresh` з `POST /api/drafts`: `start(sample, client,
  fresh=False)` → `Request.metadata` → шлюз оминає кеш. Сигнатуру
  заготовки нижче вже змінено — змініть і свою;
* у `Draft`, знімку й події `done` — `cached`, `cost`, `cost_estimated`:
  сторінка показує їх, якщо вони є;
* відмова через бюджет: шлюз підіймає `ProviderError(kind="budget")` —
  це подія `error` з `retryable=False` і часом відновлення в
  повідомленні. Чи перевіряти бюджет ще й до запуску роботи, щоб
  оператор дізнався одразу, — див. `main.py`.

Нижче — заготовка ПР10 без змін.

Фонові задачі: чернетка генерується не в HTTP-запиті, а окремо.

`POST /api/drafts` повертається одразу з номером чернетки, а робота
триває у фоні: сторінка дізнається про стан через потік подій
(`GET /api/drafts/{id}/events`) або знімок (`GET /api/drafts/{id}`) і
може скасувати її (`POST /api/drafts/{id}/cancel`). Так чернетка не
прив'язана до одного HTTP-зʼєднання: сторінку можна перезавантажити,
а потім відкрити ту саму чернетку й побачити все, що встигло статися.

Модуль не знає, як влаштований шлюз і що саме генерується; він знає
лише `Draft`, події і те, що робота виконується не в потоці запиту.

Що тут дано готовим — реєстр чернеток і події. `emit` додає подію до
журналу чернетки й будить тих, хто чекає; `subscribe` віддає події
від указаного номера і далі в міру надходження, доки чернетка не
дійде до кінцевого статусу. Веб-рівень перетворює це на
server-sent events; реалізація фонової роботи до цього не торкається.

Заготовки — `start` і `cancel`. Рішення з розділу 2 практичної роботи:

* у чому виконувати: потік на чернетку, пул потоків, asyncio-задача?
  Що обмежує кількість одночасних робіт — це чи шлюз;
* порядок дій у фоновій роботі: яку подію й який статус виставити
  спершу, що робити з фрагментами, що — з результатом і з помилкою.
  Жоден виняток не має лишити чернетку в стані `running`;
* скасування — спільне: фонова робота сама перевіряє прапорець між
  фрагментами й спробами. Хто його ставить, де він живе, що
  відбувається з текстом, який уже надіслано, і чи може скасована
  чернетка потім стати `done`, якщо потік завершився пізніше;
* `cancel` для чернетки, яка вже завершена чи не існує, — помилка якого
  роду і з яким HTTP-статусом;
* що буває зі старими чернетками: реєстр у памʼяті росте з кожною.
  Скільки тримати і коли прибирати.

Події, які розуміє сторінка (інші вона показує як є):

    status     {"status": "running", "attempt": 1, "provider": "openai",
                "fallback": false, "detail": "…"}
    delta      {"text": "…"}                      приріст тексту
    done       {"text": "…", "provider": "…", "model": "…",
                "usage": {...}, "elapsed": {...}, "attempts": 2,
                "fallback": false, "warnings": ["…"]}
    error      {"kind": "timeout", "message": "…", "retryable": true}
    cancelled  {}

Поля — відправна точка; змінюючи їх, змініть і сторінку.
"""

import datetime
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from . import drafting, gateway, telemetry
from .providers.base import ProviderError

# Статуси, після яких подій більше не буде.
TERMINAL = {"done", "failed", "cancelled"}
_log = logging.getLogger("app.jobs")


@dataclass
class Draft:
    """Стан однієї чернетки — те, що зберігається в реєстрі і йде на
    сторінку як знімок.

    `status` — `queued`, `running`, `done`, `failed`, `cancelled`.
    `text` — накопичений текст. `attempts`, `provider`, `model`,
    `fallback` — звідки взявся результат. `error` — остаточна помилка,
    якщо є: `kind`, `message`, `retryable`. `elapsed` — секунди, наприклад
    `{"first_token": 0.8, "total": 4.1}`. `warnings` — зауваження до
    чернетки з `drafting.check`. `events` — журнал подій для потоку;
    у знімок на сторінку не входить.
    """

    draft_id: str
    request_id: str
    client: str
    status: str = "queued"
    text: str = ""
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    fallback: bool = False
    error: dict | None = None
    usage: dict | None = None
    elapsed: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    cached: bool = False
    cost: float | None = None
    cost_estimated: bool = False
    fresh: bool = False
    created_at: str = ""
    updated_at: str = ""
    events: list[dict] = field(default_factory=list)


_drafts: dict[str, Draft] = {}
_order: list[str] = []
_lock = threading.RLock()
_changed = threading.Condition(_lock)
_cancellations: dict[str, threading.Event] = {}


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def new_draft(request_id: str, client: str) -> Draft:
    """Створити чернетку зі статусом `queued` і зареєструвати її.
    Роботу не запускає — це справа `start`."""
    draft = Draft(draft_id=uuid.uuid4().hex[:8], request_id=request_id, client=client,
                  created_at=_now(), updated_at=_now())
    with _lock:
        _drafts[draft.draft_id] = draft
        _order.insert(0, draft.draft_id)
    return draft


def get(draft_id: str) -> Draft | None:
    with _lock:
        return _drafts.get(draft_id)


def list_drafts(limit: int = 50) -> list[Draft]:
    """Чернетки, нові першими."""
    with _lock:
        return [_drafts[i] for i in _order[:limit]]


def emit(draft: Draft, event: str, data: dict | None = None) -> None:
    """Додати подію до журналу чернетки й розбудити підписників.

    Статус і текст чернетки ця функція не змінює — це робить фонова
    робота разом із подією, під тим самим замком, якщо їй потрібна
    узгодженість між знімком і потоком.
    """
    with _changed:
        draft.events.append({"id": len(draft.events) + 1, "event": event, "data": data or {}})
        draft.updated_at = _now()
        _changed.notify_all()


def subscribe(draft_id: str, since: int = 0, heartbeat: float = 15.0) -> Iterator[dict | None]:
    """Віддавати події чернетки, починаючи з номера `since + 1`.

    Спершу — ті, що вже є, потім — нові в міру надходження. Завершується,
    коли чернетка в кінцевому статусі й усі події віддано, або коли
    чернетки немає. Якщо подій немає довше за `heartbeat` секунд, віддає
    `None`: веб-рівень надсилає порожній коментар, щоб зʼєднання не
    закрив проксі і щоб помітити, що клієнт відключився.
    """
    while True:
        with _changed:
            draft = _drafts.get(draft_id)
            if draft is None:
                return
            pending = draft.events[since:]
            if not pending:
                if draft.status in TERMINAL:
                    return
                _changed.wait(timeout=heartbeat)
                pending = draft.events[since:]
        if not pending:
            yield None
            continue
        for event in pending:
            yield event
        since += len(pending)


def start(sample: dict, client: str, fresh: bool = False) -> Draft:
    """Створити чернетку для звернення `sample` і запустити генерацію у
    фоні. Повертає чернетку одразу — найімовірніше, ще `queued`.
    `fresh` — оминути кеш (кнопка «Повторити»)."""
    request_id = str(sample.get("id", ""))
    request = drafting.build_request(sample)
    draft = new_draft(request_id, client)
    cancellation = threading.Event()
    _cancellations[draft.draft_id] = cancellation
    request.metadata.update({
        "draft_id": draft.draft_id,
        "http_request_id": telemetry.request_id_var.get(),
        "client": client,
        "fresh": fresh,
    })
    draft.fresh = fresh
    log_data = {
        "request_id": telemetry.request_id_var.get(),
        "draft_id": draft.draft_id,
        "client": client,
        "sample_id": request_id,
        "fresh": fresh,
        "prompt_version": drafting.PROMPT_VERSION,
        "input_chars": request.metadata.get("input_chars"),
        "facts_count": request.metadata.get("facts_count"),
    }
    _log.info("draft.created", extra=log_data)
    emit(draft, "status", {"status": "queued", "draft_id": draft.draft_id})

    def work() -> None:
        started = time.perf_counter()

        def on_status(data: dict) -> None:
            if cancellation.is_set():
                return
            with _changed:
                if draft.status in TERMINAL:
                    return
                if data.get("attempt"):
                    draft.attempts = data["attempt"]
                if data.get("provider"):
                    draft.provider = data["provider"]
                if data.get("fallback") is not None:
                    draft.fallback = data["fallback"]
                draft.status = data.get("status", "running")
                emit(draft, "status", data)

        def on_delta(text: str) -> None:
            if cancellation.is_set() or not text:
                return
            with _changed:
                if draft.status in TERMINAL:
                    return
                if "first_token" not in draft.elapsed:
                    draft.elapsed["first_token"] = time.perf_counter() - started
                draft.text += text
                draft.status = "running"
                emit(draft, "delta", {"text": text})

        try:
            if cancellation.is_set():
                return
            with _changed:
                draft.status = "running"
            result = gateway.generate(request, on_delta, on_status, cancellation.is_set)
            if cancellation.is_set():
                return
            warnings = drafting.check(result.text, sample)
            with _changed:
                if draft.status in TERMINAL:
                    return
                draft.text = result.text
                draft.provider = result.provider
                draft.model = result.model
                draft.usage = result.usage
                draft.cached = result.cached
                draft.cost = result.cost
                draft.cost_estimated = result.cost_estimated
                draft.fallback = draft.fallback or result.provider != gateway.PROVIDER
                draft.elapsed["total"] = time.perf_counter() - started
                draft.status = "done"
                draft.warnings = warnings
                emit(draft, "done", {
                    "text": result.text,
                    "provider": result.provider,
                    "model": result.model,
                    "usage": result.usage,
                    "elapsed": draft.elapsed,
                    "attempts": draft.attempts,
                    "fallback": draft.fallback,
                    "cached": result.cached,
                    "cost": result.cost,
                    "cost_estimated": result.cost_estimated,
                    "warnings": warnings,
                })
            _log.info("draft.done", extra={
                **log_data, "attempts": draft.attempts, "provider": result.provider,
                "model": result.model, "fallback": draft.fallback, "cached": result.cached,
                "cost": result.cost, "cost_estimated": result.cost_estimated,
                "duration": draft.elapsed.get("total"),
            })
        except ProviderError as exc:
            if cancellation.is_set() or exc.kind == "cancelled":
                with _changed:
                    if draft.status in TERMINAL:
                        return
                    draft.status = "cancelled"
                    emit(draft, "cancelled", {})
                _log.warning("draft.cancelled", extra={**log_data, "attempts": draft.attempts})
                return
            error = {"kind": exc.kind, "message": exc.message, "retryable": exc.retryable}
            if exc.detail:
                error["detail"] = exc.detail
            with _changed:
                if draft.status in TERMINAL:
                    return
                draft.status = "failed"
                draft.error = error
                emit(draft, "error", {**error, "request_id": log_data["request_id"]})
            _log.error("draft.failed", extra={
                **log_data, "attempts": draft.attempts, "failure_kind": exc.kind,
                "failure_detail": exc.detail,
                "retryable": exc.retryable,
            })
        except Exception as exc:
            error = {
                "kind": "unknown",
                "message": "Не вдалося створити чернетку.",
                "retryable": False,
                "detail": type(exc).__name__,
            }
            with _changed:
                if draft.status in TERMINAL:
                    return
                draft.status = "failed"
                draft.error = error
                emit(draft, "error", {**error, "request_id": log_data["request_id"]})
            _log.error("draft.failed", extra={
                **log_data, "attempts": draft.attempts, "failure_kind": type(exc).__name__,
                "failure_detail": type(exc).__name__,
            })
        finally:
            _cancellations.pop(draft.draft_id, None)

    threading.Thread(target=work, name=f"draft-{draft.draft_id}", daemon=True).start()
    return draft


def cancel(draft_id: str) -> Draft:
    """Попросити фонову роботу зупинитися. Повертає чернетку; якщо її
    немає — підняти `KeyError`, якщо вона вже завершена — `ValueError`."""
    with _changed:
        draft = _drafts.get(draft_id)
        if draft is None:
            raise KeyError(draft_id)
        if draft.status in TERMINAL:
            raise ValueError(f"Чернетка вже має кінцевий стан: {draft.status}")
        cancellation = _cancellations.get(draft_id)
        if cancellation:
            cancellation.set()
        draft.status = "cancelled"
        draft.updated_at = _now()
        draft.events.append({"id": len(draft.events) + 1, "event": "cancelled", "data": {}})
        _changed.notify_all()
    _log.info("draft.cancelled", extra={
        "request_id": telemetry.request_id_var.get(), "sample_id": draft.request_id,
        "draft_id": draft.draft_id, "client": draft.client,
        "attempts": draft.attempts,
    })
    return draft
