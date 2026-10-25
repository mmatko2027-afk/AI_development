"""Фонові задачі: чернетка генерується не в HTTP-запиті, а окремо.

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
import threading
import uuid
import time
from typing import Iterator
from dataclasses import dataclass, field
from .drafting import build_request, check
from .gateway import generate
from .providers.base import Request, ProviderError

# Статуси, після яких подій більше не буде.
TERMINAL = {"done", "failed", "cancelled"}


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
    created_at: str = ""
    updated_at: str = ""
    events: list[dict] = field(default_factory=list)
    cancel_event: threading.Event = field(
        default_factory=threading.Event
    )


_drafts: dict[str, Draft] = {}
_order: list[str] = []
_lock = threading.Lock()
_changed = threading.Condition(_lock)


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


def _run_draft(draft: Draft, sample: dict):
    started = time.perf_counter()

    try:
        with _changed:
            if draft.cancel_event.is_set():
                return

            draft.status = "running"
            draft.events.append({
                "id": len(draft.events) + 1,
                "event": "status",
                "data": {"status": "running"}
            })
            _changed.notify_all()

        request = build_request(sample)


        def on_delta(text):
            if not text:
                return

            with _changed:
                if draft.status != "running":
                    return

                draft.text += text
                draft.events.append({
                    "id": len(draft.events) + 1,
                    "event": "delta",
                    "data": {"text": text}
                })
                _changed.notify_all()

        def on_status(data):
            with _changed:
                if draft.status != "running":
                    return

                draft.attempts = max(
                    draft.attempts,
                    data.get("attempt", 0)
                )

                if "provider" in data:
                    draft.provider = data["provider"]

                if "fallback" in data:
                    draft.fallback = data["fallback"]

                draft.events.append({
                    "id": len(draft.events) + 1,
                    "event": "status",
                    "data": data
                })
                _changed.notify_all()

        result = generate(
            request,
            on_delta,
            on_status,
            draft.cancel_event.is_set
        )
        warnings = check(result.text, sample)

        with _changed:
  
            if draft.status != "running":
                return

            draft.text = result.text
            draft.provider = result.provider
            draft.model = result.model
            draft.usage = result.usage
            draft.warnings = warnings
            draft.elapsed = {
                "total": round(time.perf_counter() - started, 2)
            }
            draft.status = "done"

            draft.events.append({
                "id": len(draft.events) + 1,
                "event": "done",
                "data": {
                    "text": draft.text,
                    "provider": draft.provider,
                    "model": draft.model,
                    "usage": draft.usage or {},
                    "elapsed": draft.elapsed,
                    "attempts": draft.attempts,
                    "fallback": draft.fallback,
                    "warnings": draft.warnings
                }
            })
            _changed.notify_all()

    except ProviderError as exc:
        with _changed:
            if draft.status == "cancelled":
                return

            draft.status = "failed"
            draft.error = {
                "kind": exc.kind,
                "message": exc.message,
                "retryable": exc.retryable
            }

            draft.events.append({
                "id": len(draft.events) + 1,
                "event": "error",
                "data": draft.error
            })
            _changed.notify_all()

    except Exception:
        with _changed:
            if draft.status == "cancelled":
                return

            draft.status = "failed"
            draft.error = {
                "kind": "internal_error",
                "message": "Не вдалося згенерувати чернетку.",
                "retryable": False
            }

            draft.events.append({
                "id": len(draft.events) + 1,
                "event": "error",
                "data": draft.error
            })
            _changed.notify_all()

def start(sample: dict, client: str) -> Draft:
    """Створити чернетку для звернення `sample` і запустити генерацію у
    фоні. Повертає чернетку одразу — найімовірніше, ще `queued`."""
    request_id = str(sample.get("id", uuid.uuid4().hex))
    draft = new_draft(request_id, client)

    thread = threading.Thread(
        target=_run_draft,
        args=(draft, sample),
        daemon=True
    )
    thread.start()

    return draft


def cancel(draft_id: str) -> Draft:
    """Попросити фонову роботу зупинитися. Повертає чернетку; якщо її
    немає або вона вже завершена — вирішуєте ви."""
    with _changed:
        draft = _drafts.get(draft_id)

        if draft is None:
            raise KeyError("Чернетку не знайдено")

        if draft.status in TERMINAL:
            raise ValueError("Чернетка вже завершена")

        draft.cancel_event.set()
        draft.status = "cancelled"
        draft.updated_at = _now()

        draft.events.append({
            "id": len(draft.events) + 1,
            "event": "cancelled",
            "data": {}
        })

        _changed.notify_all()

        return draft
