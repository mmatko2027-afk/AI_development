"""Кеш відповідей: однакове питання — одна оплата.

Повтори беруться звідусіль: «Повторити» без правок, пʼять натискань
поспіль, двоє операторів над одним зверненням. Кеш повертає готовий
результат, не звертаючись до провайдера, — і це видно операторові,
журналу й бюджету.

Модуль знає `Request` і `Result` з контракту провайдера і більше
нічого: ані хто питає, ані навіщо. Шлюз (`app/gateway.py`) перед
першою спробою питає `get(make_key(...))`, після вдалої генерації
кладе `put`; прапорець `fresh` у `Request.metadata` каже кеш оминути.

Функції нижче — заготовки. Рішення з розділу 2 практичної роботи:

* що входить у ключ: усе, що впливає на відповідь (провайдер, модель,
  версія prompt, повідомлення, температура, `max_tokens`), і нічого з
  того, що не впливає (клієнт, номер чернетки, час). Канонічне подання
  — `json.dumps(..., sort_keys=True)` → `hashlib.sha256`;
* термін життя: факти звернення змінюються; що таке «застаріло» для
  чернетки — година, день? Коли видаляти прострочене — при читанні чи
  окремо;
* сховище: словник у памʼяті (зникає з перезапуском) чи файл JSON у
  `state/` (переживає, але потребує обмеження розміру й атомарного
  запису). Що робити, коли записів більше за `MAX_ENTRIES`;
* що кешувати: результат запасного варіанту? підставного провайдера?
  відповідь, у якій `finish_reason` не `stop`;
* `fresh` оминає читання — а запис? Нова чернетка після «Повторити»
  має замінити стару в кеші чи ні;
* пʼять однакових запитів одночасно промахнуться всі — чи має другий
  чекати на перший (обʼєднання запитів у польоті) і скільки;
* лічильники для `stats`: влучання, промахи, обходи, записів зараз;
  чи скидати їх із перезапуском.
"""

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .providers.base import Request, Result

load_dotenv()

ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
TTL_SECONDS = float(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "500"))
# Порожнє значення — кеш лише в памʼяті процесу.
CACHE_FILE = os.getenv("CACHE_FILE", "state/cache.json")
_lock = threading.RLock()
_entries: dict[str, dict] = {}
_hits = 0
_misses = 0
_bypassed = 0
_loaded = False


def _load() -> None:
    global _loaded, _entries
    if _loaded:
        return
    if not ENABLED or not CACHE_FILE:
        _loaded = True
        return
    path = Path(CACHE_FILE)
    if not path.exists():
        _loaded = True
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Файл кешу має містити JSON-об'єкт: {path}")
    _entries = {key: value for key, value in data.items() if isinstance(value, dict)}
    _loaded = True


def _save() -> None:
    if not CACHE_FILE:
        return
    path = Path(CACHE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_entries, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def settings() -> dict:
    """Чинні налаштування — для сторінки й журналу."""
    return {
        "enabled": ENABLED,
        "ttl_seconds": TTL_SECONDS,
        "max_entries": MAX_ENTRIES,
        "file": CACHE_FILE or None,
    }


def make_key(request: Request, prompt_version: str) -> str:
    """Ключ кешу для запиту: однакові запити — однаковий ключ, будь-яка
    відмінність у тому, що впливає на відповідь, — інший."""
    value = {
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "model": request.model or os.getenv("LLM_MODEL", ""),
        "prompt_version": prompt_version,
        "messages": request.messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def get(key: str) -> Result | None:
    """Результат за ключем, якщо він є й не прострочений; інакше `None`.
    Коли кеш вимкнено — завжди `None`."""
    global _hits, _misses
    if not ENABLED:
        return None
    with _lock:
        _load()
        entry = _entries.get(key)
        if entry and time.time() - entry.get("created_at", 0) <= TTL_SECONDS:
            _hits += 1
            data = entry["result"]
            return Result(**data)
        _entries.pop(key, None)
        _misses += 1
        _save()
        return None


def put(key: str, result: Result) -> None:
    """Зберегти результат. Коли кеш вимкнено — нічого не робити."""
    if not ENABLED:
        return
    with _lock:
        _load()
        _entries[key] = {"created_at": time.time(), "result": asdict(result)}
        expired = [item for item, value in _entries.items()
                   if time.time() - value.get("created_at", 0) > TTL_SECONDS]
        for item in expired:
            _entries.pop(item, None)
        while len(_entries) > MAX_ENTRIES:
            oldest = min(_entries, key=lambda item: _entries[item].get("created_at", 0))
            _entries.pop(oldest)
        _save()


def note_bypass() -> None:
    """Зарахувати обхід кешу для запиту з прапорцем `fresh`."""
    global _bypassed
    with _lock:
        _bypassed += 1


def stats() -> dict:
    """Лічильники для `GET /api/usage` і сторінки: наприклад,
    `{"hits": 3, "misses": 7, "bypassed": 1, "entries": 6}`."""
    with _lock:
        _load()
        now = time.time()
        expired = [key for key, value in _entries.items()
                   if now - value.get("created_at", 0) > TTL_SECONDS]
        for key in expired:
            _entries.pop(key, None)
        if expired:
            _save()
        return {
            "hits": _hits,
            "misses": _misses,
            "bypassed": _bypassed,
            "entries": len(_entries),
        }
