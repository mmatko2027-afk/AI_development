"""Заготовка з ПР10. Замініть цей файл своїм `pr10/app/gateway.py` і
допишіть те, що перелічено нижче.

Що додати в ПР11:

* журнал: запис на кожну спробу — початок і кінець — та підсумок на
  запит. Поля: ідентифікатори запиту й чернетки (з `Request.metadata`),
  клієнт, спроба, провайдер, модель, яка відповіла, версія prompt, час
  очікування в черзі одночасності, час до першого фрагмента, тривалість,
  статус або вид збою, запасний варіант, токени, вартість, кеш. Рівень:
  `INFO` для штатного, `WARNING` для повтору й запасного варіанту,
  `ERROR` для остаточного збою. Без тексту повідомлень і без ключа;
* кеш: перед першою спробою — `cache.get(cache.make_key(...))`.
  Влучання — результат із `cached=True` без звертання до провайдера,
  але фрагменти споживачеві все одно віддаються (одним шматком чи
  кількома — ваше рішення). Промах — після вдалої генерації
  `cache.put`. Прапорець `fresh` у `metadata` оминає читання кешу
  (а запис?). Запасний варіант і підставний провайдер — кешувати чи ні;
* бюджет: `budget.check()` перед звертанням до провайдера — не перед
  кешем. Вичерпано — `ProviderError(kind="budget", retryable=False)` з
  повідомленням для людини й часом відновлення, без повторів.
  `budget.record(cost)` після кожного платного звертання — включно зі
  спробами, що завершилися збоєм після того, як токени вже витрачено;
* вартість: `costs.estimate(usage, ...)` після кожного звертання → у
  `Result.cost`, `Result.cost_estimated` і в журнал; `usage` зі
  streaming — див. провайдера.

Шлюз і далі не знає, як влаштовані журнал, кеш, бюджет і ціни: він
викликає `cache.get/put`, `budget.check/record`, `costs.estimate` і
`logging` — і все.

Нижче — заготовка ПР10 без змін.

Шлюз до моделі: єдиний шлях, яким застосунок звертається до
провайдера.

Решта застосунку не викликає провайдерів напряму. Вона каже шлюзу
«згенеруй» — і отримує або результат, або контрольовану помилку. Між
цими двома подіями шлюз робить те, що не повинен робити ані провайдер,
ані веб-рівень, ані фонова задача:

* обирає провайдера за конфігурацією (`LLM_PROVIDER`) і запасний
  варіант (`LLM_FALLBACK`);
* тримає таймаути: на спробу, на бездіяльність у streaming, на весь
  запит разом зі спробами;
* повторює спроби — лише ті збої, які має сенс повторювати, з
  наростаючою паузою й випадковою часткою, не більше `MAX_ATTEMPTS`
  і не довше за `TOTAL_TIMEOUT`;
* обмежує кількість одночасних звертань до провайдера на весь
  застосунок (`MAX_CONCURRENT`);
* перериває роботу, коли той, хто просив, передумав;
* повідомляє, що відбувається: яка спроба, який провайдер, чи це
  запасний варіант — щоб сторінка могла це показати.

Шлюз не знає, що саме генерується (чернетка відповіді чи щось інше) і
звідки прийшов запит (сторінка, тест, скрипт). Він знає лише
`providers.base`.

Функції нижче — заготовки. Рішення з розділу 2 практичної роботи:

* три таймаути — на спробу, на бездіяльність, на весь запит. Який
  ловить що, і що стається, коли спливає кожен із них посеред
  streaming, коли частину тексту вже віддано;
* які `kind` повторювати, а які ні; що робити з `retry_after` від
  провайдера, якщо він більший за ваш бюджет часу;
* чи повторювати спробу після того, як перший фрагмент уже віддано
  споживачеві. Якщо так — як сказати йому «починаємо спочатку»;
* коли вмикати запасний варіант: після всіх спроб, після першої, одразу
  на певних видах збою? Що з результату має сказати, що він запасний;
* що означає `MAX_CONCURRENT`, коли місця немає: чекати (скільки?) чи
  відмовляти одразу — і як це відрізняється від обмеження частоти для
  клієнтів у `app/limits.py`;
* як передавати фрагменти й стан назовні: зворотний виклик, генератор,
  черга? Як дізнаватися про скасування між фрагментами й між спробами;
* що потрапляє в підсумок для журналу: спроби, провайдер, модель, час
  до першого фрагмента, загальний час, токени.
"""

import os
import logging
import random
import re
import threading
import time
from typing import Callable

from dotenv import load_dotenv

from . import budget, cache, costs
from .providers.base import Provider, ProviderError, Request, Result
from .providers.mock import MockProvider
from .providers.openai_compat import OpenAICompatProvider

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai")
FALLBACK = os.getenv("LLM_FALLBACK", "none")

TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
IDLE_TIMEOUT = float(os.getenv("LLM_IDLE_TIMEOUT", "8"))
TOTAL_TIMEOUT = float(os.getenv("LLM_TOTAL_TIMEOUT", "45"))
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "0.5"))
BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))
MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "2"))

# Параметри генерації — спільні для всіх провайдерів.
MODEL = os.getenv("LLM_MODEL")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
_slots = threading.BoundedSemaphore(max(1, MAX_CONCURRENT))
_log = logging.getLogger("app.gateway")


def _safe_error_detail(error: Exception) -> str:
    """Додати пояснення винятку до діагностики, не записуючи API-ключ."""
    detail = str(error).replace("\r", " ").replace("\n", " ").strip()
    api_key = os.getenv("LLM_API_KEY", "")
    if api_key:
        detail = detail.replace(api_key, "[приховано]")
    detail = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [приховано]", detail)
    detail = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[приховано]", detail)
    detail = re.sub(r"https?://\S+", "[адреса прихована]", detail)
    if len(detail) > 240:
        detail = detail[:237] + "..."
    if detail:
        return f"{type(error).__name__}: {detail}"
    return type(error).__name__


def settings() -> dict:
    """Чинні налаштування шлюзу — для сторінки й журналу. Ключа тут
    немає і бути не може."""
    return {
        "provider": PROVIDER,
        "fallback": FALLBACK,
        "model": MODEL,
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "max_concurrent": MAX_CONCURRENT,
    }


def make_provider(name: str) -> Provider:
    """Зібрати провайдера за назвою: `openai` — з налаштувань доступу в
    `.env`, `mock` — підставний. Інша назва — та сама адреса й ключ, але
    інша модель (для запасного варіанту)."""
    if name == "mock":
        return MockProvider()
    model = MODEL or ""
    if name != "openai":
        model = name
    return OpenAICompatProvider(
        base_url=os.getenv("LLM_BASE_URL", ""),
        api_key=os.getenv("LLM_API_KEY", ""),
        model=model,
        timeout=TIMEOUT,
        idle_timeout=IDLE_TIMEOUT,
    )


def generate(request: Request, on_delta: Callable[[str], None],
             on_status: Callable[[dict], None], is_cancelled: Callable[[], bool]) -> Result:
    """Згенерувати відповідь, дотримуючись таймаутів, спроб, обмеження
    одночасності й запасного варіанту.

    `on_delta` отримує кожен фрагмент тексту, `on_status` — зміни стану
    (наприклад, `{"attempt": 2, "provider": "openai"}` або
    `{"fallback": True}`), `is_cancelled` опитується між фрагментами й
    між спробами. Сигнатура — відправна точка: якщо вам зручніше
    генератор чи черга, змініть і її, і `app/jobs.py`.

    Повертає `Result` або підіймає `ProviderError` — уже остаточний,
    після всіх спроб і запасного варіанту. Інших винятків назовні не
    виходить.
    """
    started = time.monotonic()
    request_id = str(request.metadata.get("http_request_id") or request.metadata.get("request_id") or "")
    draft_id = request.metadata.get("draft_id")
    client = request.metadata.get("client")
    prompt_version = str(request.metadata.get("prompt_version", "unknown"))
    cache_key = cache.make_key(request, prompt_version)
    fresh = bool(request.metadata.get("fresh"))
    extra = {
        "request_id": request_id,
        "draft_id": draft_id,
        "client": client,
        "prompt_version": prompt_version,
        "cache_key": cache_key,
    }

    if fresh:
        cache.note_bypass()
        _log.info("cache.bypassed", extra=extra)
    else:
        cached = cache.get(cache_key)
        if cached is not None:
            cached.cached = True
            _log.info("cache.hit", extra={**extra, "cost": cached.cost})
            on_status({"cached": True, "provider": cached.provider, "model": cached.model})
            if cached.text:
                on_delta(cached.text)
            return cached
        _log.info("cache.miss", extra=extra)

    provider_names = [PROVIDER]
    if FALLBACK and FALLBACK.lower() != "none" and FALLBACK not in provider_names:
        provider_names.append(FALLBACK)
    attempts = 0
    last_error: ProviderError | None = None
    fallback_active = False
    for provider_index, provider_name in enumerate(provider_names):
        fallback_active = provider_index > 0
        provider = make_provider(provider_name)
        if attempts >= max(1, MAX_ATTEMPTS):
            break
        attempt_limit = max(1, MAX_ATTEMPTS - attempts)
        for retry_index in range(attempt_limit):
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.")
            remaining = TOTAL_TIMEOUT - (time.monotonic() - started)
            if remaining <= 0:
                raise ProviderError("timeout", "Вичерпано загальний час очікування.", retryable=False)

            attempts += 1
            on_status({
                "status": "running",
                "attempt": attempts,
                "provider": provider.name,
                "fallback": fallback_active,
            })
            _log.info("llm.attempt.start", extra={
                **extra, "attempt": attempts, "provider": provider.name,
                "model": getattr(provider, "model", None), "fallback": fallback_active,
            })
            attempt_started = time.monotonic()
            queue_started = time.monotonic()
            provider_started = attempt_started
            queue_wait = 0.0
            acquired = False
            first_token = None
            answer = ""
            usage = None
            finish_reason = None
            attempt_cost = None
            estimated = False
            error = None
            stream_started = False
            try:
                while not acquired:
                    if is_cancelled():
                        queue_wait = time.monotonic() - queue_started
                        raise ProviderError("cancelled", "Генерацію скасовано.")
                    remaining = TOTAL_TIMEOUT - (time.monotonic() - started)
                    if remaining <= 0:
                        queue_wait = time.monotonic() - queue_started
                        raise ProviderError("timeout", "Вичерпано час очікування в черзі.")
                    acquired = _slots.acquire(timeout=min(0.2, remaining))
                queue_wait = time.monotonic() - queue_started
                provider_started = time.monotonic()
                remaining = TOTAL_TIMEOUT - (provider_started - started)
                if remaining <= 0:
                    raise ProviderError("timeout", "Вичерпано загальний час очікування.")
                if hasattr(provider, "timeout"):
                    provider.timeout = min(TIMEOUT, remaining)
                if provider.name != "mock" or costs.MOCK_BILLABLE:
                    budget.check()
                provider_request = Request(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    model=provider.model,
                    metadata={**request.metadata, "is_cancelled": is_cancelled},
                )
                stream_started = True
                for chunk in provider.stream(provider_request):
                    if is_cancelled():
                        raise ProviderError("cancelled", "Генерацію скасовано.")
                    if time.monotonic() - started > TOTAL_TIMEOUT:
                        raise ProviderError("timeout", "Вичерпано загальний час очікування.")
                    if chunk.text:
                        if first_token is None:
                            first_token = time.monotonic() - provider_started
                        answer += chunk.text
                        on_delta(chunk.text)
                    if chunk.usage:
                        usage = chunk.usage
                    if chunk.finish_reason:
                        finish_reason = chunk.finish_reason
            except budget.BudgetExceeded as exc:
                error = ProviderError("budget", str(exc), retryable=False)
            except ProviderError as exc:
                error = exc
                if answer:
                    partial_cost, partial_estimated = costs.estimate(
                        usage, provider=provider.name, model=provider.model, text=answer
                    )
                    budget.record(partial_cost)
                    attempt_cost, estimated = partial_cost, partial_estimated
            except Exception as exc:
                if not stream_started:
                    raise
                error = ProviderError("unknown", "Сталася неочікувана помилка провайдера.",
                                      detail=_safe_error_detail(exc))
            finally:
                if acquired:
                    _slots.release()

            if error is None:
                attempt_cost, estimated = costs.estimate(
                    usage, provider=provider.name, model=provider.model, text=answer or None
                )
                budget.record(attempt_cost)
                duration = time.monotonic() - provider_started
                result = Result(
                    text=answer,
                    provider=provider.name,
                    model=provider.model,
                    usage=usage,
                    finish_reason=finish_reason,
                    elapsed=duration,
                    cost=attempt_cost,
                    cost_estimated=estimated,
                )
                _log.info("llm.attempt.done", extra={
                    **extra, "attempt": attempts, "provider": provider.name, "model": result.model,
                    "queue_wait": round(queue_wait, 4),
                    "first_token": round(first_token, 4) if first_token is not None else None,
                    "duration": round(duration, 4), "status": "success",
                    "fallback": fallback_active, "usage": usage, "cost": attempt_cost,
                    "cost_estimated": estimated,
                })
                _log.info("llm.done", extra={
                    **extra, "attempts": attempts, "provider": provider.name, "model": result.model,
                    "fallback": fallback_active, "usage": usage, "cost": attempt_cost,
                    "cost_estimated": estimated, "duration": round(time.monotonic() - started, 4),
                })
                try:
                    cache.put(cache_key, result)
                except (OSError, TypeError, ValueError) as exc:
                    _log.error("cache.write_failed", extra={**extra, "error": type(exc).__name__})
                return result

            last_error = error
            duration = time.monotonic() - provider_started
            retrying = bool(error and error.retryable and retry_index + 1 < attempt_limit and not answer)
            will_fallback = bool(
                error and not answer and provider_index + 1 < len(provider_names)
                and (error.retryable or error.kind in {"auth", "bad_request", "unknown"})
            )
            _log.log(
                logging.WARNING if retrying or will_fallback else logging.ERROR,
                "llm.attempt.done",
                extra={
                    **extra, "attempt": attempts, "provider": provider.name,
                    "model": getattr(provider, "model", None),
                    "queue_wait": round(queue_wait, 4),
                    "first_token": round(first_token, 4) if first_token is not None else None,
                    "duration": round(duration, 4), "status": "failed",
                    "failure_kind": error.kind if error else "unknown",
                    "failure_detail": error.detail if error else None,
                    "retryable": error.retryable if error else False,
                    "fallback": fallback_active, "usage": usage, "cost": attempt_cost,
                    "cost_estimated": estimated,
                },
            )
            if error and error.kind == "budget":
                raise error
            if is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.") from error
            if answer or not error or not error.retryable or retry_index + 1 >= attempt_limit:
                break
            wait = error.retry_after if error.retry_after is not None else min(
                BACKOFF_MAX, BACKOFF_BASE * (2 ** retry_index)
            )
            wait = max(0.0, min(wait + random.uniform(0, BACKOFF_BASE), TOTAL_TIMEOUT - (time.monotonic() - started)))
            on_status({"status": "running", "attempt": attempts, "provider": provider.name,
                       "fallback": fallback_active, "retrying": True})
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                if is_cancelled():
                    raise ProviderError("cancelled", "Генерацію скасовано.")
                time.sleep(min(0.1, deadline - time.monotonic()))

        if last_error and not answer and (last_error.retryable or last_error.kind in {"auth", "bad_request", "unknown"}):
            if provider_index + 1 < len(provider_names) and not is_cancelled():
                _log.warning("llm.fallback", extra={
                    **extra, "provider": provider.name, "failure_kind": last_error.kind,
                    "fallback_provider": provider_names[provider_index + 1],
                })
                on_status({"fallback": True, "provider": provider_names[provider_index + 1]})
                continue
        break

    if last_error:
        _log.error("llm.failed", extra={
            **extra,             "attempts": attempts, "failure_kind": last_error.kind,
            "failure_detail": last_error.detail,
            "retryable": last_error.retryable, "duration": round(time.monotonic() - started, 4),
        })
        raise last_error
    raise ProviderError("unknown", "Не вдалося отримати відповідь від провайдера.")
