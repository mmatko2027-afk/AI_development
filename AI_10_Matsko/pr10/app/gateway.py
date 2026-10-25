"""Шлюз до моделі: єдиний шлях, яким застосунок звертається до
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
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable

from dotenv import load_dotenv

from .providers.base import Provider, Request, Result, ProviderError
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
_semaphore = threading.BoundedSemaphore(max(1, MAX_CONCURRENT))

def settings() -> dict:
    """Чинні налаштування шлюзу — для сторінки й журналу. Ключа тут
    немає і бути не може."""
    return {
        "provider": PROVIDER,
        "fallback": FALLBACK,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "backoff_base": BACKOFF_BASE,
        "backoff_max": BACKOFF_MAX,
        "max_concurrent": MAX_CONCURRENT,
    }


def make_provider(name: str) -> Provider:
    """Зібрати провайдера за назвою: `openai` — з налаштувань доступу в
    `.env`, `mock` — підставний. Інша назва — та сама адреса й ключ, але
    інша модель (для запасного варіанту)."""
    if name == "mock":
        return MockProvider()

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv(
        "OPENAI_BASE_URL",
        "https://api.openai.com/v1",
    )

    model = MODEL if name == "openai" else name

    return OpenAICompatProvider(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=TIMEOUT,
    )

def _cancelled() -> ProviderError:
    return ProviderError(
        "cancelled",
        "Запит скасовано.",
        retryable=False,
    )

def _wait(seconds: float, is_cancelled: Callable[[], bool]) -> None:
    """Чекати, перевіряючи скасування."""

    end_time = time.monotonic() + max(0, seconds)

    while time.monotonic() < end_time:
        if is_cancelled():
            raise _cancelled()

        remaining = end_time - time.monotonic()
        time.sleep(min(0.1, max(0, remaining)))

    if is_cancelled():
        raise _cancelled()

def _future_result(
    future,
    timeout: float,
    is_cancelled: Callable[[], bool],
):
    """Очікувати результат із перевіркою тайм-ауту та скасування."""
    end_time = time.monotonic() + max(0, timeout)

    while True:
        if is_cancelled():
            future.cancel()
            raise _cancelled()

        remaining = end_time - time.monotonic()

        if remaining <= 0:
            future.cancel()
            raise TimeoutError()

        try:
            return future.result(timeout=min(0.1, remaining))
        except TimeoutError:

            if future.done():
                raise


def _run_attempt(
    provider: Provider,
    request: Request,
    on_delta: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    deadline: float,
) -> Result:
    """Виконати одну потокову спробу."""

    started = time.monotonic()
    attempt_deadline = min(deadline, started + TIMEOUT)

    executor = ThreadPoolExecutor(max_workers=1)
    stream = None

    text_parts = []
    usage = None
    finish_reason = None
    first_chunk = True

    try:

        stream_future = executor.submit(provider.stream, request)

        try:
            stream = _future_result(
                stream_future,
                attempt_deadline - time.monotonic(),
                is_cancelled,
            )
        except TimeoutError as error:
            raise ProviderError(
                "timeout",
                "Час створення потоку відповіді вичерпано.",
                retryable=True,
            ) from error

        iterator = iter(stream)

        while True:
            if is_cancelled():
                raise _cancelled()

            now = time.monotonic()
            remaining_total = deadline - now
            remaining_attempt = attempt_deadline - now

            if remaining_total <= 0:
                raise ProviderError(
                    "timeout",
                    "Перевищено загальний час виконання запиту.",
                    retryable=False,
                )

            if remaining_attempt <= 0:
                raise ProviderError(
                    "timeout",
                    "Час однієї спроби генерації вичерпано.",
                    retryable=True,
                )


            if first_chunk:
                wait_timeout = remaining_attempt
            else:
                wait_timeout = min(
                    IDLE_TIMEOUT,
                    remaining_attempt,
                )

            future = executor.submit(next, iterator)

            try:
                chunk = _future_result(
                    future,
                    wait_timeout,
                    is_cancelled,
                )
            except StopIteration:
                break
            except TimeoutError as error:
                future.cancel()

                if time.monotonic() >= deadline:
                    raise ProviderError(
                        "timeout",
                        "Перевищено загальний час виконання запиту.",
                        retryable=False,
                    ) from error

                if time.monotonic() >= attempt_deadline:
                    message = "Час однієї спроби генерації вичерпано."
                else:
                    message = "Час очікування фрагмента вичерпано."

                raise ProviderError(
                    "timeout",
                    message,
                    retryable=True,
                ) from error

            first_chunk = False

            if chunk.text:
                text_parts.append(chunk.text)
                on_delta(chunk.text)

            if chunk.usage is not None:
                usage = chunk.usage

            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason

        if is_cancelled():
            raise _cancelled()

        return Result(
            text="".join(text_parts),
            provider=provider.name,
            model=getattr(provider, "model", MODEL),
            usage=usage,
            finish_reason=finish_reason or "stop",
            elapsed=time.monotonic() - started,
        )

    finally:
        if stream is not None:
            close = getattr(stream, "close", None)

            if close is not None:
                try:
                    close()
                except Exception:
                    pass

        executor.shutdown(wait=False, cancel_futures=True)

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
    deadline = started + TOTAL_TIMEOUT

    if is_cancelled():
        raise _cancelled()

    semaphore_timeout = max(
        0,
        min(TIMEOUT, deadline - time.monotonic()),
    )

    if not _semaphore.acquire(timeout=semaphore_timeout):
        raise ProviderError(
            "unavailable",
            "Зараз усі місця для запитів зайняті. Спробуйте пізніше.",
            retryable=True,
        )

    try:
        providers = [PROVIDER]

        if (
            FALLBACK
            and FALLBACK.lower() != "none"
            and FALLBACK != PROVIDER
        ):
            providers.append(FALLBACK)

        last_error = None

        for provider_index, provider_name in enumerate(providers):
            if is_cancelled():
                raise _cancelled()

            if time.monotonic() >= deadline:
                raise ProviderError(
                    "timeout",
                    "Перевищено загальний час виконання запиту.",
                    retryable=False,
                )

            provider = make_provider(provider_name)
            is_fallback = provider_index > 0

            if is_fallback:
                on_status({
                    "fallback": True,
                    "provider": provider_name,
                })


            attempts = 1 if is_fallback else max(1, MAX_ATTEMPTS)

            for attempt in range(1, attempts + 1):
                if is_cancelled():
                    raise _cancelled()

                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    raise ProviderError(
                        "timeout",
                        "Перевищено загальний час виконання запиту.",
                        retryable=False,
                    )

                on_status({
                    "attempt": attempt,
                    "provider": provider_name,
                    "fallback": is_fallback,
                })

                received_text = False

                def send_delta(delta: str) -> None:
                    nonlocal received_text

                    if delta:
                        received_text = True
                        on_delta(delta)

                try:
                    result = _run_attempt(
                        provider,
                        request,
                        send_delta,
                        is_cancelled,
                        deadline,
                    )

                    on_status({
                        "done": True,
                        "provider": result.provider,
                        "model": result.model,
                        "elapsed": result.elapsed,
                        "usage": result.usage,
                        "attempt": attempt,
                        "fallback": is_fallback,
                    })

                    return result

                except ProviderError as error:
                    last_error = error

                    if error.kind == "cancelled":
                        raise

                    if received_text:
                        raise

                    if not error.retryable or attempt >= attempts:
                        break

                    remaining = deadline - time.monotonic()

                    if remaining <= 0:
                        break

                    backoff = min(
                        BACKOFF_BASE * (2 ** (attempt - 1)),
                        BACKOFF_MAX,
                    )

                    if error.retry_after is not None:
                        backoff = max(backoff, error.retry_after)

                    backoff += random.uniform(0, 0.25)
                    backoff = min(backoff, remaining)

                    on_status({
                        "retry": True,
                        "attempt": attempt,
                        "delay": backoff,
                        "kind": error.kind,
                    })

                    _wait(backoff, is_cancelled)

        if is_cancelled():
            raise _cancelled()

        if time.monotonic() >= deadline:
            raise ProviderError(
                "timeout",
                "Перевищено загальний час виконання запиту.",
                retryable=False,
            )

        if last_error is not None:
            raise last_error

        raise ProviderError(
            "unknown",
            "Не вдалося отримати відповідь від провайдера.",
            retryable=False,
        )

    except ProviderError:
        raise

    except Exception as error:
        raise ProviderError(
            "unknown",
            "Під час генерації сталася внутрішня помилка.",
            retryable=False,
            detail=type(error).__name__,
        ) from error

    finally:
        _semaphore.release()
