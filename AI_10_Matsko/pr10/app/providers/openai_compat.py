"""Справжній провайдер: OpenAI-сумісний API — той самий, що в ПР3–ПР9.

Єдиний модуль застосунку, який імпортує SDK `openai`. Його справа —
перетворити наш `Request` на виклик API, а відповідь і збої API — на
`Result`, `Chunk` і `ProviderError` з `base.py`. Що робити зі збоєм —
повторити, зачекати, перейти на запасний варіант — тут не вирішується:
це справа шлюзу.

Клас нижче — заготовка. Рішення з розділу 2 практичної роботи:

* який виняток SDK у який `kind` перетворюється і який із них
  `retryable`: `APITimeoutError`, `RateLimitError` (429),
  `APIConnectionError`, `InternalServerError` (5xx),
  `AuthenticationError` (401), `BadRequestError` (400). Де взяти
  `Retry-After`, якщо провайдер його надіслав;
* таймаут: SDK приймає його при створенні клієнта і на окремий виклик
  (`with_options`). Що передавати — `LLM_TIMEOUT` на спробу чи щось
  інше — і хто рахує таймаут бездіяльності в streaming: SDK такого не
  вміє, це робиться у споживача фрагментів;
* `max_retries` SDK за замовчуванням не нульовий — SDK сам мовчки
  повторює деякі запити. Залишити це йому чи вимкнути й повторювати у
  шлюзі, де видно спроби, бюджет і запасний варіант;
* streaming: `stream=True` віддає обʼєкт, по якому йдуть фрагментами
  (`choices[0].delta.content`). Як закрити його, якщо споживач
  перестав читати (скасування, таймаут бездіяльності), і чи приходить
  `usage` в останньому фрагменті (`stream_options`), а якщо ні —
  що записати;
* що з відповіді потрапляє в `detail` для журналу, а що — у `message`
  для людини. Ключ і повний текст запиту — ні туди, ні туди.

Обробку збоїв із ПР3–ПР9 перенесіть сюди — але тепер вона не підіймає
власний `LLMError`, а зводить усе до `ProviderError`.
"""

from typing import Iterator

from .base import Chunk, ProviderError, Request, Result

import time
from typing import Iterator

from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
class OpenAICompatProvider:
    """Провайдер поверх OpenAI-сумісного API."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )
    def _handle_error(self, error: Exception) -> ProviderError:
        """Перетворити помилку API на ProviderError."""

        if isinstance(error, APITimeoutError):
            kind = "timeout"
            message = "Час очікування відповіді вичерпано."
            retryable = True

        elif isinstance(error, RateLimitError):
            kind = "rate_limit"
            message = "Забагато запитів. Спробуйте пізніше."
            retryable = True

        elif isinstance(error, APIConnectionError):
            kind = "unavailable"
            message = "Не вдалося з'єднатися з сервісом."
            retryable = True

        elif isinstance(error, InternalServerError):
            kind = "unavailable"
            message = "Сервіс тимчасово недоступний."
            retryable = True

        elif isinstance(error, AuthenticationError):
            kind = "auth"
            message = "Помилка авторизації провайдера."
            retryable = False

        elif isinstance(error, BadRequestError):
            kind = "bad_request"
            message = "Провайдер відхилив запит."
            retryable = False

        elif isinstance(error, APIStatusError):
            kind = "unavailable" if error.status_code >= 500 else "unknown"
            message = "Помилка під час звернення до провайдера."
            retryable = error.status_code >= 500

        else:
            kind = "unknown"
            message = "Сталася невідома помилка провайдера."
            retryable = False

        retry_after = None
        response = getattr(error, "response", None)

        if response is not None:
            value = response.headers.get("retry-after")
            if value:
                try:
                    retry_after = float(value)
                except ValueError:
                    retry_after = None

        detail = type(error).__name__

        if isinstance(error, APIStatusError):
            detail += f", status={error.status_code}"

        return ProviderError(
            kind,
            message,
            retryable=retryable,
            retry_after=retry_after,
            detail=detail,
        )

    def complete(self, request: Request) -> Result:
        """Отримати відповідь цілком."""
        start = time.monotonic()
        model = request.model or self.model

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                timeout=self.timeout,
            )

            choice = response.choices[0]
            text = choice.message.content or ""

            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return Result(
                text=text,
                provider=self.name,
                model=response.model or model,
                usage=usage,
                finish_reason=choice.finish_reason,
                elapsed=time.monotonic() - start,
            )

        except ProviderError:
            raise
        except Exception as error:
            raise self._handle_error(error) from error

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Віддавати фрагменти відповіді в міру надходження."""
        model = request.model or self.model
        stream = None

        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                timeout=self.timeout,
                stream=True,
                stream_options={"include_usage": True},
            )

            for part in stream:
                usage = None

                if part.usage:
                    usage = {
                        "prompt_tokens": part.usage.prompt_tokens,
                        "completion_tokens": part.usage.completion_tokens,
                        "total_tokens": part.usage.total_tokens,
                    }

                text = ""
                finish_reason = None

                if part.choices:
                    choice = part.choices[0]
                    text = choice.delta.content or ""
                    finish_reason = choice.finish_reason

                yield Chunk(
                    text=text,
                    usage=usage,
                    finish_reason=finish_reason,
                )

        except ProviderError:
            raise
        except Exception as error:
            raise self._handle_error(error) from error

        finally:
            if stream is not None:
                stream.close()
