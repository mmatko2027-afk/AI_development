"""Заготовка з ПР10. Замініть цей файл своїм
`pr10/app/providers/openai_compat.py`.

Що перевірити в ПР11: `usage`. Для вартості потрібні токени на вході й
на виході, а в режимі streaming SDK віддає їх лише в останньому
фрагменті й лише з `stream_options={"include_usage": True}` — і не
кожен провайдер це підтримує (скрипт перевірки середовища покаже).
Якщо `usage` в потоці немає, `Chunk.usage` лишається `None`, і вартість
оцінює шлюз через `costs.estimate` за текстом. Кешовані провайдером
вхідні токени — у `usage.prompt_tokens_details.cached_tokens`, якщо є;
передайте їх далі як є. Журнал у провайдері не потрібен: його веде
шлюз.

Нижче — заготовка ПР10 без змін.

Справжній провайдер: OpenAI-сумісний API — той самий, що в ПР3–ПР9.

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


class OpenAICompatProvider:
    """Провайдер поверх OpenAI-сумісного API."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float,
                 idle_timeout: float = 8.0) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.idle_timeout = idle_timeout

    def _client(self):
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError(
                "unknown",
                "У цьому середовищі не встановлено пакет openai. "
                "Запустіть python -m pip install -r requirements.txt і перезапустіть застосунок.",
                detail="missing_dependency: openai",
            ) from exc
        if not self.api_key:
            raise ProviderError("auth", "Налаштуйте LLM_API_KEY у файлі .env.")
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(self.timeout, read=min(self.timeout, self.idle_timeout)),
            max_retries=0,
        )

    @staticmethod
    def _provider_error(exc: Exception) -> ProviderError:
        name = type(exc).__name__
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            status_code = getattr(exc, "status_code", None)
        mapping = {
            "APITimeoutError": ("timeout", True),
            "RateLimitError": ("rate_limit", True),
            "APIConnectionError": ("unavailable", True),
            "InternalServerError": ("unavailable", True),
            "AuthenticationError": ("auth", False),
            "PermissionDeniedError": ("auth", False),
            "BadRequestError": ("bad_request", False),
        }
        kind, retryable = mapping.get(name, ("unknown", False))
        if status_code == 401 or status_code == 403:
            kind, retryable = "auth", False
        elif status_code == 429:
            kind, retryable = "rate_limit", True
        elif status_code == 400 or status_code == 404 or status_code == 422:
            kind, retryable = "bad_request", False
        elif isinstance(status_code, int) and status_code >= 500:
            kind, retryable = "unavailable", True

        headers = getattr(response, "headers", {}) or {}
        retry_after = None
        try:
            retry_after = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            pass
        messages = {
            "timeout": "Провайдер не відповів вчасно.",
            "rate_limit": "Провайдер обмежив частоту запитів.",
            "unavailable": "Провайдер тимчасово недоступний.",
            "auth": "Не вдалося пройти автентифікацію у провайдера.",
            "bad_request": "Провайдер не прийняв параметри запиту.",
            "unknown": "Не вдалося отримати відповідь від провайдера.",
        }
        message = messages[kind]
        if kind == "bad_request" and status_code == 404:
            message = "Провайдер не знайшов модель або endpoint. Перевірте назву моделі та доступ до неї."
        elif kind == "bad_request" and status_code in {400, 422}:
            message = "Провайдер відхилив запит. Перевірте модель і параметри запиту."
        elif kind == "auth" and status_code == 403:
            message = "Ключ не має доступу до цієї моделі або endpoint."

        detail = name
        if status_code is not None:
            detail = f"{name} HTTP {status_code}"
        return ProviderError(kind, message, retryable=retryable,
                             retry_after=retry_after, detail=detail)

    def _messages(self, request: Request) -> list[dict]:
        return request.messages

    def complete(self, request: Request) -> Result:
        """Отримати відповідь цілком."""
        client = self._client()
        try:
            response = client.chat.completions.create(
                model=request.model or self.model,
                messages=self._messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception as exc:
            raise self._provider_error(exc) from exc
        choice = response.choices[0]
        usage = response.usage.model_dump(exclude_none=True) if response.usage else None
        return Result(
            text=choice.message.content or "",
            provider=self.name,
            model=response.model or request.model or self.model,
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Віддавати фрагменти відповіді в міру надходження."""
        client = self._client()
        stream = None
        try:
            stream = client.chat.completions.create(
                model=request.model or self.model,
                messages=self._messages(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            for item in stream:
                if not item.choices:
                    usage = item.usage.model_dump(exclude_none=True) if item.usage else None
                    if usage:
                        yield Chunk(text="", usage=usage)
                    continue
                choice = item.choices[0]
                content = choice.delta.content or ""
                usage = item.usage.model_dump(exclude_none=True) if item.usage else None
                yield Chunk(text=content, usage=usage, finish_reason=choice.finish_reason)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._provider_error(exc) from exc
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
