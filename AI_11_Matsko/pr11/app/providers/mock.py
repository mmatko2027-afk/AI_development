"""Заготовка з ПР10. Замініть цей файл своїм `pr10/app/providers/mock.py`.

Що перевірити в ПР11: підставний провайдер має повертати правдоподібний
`usage` (наприклад, за кількістю слів у запиті й відповіді), інакше
вартість, бюджет і порівняння з кешем не перевірити без ключа. За
замовчуванням він рахується за цінами справжнього (`MOCK_BILLABLE=1`).

Нижче — заготовка ПР10 без змін.

Підставний провайдер: поводиться як справжній, але без мережі й
ключа — і збоїть тоді, коли ви просите.

Навіщо він потрібен:

* застосунок запускається й демонструється без ключа і без доступу до
  провайдера — у поїзді, на захисті, в автоматичних тестах;
* збої відтворюються на вимогу. Дочекатися справжнього 429 чи обриву
  посеред streaming можна довго; тут вони стаються за налаштуванням;
* безкоштовний ліміт ключа не витрачається на перевірку кнопок.

Поводитися він має так само, як справжній: ті самі `Result`, `Chunk`,
`ProviderError`; фрагменти — із затримкою, а не всі одразу, інакше
сторінка не покаже, як виглядає streaming. Текст відповіді не мусить
бути розумним — досить правдоподібного: у `samples/requests.json` для
кожного звернення є `reply_example`, а яке саме звернення зараз
обробляється, видно з `Request.metadata`.

Клас нижче — заготовка. Рішення з розділу 2 практичної роботи:

* звідки текст: `reply_example` за `metadata["request_id"]`, щось
  складене з останнього повідомлення користувача чи сталий рядок — і
  що робити, коли звернення невідоме;
* як відтворити кожен вид збою з `MOCK_FAILURE_KIND`: `timeout` —
  мовчати довше за таймаут чи одразу підняти `ProviderError("timeout")`?
  Перше чесніше, бо перевіряє ваш таймаут, а не лише обробку помилки;
  `rate_limit` — з яким `retry_after`; `mid_stream` — обрив після
  кількох фрагментів, коли частину тексту вже показано;
* випадковість: `MOCK_FAILURE_RATE=0.3` дає різні прогони — зручно для
  демонстрації, незручно для тестів. Чи потрібен спосіб зробити збій
  детермінованим (наприклад, через `metadata`);
* чи має підставний провайдер поважати скасування: якщо споживач
  перестав читати фрагменти, затримки мають припинитися.

Значення нижче читаються з `.env`; їхній зміст описано в
`.env.example`.
"""

import os
import random
import time
from typing import Iterator

from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result
from ..samples import all_samples

load_dotenv()

FIRST_DELAY_MS = int(os.getenv("MOCK_FIRST_DELAY_MS", "600"))
DELAY_MS = int(os.getenv("MOCK_DELAY_MS", "40"))
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0"))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable")


class MockProvider:
    """Провайдер без моделі: канонічні тексти, затримки, збої на вимогу."""

    name = "mock"
    model = "mock-1"

    def complete(self, request: Request) -> Result:
        """Повернути відповідь цілком — після тих самих затримок і з
        тими самими збоями, що й у `stream`."""
        text = ""
        usage = None
        finish_reason = None
        for chunk in self.stream(request):
            text += chunk.text
            usage = chunk.usage or usage
            finish_reason = chunk.finish_reason or finish_reason
        return Result(
            text=text,
            provider=self.name,
            model=self.model,
            usage=usage,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _pause(milliseconds: int, request: Request) -> None:
        deadline = time.monotonic() + milliseconds / 1000
        is_cancelled = request.metadata.get("is_cancelled")
        while time.monotonic() < deadline:
            if callable(is_cancelled) and is_cancelled():
                raise ProviderError("cancelled", "Генерацію скасовано.")
            time.sleep(min(0.05, deadline - time.monotonic()))

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Віддавати текст фрагментами з затримкою `DELAY_MS` між ними."""
        fail_this_request = random.random() < FAILURE_RATE
        if fail_this_request:
            kind = FAILURE_KIND
            if kind == "timeout":
                raise ProviderError("timeout", "Підставний таймаут.", retryable=True)
            if kind == "rate_limit":
                raise ProviderError("rate_limit", "Підставний ліміт провайдера.",
                                    retryable=True, retry_after=1)
            if kind == "auth":
                raise ProviderError("auth", "Помилка доступу до провайдера.")
            if kind == "bad_request":
                raise ProviderError("bad_request", "Провайдер не прийняв запит.")
            if kind != "mid_stream":
                raise ProviderError("unavailable", "Підставний провайдер недоступний.",
                                    retryable=True)

        if FIRST_DELAY_MS > 0:
            self._pause(FIRST_DELAY_MS, request)

        request_id = str(request.metadata.get("request_id", ""))
        sample = next((item for item in all_samples() if str(item["id"]) == request_id), None)
        if sample:
            text = sample.get("reply_example", "")
        elif request.messages:
            text = "Чернетка відповіді: " + str(request.messages[-1].get("content", ""))[:180]
        else:
            text = "Будь ласка, уточніть деталі звернення."
        if not text:
            text = "Будь ласка, уточніть деталі звернення."

        pieces = [text[index:index + 32] for index in range(0, len(text), 32)]
        for index, piece in enumerate(pieces):
            if index and DELAY_MS > 0:
                self._pause(DELAY_MS, request)
            yield Chunk(text=piece)
            if fail_this_request and FAILURE_KIND == "mid_stream" and index == 0:
                raise ProviderError("unavailable", "Підставний потік обірвався.",
                                    retryable=True)

        prompt_words = sum(len(str(message.get("content", "")).split())
                           for message in request.messages)
        completion_words = len(text.split())
        usage = {
            "prompt_tokens": max(1, round(prompt_words * 1.3)),
            "completion_tokens": max(1, round(completion_words * 1.3)),
        }
        yield Chunk(text="", usage=usage, finish_reason="stop")
