"""Підставний провайдер: поводиться як справжній, але без мережі й
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
import time
import random
from typing import Iterator

from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result

load_dotenv()

FIRST_DELAY_MS = int(os.getenv("MOCK_FIRST_DELAY_MS", "600"))
DELAY_MS = int(os.getenv("MOCK_DELAY_MS", "40"))
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0"))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable")


class MockProvider:
    """Провайдер без моделі: канонічні тексти, затримки, збої на вимогу."""

    name = "mock"
    model = "mock-1"

    
    def _get_text(self, request: Request) -> str:
        """Отримати приклад відповіді для потрібного звернення."""

        from ..samples import get

        request_id = request.metadata.get("request_id")
        if request_id is None:
            request_id = request.metadata.get("sample_id")

        sample = get(str(request_id)) if request_id is not None else None

        if sample is not None:
            return sample.get(
                "reply_example",
                "Дякуємо за звернення. Ми уточнимо інформацію."
            )

        return (
            "Дякуємо за звернення. "
            "Ваше питання потребує додаткового уточнення."
        )


    def _check_failure(self) -> None:
        """За потреби відтворити помилку."""

        if random.random() >= FAILURE_RATE:
            return

        if FAILURE_KIND == "timeout":
            time.sleep(max(FIRST_DELAY_MS / 1000, 1))
            raise ProviderError(
                "timeout",
                "Час очікування відповіді вичерпано.",
                retryable=True,
            )

        if FAILURE_KIND == "rate_limit":
            raise ProviderError(
                "rate_limit",
                "Перевищено ліміт запитів.",
                retryable=True,
                retry_after=1,
            )

        if FAILURE_KIND == "mid_stream":
            return

        raise ProviderError(
            FAILURE_KIND,
            "Підставний провайдер імітував помилку.",
            retryable=FAILURE_KIND in ("unavailable", "timeout", "rate_limit"),
        )


    def complete(self, request: Request) -> Result:
        """Повернути відповідь цілком — після тих самих затримок і з
        тими самими збоями, що й у `stream`."""
        start = time.monotonic()
        time.sleep(FIRST_DELAY_MS / 1000)
        self._check_failure()

        return Result(
            text=self._get_text(request),
            provider=self.name,
            model=self.model,
            usage=None,
            finish_reason="stop",
            elapsed=time.monotonic() - start,
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Віддавати текст фрагментами з затримкою `DELAY_MS` між ними."""
        start = time.monotonic()
        time.sleep(FIRST_DELAY_MS / 1000)
        self._check_failure()

        text = self._get_text(request)
        words = text.split()
        mid_stream = (
            FAILURE_KIND == "mid_stream"
            and random.random() < FAILURE_RATE
        )

        for index, word in enumerate(words):
            if index > 0:
                time.sleep(DELAY_MS / 1000)

            if mid_stream and index >= max(1, len(words) // 2):
                raise ProviderError(
                    "unavailable",
                    "З'єднання перервано під час відповіді.",
                    retryable=True,
                    detail="mock mid_stream failure",
                )

            chunk_text = word
            if index < len(words) - 1:
                chunk_text += " "

            yield Chunk(text=chunk_text)

        yield Chunk(
            text="",
            usage=None,
            finish_reason="stop",
        )
