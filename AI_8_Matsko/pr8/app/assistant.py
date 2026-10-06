"""Цикл помічника: від питання клієнта до відповіді, спертої на дані
магазину.

Єдине місце, яке знає послідовність: спитати модель з описами
інструментів (`llm.chat`), виконати виклики, які вона запропонувала
(`tools.call`), повернути їй результати і спитати ще раз — доки вона не
відповість текстом або доки не вичерпано обмеження. Учасники одне про
одного не знають: модель — про сервіс магазину, інструменти — про API
моделі, веб-рівень — ні про що з цього.

Функція `answer` — заготовка. Рішення з розділу 2 практичної роботи:

* скільки разів можна повернутися до моделі і що робити, коли ліміт
  вичерпано, а відповіді текстом так і немає;
* що робити, якщо модель запропонувала кілька викликів одразу: виконати
  всі, частину, жодного? Чи залежить це від того, що саме вони роблять?
* як звʼязати результат із викликом, до якого він належить;
* що потрапляє в журнал викликів (`ToolTrace`) і що з нього показувати
  на сторінці;
* що рахувати окремо: час моделі й час інструментів — це різні витрати.
"""

import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from . import llm, tools

load_dotenv()

# Скільки разів можна звернутися до моделі за одне питання клієнта.
MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))


@dataclass
class ToolTrace:
    """Запис про один виклик інструмента — для сторінки й для перевірки.

    `round` — на якому звертанні до моделі його запропоновано (з 1).
    `name` і `arguments` — як їх повернула модель, до будь-яких
    перевірок. `status` і `reason` — що з викликом зробив ваш код.
    `result` — що повернулося моделі. `elapsed` — час виконання, секунди.
    """

    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    """Результат циклу — те, з чим працює веб-рівень.

    `text` — відповідь клієнтові. `calls` — журнал викликів інструментів
    у порядку виконання. `rounds` — скільки разів звертались до моделі.
    `stopped` — чому цикл завершився, наприклад `answer` (модель
    відповіла текстом) чи `limit` (вичерпано обмеження). `elapsed` — час
    за видами, наприклад `{"model": 3.4, "tools": 0.02}`. `usage` —
    токени запиту й відповіді, сумарно за всі звертання.
    """

    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(question: str, customer_id: str) -> Answer:
    """Відповісти клієнтові, за потреби викликаючи інструменти.

    `customer_id` — клієнт, який увійшов; його передає веб-рівень, а не
    модель.
    """
    messages = llm.build_messages(question)

    context = tools.Context(
        customer_id=customer_id
    )

    calls = []
    total_model_time = 0
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for round_number in range(1, MAX_ROUNDS + 1):
        try:
            result = llm.chat(
                messages=messages,
                tools=tools.specs(),
                tool_choice="auto",
            )
        except llm.LLMError as exc:
            return Answer(
                text=f"Не вдалося отримати відповідь від мовної моделі: {exc}",
                calls=calls,
                rounds=round_number,
                stopped="model_error",
                elapsed={
                    "model": total_model_time
                },
                usage=total_usage,
            )

        total_model_time += result.get("elapsed", 0)

        usage = result.get("usage")

        if usage:
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)

        message = result["message"]

        messages.append(message)

        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            return Answer(
                text=message.get("content") or "",
                calls=calls,
                rounds=round_number,
                stopped="answer",
                model=result.get("model"),
                elapsed={
                    "model": total_model_time
                },
                usage=total_usage,
            )

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            raw_arguments = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]

            start_tool = time.perf_counter()

            tool_result = tools.call(
                tool_name,
                raw_arguments,
                context,
            )

            tool_elapsed = time.perf_counter() - start_tool

            calls.append(
                ToolTrace(
                    round=round_number,
                    name=tool_name,
                    arguments=raw_arguments,
                    status=tool_result.status,
                    reason=tool_result.reason,
                    result=tool_result.content,
                    elapsed=tool_elapsed,
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(tool_result.content),
                }
            )

    return Answer(
        text="Не вдалося отримати остаточну відповідь у встановлену кількість раундів.",
        calls=calls,
        rounds=MAX_ROUNDS,
        stopped="max_rounds",
        elapsed={
            "model": total_model_time
        },
        usage=total_usage,
    )