"""Вартість виклику: токени → гроші.

Провайдер тарифікує токени окремо на вході й на виході; кешовані ним
вхідні токени — дешевше. Ціни — з `.env`, за мільйон токенів; значення
зі зразка орієнтовні, чинні — на сторінці цін провайдера. Безкоштовний
рівень не скасовує розрахунку: рахуємо так, ніби платимо, інакше нема
чого обмежувати й порівнювати.

Модуль знає лише ціни та `usage`; про провайдерів, чернетки й журнал
він не знає. Шлюз викликає `estimate` після кожного звертання і кладе
результат у `Result.cost` та в журнал; ліміт (`budget.py`) додає його
до витрат дня.

Функція нижче — заготовка. Рішення з розділу 2 практичної роботи:

* назви полів `usage`: OpenAI-сумісний API віддає `prompt_tokens`,
  `completion_tokens`, іноді `prompt_tokens_details.cached_tokens`;
  підставний провайдер — те, що ви в ньому напишете. Що робити з
  полями, яких немає;
* `usage` не прийшов (потік обірвано, провайдер не підтримує
  `stream_options`): повернути `None`, нуль чи оцінку за довжиною
  тексту (приблизно 4 символи на токен для англійської, для
  української помітно менше)? Якщо оцінку — вона має бути позначена
  як оцінка, а не факт, у `Result` і в журналі;
* чия ціна: модель запасного варіанту може коштувати інакше, ніж
  основна; підставний провайдер за замовчуванням рахується за цінами
  справжнього (`MOCK_BILLABLE=1`), щоб бюджет можна було перевірити
  без ключа — чи лишати так;
* одиниці й округлення: долари з шістьма знаками, центи, мікродолари?
  Суми за день складаються з тисяч дрібних чисел.
"""

import os
import math

from dotenv import load_dotenv

load_dotenv()

# Ціни за мільйон токенів, у доларах.
PRICE_INPUT = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0.30"))
PRICE_OUTPUT = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "2.50"))
_cached = os.getenv("LLM_PRICE_CACHED_INPUT_PER_1M", "")
PRICE_CACHED_INPUT = float(_cached) if _cached else PRICE_INPUT * 0.25
MOCK_BILLABLE = os.getenv("MOCK_BILLABLE", "1") == "1"


def settings() -> dict:
    """Чинні ціни — для сторінки й журналу."""
    return {
        "input_per_1m": PRICE_INPUT,
        "output_per_1m": PRICE_OUTPUT,
        "cached_input_per_1m": PRICE_CACHED_INPUT,
        "mock_billable": MOCK_BILLABLE,
        "currency": "USD",
    }


def estimate(usage: dict | None, *, provider: str | None = None, model: str | None = None,
             text: str | None = None) -> tuple[float | None, bool]:
    """Оцінити вартість одного звертання до провайдера.

    `usage` — те, що повернув провайдер (може бути `None`); `provider` і
    `model` — хто відповідав; `text` — текст відповіді, якщо доведеться
    оцінювати без `usage`.

    Повертає пару: вартість у доларах (або `None`, якщо її не визначити)
    і ознаку «це оцінка, а не розрахунок за фактичним `usage`».
    """
    if provider == "mock" and not MOCK_BILLABLE:
        return 0.0, False

    def number(value) -> int | None:
        try:
            result = int(value)
        except (TypeError, ValueError):
            return None
        return max(0, result)

    if usage:
        input_tokens = number(usage.get("prompt_tokens"))
        output_tokens = number(usage.get("completion_tokens"))
        if input_tokens is not None and output_tokens is not None:
            details = usage.get("prompt_tokens_details") or {}
            cached_tokens = number(details.get("cached_tokens")) or 0
            cached_tokens = min(input_tokens, cached_tokens)
            regular_tokens = input_tokens - cached_tokens
            cost = (
                regular_tokens * PRICE_INPUT
                + cached_tokens * PRICE_CACHED_INPUT
                + output_tokens * PRICE_OUTPUT
            ) / 1_000_000
            return round(cost, 8), False

    if text is None:
        return None, False
    estimated_output_tokens = max(1, math.ceil(len(text) / 3))
    estimated_cost = estimated_output_tokens * PRICE_OUTPUT / 1_000_000
    return round(estimated_cost, 8), True
