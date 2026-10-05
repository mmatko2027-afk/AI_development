"""Конвеєр обробки документа: від байтів файлу до рішення, що з ним робити.

Єдине місце, яке знає всі кроки послідовно: підготувати зображення
(`images.prepare`), вилучити поля (`llm.extract`), перевірити їх
правилами (`rules.check`) і вирішити долю документа. Учасники одне про
одного не знають: модуль зображень — про модель, модель — про правила,
правила — про веб-рівень.

Рішення про документ — одне з трьох:

* `auto` — дані можна передати в реєстр платежів без людини;
* `review` — дані є, але їх має переглянути людина: причини перелічено;
* `reject` — це не рахунок на оплату або з документа нічого не взяти.

Рішення ухвалює код за результатами перевірок, а не модель. Функції
нижче — заготовки:

* які перевірки мають бути пройдені для `auto` і чи всі проблеми
  однаково важкі: неправильна одиниця виміру і неправильний IBAN
  коштують по-різному;
* що робити з полем, якого модель не прочитала, і з полем, яке вона
  прочитала, але про яке сама сказала, що не впевнена;
* що вимірювати окремо: підготовку, вилучення, перевірки.
"""

import time
from dataclasses import dataclass, field

from . import images, llm, rules
from .rules import Issue


@dataclass
class Result:
    """Результат конвеєра — те, з чим працює веб-рівень.

    `decision` — `auto`, `review` або `reject`; `reasons` — чому саме так,
    словами для людини. `document` — поля, які повернула модель і які
    пройшли схему (або `None`, якщо до цього не дійшло). `issues` —
    проблеми, знайдені правилами. `image` — розміри до й після підготовки.
    `elapsed` — час за етапами, наприклад `{"prepare": 0.05,
    "extraction": 6.8, "checks": 0.001}`. `usage` — токени, якщо модель
    викликалась.
    """

    decision: str
    reasons: list[str] = field(default_factory=list)
    document: dict | None = None
    issues: list[Issue] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def decide(document: dict, issues: list[Issue]) -> tuple[str, list[str]]:
    """Вирішити долю документа за вилученими полями й знайденими
    проблемами. Повертає рішення і його причини."""
    reasons = []

    if document.get("document_type") != "invoice":
        return "reject", ["Документ не є рахунком на оплату."]

    if issues:
        for issue in issues:
            reasons.append(issue.message)

        return "review", reasons

    return "auto", ["Усі перевірки пройдено успішно."]


def process(content: bytes) -> Result:
    """Обробити файл: підготувати → вилучити → перевірити → вирішити."""
    start = time.perf_counter()

    try:
        image = images.prepare(content)
    except images.ImageError as error:
        return Result(
            decision="reject",
            reasons=[str(error)],
        )

    prepare_time = time.perf_counter() - start

    start = time.perf_counter()

    try:
        result = llm.extract(image)
    except llm.LLMError as error:
        return Result(
            decision="review",
            reasons=[str(error)],
            image={
                "original": image.original,
                "sent": image.sent,
            },
            elapsed={
                "prepare": round(prepare_time, 3),
                "extraction": round(time.perf_counter() - start, 3),
            },
        )

    extraction_time = time.perf_counter() - start

    document = result["data"]

    start = time.perf_counter()

    issues = rules.check(document)

    checks_time = time.perf_counter() - start

    decision, reasons = decide(document, issues)

    return Result(
        decision=decision,
        reasons=reasons,
        document=document,
        issues=issues,
        image={
            "original": image.original,
            "sent": image.sent,
        },
        model=result["model"],
        elapsed={
            "prepare": round(prepare_time, 3),
            "extraction": round(extraction_time, 3),
            "checks": round(checks_time, 3),
        },
        usage=result["usage"],
    )
