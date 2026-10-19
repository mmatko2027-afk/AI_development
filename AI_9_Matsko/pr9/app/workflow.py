"""Workflow обробки звернення клієнта: кроки, переходи, обмеження.

Єдине місце, яке знає послідовність. Тут вирішується, який крок іде
наступним, з якими даними і що робити з його результатом. Модель
(`app/llm.py`) виконує окремі кроки й не знає, що буде далі.
Інструменти (`app/tools.py`) виконують виклики й не знають, хто їх
просить — модель чи цей модуль. Сховище (`app/store.py`) зберігає
стан (`app/state.py`) і не знає, що в ньому. Веб-рівень бачить лише
`start`, `resume` і `Run`.

Звернення не обробляється за один виклик. `start` веде його, доки це
можливо без людини: до кінцевого стану або до паузи на підтвердження
оператором. `resume` продовжує після рішення оператора — можливо, за
годину й після перезапуску застосунку, тож усе потрібне для
продовження має бути в збереженому стані, а не в памʼяті процесу.

Функції нижче — заготовки. Рішення з розділу 2 практичної роботи:

* які кроки, у якому порядку, якого типу кожен (модель, інструмент,
  код, людина) і за якими умовами переходити між ними; де модель
  можна прибрати, бо код упорається сам;
* які статуси має звернення і які переходи між ними дозволені;
  кінцеві статуси — які саме, і чи в кожен веде хоч одна гілка;
* що передається між кроками (`Run.data`) і що з цього потрібно після
  паузи;
* які інструменти доступні кожному кроку і як це перевіряє код;
* як дія зі змінами виконується рівно один раз: подвійне підтвердження,
  повторне звернення, перезапуск між виконанням і записом стану;
* що перевірити ще раз після підтвердження, перш ніж виконувати;
* як рахуються обмеження — кроки, час автоматичної частини, токени — і
  що відбувається, коли якесь вичерпано посеред сценарію;
* як перевіряється відповідь клієнтові і що робити, якщо вона не
  пройшла перевірку.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv

from . import llm, schema, store, tools
from .state import Run, Step

load_dotenv()

# Обмеження одного звернення. Значення — відправна точка, а не
# відповідь: звідки взяти свої, вирішуєте ви.
MAX_STEPS = int(os.getenv("WORKFLOW_MAX_STEPS", "12"))
TIMEOUT = float(os.getenv("WORKFLOW_TIMEOUT", "60"))
TOKEN_BUDGET = int(os.getenv("WORKFLOW_TOKEN_BUDGET", "8000"))


def limits() -> dict:
    """Обмеження одного звернення — для сторінки."""
    return {"max_steps": MAX_STEPS, "timeout": TIMEOUT, "token_budget": TOKEN_BUDGET}

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save(run: Run) -> None:
    run.updated_at = _now()
    store.save(run)


def _add_step(
    run: Run,
    name: str,
    kind: str,
    status: str,
    detail: str = "",
    input_data=None,
    output_data=None,
    elapsed=None,
    usage=None,
) -> None:
    """Додає крок до журналу та зберігає стан."""
    run.steps.append(
        Step(
            n=len(run.steps) + 1,
            name=name,
            kind=kind,
            status=status,
            detail=detail,
            input=input_data,
            output=output_data,
            elapsed=elapsed,
            usage=usage,
            at=_now(),
        )
    )
    _save(run)


def _check_limits(run: Run, started: float) -> None:
    """Перевіряє ліміти перед виконанням наступного кроку."""
    if len(run.steps) >= MAX_STEPS:
        raise RuntimeError("Вичерпано ліміт кроків workflow.")

    if time.perf_counter() - started >= TIMEOUT:
        raise RuntimeError("Вичерпано ліміт часу workflow.")

    if run.usage.get("total_tokens", 0) >= TOKEN_BUDGET:
        raise RuntimeError("Вичерпано бюджет токенів workflow.")


def _add_usage(run: Run, usage: dict | None) -> None:
    """Додає використання токенів до загальної статистики."""
    if not usage:
        return

    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        run.usage[key] = run.usage.get(key, 0) + int(usage.get(key, 0))


def _finish(
    run: Run,
    status: str,
    reply: str,
    reason: str | None = None,
) -> Run:
    """Завершує звернення та зберігає результат."""
    run.status = status
    run.reply = reply
    run.stopped = reason
    run.pending = None
    _save(run)
    return run


def _call_tool(
    run: Run,
    name: str,
    arguments: dict,
    started: float,
) -> tools.ToolResult:
    """Викликає інструмент через спільну перевірку tools.call()."""
    _check_limits(run, started)

    before = time.perf_counter()

    result = tools.call(
        name,
        json.dumps(arguments, ensure_ascii=False),
        tools.Context(customer_id=run.customer_id),
    )

    elapsed = time.perf_counter() - before

    _add_step(
        run,
        name,
        "tool",
        result.status,
        detail=result.reason or "Інструмент виконано.",
        input_data=arguments,
        output_data=result.content,
        elapsed=elapsed,
    )

    if result.status == "error":
        raise RuntimeError(
            result.reason or "Сервіс магазину не виконав операцію."
        )

    return result


def _get_order(
    run: Run,
    order_id: str,
    started: float,
) -> dict:
    """Отримує замовлення з перевіркою власника через tools.call()."""
    result = _call_tool(
        run,
        "get_order",
        {"order_id": order_id},
        started,
    )

    if result.status != "ok" or not isinstance(result.content, dict):
        raise RuntimeError(
            result.reason or "Не вдалося перевірити замовлення."
        )

    return result.content


def _parse_request(run: Run, started: float) -> dict:
    """Крок моделі: перетворює текст звернення на структуровані дані."""
    _check_limits(run, started)

    messages = [
        {
            "role": "system",
            "content": (
                "Проаналізуй звернення клієнта магазину та поверни "
                "лише дані за наданою JSON-схемою. "
                "intent: return, cancel_order, order_status або other. "
                "Не вигадуй номер замовлення, назву товару чи причину. "
                "Якщо значення невідоме, використовуй null. "
                "Категорії повернення: not_suitable, defect, "
                "wrong_item, damaged_delivery або other."
            ),
        },
        {"role": "user", "content": run.text},
    ]

    before = time.perf_counter()
    response = llm.complete(messages, schema=schema.request_schema())
    elapsed = time.perf_counter() - before

    usage = response.get("usage")
    _add_usage(run, usage)

    if run.usage.get("total_tokens", 0) > TOKEN_BUDGET:
        raise RuntimeError("Вичерпано бюджет токенів workflow.")

    parsed = schema.validate(response.get("text", ""))

    _add_step(
        run,
        "parse_request",
        "model",
        "ok",
        detail="Звернення розібрано та перевірено за схемою.",
        input_data={"text": run.text},
        output_data=parsed,
        elapsed=elapsed,
        usage=usage,
    )

    return parsed


def _prepare_action(
    run: Run,
    parsed: dict,
    started: float,
) -> Run:
    """Визначає наступний крок на основі результату розбору."""
    intent = parsed["intent"]
    order_id = parsed.get("order_number")

    if intent == "other":
        return _finish(
            run,
            "done",
            "Дякуємо за звернення. Для цього питання потрібна "
            "консультація оператора.",
        )

    if not order_id:
        return _finish(
            run,
            "needs_info",
            "Будь ласка, уточніть номер замовлення.",
        )

    order = _get_order(run, order_id, started)

    run.data["parsed_request"] = parsed
    run.data["order"] = order
    _save(run)

    if intent == "order_status":
        return _finish(
            run,
            "done",
            f"Замовлення {order['order_id']}: "
            f"{order['status_label']}. Сума: {order['total']} грн.",
        )

    if intent == "cancel_order":
        if order["status"] != "preparing":
            return _finish(
                run,
                "needs_info",
                "Замовлення не можна скасувати автоматично, "
                f"оскільки його статус — {order['status_label']}.",
            )

        action = {
            "tool": "cancel_order",
            "arguments": {"order_id": order_id},
        }

    elif intent == "return":
        product_name = parsed.get("product_name")
        category = parsed.get("return_category")

        if not product_name or not category:
            return _finish(
                run,
                "needs_info",
                "Уточніть назву товару та причину повернення.",
            )

        if category == "other":
            return _finish(
                run,
                "needs_info",
                "Причину повернення потрібно додатково уточнити.",
            )

        if order["status"] != "delivered":
            return _finish(
                run,
                "needs_info",
                "Повернення оформлюється лише після отримання "
                f"замовлення. Поточний статус: {order['status_label']}.",
            )

        matched_sku = None

        for item in order.get("items", []):
            product_result = _call_tool(
                run,
                "get_product",
                {"sku": item["sku"]},
                started,
            )

            if product_result.status != "ok":
                continue

            product = product_result.content

            if (
                isinstance(product, dict)
                and product.get("name", "").casefold()
                == product_name.casefold()
            ):
                matched_sku = item["sku"]
                break

        if matched_sku is None:
            return _finish(
                run,
                "needs_info",
                "Не вдалося знайти в замовленні товар із зазначеною "
                "назвою. Уточніть його назву.",
            )

        reason_map = {
            "not_suitable": "not_suitable",
            "defect": "defect",
            "wrong_item": "wrong_item",
            "damaged_delivery": "damaged",
        }

        action = {
            "tool": "create_return",
            "arguments": {
                "order_id": order_id,
                "sku": matched_sku,
                "reason": reason_map[category],
                "quantity": 1,
                "comment": parsed.get("return_reason") or "",
            },
        }

    else:
        return _finish(
            run,
            "needs_info",
            "Не вдалося визначити тип звернення. Уточніть Ваш запит.",
        )

    run.pending = {
        "state": "awaiting_approval",
        "action": action,
        "order_id": order_id,
    }
    run.status = "awaiting_approval"
    run.reply = (
        "Дію підготовлено. Очікується підтвердження оператора."
    )

    _add_step(
        run,
        "wait_for_approval",
        "human",
        "waiting",
        detail=f"Очікується підтвердження: {action['tool']}.",
        input_data=action,
    )

    _save(run)
    return run



def start(text: str, customer_id: str) -> Run:
    """Прийняти нове звернення й вести його, доки це можливо без людини.

    Повертає збережений стан: кінцевий або `awaiting_approval`.
    `customer_id` передає веб-рівень — це клієнт, який увійшов.
    """
    run = Run(
        run_id=str(uuid.uuid4()),
        customer_id=customer_id,
        text=text,
        status="running",
        created_at=_now(),
        updated_at=_now(),
    )
    _save(run)

    started = time.perf_counter()

    try:
        parsed = _parse_request(run, started)
        run.data["parsed_request"] = parsed
        _save(run)

        return _prepare_action(run, parsed, started)

    except Exception as error:
        return _finish(
            run,
            "failed",
            "Не вдалося автоматично обробити звернення. "
            "Потрібна перевірка оператором.",
            str(error),
        )



def resume(run_id: str, approve: bool, comment: str = "") -> Run:
    """Продовжити звернення після рішення оператора.

    Викликається, можливо, через години й після перезапуску застосунку —
    і, можливо, двічі для того самого звернення.
    """
    run = store.load(run_id)

    if run is None:
        raise ValueError("Звернення не знайдено.")

    if run.status != "awaiting_approval" or not run.pending:
        return run

    if run.pending.get("state") != "awaiting_approval":
        return run

    if not approve:
        return _finish(
            run,
            "rejected",
            "Оператор відхилив виконання дії."
            + (f" Коментар: {comment.strip()}" if comment.strip() else ""),
            "Дію відхилено оператором.",
        )

    action = run.pending.get("action", {})
    tool_name = action.get("tool")
    arguments = action.get("arguments", {})

    if tool_name not in {"cancel_order", "create_return"}:
        return _finish(
            run,
            "failed",
            "Операція не підтримується. Потрібна перевірка оператором.",
            "Невідомий інструмент у pending.",
        )

    run.pending["state"] = "executing"
    run.pending["operator_comment"] = comment
    _save(run)

    started = time.perf_counter()

    try:
        order_id = arguments["order_id"]
        order = _get_order(run, order_id, started)

        if tool_name == "cancel_order" and order["status"] != "preparing":
            return _finish(
                run,
                "failed",
                "Замовлення вже не можна скасувати: "
                f"статус — {order['status_label']}.",
                "Повторна перевірка статусу не пройдена.",
            )

        if tool_name == "create_return":
            if order["status"] != "delivered":
                return _finish(
                    run,
                    "failed",
                    "Замовлення більше не відповідає умовам повернення.",
                    "Повторна перевірка статусу не пройдена.",
                )

            if not any(
                item.get("sku") == arguments.get("sku")
                for item in order.get("items", [])
            ):
                return _finish(
                    run,
                    "failed",
                    "Товар не знайдено в замовленні.",
                    "Повторна перевірка товару не пройдена.",
                )

        result = _call_tool(run, tool_name, arguments, started)

        if result.status != "ok":
            return _finish(
                run,
                "failed",
                "Не вдалося виконати операцію. "
                "Потрібна перевірка оператором.",
                result.reason,
            )

        if tool_name == "cancel_order":
            reply = (
                f"Замовлення {order_id} скасовано. "
                f"Статус: {result.content['status_label']}."
            )
        else:
            reply = (
                "Заявку на повернення створено. "
                f"Номер заявки: {result.content['return_id']}. "
                f"{result.content['next_steps']}"
            )

        _add_step(
            run,
            "complete_action",
            "code",
            "ok",
            detail="Дію виконано після підтвердження оператора.",
            output_data=result.content,
        )

        return _finish(run, "done", reply)

    except Exception as error:
        return _finish(
            run,
            "failed",
            "Не вдалося завершити операцію. "
            "Перевірте її стан перед повторною обробкою.",
            str(error),
        )
