"""Інструменти помічника: що модель може попросити виконати і як це
перевіряється.

Єдине місце застосунку, яке знає і про модель, і про сервіс магазину
(`shop/service.py`). Тут описано інструменти для моделі, тут перевіряють
аргументи, які вона повернула, і тут викликають сервіс. Модуль моделі
(`app/llm.py`) не знає, які інструменти існують, а сервіс не знає, що
його викликає модель.

Набір можливостей помічника задано (розділ 2 практичної роботи):

* перелік замовлень клієнта і стан конкретного замовлення;
* пошук товарів у каталозі й картка товару;
* наявність товару на складі;
* вартість і строк доставки;
* заявка на повернення товару.

Скасування замовлення — на ваш розсуд, з обґрунтуванням. Адміністративні
операції сервісу моделі недоступні за жодних умов.

Функції нижче — заготовки. Рішення з розділу 2 практичної роботи:

* скільки інструментів і які саме: одна функція сервісу — один
  інструмент, чи інакше; як їх назвати й описати, щоб модель обирала
  правильно — і коли не обирала жодного;
* які параметри має кожен інструмент і як їх обмежити схемою: типи,
  переліки допустимих значень, формати, межі;
* звідки береться, хто клієнт: з аргументів, які заповнила модель, чи з
  контексту сеансу, який модель не бачить і не може змінити;
* що перевірити після схеми: чи існує замовлення, чиє воно, чи
  дозволена операція;
* що повертати моделі з результату сервісу: запис цілком чи лише
  потрібні поля — і чому;
* як повідомляти моделі про відмову й помилку так, щоб вона могла
  пояснити їх клієнтові, не вигадуючи.
"""

import json

from dataclasses import dataclass

from shop import service


@dataclass
class Context:
    """Те, що відомо про сеанс незалежно від моделі.

    `customer_id` — клієнт, який увійшов. У справжньому застосунку його
    бере веб-рівень із сесії після автентифікації; тут його обирають на
    сторінці. Модель цього поля не бачить і змінити не може.
    """

    customer_id: str


@dataclass
class ToolResult:
    """Підсумок одного виклику інструмента.

    `status` — що сталося, наприклад: `ok` — виконано; `rejected` —
    виклик не пройшов ваших перевірок, сервіс не викликався; `error` —
    сервіс відмовив або не відповів. `content` — те, що піде моделі як
    результат виклику. `reason` — пояснення для журналу й сторінки, якщо
    виклик відхилено або він завершився помилкою. `arguments` —
    аргументи після перевірки, якщо до неї дійшло. Склад полів — ваше
    рішення; `app/assistant.py` і сторінка працюють з тим, що тут буде.
    """

    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None


def specs() -> list[dict]:
    """Повернути описи інструментів у форматі, який приймає API моделі.

    Кожен опис — назва, пояснення для моделі, коли інструмент потрібен, і
    JSON Schema параметрів. Модель бачить лише це: ані коду функцій, ані
    сервісу.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "list_orders",
                "description": (
                    "Повертає список замовлень поточного клієнта. "
                    "Використовуй для питання про замовлення клієнта."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": (
                    "Повертає основну інформацію про конкретне замовлення. "
                    "Використовуй, коли клієнт питає про стан свого замовлення."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Номер замовлення.",
                        }
                    },
                    "required": ["order_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": (
                    "Шукає товари в каталозі за назвою або описом."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Що потрібно знайти.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Категорія товару, якщо відома.",
                        },
                        "max_price": {
                            "type": "number",
                            "description": "Максимальна ціна, якщо її вказав клієнт.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Максимальна кількість результатів.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_product",
                "description": (
                    "Повертає інформацію про конкретний товар за його SKU."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sku": {
                            "type": "string",
                            "description": "Артикул товару.",
                        }
                    },
                    "required": ["sku"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock",
                "description": (
                    "Повертає кількість товару на складі та інформацію "
                    "про очікувану поставку."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sku": {
                            "type": "string",
                            "description": "Артикул товару.",
                        }
                    },
                    "required": ["sku"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delivery_quote",
                "description": (
                    "Розраховує вартість і строк доставки товарів "
                    "у вказане місто."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Місто доставки в Україні.",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["branch", "courier", "pickup"],
                            "description": "Спосіб доставки.",
                        },
                        "items": {
                            "type": "array",
                            "description": "Товари для доставки.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sku": {
                                        "type": "string"
                                    },
                                    "quantity": {
                                        "type": "integer",
                                        "minimum": 1
                                    },
                                },
                                "required": ["sku", "quantity"],
                            },
                        },
                    },
                    "required": ["city", "method", "items"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_return",
                "description": (
                    "Створює заявку на повернення товару з отриманого "
                    "замовлення клієнта."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Номер замовлення.",
                        },
                        "sku": {
                            "type": "string",
                            "description": "Артикул товару.",
                        },
                        "reason": {
                            "type": "string",
                            "enum": [
                                "not_suitable",
                                "defect",
                                "wrong_item",
                                "damaged",
                            ],
                            "description": "Причина повернення.",
                        },
                        "quantity": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Кількість товарів.",
                        },
                        "comment": {
                            "type": "string",
                            "description": "Додатковий коментар клієнта.",
                        },
                    },
                    "required": [
                        "order_id",
                        "sku",
                        "reason",
                        "quantity",
                    ],
                },
            },
        },
    ]

def make_result(status, content=None, reason=None, arguments=None):
    return ToolResult(
        status=status,
        content=content,
        reason=reason,
        arguments=arguments,
    )

def call(name: str, raw_arguments: str, ctx: Context) -> ToolResult:
    """Виконати виклик інструмента, який запропонувала модель.

    `name` і `raw_arguments` — як їх повернула модель: назва й рядок з
    аргументами у JSON. Ні тому, ні іншому не можна вірити наперед:
    назви може не бути серед ваших інструментів, рядок може не бути
    JSON, аргументи можуть не пройти схему або стосуватися чужого
    замовлення.

    Жоден виняток звідси не має вийти назовні: будь-який результат —
    `ToolResult`.
    """
    allowed_tools = [
        "list_orders",
        "get_order",
        "search_products",
        "get_product",
        "get_stock",
        "delivery_quote",
        "create_return",
    ]

    if name not in allowed_tools:
        return make_result(
            "rejected",
            reason="Такого інструмента немає.",
        )

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return make_result(
            "rejected",
            reason="Аргументи мають бути правильним JSON.",
        )

    if not isinstance(arguments, dict):
        return make_result(
            "rejected",
            reason="Аргументи повинні бути JSON-об'єктом.",
        )

    customer_id = ctx.customer_id

    try:

        if name == "list_orders":

            orders = service.list_orders(customer_id)

            result = []

            for order in orders:
                result.append({
                    "order_id": order["order_id"],
                    "date": order["created_at"],
                    "status": order["status"],
                    "status_label": order["status_label"],
                    "total": order["total"],
                    "items_count": order["items_count"],
                })

            return make_result(
                "ok",
                result,
                arguments=arguments,
            )

        if name == "get_order":

            order_id = arguments.get("order_id")

            if not order_id:
                return make_result(
                    "rejected",
                    reason="Не вказано номер замовлення.",
                    arguments=arguments,
                )

            order = service.get_order(order_id)

            if order["customer_id"] != customer_id:
                return make_result(
                    "rejected",
                    reason="Це замовлення належить іншому клієнту.",
                    arguments=arguments,
                )

            result = {
                "order_id": order["order_id"],
                "date": order["date"],
                "status": order["status"],
                "total": order["total"],
                "items": order["items"],
                "delivery": order["delivery"],
            }

            return make_result(
                "ok",
                result,
                arguments=arguments,
            )

        if name == "search_products":

            query = arguments.get("query")

            if not query:
                return make_result(
                    "rejected",
                    reason="Не вказано, що потрібно знайти.",
                    arguments=arguments,
                )

            products = service.search_products(
                query=query,
                category=arguments.get("category"),
                max_price=arguments.get("max_price"),
                limit=arguments.get("limit", 5),
            )

            result = []

            for product in products:
                result.append({
                    "sku": product["sku"],
                    "name": product["name"],
                    "category": product["category"],
                    "price": product["price"],
                    "description": product["description"],
                })

            return make_result(
                "ok",
                result,
                arguments=arguments,
            )

        if name == "get_product":

            sku = arguments.get("sku")

            if not sku:
                return make_result(
                    "rejected",
                    reason="Не вказано SKU товару.",
                    arguments=arguments,
                )

            product = service.get_product(sku)

            result = {
                "sku": product["sku"],
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "weight": product["weight"],
                "warranty": product["warranty"],
                "returnable": product["returnable"],
                "description": product["description"],
            }

            return make_result(
                "ok",
                result,
                arguments=arguments,
            )

        if name == "get_stock":

            sku = arguments.get("sku")

            if not sku:
                return make_result(
                    "rejected",
                    reason="Не вказано SKU товару.",
                    arguments=arguments,
                )

            stock = service.get_stock(sku)

            result = {
                "sku": sku,
                "available": stock["available"],
                "incoming": stock["incoming"],
            }

            return make_result(
                "ok",
                result,
                arguments=arguments,
            )

        if name == "delivery_quote":

            city = arguments.get("city")
            method = arguments.get("method")
            items = arguments.get("items")

            if not city or not method or not items:
                return make_result(
                    "rejected",
                    reason="Для доставки потрібні місто, спосіб доставки та товари.",
                    arguments=arguments,
                )

            quote = service.delivery_quote(
                city=city,
                method=method,
                items=items,
            )

            return make_result(
                "ok",
                quote,
                arguments=arguments,
            )

        if name == "create_return":

            order_id = arguments.get("order_id")
            sku = arguments.get("sku")
            reason = arguments.get("reason")
            quantity = arguments.get("quantity", 1)
            comment = arguments.get("comment")

            if not order_id or not sku or not reason:
                return make_result(
                    "rejected",
                    reason="Для повернення потрібні замовлення, товар і причина.",
                    arguments=arguments,
                )

            order = service.get_order(order_id)

            if order["customer_id"] != customer_id:
                return make_result(
                    "rejected",
                    reason="Це замовлення належить іншому клієнту.",
                    arguments=arguments,
                )

            return_data = service.create_return(
                order_id=order_id,
                sku=sku,
                reason=reason,
                quantity=quantity,
                comment=comment,
            )

            return make_result(
                "ok",
                return_data,
                arguments=arguments,
            )

    except service.ShopError as error:
        return make_result(
            "error",
            reason=str(error),
            arguments=arguments,
        )

    except Exception as error:
        return make_result(
            "error",
            reason="Внутрішня помилка сервісу магазину.",
            arguments=arguments,
        )

    return make_result(
        "rejected",
        reason="Виклик не був виконаний.",
        arguments=arguments,
    )
