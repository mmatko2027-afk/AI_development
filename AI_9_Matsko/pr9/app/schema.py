"""Контракт кроку розбору: що модель має витягти зі звернення клієнта
і як це перевіряється.

Окремий модуль, як у ПР4, ПР6 і ПР7: схема — домовленість між моделлю
та рештою workflow. Наступні кроки працюють лише з тим, що пройшло
перевірку тут, і не бачать сирого тексту відповіді моделі.

Що має дати розбір (розділ 2 практичної роботи): вид звернення —
повернення товару, скасування замовлення, питання про стан замовлення
або щось інше; номер замовлення, якщо клієнт його назвав; товар, як
клієнт його назвав; причину повернення.

Рішення — ваші:

* які поля й що з них обовʼязкове; як модель каже «номер не названо» і
  «не розумію, про що звернення» — і чи це одне й те саме;
* причина повернення: категорія сервісу (`not_suitable`, `defect`, …),
  слова клієнта чи обидва? Хто вирішує категорію — модель чи ваш код за
  словами клієнта? Згадайте q11 із ПР8;
* що робити зі зверненням, у якому дві дії, — схема це дозволяє чи ні;
* чи просити від моделі щось понад потрібне наступним крокам — і чи буде
  кому цим скористатися;
* що в цій схемі статичне, а що будується з даних: наприклад, вибір
  товару з позицій конкретного замовлення — це окремий крок із власною
  схемою, `enum` якої складається з результату інструмента.

Описати схему можна JSON Schema вручну або pydantic-моделлю.
"""
import json


def request_schema() -> dict:
    """Повернути JSON Schema результату розбору звернення.

    Та сама схема передається моделі (`response_format`) і перевіряє
    те, що повернулося.
    """
    return {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [
                    "return",
                    "cancel_order",
                    "order_status",
                    "multiple",
                    "other",
                ],
                "description": "Основний вид звернення клієнта; multiple — якщо в одному зверненні кілька окремих дій.",
            },
            "order_number": {
                "type": ["string", "null"],
                "description": "Номер замовлення або null, якщо його не названо.",
            },
            "product_name": {
                "type": ["string", "null"],
                "description": "Назва товару зі слів клієнта або null.",
            },
            "quantity": {
                "type": ["integer", "null"],
                "description": "Кількість одиниць товару для повернення або null, якщо не вказано.",
            },
            "return_category": {
                "type": ["string", "null"],
                "enum": [
                    "not_suitable",
                    "defect",
                    "wrong_item",
                    "damaged_delivery",
                    "other",
                    None,
                ],
                "description": "Категорія причини повернення або null.",
            },
            "return_reason": {
                "type": ["string", "null"],
                "description": "Причина повернення словами клієнта або null.",
            },
        },
        "required": [
            "intent",
            "order_number",
            "product_name",
            "quantity",
            "return_category",
            "return_reason",
        ],
        "additionalProperties": False,
    }


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі на кроці розбору й повернути
    дані, якими можуть користуватися наступні кроки.

    Якщо текст не є JSON або не проходить схему — підняти помилку з
    поясненням, що саме не так.

    Схема перевіряє форму. Чи існує замовлення і чиє воно — перевірка
    змісту, і її місце на наступних кроках workflow.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise ValueError("Відповідь моделі не є коректним JSON.")

    if not isinstance(data, dict):
        raise ValueError("Результат розбору має бути JSON-об'єктом.")

    schema = request_schema()
    properties = schema["properties"]

    for field in schema["required"]:
        if field not in data:
            raise ValueError(f"Відсутнє обов'язкове поле: {field}")

    for field in data:
        if field not in properties:
            raise ValueError(f"Невідоме поле: {field}")

    for field, rules in properties.items():
        value = data[field]
        raw_types = rules["type"]
        allowed_types = [raw_types] if isinstance(raw_types, str) else raw_types

        if value is None:
            if "null" not in allowed_types:
                raise ValueError(f"Поле {field} не може бути null.")
        elif "string" in allowed_types:
            if not isinstance(value, str):
                raise ValueError(f"Поле {field} має бути рядком.")
            cleaned = value.strip()
            if not cleaned:
                if "null" not in allowed_types:
                    raise ValueError(f"Поле {field} не може бути порожнім.")
                data[field] = None
                value = None
            else:
                if field == "order_number":
                    cleaned = cleaned.lstrip("#№ ")
                data[field] = cleaned
                value = cleaned
        elif "integer" in allowed_types:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Поле {field} має бути цілим числом.")
            if value < 1:
                raise ValueError(f"Поле {field} має бути не меншим за 1.")

        if value is not None and "enum" in rules:
            if value not in rules["enum"]:
                raise ValueError(
                    f"Недопустиме значення поля {field}: {value}"
                )

    return data


def item_selection_schema(skus: list[str]) -> dict:
    """Динамічна JSON Schema для вибору товару з позицій конкретного замовлення."""
    return {
        "type": "object",
        "properties": {
            "sku": {
                "type": ["string", "null"],
                "enum": [*skus, None],
                "description": "SKU товару зі списку позицій замовлення або null, якщо жоден не підходить.",
            }
        },
        "required": ["sku"],
        "additionalProperties": False,
    }


def validate_item_selection(raw: str, skus: list[str]) -> dict:
    """Перевірити вибір товару за динамічною схемою."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise ValueError("Відповідь моделі при виборі товару не є коректним JSON.")

    if not isinstance(data, dict) or set(data.keys()) != {"sku"}:
        raise ValueError("Результат вибору товару має містити лише поле sku.")

    sku = data["sku"]
    if sku is not None and (not isinstance(sku, str) or sku not in skus):
        raise ValueError(f"Обраний SKU {sku!r} відсутній у замовленні.")

    return {"sku": sku}
