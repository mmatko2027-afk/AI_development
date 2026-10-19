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
                    "other"
                ],
                "description": "Основний вид звернення клієнта."
            },
            "order_number": {
                "type": ["string", "null"],
                "description": "Номер замовлення або null, якщо його не названо."
            },
            "product_name": {
                "type": ["string", "null"],
                "description": "Назва товару зі слів клієнта або null."
            },
            "return_category": {
                "type": ["string", "null"],
                "enum": [
                    "not_suitable",
                    "defect",
                    "wrong_item",
                    "damaged_delivery",
                    "other",
                    None
                ],
                "description": "Категорія причини повернення або null."
            },
            "return_reason": {
                "type": ["string", "null"],
                "description": "Причина повернення словами клієнта або null."
            }
        },
        "required": [
            "intent",
            "order_number",
            "product_name",
            "return_category",
            "return_reason"
        ],
        "additionalProperties": False
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
        allowed_types = rules["type"]

        if value is None:
            if "null" not in allowed_types:
                raise ValueError(f"Поле {field} не може бути null.")
        elif "string" in allowed_types and not isinstance(value, str):
            raise ValueError(f"Поле {field} має бути рядком.")

        if value is not None and "enum" in rules:
            if value not in rules["enum"]:
                raise ValueError(
                    f"Недопустиме значення поля {field}: {value}"
                )

    return data
