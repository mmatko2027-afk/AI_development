"""Програмні правила: перевірка змісту вилучених даних.

Схема (`app/schema.py`) каже, що IBAN — рядок, а сума — число. Чи це
справжній IBAN і чи сходиться сума з позиціями, схема не знає. Це знає
документ: у рахунку багато надлишковості — контрольні розряди в кодах,
підсумки, що мають дорівнювати сумі рядків, ПДВ, що має дорівнювати
п'ятій частині. Модель, яка прочитала 8 замість 3, цю надлишковість
порушує, і код це бачить. Правила тут не знають ні про модель, ні про
веб-рівень: на вході — перевірений за схемою словник, на виході — перелік
знайдених проблем.

Набір правил задано (розділ 2 практичної роботи):

* IBAN: формат і контрольне число за ISO 13616 (залишок mod 97);
* код постачальника й покупця: 8 цифр ЄДРПОУ з контрольним розрядом
  або 10 цифр РНОКПП;
* арифметика: кількість × ціна = сума в кожному рядку; сума рядків =
  разом без ПДВ; ПДВ = 20 % (або нуль для неплатника); разом + ПДВ =
  до сплати — з допуском на округлення, який ви оберете й обґрунтуєте;
* дати: існують, не з майбутнього, строк дії не раніше дати рахунку;
* покупець — ваша компанія (`reference/company.json`);
* постачальник відомий, і IBAN збігається з довідником
  (`reference/suppliers.json`);
* обовʼязкові поля присутні.

Рішення — ваші: які правила дають помилку, а які — попередження; чи
перевіряти правило, якщо потрібного поля немає; як рахувати гроші, щоб
0,1 + 0,2 не дало 0,30000000000000004.
"""

import json
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path


REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MONEY_STEP = Decimal("0.01")


@dataclass
class Issue:
    """Проблема, знайдена правилом."""

    field: str
    rule: str
    message: str
    severity: str = "error"


def load_reference() -> tuple[dict, list[dict]]:
    """Прочитати реквізити своєї компанії й довідник постачальників."""
    company = json.loads(
        (REFERENCE_DIR / "company.json").read_text(encoding="utf-8")
    )
    suppliers = json.loads(
        (REFERENCE_DIR / "suppliers.json").read_text(encoding="utf-8")
    )
    return company, suppliers


def clean_text(value: str) -> str:
    value = value.replace("ʼ", "'")
    value = value.replace("’", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", str(value))


def _money(value: Decimal) -> Decimal:
    """Округлити суму до копійок."""
    return value.quantize(MONEY_STEP, rounding=ROUND_HALF_UP)


def _valid_iban(value: str) -> bool:
    iban = _compact(value).upper()

    if not re.fullmatch(r"UA[0-9]{27}", iban):
        return False

    rearranged = iban[4:] + iban[:4]
    remainder = 0

    for character in rearranged:
        if character.isalpha():
            expanded = str(ord(character) - ord("A") + 10)
        else:
            expanded = character

        for digit in expanded:
            remainder = (remainder * 10 + int(digit)) % 97

    return remainder == 1


def _valid_edrpou(value: str) -> bool:
    code = _compact(value)

    if not re.fullmatch(r"[0-9]{8}", code):
        return False

    weights = [1, 2, 3, 4, 5, 6, 7]

    if code[0] in "345":
        weights = [7, 1, 2, 3, 4, 5, 6]

    total = sum(
        weight * int(digit)
        for weight, digit in zip(weights, code[:7])
    )

    check_digit = total % 11

    if check_digit == 10:
        total = sum(
            (weight + 2) * int(digit)
            for weight, digit in zip(weights, code[:7])
        )
        check_digit = total % 11 % 10

    return int(code[-1]) == check_digit


def _decimal(value: object) -> Decimal:
    normalized = re.sub(r"\s+", "", str(value)).replace(",", ".")
    number = Decimal(normalized)

    if not number.is_finite():
        raise InvalidOperation

    return number


def check(document: dict) -> list[Issue]:
    """Застосувати правила до вилученого документа."""
    issues = []

    company, suppliers = load_reference()

    if document.get("document_type") != "invoice":
        issues.append(
            Issue(
                "document_type",
                "document_type",
                "Документ не є рахунком."
            )
        )
        return issues

    # Перевірка обов'язкових полів верхнього рівня.
    required_fields = [
        "invoice_number",
        "date",
        "supplier",
        "buyer",
        "items",
        "total_without_vat",
        "vat",
        "total_due",
    ]

    for field in required_fields:
        value = document.get(field)

        if value is None or value == "" or value == [] or value == {}:
            issues.append(
                Issue(
                    field,
                    "required",
                    "Обов'язкове поле відсутнє."
                )
            )

    supplier = document.get("supplier") or {}
    buyer = document.get("buyer") or {}

    # Перевірка обов'язкових полів постачальника.
    supplier_required = ["name", "code", "iban"]

    for field in supplier_required:
        value = supplier.get(field)

        if value is None or not str(value).strip():
            issues.append(
                Issue(
                    f"supplier.{field}",
                    "required",
                    "Обов'язкове поле постачальника відсутнє."
                )
            )

    # Перевірка обов'язкових полів покупця.
    buyer_required = ["name", "code"]

    for field in buyer_required:
        value = buyer.get(field)

        if value is None or not str(value).strip():
            issues.append(
                Issue(
                    f"buyer.{field}",
                    "required",
                    "Обов'язкове поле покупця відсутнє."
                )
            )

    # Перевірка дати рахунку.
    invoice_date = None
    date_value = document.get("date")

    if date_value:
        try:
            invoice_date = date.fromisoformat(date_value)

            if invoice_date > date.today():
                issues.append(
                    Issue(
                        "date",
                        "date",
                        "Дата рахунку знаходиться в майбутньому."
                    )
                )

        except (ValueError, TypeError):
            issues.append(
                Issue(
                    "date",
                    "date",
                    "Неправильний формат дати."
                )
            )

    # Перевірка строку дії рахунку.
    valid_until = document.get("valid_until")

    if valid_until and invoice_date is not None:
        try:
            valid_date = date.fromisoformat(valid_until)

            if valid_date < invoice_date:
                issues.append(
                    Issue(
                        "valid_until",
                        "date",
                        "Строк дії не може бути раніше дати рахунку."
                    )
                )

        except (ValueError, TypeError):
            issues.append(
                Issue(
                    "valid_until",
                    "date",
                    "Неправильний формат строку дії."
                )
            )

    # Перевірка покупця.
    buyer_code = buyer.get("code")

    if buyer_code:
        compact_code = _compact(buyer_code)

        if re.fullmatch(r"[0-9]{8}", compact_code):
            if not _valid_edrpou(compact_code):
                issues.append(
                    Issue(
                        "buyer.code",
                        "edrpou_checksum",
                        "Неправильний контрольний розряд ЄДРПОУ покупця."
                    )
                )

        elif not re.fullmatch(r"[0-9]{10}", compact_code):
            issues.append(
                Issue(
                    "buyer.code",
                    "code_format",
                    "Код має бути 8 цифр ЄДРПОУ або 10 цифр РНОКПП."
                )
            )

    if buyer.get("name"):
        if clean_text(buyer["name"]).lower() != clean_text(
            company["name"]
        ).lower():
            issues.append(
                Issue(
                    "buyer.name",
                    "buyer",
                    "Неправильний покупець."
                )
            )

    if buyer_code:
        if _compact(buyer_code) != _compact(company["code"]):
            issues.append(
                Issue(
                    "buyer.code",
                    "buyer",
                    "Неправильний код покупця."
                )
            )

    # Перевірка постачальника.
    supplier_name = supplier.get("name")
    supplier_code = supplier.get("code")
    supplier_iban = supplier.get("iban")

    if supplier_code:
        compact_code = _compact(supplier_code)

        if re.fullmatch(r"[0-9]{8}", compact_code):
            if not _valid_edrpou(compact_code):
                issues.append(
                    Issue(
                        "supplier.code",
                        "edrpou_checksum",
                        "Неправильний контрольний розряд ЄДРПОУ постачальника."
                    )
                )

        elif not re.fullmatch(r"[0-9]{10}", compact_code):
            issues.append(
                Issue(
                    "supplier.code",
                    "code_format",
                    "Код має бути 8 цифр ЄДРПОУ або 10 цифр РНОКПП."
                )
            )

    # Перевірка формату та контрольного числа IBAN.
    if supplier_iban:
        compact_iban = _compact(supplier_iban).upper()

        if not re.fullmatch(r"UA[0-9]{27}", compact_iban):
            issues.append(
                Issue(
                    "supplier.iban",
                    "iban_format",
                    "IBAN має містити UA та 27 цифр."
                )
            )

        elif not _valid_iban(supplier_iban):
            issues.append(
                Issue(
                    "supplier.iban",
                    "iban_checksum",
                    "Контрольна сума IBAN неправильна."
                )
            )

    # Пошук постачальника в довіднику.
    found_supplier = None

    if supplier_name or supplier_code:
        for item in suppliers:
            name_match = (
                supplier_name
                and clean_text(item["name"]).lower()
                == clean_text(supplier_name).lower()
            )

            code_match = (
                supplier_code
                and _compact(item["code"]) == _compact(supplier_code)
            )

            if name_match or code_match:
                found_supplier = item
                break

        if found_supplier is None:
            issues.append(
                Issue(
                    "supplier",
                    "supplier",
                    "Постачальника немає у довіднику."
                )
            )
        else:
            # Назва та код перевіряються окремо.
            if supplier_name:
                if clean_text(supplier_name).lower() != clean_text(
                    found_supplier["name"]
                ).lower():
                    issues.append(
                        Issue(
                            "supplier.name",
                            "supplier_name",
                            "Назва постачальника не відповідає довіднику."
                        )
                    )

            if supplier_code:
                if _compact(supplier_code) != _compact(
                    found_supplier["code"]
                ):
                    issues.append(
                        Issue(
                            "supplier.code",
                            "supplier_code",
                            "Код постачальника не відповідає довіднику."
                        )
                    )

            # Перевірка IBAN за довідником.
            if supplier_iban:
                iban1 = _compact(supplier_iban).upper()
                iban2 = _compact(found_supplier["iban"]).upper()

                if iban1 != iban2:
                    issues.append(
                        Issue(
                            "supplier.iban",
                            "iban",
                            "IBAN не відповідає довіднику."
                        )
                    )

    # Перевірка позицій рахунку.
    items = document.get("items") or []

    total_items = Decimal("0")
    items_valid = True

    for number, item in enumerate(items):
        try:
            quantity = _decimal(item.get("quantity"))
            price = _decimal(item.get("price"))
            amount = _decimal(item.get("amount"))

            calculated = _money(quantity * price)
            amount = _money(amount)

            if calculated != amount:
                issues.append(
                    Issue(
                        f"items.{number}.amount",
                        "arithmetic",
                        "Кількість × ціна не дорівнює сумі."
                    )
                )

            total_items += amount

        except (InvalidOperation, TypeError, ValueError):
            items_valid = False
            issues.append(
                Issue(
                    f"items.{number}",
                    "number",
                    "Неправильні числові значення."
                )
            )

    # Перевірка суми без ПДВ.
    total_without_vat = None
    total_value = document.get("total_without_vat")

    if total_value is not None and total_value != "":
        try:
            total_without_vat = _decimal(total_value)

        except (InvalidOperation, TypeError, ValueError):
            issues.append(
                Issue(
                    "total_without_vat",
                    "number",
                    "Неправильне числове значення суми без ПДВ."
                )
            )

    if (
        items_valid
        and total_without_vat is not None
        and _money(total_items) != _money(total_without_vat)
    ):
        issues.append(
            Issue(
                "total_without_vat",
                "arithmetic",
                "Сума позицій не дорівнює сумі без ПДВ."
            )
        )

    # Перевірка ПДВ.
    vat = None
    vat_value = document.get("vat")

    if vat_value is not None and vat_value != "":
        try:
            vat = _decimal(vat_value)

        except (InvalidOperation, TypeError, ValueError):
            issues.append(
                Issue(
                    "vat",
                    "number",
                    "Неправильне числове значення ПДВ."
                )
            )

    if (
        found_supplier is not None
        and total_without_vat is not None
        and vat is not None
    ):
        if found_supplier["vat_payer"]:
            expected_vat = _money(
                total_without_vat * Decimal("0.20")
            )
        else:
            expected_vat = Decimal("0.00")

        if _money(vat) != expected_vat:
            issues.append(
                Issue(
                    "vat",
                    "vat",
                    "Неправильно розрахований ПДВ."
                )
            )

    # Перевірка загальної суми до сплати.
    total_due = None
    total_due_value = document.get("total_due")

    if total_due_value is not None and total_due_value != "":
        try:
            total_due = _decimal(total_due_value)

        except (InvalidOperation, TypeError, ValueError):
            issues.append(
                Issue(
                    "total_due",
                    "number",
                    "Неправильне числове значення суми до сплати."
                )
            )

    if (
        total_due is not None
        and total_without_vat is not None
        and vat is not None
    ):
        expected_total = _money(total_without_vat + vat)

        if _money(total_due) != expected_total:
            issues.append(
                Issue(
                    "total_due",
                    "arithmetic",
                    "Сума без ПДВ + ПДВ не дорівнює сумі до сплати."
                )
            )

    return issues
