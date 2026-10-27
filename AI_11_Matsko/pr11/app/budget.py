"""Денний ліміт витрат: скільки застосунок може витратити за добу.

Обмеження частоти з ПР10 рахує запити за хвилину і не завадить
витратити місячний бюджет рівномірним потоком. Тут рахуються гроші за
день на весь застосунок. Ліній дві: мʼяка — попередження на частці
`WARN_AT` (оператор бачить, журнал отримує `WARNING`), жорстка —
відмова після вичерпання.

Модуль знає лише суму, ліміт і день. Шлюз (`app/gateway.py`) викликає
`check()` перед платним звертанням до провайдера і `record(cost)`
після нього; веб-рівень показує `status()` на сторінці.

Функції нижче — заготовки. Рішення з розділу 2 практичної роботи:

* де перевіряти: перед кожним платним викликом у шлюзі — обовʼязково
  (влучання в кеш безкоштовне й перевірки не потребує); чи ще й до
  запуску фонової роботи, щоб оператор дізнався одразу, а не подією
  через секунду;
* що з двома одночасними викликами на залишок бюджету: обидва пройдуть
  перевірку. Резервувати очікувану вартість наперед і уточнювати після
  — чи змиритися з невеликим перевищенням;
* стан на диску: лічильник у памʼяті обнулюється з перезапуском — і
  перезапуск «поповнює» бюджет. Файл у `state/` з датою і сумою,
  атомарний запис, як у ПР9;
* межа доби: за яким часом — локальним чи UTC; коли саме відновлюється
  ліміт і як це сказати операторові (`resets_at`);
* чернетка, яка вже генерується, коли бюджет вичерпався посеред неї:
  дати завершити чи перервати;
* вартість `None` (не визначено): рахувати як нуль, як оцінку, чи
  відмовляти, бо не знаємо, скільки витратили.
"""

import os
import datetime
import json
import logging
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DAILY_USD = float(os.getenv("BUDGET_DAILY_USD", "1.00"))
WARN_AT = float(os.getenv("BUDGET_WARN_AT", "0.8"))
BUDGET_FILE = os.getenv("BUDGET_FILE", "state/budget.json")
_lock = threading.RLock()
_log = logging.getLogger("app.budget")


def _now() -> datetime.datetime:
    return datetime.datetime.now().astimezone()


def _read(day: str) -> float:
    if not BUDGET_FILE:
        return 0.0
    path = Path(BUDGET_FILE)
    if not path.exists():
        return 0.0
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("day") != day:
        return 0.0
    return max(0.0, float(data.get("spent", 0.0)))


def _write(day: str, spent: float) -> None:
    if not BUDGET_FILE:
        return
    path = Path(BUDGET_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"day": day, "spent": spent}), encoding="utf-8")
    temporary.replace(path)


def _resets_at(now: datetime.datetime) -> str:
    tomorrow = now.date() + datetime.timedelta(days=1)
    return datetime.datetime.combine(tomorrow, datetime.time.min, now.tzinfo).isoformat()


class BudgetExceeded(Exception):
    """Денний ліміт вичерпано. `spent`, `limit` — у доларах; `resets_at` —
    коли ліміт відновиться, у форматі ISO."""

    def __init__(self, spent: float, limit: float, resets_at: str) -> None:
        super().__init__(f"денний ліміт витрат {limit:.2f} $ вичерпано (витрачено {spent:.4f} $); "
                         f"відновиться {resets_at}")
        self.spent = spent
        self.limit = limit
        self.resets_at = resets_at


def settings() -> dict:
    """Чинні налаштування — для сторінки."""
    return {"daily_usd": DAILY_USD, "warn_at": WARN_AT, "file": BUDGET_FILE}


def check(expected_cost: float | None = None) -> None:
    """Переконатися, що платний виклик ще дозволений; інакше підняти
    `BudgetExceeded`. `expected_cost` — очікувана вартість, якщо ви
    вирішите резервувати її наперед."""
    now = _now()
    with _lock:
        spent = _read(now.date().isoformat())
        expected = max(0.0, expected_cost or 0.0)
        if DAILY_USD >= 0 and spent + expected > DAILY_USD:
            raise BudgetExceeded(spent, DAILY_USD, _resets_at(now))


def record(cost: float | None) -> None:
    """Зарахувати вартість виконаного звертання до витрат дня."""
    if cost is None:
        return
    now = _now()
    with _lock:
        day = now.date().isoformat()
        previous = _read(day)
        spent = previous + max(0.0, float(cost))
        _write(now.date().isoformat(), round(spent, 8))
    threshold = DAILY_USD * WARN_AT
    if DAILY_USD > 0 and previous < threshold <= spent:
        _log.warning("budget.warning", extra={
            "day": day,
            "spent": round(spent, 8),
            "limit": DAILY_USD,
            "warn_at": WARN_AT,
            "resets_at": _resets_at(now),
        })


def status() -> dict:
    """Стан для `GET /api/usage` і сторінки: наприклад,
    `{"day": "2026-10-23", "spent": 0.1234, "limit": 1.0, "remaining": 0.8766,
    "warning": False, "exceeded": False, "resets_at": "2026-10-24T00:00:00+03:00"}`."""
    now = _now()
    with _lock:
        spent = _read(now.date().isoformat())
    warning_at = DAILY_USD * WARN_AT
    return {
        "day": now.date().isoformat(),
        "spent": round(spent, 8),
        "limit": DAILY_USD,
        "remaining": max(0.0, round(DAILY_USD - spent, 8)),
        "warning": spent >= warning_at if DAILY_USD > 0 else True,
        "exceeded": spent >= DAILY_USD,
        "resets_at": _resets_at(now),
    }
