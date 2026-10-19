"""Сховище стану звернень: зберегти, прочитати, перелічити.

Стан живе поза памʼяттю процесу, бо звернення чекає на оператора
довше, ніж живе один процес застосунку. Модуль не знає ані про модель,
ані про магазин, ані про кроки workflow: він зберігає `Run` і
повертає його таким, яким зберіг.

Каталог — `RUNS_DIR` з `.env` (за замовчуванням `runs/` у папці
роботи). До репозиторію він не потрапляє: там тексти звернень клієнтів.

Функції нижче — заготовки. Рішення з розділу 2 практичної роботи:

* у чому зберігати: JSON-файл на звернення чи SQLite (`sqlite3` зі
  стандартної бібліотеки) — і що це змінює для переліку за статусом;
* як записувати, щоб збій посередині запису не лишив зіпсованого файлу;
* як перетворити `Run` на JSON і назад, разом зі списком `Step`;
* що робити, якщо два запити одночасно змінюють те саме звернення —
  наприклад, оператор двічі натиснув «підтвердити». Чи це справа
  сховища, чи workflow?
"""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .state import Run, Step

load_dotenv()

RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))

def _get_path(run_id: str) -> Path:
    """Повернути шлях до файлу звернення."""
    if (
        not run_id
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError("Некоректний ідентифікатор звернення.")

    return RUNS_DIR / f"{run_id}.json"

def save(run: Run) -> None:
    """Зберегти стан звернення, замінивши попередній."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = _get_path(run.run_id)

    data = asdict(run)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=RUNS_DIR,
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(data, temp_file, ensure_ascii=False, indent=2)

        os.replace(temp_path, path)

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def load(run_id: str) -> Run | None:
    """Прочитати стан звернення; `None`, якщо такого немає."""
    path = _get_path(run_id)

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    data["steps"] = [
        Step(**step_data)
        for step_data in data.get("steps", [])
    ]

    return Run(**data)


def list_runs(status: str | None = None) -> list[Run]:
    """Усі звернення, за потреби — лише з певним статусом; нові першими."""
    if not RUNS_DIR.exists():
        return []

    runs = []

    for path in RUNS_DIR.glob("*.json"):
        run = load(path.stem)

        if run is not None:
            if status is None or run.status == status:
                runs.append(run)

    runs.sort(
        key=lambda run: run.updated_at or run.created_at or "",
        reverse=True,
    )

    return runs
