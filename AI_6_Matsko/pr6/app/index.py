"""Векторний індекс: зберігання векторів фрагментів і пошук найближчих.

Цей модуль — той самий, що в ПР5, і тут він лишений заготовкою навмисно.
Замініть файл своєю реалізацією з `pr5/app/` цілком: у ПР6 вона не
змінюється, а лише використовується. Якщо в ПР5 ви перейменували
функції або поля — узгодьте з ними виклики в нових модулях.

Індекс — це вектори всіх фрагментів, самі фрагменти з метаданими й назва
моделі, якою вектори отримано. Він будується окремою командою
(`ingest.py`) і зберігається на диску, а застосунок при старті лише
читає його: перераховувати ембедінги колекції на кожен запуск — марна
трата часу, а на кожен запит — тим паче.

Веб-рівень (`app/main.py`) звертається сюди з вектором запиту й отримує
список влучень із оцінкою схожості. Звідки береться вектор — не справа
індексу; що показувати клієнтові — не його справа теж.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення з
розділу 2 практичної роботи:

* яку міру схожості взяти й що зробити з векторами перед порівнянням;
* як шукати: перебір усіх векторів (для сотень фрагментів — цілком
  доречно) чи бібліотека наближеного пошуку;
* у якому вигляді зберігати індекс на диску і що обовʼязково покласти
  поруч із векторами, щоб потім не переплутати, чиї вони;
* де застосовувати фільтри за метаданими — до ранжування чи після — і що
  станеться з top-k у кожному з варіантів;
* що робити з кількома фрагментами одного документа у видачі;
* чи потрібен поріг схожості й звідки взяти його значення.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from app.documents import Chunk
from app.embeddings import embed_query

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index"

# Скільки результатів повертати за замовчуванням і чи відсікати слабкі.
# Порожній поріг означає «без порога».
DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
_threshold = os.getenv("SIMILARITY_THRESHOLD", "").strip()
SIMILARITY_THRESHOLD: float | None = float(_threshold) if _threshold else None


@dataclass
class Hit:
    """Одне влучення пошуку: фрагмент і оцінка його схожості із запитом."""

    chunk: Chunk
    score: float


@dataclass
class SearchIndex:
    """Індекс у памʼяті.

    `vectors` — масив (кількість фрагментів × розмірність); рядок i
    відповідає `chunks[i]`. `model_name` — модель, якою отримано вектори:
    без неї індекс, збудований однією моделлю, мовчки шукатиме векторами
    іншої.
    """

    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


def build(chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> SearchIndex:
    """Зібрати індекс із фрагментів і їхніх векторів.

    Тут перевіряється, що векторів стільки ж, скільки фрагментів, і
    вирішується, що зробити з векторами перед пошуком (наприклад,
    нормалізувати, якщо схожість — косинусна).
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            "Кількість фрагментів не відповідає кількості векторів."
        )

    vectors = np.asarray(vectors, dtype=np.float32)

    
    if len(vectors) > 0:
        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1.0
        vectors = vectors / norms

    return SearchIndex(
        chunks=chunks,
        vectors=vectors,
        model_name=model_name
    )
    


def save(index: SearchIndex, path: Path = INDEX_DIR) -> None:
    """Зберегти індекс на диск.

    Формат — ваше рішення: `numpy` для векторів і JSON для фрагментів з
    метаданими — цілком достатньо. Назву моделі зберігайте обовʼязково.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    np.save(path / "vectors.npy", index.vectors)


    chunks_data = []

    for chunk in index.chunks:
        chunks_data.append({
            "text": chunk.text,
            "source": chunk.source,
            "metadata": chunk.metadata
        })

    data = {
        "model_name": index.model_name,
        "chunks": chunks_data,
        "extra": index.extra
    }
    with open(
        path / "index.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Індекс збережено у: {path}")
    


def load(path: Path = INDEX_DIR) -> SearchIndex:
    """Прочитати індекс із диска.

    Якщо індексу немає — підняти зрозумілу помилку: веб-рівень має
    сказати користувачеві «індекс не збудовано», а не впасти з
    `FileNotFoundError` десь усередині.
    """
    path = Path(path)

    vectors_path = path / "vectors.npy"
    json_path = path / "index.json"

    if not vectors_path.exists():
        raise FileNotFoundError(
            "Індекс не збудовано: файл vectors.npy не знайдено. "
            "Спочатку виконайте python ingest.py"
        )

    if not json_path.exists():
        raise FileNotFoundError(
            "Індекс не збудовано: файл index.json не знайдено. "
            "Спочатку виконайте python ingest.py"
        )

    vectors = np.load(vectors_path)
    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    chunks = []

    for item in data["chunks"]:
        chunks.append(
            Chunk(
                text=item["text"],
                source=item["source"],
                metadata=item["metadata"]
            )
        )

    if len(chunks) != len(vectors):
        raise ValueError(
            "Кількість фрагментів у index.json "
            "не відповідає кількості векторів."
        )
    return SearchIndex(
        chunks=chunks,
        vectors=vectors,
        model_name=data["model_name"],
        extra=data.get("extra", {})
    )
    


def search(
    index: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[Hit]:
    """Знайти фрагменти, найближчі до вектора запиту.

    `filters` — умови на метадані фрагмента, наприклад
    `{"category": "інструкція", "audience": "клієнти"}`. Як трактувати
    поле з датою і що робити з невідомим полем — ваше рішення.

    Повертає щонайбільше `top_k` влучень, упорядкованих за спаданням
    оцінки. Оцінка — та сама міра схожості, що й для ранжування, і саме
    вона показується користувачеві.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k повинен бути більшим за 0."
        )

    filters = filters or {}

    allowed_filters = {
        "title",
        "category",
        "product",
        "audience",
        "updated",
        "status",
        "section"
    }
    for key in filters:
        if key not in allowed_filters:
            raise ValueError(
                f"Невідомий фільтр: {key}. "
                f"Дозволені: {', '.join(sorted(allowed_filters))}"
            )

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    )

    norm = np.linalg.norm(query_vector)

    if norm != 0:
        query_vector = query_vector / norm

    if len(index.vectors) == 0:
        return []
    scores = index.vectors @ query_vector

    results = []

    for number, score in enumerate(scores):

        chunk = index.chunks[number]

        matches = True

        for key, expected_value in filters.items():
            actual_value = chunk.metadata.get(key, "")

            actual_value = str(
                actual_value
            ).strip().lower()

            expected_value = str(
                expected_value
            ).strip().lower()

            if actual_value != expected_value:
                matches = False
                break

        if not matches:
            continue

        score = float(score)

        if threshold is not None and score < threshold:
            continue
        results.append(
            Hit(
                chunk=chunk,
                score=score
            )
        )

    results.sort(
        key=lambda hit: hit.score,
        reverse=True
    )

    return results[:top_k]
