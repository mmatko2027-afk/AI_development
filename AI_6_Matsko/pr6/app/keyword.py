"""Пошук за ключовими словами — точка порівняння для семантичного.

Цей модуль — той самий, що в ПР5, і тут він лишений заготовкою навмисно.
Замініть файл своєю реалізацією з `pr5/app/` цілком: у ПР6 вона не
змінюється, а лише використовується. Якщо в ПР5 ви перейменували
функції або поля — узгодьте з ними виклики в нових модулях.

Той самий набір фрагментів, той самий формат влучень (`Hit`), ті самі
фільтри — інший спосіб ранжувати. Без цього модуля не буде з чим
порівнювати семантичний пошук, а без порівняння — не буде відповіді на
питання, чи він узагалі потрібен для цієї колекції.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення:

* як розбивати текст на слова: що робити з регістром, розділовими
  знаками, апострофом у «звʼязок», числами й артикулами на кшталт
  `OR-X2-BLK`;
* чи зводити слова до основи — і чим, якщо так; без цього «повернути»
  й «повернення» для пошуку різні слова;
* чим ранжувати: кількістю збігів, TF-IDF, BM25 (пакет `rank_bm25`
  вже в залежностях);
* у яких одиницях оцінка й чи можна її порівнювати з оцінкою
  семантичного пошуку.
"""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit

@dataclass
class KeywordIndex:
    """Індекс для пошуку за словами. Що саме тут зберігати — вирішуєте ви."""

    chunks: list[Chunk]
    extra: dict = field(default_factory=dict)
    bm25: object = None
    


def tokenize(text: str) -> list[str]:
    """Розбити текст на слова для індексування й для запиту.

    Одна й та сама функція для обох: запит і фрагмент мають розбиватися
    однаково, інакше збігів не буде.
    """

    text = text.lower()

    pattern = (
        r"[a-zа-яіїєґ0-9]+"
        r"(?:[-_][a-zа-яіїєґ0-9]+)*"
    )

    return re.findall(pattern, text)
    


def build(chunks: list[Chunk]) -> KeywordIndex:
    """Зібрати індекс за словами з тих самих фрагментів, що й векторний."""
    documents_tokens = []

    for chunk in chunks:
        tokens = tokenize(chunk.text)
        documents_tokens.append(tokens)

    bm25 = BM25Okapi(documents_tokens)

    return KeywordIndex(
        chunks=chunks,
        bm25=bm25
    )


def search(
    index: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти за словами запиту.

    Формат результату — той самий `Hit`, що й у векторного пошуку, з тими
    самими фільтрами за метаданими, щоб сторінка показувала обидва
    способи поруч.
    """
    query = query.strip()

    if not query:
        raise ValueError(
            "Пошуковий запит не може бути порожнім."
        )

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
        "section",
    }

    for key in filters:
        if key not in allowed_filters:
            raise ValueError(
                f"Невідомий фільтр: {key}"
            )

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = index.bm25.get_scores(query_tokens)

    results = []

    for number, score in enumerate(scores):

        chunk = index.chunks[number]

        matches = True

        for key, expected_value in filters.items():

            actual_value = chunk.metadata.get(key, "")

            if str(actual_value).strip().lower() != str(
                expected_value
            ).strip().lower():
                matches = False
                break

        if not matches:
            continue

        results.append(
            Hit(
                chunk=chunk,
                score=float(score)
            )
        )

    results.sort(
        key=lambda hit: hit.score,
        reverse=True
    )

    return results[:top_k]