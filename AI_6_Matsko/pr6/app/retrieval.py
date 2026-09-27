"""Відбір фрагментів для моделі та збирання контексту.

Між пошуком із ПР5 і мовною моделлю стоїть шар, якого в ПР5 не було: він
вирішує, *що саме* модель побачить. Пошук повертає влучення з оцінками —
стільки, скільки попросили; модель потребує тексту — обмеженого за
розміром, упорядкованого, з позначками джерел, на які вона зможе
послатися. Цей шар — тут.

Сам пошук сюди імпортується, а не переписується: `index.search` і
`keyword.search` — ваші модулі з ПР5. Веб-рівень сюди не звертається;
цей модуль викликає `app/rag.py`.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення з
розділу 2 практичної роботи:

* яким пошуком добирати фрагменти — семантичним, за словами, обома
  (і як тоді злити дві видачі в одну);
* які фільтри накладає код **завжди**, незалежно від того, що просив
  користувач: аудиторія, статус документа;
* скільки фрагментів віддавати моделі, чи згортати кілька фрагментів
  одного документа й чи потрібен поріг «нижче — не віддавати»;
* у якому порядку ставити фрагменти в контекст і як їх позначати, щоб
  модель могла послатися на конкретний;
* як умістити контекст у бюджет і що відкидати, коли він не вміщується.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .documents import Chunk
from .index import Hit, SearchIndex, search
from .keyword import KeywordIndex
from .embeddings import embed_query

load_dotenv()

# Скільки фрагментів іде в контекст моделі й скільки токенів на них є.
# Значення — відправна точка; їхнє обґрунтування — результат вашої
# перевірки на наборі запитань, а не здогадка.
CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "4"))
CONTEXT_BUDGET = int(os.getenv("RAG_CONTEXT_BUDGET", "1500"))


@dataclass
class Source:
    """Джерело, показане моделі й користувачеві.

    `ref` — номер, під яким фрагмент стоїть у контексті й на який модель
    посилається у відповіді. `chunk` — сам фрагмент із метаданими;
    `score` — оцінка пошуку, за якою його відібрано.
    """

    ref: int
    chunk: Chunk
    score: float

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def retrieve(
    query: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти, які варто показати моделі.

    Заготовка. Тут викликається ваш пошук із ПР5 і ухвалюється, що з його
    видачі піде далі: які фільтри код додає сам (внутрішні документи не
    мають дійти до моделі, коли питає клієнт, — незалежно від `filters`),
    чи відкидати влучення нижче за поріг, чи згортати кілька фрагментів
    одного документа, скільки лишати.

    Повертає влучення, упорядковані за спаданням оцінки. Порожній список —
    коректний результат: він означає «у базі знань про це немає», і що
    робити далі, вирішує `app/rag.py`.
    """
    if index is None:
        raise ValueError("Індекс не завантажено.")

    query = query.strip()

    if not query:
        raise ValueError("Пошуковий запит не може бути порожнім.")

    safe_filters = dict(filters or {})

    safe_filters["audience"] = "клієнти"
    safe_filters["status"] = "чинний"

    search_top_k = max(CONTEXT_CHUNKS * 2, 8)

    hits = search(
        index,
        query_vector=embed_query(query),
        top_k=search_top_k,
        filters=safe_filters,
    )

    hits = hits[:CONTEXT_CHUNKS]

    return hits

def build_context(hits: list[Hit], budget: int = CONTEXT_BUDGET) -> tuple[str, list[Source]]:
    """Зібрати з влучень текст контексту для моделі та перелік джерел.

    Заготовка. Кожен фрагмент у контексті має позначку — номер і те, що
    допоможе моделі й користувачеві зрозуміти, звідки він: назва
    документа, розділ, дата редакції. Текст фрагмента передається як є,
    без переказу. Порядок і формат позначок — ваше рішення; від них
    залежить, чи зможе модель послатися на джерело так, щоб код це потім
    перевірив.

    Контекст має вміститися в `budget` токенів. Оцінку кількості токенів ви
    вже писали в ПР4. Повертає текст контексту й список `Source` у тому
    самому порядку, що й у тексті.
    """
    if budget <= 0:
        raise ValueError("Бюджет контексту повинен бути більшим за 0.")

    context_parts = []
    sources = []

    used_tokens = 0
    ref = 1

    for hit in hits:
        chunk = hit.chunk

        title = chunk.metadata.get("title", chunk.source)
        section = chunk.metadata.get("section", "без розділу")
        updated = chunk.metadata.get("updated", "дата не вказана")

        part = (
            f"[{ref}]\n"
            f"Документ: {title}\n"
            f"Розділ: {section}\n"
            f"Дата редакції: {updated}\n"
            f"Текст:\n"
            f"{chunk.text}\n"
        )

        part_tokens = estimate_tokens(part)

        if used_tokens + part_tokens > budget:
            continue

        source = Source(
            ref=ref,
            chunk=chunk,
            score=hit.score,
        )

        context_parts.append(part)
        sources.append(source)

        used_tokens += part_tokens
        ref += 1

        if len(sources) >= CONTEXT_CHUNKS:
            break

    context = "\n".join(context_parts)

    return context, sources