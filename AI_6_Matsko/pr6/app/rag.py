"""Конвеєр відповіді за базою знань: від питання до відповіді з джерелами.

Це єдине місце, яке знає всі кроки послідовно: знайти фрагменти
(`retrieval.retrieve`), зібрати з них контекст (`retrieval.build_context`),
спитати модель (`llm.ask`), перевірити те, що вона повернула, і скласти
результат для веб-рівня. Модулі-учасники одне про одного не знають:
пошук не знає про модель, модель — про індекс, веб-рівень — ні про що з
цього.

Функція `answer` — заготовка. Порядок кроків очевидний; рішення — ні:

* що робити, коли пошук не повернув нічого придатного: не викликати
  модель і відповісти наперед заданим текстом, чи викликати все одно —
  і чому;
* чи довіряти полю «відповідь знайдено», яке заповнила сама модель, і
  чим його перевірити: наприклад, чи існують названі нею номери джерел
  серед показаних, чи не порожній їх перелік;
* що показувати як джерела: лише ті фрагменти, на які модель послалася,
  чи всі, які їй показали;
* що потрапляє в `retrieved` — те, що бачила модель, — і чи віддавати
  це користувачеві чи лише вам для налагодження;
* що рахувати окремо: час пошуку і час генерації — це різні витрати, і
  на сторінці вони показуються окремо.
"""

import time
from dataclasses import dataclass, field

from .index import Hit, SearchIndex
from .keyword import KeywordIndex
from .retrieval import Source, retrieve, build_context
from . import llm


@dataclass
class Answer:
    """Результат конвеєра — те, з чим працює веб-рівень.

    `text` — відповідь для клієнта. `found` — чи відповідь спирається на
    базу знань (після вашої перевірки, а не зі слів моделі). `sources` —
    джерела, які показуються під відповіддю. `retrieved` — фрагменти,
    які бачила модель, для налагодження: за ними видно, чия це помилка —
    пошуку чи генерації. `elapsed` — час за етапами, наприклад
    `{"retrieval": 0.04, "generation": 1.9}`. `usage` — токени запиту й
    відповіді, якщо модель викликалась.
    """

    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(
    question: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> Answer:
    """Відповісти на питання за базою знань.

    Заготовка: знайти → відібрати → зібрати контекст → спитати модель →
    перевірити → скласти `Answer`. Що саме відбувається на кожній стрілці,
    описано вгорі модуля й у розділі 2 практичної роботи.

    `filters` — те, що передала сторінка (наприклад, товар). Запобіжники,
    які не залежать від користувача, — не тут і не в аргументах, а в
    `retrieval.retrieve`.
    """
    question = question.strip()

    if not question:
        raise ValueError("Питання не може бути порожнім.")

    if index is None:
        raise ValueError("Індекс не завантажено.")

    retrieval_start = time.perf_counter()

    hits = retrieve(
        question,
        index,
        keyword_index,
        filters=filters,
    )

    if not hits:
        retrieval_time = time.perf_counter() - retrieval_start

        return Answer(
            text=(
                "У базі знань немає достатньої інформації "
                "для відповіді на це питання."
            ),
            found=False,
            sources=[],
            retrieved=[],
            model=None,
            elapsed={
                "retrieval": round(retrieval_time, 3),
                "generation": 0,
            },
            usage=None,
        )

    context, sources = build_context(hits)

    retrieval_time = time.perf_counter() - retrieval_start

    if not sources or not context.strip():
        return Answer(
            text=(
                "У базі знань немає достатньої інформації "
                "для відповіді на це питання."
            ),
            found=False,
            sources=[],
            retrieved=hits,
            model=None,
            elapsed={
                "retrieval": round(retrieval_time, 3),
                "generation": 0,
            },
            usage=None,
        )

    generation_start = time.perf_counter()

    result = llm.ask(
        question,
        context,
    )

    generation_time = time.perf_counter() - generation_start

    source_numbers = []

    for source in sources:
        source_numbers.append(source.ref)

    model_sources = result.get("sources", [])

    for source_number in model_sources:
        if source_number not in source_numbers:
            raise llm.LLMError(
                "Модель послалася на джерело, "
                "якого не було в контексті."
            )

    selected_sources = []

    for source in sources:
        if source.ref in model_sources:
            selected_sources.append(source)

    found = bool(result.get("found", False))

    if found and not selected_sources:
        found = False

    if not found:
        selected_sources = []

    return Answer(
        text=result["answer"],
        found=found,
        sources=selected_sources,
        retrieved=hits,
        model=result.get("model"),
        elapsed={
            "retrieval": round(retrieval_time, 3),
            "generation": round(generation_time, 3),
        },
        usage=result.get("usage"),
    )
    
