"""Модуль ембедінгів: єдине місце застосунку, яке знає, яка модель
перетворює текст на вектор і як її викликати.

Цей модуль — той самий, що в ПР5, і тут він лишений заготовкою навмисно.
Замініть файл своєю реалізацією з `pr5/app/` цілком: у ПР6 вона не
змінюється, а лише використовується. Якщо в ПР5 ви перейменували
функції або поля — узгодьте з ними виклики в нових модулях.

Індекс (`app/index.py`) і веб-рівень (`app/main.py`) отримують звідси
готові вектори й не знають, локальна це модель чи API. Замінивши модель
тут, ви не змінюєте решту коду — але маєте перебудувати індекс: вектори
різних моделей непорівнянні.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення з
розділу 2 практичної роботи:

* яку модель узяти й чому саме її: мови, розмірність, довжина входу,
  розмір, швидкість на CPU;
* чи потрібні моделі префікси для запиту й для тексту (у деяких моделей
  запит і документ кодуються по-різному — дивіться картку моделі);
* чи нормалізувати вектори і де — тут чи в індексі;
* як кодувати кілька сотень фрагментів: по одному чи пакетами.

Назва моделі читається з `.env`; значення за замовчуванням — відправна
точка, а не рекомендація.
"""

import os

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Назва моделі ембедінгів. Для локальної моделі — ідентифікатор на
# Hugging Face Hub; для моделі через API — назва в провайдера.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

_model = None
def get_model():
    """Повернути готову до роботи модель.

    Локальна модель вантажиться з диска секунди й займає сотні мегабайт
    памʼяті — створюйте її один раз, а не на кожен запит. Для моделі через
    API тут створюється клієнт (ключ — із `.env`, як у ПР3).
    """
    global _model

    if _model is None:
        print(
            f"Завантаження моделі: {MODEL_NAME}"
        )
        _model = SentenceTransformer(
            MODEL_NAME
        )
        print("Модель завантажена.")
    return _model

def embed_passages(texts: list[str]) -> np.ndarray:
    """Перетворити тексти фрагментів на вектори.

    Повертає масив розміру (кількість текстів × розмірність моделі).
    Викликається під час індексування — для всієї колекції одразу.
    """
    if not texts:
        return np.empty(
            (0, 0),
            dtype=np.float32
        )

    model = get_model()

    prepared_texts = [
        f"passage: {text}"
        for text in texts
    ]

    vectors = model.encode(
        prepared_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return vectors.astype(np.float32)
    


def embed_query(text: str) -> np.ndarray:
    """Перетворити запит користувача на вектор тієї самої розмірності.

    Окрема функція навмисно: у моделей із префіксами запит кодується не
    так, як фрагмент, і саме тут це видно.
    """
    text = text.strip()

    if not text:
        raise ValueError(
            "Пошуковий запит не може бути порожнім."
        )

    model = get_model()

    prepared_query = f"query: {text}"

    vector = model.encode(
        [prepared_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return vector[0].astype(
        np.float32
    )
