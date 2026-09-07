"""Модуль inference: єдине місце застосунку, яке знає про модель.

Тут живуть ваги, поріг упевненості й формат «сирого» результату моделі.
Веб-рівень (`app/main.py`) отримує звідси готовий структурований список
знайдених обʼєктів і нічого не знає ані про `ultralytics`, ані про те,
у якому вигляді модель віддає рамки.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши по дорозі
рішення з розділу 2 практичної роботи:

* де саме завантажувати ваги, щоб це сталося **один раз**, а не на кожен запит;
* яким узяти поріг упевненості й чи дозволяти змінювати його ззовні;
* у якому вигляді віддавати результат: які поля, які одиниці координат;
* як виміряти час inference і що саме до нього зараховувати;
* як повестися, коли надійшов не той файл — не зображення або порожній.

Довідка про модель: https://docs.ultralytics.com/
"""

import io
import time
from xml.parsers.expat import model

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25


class DetectionError(Exception):
    """Помилка детекції, зрозуміла веб-рівню.

    Заготовка. Вирішіть, чи достатньо одного типу помилки, чи їх варто
    розрізняти — некоректний файл, збій моделі, — і що з цього має
    побачити користувач.
    """
_model = None

def load_model():
    """Повернути готову до роботи модель.

    Завантаження ваг коштує дорого. Подумайте, як зробити так, щоб воно
    відбулося один раз за час життя застосунку.
    """
    global _model
    if _model is None:
        _model = YOLO(WEIGHTS)
    return _model


def detect(image_bytes: bytes, confidence: float = DEFAULT_CONFIDENCE):
    """Знайти обʼєкти на зображенні.

    Приймає байти завантаженого файлу, повертає структурований результат:
    для кожного знайденого обʼєкта — клас, рамку й упевненість, а також
    їхню кількість і час виконання. Точний склад полів — ваше рішення.
    """
    if not image_bytes:
        raise DetectionError("Файл порожній")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise DetectionError("Завантажений файл не є зображенням")

    if not 0 <= confidence <= 1:
        raise DetectionError(
            "Поріг упевненості має бути від 0 до 1"
        )

    detector = load_model()
    start_time = time.perf_counter()

    results = detector.predict(
        source=image,
        conf=confidence,
        verbose=False,
    )

    elapsed = (time.perf_counter() - start_time) * 1000

    detections = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0])
            confidence_value = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": result.names[class_id],
                "confidence": round(confidence_value, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                },
            })

    return {
        "count": len(detections),
        "inference_time_ms": round(elapsed, 2),
        "detections": detections,
    }
