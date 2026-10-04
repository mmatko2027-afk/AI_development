"""Підготовка зображення до надсилання моделі.

Файл, який надіслав користувач, — недовірені байти: це може бути не
зображення, зображення завбільшки 40 мегапікселів, фото, повернуте на
бік (орієнтація записана в EXIF, а не в пікселях), або знімок 300 × 200,
на якому цифр не розібрати. Модуль вирішує, що з цього можна віддати
моделі і в якому вигляді. Про модель він не знає: повертає підготовлені
байти й відомості про них, а як вкласти їх у запит — справа `app/llm.py`.

Функція `prepare` — заготовка. Рішення з розділу 2 практичної роботи:

* що вважати неприйнятним файлом і яку помилку повернути: не
  зображення, порожній файл, завеликий файл, завеликий у пікселях;
* чи враховувати орієнтацію з EXIF (фото з телефона);
* до якого розміру зменшувати і чи зменшувати взагалі — ціна зображення
  в токенах росте з роздільністю, а дрібний шрифт зі зменшенням
  зливається;
* у якому форматі надсилати: PNG без втрат чи JPEG, який менший;
* чи відсіювати завідомо непридатні знімки ще до моделі (надто малі,
  надто розмиті) — і що тоді сказати користувачеві.
"""

import os
from dataclasses import dataclass
from io import BytesIO

from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()

UPLOAD_MAX_BYTES = int(float(os.getenv("UPLOAD_MAX_MB", "10")) * 1024 * 1024)
IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))

MAX_PIXELS = 40_000_000
MIN_WIDTH = 300
MIN_HEIGHT = 300
class ImageError(Exception):
    """Файл не можна віддати моделі. Повідомлення — для користувача."""


@dataclass
class PreparedImage:
    """Зображення, готове до надсилання.

    `data` і `mime` — те, що піде моделі. `original` і `sent` — розміри
    до й після підготовки, наприклад `{"width": 1240, "height": 1754,
    "bytes": 179639}`: сторінка показує їх поруч із токенами, щоб було
    видно, скільки коштує роздільність.
    """

    data: bytes
    mime: str
    original: dict
    sent: dict


def prepare(content: bytes) -> PreparedImage:
    """Перевірити байти, що надійшли, і підготувати зображення для моделі.

    Непридатний файл — `ImageError` з поясненням, що не так.
    """
    if not content:
        raise ImageError("Файл порожній.")
    if len(content) > UPLOAD_MAX_BYTES:
        max_mb = UPLOAD_MAX_BYTES / 1024 / 1024
        raise ImageError(
            f"Файл занадто великий. Максимальний розмір — {max_mb:.1f} МБ."
        )
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except UnidentifiedImageError:
        raise ImageError("Файл не є коректним зображенням.")
    except Exception as exc:
        raise ImageError(f"Не вдалося відкрити зображення: {exc}")
    original_width, original_height = image.size
    original = {
        "width": original_width,
        "height": original_height,
        "bytes": len(content),
    }

    if original_width * original_height > MAX_PIXELS:
        raise ImageError(
            "Зображення має занадто велику кількість пікселів."
        )

    if original_width < MIN_WIDTH or original_height < MIN_HEIGHT:
        raise ImageError(
            "Зображення занадто маленьке. "
            "Будь ласка, завантажте чіткіше фото рахунку."
        )

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    image.thumbnail(
        (IMAGE_MAX_SIDE, IMAGE_MAX_SIDE),
        Image.Resampling.LANCZOS,
    )

    output = BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=90,
        optimize=True,
    )

    prepared_data = output.getvalue()
    sent = {
        "width": image.width,
        "height": image.height,
        "bytes": len(prepared_data),
    }

    return PreparedImage(
        data=prepared_data,
        mime="image/jpeg",
        original=original,
        sent=sent,
    )
