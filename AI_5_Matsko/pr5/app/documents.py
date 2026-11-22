"""Колекція документів: читання, метадані, поділ на фрагменти.

Це єдине місце, яке знає, як влаштовані файли в `docs/`: де в них
метадані, як розмічено текст, за якими межами його ділити. Решта
застосунку працює з готовими фрагментами (`Chunk`) і не читає файлів.

Розбір блоку метаданих реалізовано: це формат файлів, а не предмет
роботи. Поділ на фрагменти — заготовка: розмір, межі, перекриття і те,
що саме потрапляє в кожен фрагмент, — рішення з розділу 2 практичної
роботи.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"

# Параметри поділу — відправна точка, а не рекомендація. У чому їх
# рахувати (символи, слова, токени моделі) і як застосовувати, коли
# ділите за заголовками, — ваше рішення.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


@dataclass
class Chunk:
    """Фрагмент документа — одиниця індексування й пошуку.

    `text` — те, що перетворюється на вектор і показується в результатах.
    `source` — імʼя файлу, з якого взято фрагмент.
    `metadata` — поля з блоку метаданих файлу (title, category, product,
    audience, updated, status) плюс те, що ви вирішите додати самі:
    заголовок розділу, порядковий номер фрагмента, позицію в документі.
    Фільтри пошуку працюють саме з цим словником.
    """

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Відокремити блок метаданих від тексту документа.

    Блок — рядки `ключ: значення` між двома рядками `---` на початку
    файлу. Повертає словник метаданих і решту тексту. Порожні значення
    (`product:` без нічого) стають порожнім рядком. Якщо блоку немає —
    порожній словник і текст як є.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    metadata: dict = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return metadata, body
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return {}, raw



def get_overlap(text: str, max_length: int) -> str:
    """Get the last complete words within the overlap limit."""

    if max_length <= 0:
        return ""

    words = text.split()
    overlap_words = []
    current_length = 0

    for word in reversed(words):
        added_length = len(word)

        if overlap_words:
            added_length += 1

        if current_length + added_length > max_length:
            break

        overlap_words.insert(0, word)
        current_length += added_length

    return " ".join(overlap_words)



def split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    """Split a long paragraph without breaking words."""

    words = paragraph.split()
    parts = []
    current_words = []

    overlap_size = max(
        0,
        min(CHUNK_OVERLAP, limit - 1)
    )

    for word in words:
        candidate = " ".join(current_words + [word])

        if current_words and len(candidate) > limit:
            parts.append(" ".join(current_words))

            overlap = get_overlap(
                " ".join(current_words),
                overlap_size
            )

            current_words = overlap.split()

        current_words.append(word)

    if current_words:
        parts.append(" ".join(current_words))

    return parts



def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict, str]]:
    """Прочитати всі документи колекції.

    Повертає список трійок (імʼя файлу, метадані, текст) для кожного
    `*.md` у папці, крім `README.md` — він описує колекцію, а не є її
    частиною. Порядок — за іменем файлу, щоб індекс будувався однаково
    від запуску до запуску.
    """
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append((path.name, metadata, body))
    return documents


def split(text: str, source: str, metadata: dict) -> list[Chunk]:
    """Поділити текст документа на фрагменти.

    Заготовка. Тут ухвалюються рішення, від яких залежить якість пошуку:

    * за якими межами ділити — за заголовками розділів, за абзацами, за
      фіксованою кількістю символів чи токенів, чи комбінуючи;
    * якого розміру фрагмент і чи потрібне перекриття між сусідніми;
    * що класти в текст фрагмента, крім самого уривка: назву документа,
      заголовок розділу — те, без чого уривок «Утримуйте кнопку 10
      секунд» не каже, про який пристрій ідеться;
    * що класти в метадані фрагмента понад метадані документа.

    Модель ембедінгів має обмеження на довжину входу (у картці моделі);
    усе, що довше, вона мовчки обріже.
    """
    text = text.strip()

    if not text:
        return []

    chunks = []
    sections = ("\n" + text).split("\n## ")

    for section_number, section in enumerate(sections):
        section = section.strip()

        if not section:
            continue

        if section_number > 0:
            lines = section.splitlines()
            section_title = lines[0].strip()
            section_body = "\n".join(lines[1:]).strip()
        else:
            section_title = ""
            section_body = section

        prefix = f"{section_title}\n" if section_title else ""

        limit = CHUNK_SIZE - len(prefix)

        if limit <= 0:
            raise ValueError("Section title is too long for CHUNK_SIZE")

        if not section_body:
            continue

        overlap_size = max(
            0,
            min(CHUNK_OVERLAP, limit - 1)
        )

        paragraphs = [
            paragraph.strip()
            for paragraph in section_body.split("\n\n")
            if paragraph.strip()
        ]

        chunk_metadata = {
            **metadata,
            "section": section_title
        }

        current_text = ""

        for paragraph in paragraphs:

            if len(paragraph) > limit:
                if current_text:
                    chunks.append(
                        Chunk(
                            text=f"{prefix}{current_text}",
                            source=source,
                            metadata=chunk_metadata
                        )
                    )
                    current_text = ""

                parts = split_long_paragraph(paragraph, limit)

                for part in parts:
                    chunks.append(
                        Chunk(
                            text=f"{prefix}{part}",
                            source=source,
                            metadata=chunk_metadata
                        )
                    )

                continue

            if not current_text:
                current_text = paragraph

            elif len(current_text) + len(paragraph) + 2 <= limit:
                current_text += "\n\n" + paragraph

            else:
                chunks.append(
                    Chunk(
                        text=f"{prefix}{current_text}",
                        source=source,
                        metadata=chunk_metadata
                    )
                )

                overlap = get_overlap(
                    current_text,
                    overlap_size
                )

                if overlap:
                    current_text = overlap + "\n\n" + paragraph
                else:
                    current_text = paragraph

        if current_text:
            chunks.append(
                Chunk(
                    text=f"{prefix}{current_text}",
                    source=source,
                    metadata=chunk_metadata
                )
            )

    return chunks




def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Прочитати колекцію й повернути всі її фрагменти."""
    chunks: list[Chunk] = []
    for source, metadata, body in load_documents(docs_dir):
        chunks.extend(split(body, source, metadata))
    return chunks
