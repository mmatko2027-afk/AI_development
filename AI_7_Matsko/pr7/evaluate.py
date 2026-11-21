"""Перевірка зразків рахунків."""

from pathlib import Path

from app import extraction
from app import extraction, images, llm

SAMPLES_DIR = Path("samples")


def main():

    folders = ["clean", "degraded", "own"]

    for folder_name in folders:
        folder = SAMPLES_DIR / folder_name

        if not folder.exists():
            continue

        print()
        print("=" * 50)
        print(folder_name)
        print("=" * 50)

        for file in sorted(folder.iterdir()):

            if file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
                continue

            print()
            print("Файл:", file.name)

            content = file.read_bytes()

            try:
                result = extraction.process(content)

            except (images.ImageError, llm.LLMError) as exc:
                print(f"Помилка обробки {file_path}: {exc}")
                continue

            print("Рішення:", result.decision)

            if result.reasons:
                print("Причини:")

                for reason in result.reasons:
                    print("-", reason)


if __name__ == "__main__":
    main()
