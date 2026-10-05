"""Перевірка зразків рахунків."""

from pathlib import Path

from app import extraction


SAMPLES_DIR = Path("samples")


def main():

    folders = [
        "clean",
        "blur",
        "crop",
        "faded",
        "jpeg",
        "lowres",
        "photo",
    ]

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

            result = extraction.process(content)

            print("Рішення:", result.decision)

            if result.reasons:
                print("Причини:")

                for reason in result.reasons:
                    print("-", reason)


if __name__ == "__main__":
    main()
