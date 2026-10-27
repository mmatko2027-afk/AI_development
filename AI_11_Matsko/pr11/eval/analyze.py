"""Аналіз журналу: повільні, невдалі й повторні виклики, витрати, кеш.

Запуск із папки pr11:

    python eval/analyze.py                       # logs/app.jsonl, усі розділи
    python eval/analyze.py logs/app.jsonl --since 2026-10-23 --section slow

Читання файлу, відбір за датою і допоміжні функції готові. Розділи
звіту — заготовки: поля, за якими вони працюють, — ваші, бо назви
подій і полів ви обрали, коли писали записи в шлюзі й фонових задачах.
`summary` працює на будь-якому журналі й показує, що в ньому взагалі є.

Орієнтир для кожного розділу — питання, на яке він відповідає; воно в
докстрінгу. Відповідь — числа й кілька рядків журналу, а не переказ.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path, since: str | None = None) -> list[dict]:
    """Прочитати JSON lines. Рядки, які не розбираються, рахуються й
    пропускаються; `since` — дата або час у форматі ISO, від якого брати
    записи (порівнюється з полем `ts` як рядок)."""
    records: list[dict] = []
    broken = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                broken += 1
                continue
            if since and str(record.get("ts", "")) < since:
                continue
            records.append(record)
    if broken:
        print(f"(пропущено рядків, що не розбираються: {broken})", file=sys.stderr)
    return records


def percentile(values: list[float], p: float) -> float | None:
    """p-й процентиль (0 ≤ p ≤ 1) за найближчим рангом; `None` для
    порожнього списку."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def group(records: list[dict], key: str) -> dict:
    """Згрупувати записи за значенням поля."""
    groups: dict = defaultdict(list)
    for record in records:
        groups[record.get(key)].append(record)
    return groups


def summary(records: list[dict]) -> None:
    """Що взагалі є в журналі: скільки записів, за який час, яких подій і
    рівнів скільки. Працює на будь-якому журналі."""
    if not records:
        print("записів немає")
        return
    stamps = [r.get("ts", "") for r in records]
    print(f"записів: {len(records)}; від {min(stamps)} до {max(stamps)}")
    print("рівні:", dict(Counter(r.get("level") for r in records)))
    print("події:")
    for event, count in Counter(r.get("event") for r in records).most_common():
        print(f"  {count:>6}  {event}")


def slow(records: list[dict]) -> None:
    """Які виклики найповільніші й що таке «повільно» в цих числах?

    Розподіл тривалості звертань до провайдера (медіана, p95, максимум),
    а не середнє; десятка найдовших із ідентифікаторами, провайдером,
    моделлю, спробою; де пройшов час — у черзі, до першого фрагмента чи
    в генерації; розріз за провайдером і годиною.
    """
    attempts = [row for row in records if row.get("event") == "llm.attempt.done"]
    durations = [float(row["duration"]) for row in attempts
                 if isinstance(row.get("duration"), (int, float))]
    if not durations:
        print("записів тривалості звертань немає")
        return
    print(f"спроб: {len(durations)}; медіана {percentile(durations, 0.5):.3f} с; "
          f"p95 {percentile(durations, 0.95):.3f} с; максимум {max(durations):.3f} с")
    print("за провайдером:")
    for provider, rows in sorted(group(attempts, "provider").items(), key=lambda item: str(item[0])):
        values = [float(row["duration"]) for row in rows
                  if isinstance(row.get("duration"), (int, float))]
        if values:
            print(f"  {provider}: {len(values)}; медіана {percentile(values, 0.5):.3f} с; "
                  f"p95 {percentile(values, 0.95):.3f} с")
    print("найповільніші:")
    for row in sorted(attempts, key=lambda item: item.get("duration", 0), reverse=True)[:10]:
        print(
            f"  request={row.get('request_id')} draft={row.get('draft_id')} "
            f"provider={row.get('provider')} model={row.get('model')} attempt={row.get('attempt')} "
            f"queue={row.get('queue_wait')}s first={row.get('first_token')}s "
            f"duration={row.get('duration')}s"
        )


def failed(records: list[dict]) -> None:
    """Що й чому не вдалося?

    Невдалі спроби за видом збою; остаточні збої чернеток; частка
    чернеток із невдалою першою спробою, які все ж завершилися успіхом
    (чи окупилися повтори); вартість спроб, які нічого не дали.
    """
    attempts = [row for row in records if row.get("event") == "llm.attempt.done"]
    failed_attempts = [row for row in attempts if row.get("status") == "failed"]
    kinds = Counter(row.get("failure_kind", "unknown") for row in failed_attempts)
    print(f"невдалих спроб: {len(failed_attempts)}; за видом: {dict(kinds)}")
    final_failures = [row for row in records if row.get("event") == "draft.failed"]
    print(f"чернеток завершилися невдачею: {len(final_failures)}")
    by_draft: dict[str, list[dict]] = defaultdict(list)
    for row in attempts:
        if row.get("draft_id"):
            by_draft[str(row["draft_id"])].append(row)
    recovered = 0
    with_failed_first = 0
    for draft_id, rows in by_draft.items():
        rows.sort(key=lambda row: row.get("attempt", 0))
        if rows and rows[0].get("status") == "failed":
            with_failed_first += 1
            if any(row.get("status") == "success" for row in rows[1:]):
                recovered += 1
    print(f"чернеток із невдалою першою спробою, урятовано повтором: "
          f"{recovered}/{with_failed_first}")
    wasted = sum(float(row["cost"]) for row in failed_attempts
                 if isinstance(row.get("cost"), (int, float)))
    print(f"вартість невдалих спроб: {wasted:.6f} $")


def repeated(records: list[dict]) -> None:
    """Що ми робили двічі?

    Той самий ключ кешу кілька разів за короткий час (кандидати на кеш
    і питання «чому»); чернетки з кількома спробами (повтори шлюзу);
    клієнти з серіями замовлень за хвилину.
    """
    cache_events = [row for row in records if row.get("event") in {
        "cache.hit", "cache.miss", "cache.bypassed"
    } and row.get("cache_key")]
    key_counts = Counter(row["cache_key"] for row in cache_events)
    repeated_keys = [(key, count) for key, count in key_counts.most_common() if count > 1]
    print(f"ключів кешу, що зустрічалися повторно: {len(repeated_keys)}")
    for key, count in repeated_keys[:10]:
        print(f"  {key[:12]}…: {count} звертань")
    attempts_by_draft: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        if row.get("event") == "llm.attempt.done" and row.get("draft_id"):
            attempts_by_draft[str(row["draft_id"])].append(row)
    retried = [(draft, rows) for draft, rows in attempts_by_draft.items() if len(rows) > 1]
    print(f"чернеток із кількома спробами: {len(retried)}")
    for draft, rows in retried[:10]:
        print(f"  {draft}: {len(rows)} спроб")
    client_hits = Counter(row.get("client") for row in records
                          if row.get("event") == "draft.created" and row.get("client"))
    print(f"клієнтів із кількома замовленнями: "
          f"{sum(1 for count in client_hits.values() if count > 1)}")


def spend(records: list[dict]) -> None:
    """Скільки це коштувало?

    Вартість і токени за день, за модель, за клієнта; середня вартість
    чернетки; частка вартості, яка припала на спроби зі збоєм і на
    запасний варіант; скільки записів без вартості або з оцінкою.
    """
    attempts = [row for row in records if row.get("event") == "llm.attempt.done"]
    known_costs = [row for row in attempts if isinstance(row.get("cost"), (int, float))]
    total = sum(row["cost"] for row in known_costs)
    print(f"вартість: {total:.6f} $; спроб із відомою вартістю: {len(known_costs)}/{len(attempts)}")
    print(f"середня вартість відомої спроби: "
          f"{total / len(known_costs):.6f} $" if known_costs else "середня вартість: немає даних")
    for label, key in (("модель", "model"), ("клієнт", "client")):
        print(f"за {label}:")
        totals: dict[str, float] = defaultdict(float)
        for row in known_costs:
            totals[str(row.get(key) or "невідомо")] += row["cost"]
        for name, value in sorted(totals.items(), key=lambda item: item[1], reverse=True):
            print(f"  {name}: {value:.6f} $")
    daily: dict[str, float] = defaultdict(float)
    for row in known_costs:
        day = str(row.get("ts", ""))[:10] or "невідомо"
        daily[day] += row["cost"]
    print("за днем:")
    for day, value in sorted(daily.items()):
        print(f"  {day}: {value:.6f} $")
    failed_cost = sum(row["cost"] for row in known_costs if row.get("status") == "failed")
    fallback_cost = sum(row["cost"] for row in known_costs if row.get("fallback"))
    print(f"невдалі спроби: {failed_cost:.6f} $; запасний варіант: {fallback_cost:.6f} $")
    estimated = sum(bool(row.get("cost_estimated")) for row in known_costs)
    missing = len(attempts) - len(known_costs)
    print(f"оцінених вартостей: {estimated}; без вартості: {missing}")
    usage_rows = [row.get("usage") or {} for row in attempts if isinstance(row.get("usage"), dict)]
    input_tokens = sum(row.get("prompt_tokens", 0) or 0 for row in usage_rows)
    output_tokens = sum(row.get("completion_tokens", 0) or 0 for row in usage_rows)
    print(f"токени: вхід {input_tokens}; вихід {output_tokens}")


def cache_report(records: list[dict]) -> None:
    """Що дав кеш?

    Влучання, промахи, обходи (`fresh`) і частка влучань; скільки
    звертань до провайдера й грошей заощаджено; які звернення
    повторювалися найчастіше.
    """
    hits = [row for row in records if row.get("event") == "cache.hit"]
    misses = [row for row in records if row.get("event") == "cache.miss"]
    bypassed = [row for row in records if row.get("event") == "cache.bypassed"]
    looked_up = len(hits) + len(misses)
    hit_rate = len(hits) / looked_up if looked_up else 0
    provider_attempts = [row for row in records if row.get("event") == "llm.attempt.done"]
    saved_cost = sum(row.get("cost", 0) or 0 for row in hits
                     if isinstance(row.get("cost", 0), (int, float)))
    print(f"влучання: {len(hits)}; промахи: {len(misses)}; обходи fresh: {len(bypassed)}; "
          f"частка влучань: {hit_rate:.1%}")
    print(f"звертань до провайдера: {len(provider_attempts)}; "
          f"вартість уже кешованих відповідей: {saved_cost:.6f} $")
    keys = Counter(row.get("cache_key") for row in records
                   if row.get("event") in {"cache.hit", "cache.miss", "cache.bypassed"}
                   and row.get("cache_key"))
    print("найчастіші звернення:")
    for key, count in keys.most_common(10):
        if count > 1:
            print(f"  {str(key)[:12]}…: {count}")
    if not keys:
        print("  немає даних про ключі кешу")


SECTIONS = {
    "summary": summary,
    "slow": slow,
    "failed": failed,
    "repeated": repeated,
    "spend": spend,
    "cache": cache_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", default="logs/app.jsonl")
    parser.add_argument("--since", help="брати записи від цієї дати/часу (ISO), наприклад 2026-10-23")
    parser.add_argument("--section", choices=["all", *SECTIONS], default="all")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"журналу немає: {path}", file=sys.stderr)
        return 1
    records = load(path, args.since)

    names = list(SECTIONS) if args.section == "all" else [args.section]
    for name in names:
        print(f"\n=== {name}")
        SECTIONS[name](records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
