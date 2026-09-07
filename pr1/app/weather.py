"""Модуль інтеграції із зовнішнім API погоди.

Це єдине місце застосунку, яке знає про HTTP: адреси сервісів, параметри
запиту, коди відповіді й формат JSON. Веб-рівень (`app/main.py`) отримує
звідси готовий результат або зрозумілу помилку і нічого не знає про
`requests`.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши по дорозі рішення
з розділу 4 практичної роботи:

* як передати параметри запиту, не склеюючи URL вручну;
* яке обмеження часу (timeout) поставити й що робити, коли воно спрацювало;
* чи однаково реагувати на помилку клієнта (4xx) і сервера (5xx);
* як повестися, коли міста не знайдено або у відповіді немає потрібних полів;
* що саме віддавати назовні при успіху і як позначати помилку.

Реальні відповіді обох сервісів збережено в папці `samples/` — подивіться їх
перед тим, як писати розбір відповіді.
"""

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherError(Exception):
    """Помилка отримання погоди, зрозуміла веб-рівню.

    Заготовка. Вирішіть, чи достатньо одного типу помилки, чи їх варто
    розрізняти — місто не знайдено, сервіс недоступний, відповідь не та,
    якої очікували. Від цього залежить, який HTTP-статус поверне застосунок
    і що побачить користувач.
    """


def find_city(name: str):

    """Знайти координати міста за його назвою.

    Що саме повертати — вирішіть самі: пару чисел, словник, окремий тип.
    Врахуйте випадок, коли міста з такою назвою немає.
    """
    try:
        result = requests.get(
            GEOCODING_URL,
            params={
                "name": name,
                "count": 1,
                "language": "uk",
                "format": "json"
            },
            timeout=10
        )
    except requests.Timeout:
        raise WeatherError("Час очікування сервісу вичерпано.")
    except requests.RequestException:
        raise WeatherError("Не вдалося підключитися до сервісу.")

    if result.status_code >= 500:
        raise WeatherError("Сервіс геокодування тимчасово недоступний.")

    if result.status_code >= 400:
        raise WeatherError("Помилка запиту до сервісу геокодування.")

    data = result.json()

    if "results" not in data or not data["results"]:
        raise WeatherError(f"Місто '{name}' не знайдено.")

    city_data = data["results"][0]

    if "latitude" not in city_data or "longitude" not in city_data:
        raise WeatherError("У відповіді немає координат міста.")

    return {
        "name": city_data["name"],
        "latitude": city_data["latitude"],
        "longitude": city_data["longitude"]
    }


def get_current_weather(city: str):
    """Повернути поточну погоду в місті: температуру й швидкість вітру.

    Це функція, яку викликає веб-рівень. Вона поєднує геокодування і запит
    прогнозу та віддає результат у зручному для застосунку вигляді.
    """
    location = find_city(city)

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": "temperature_2m,wind_speed_10m"
    }

    try:
        result = requests.get(
            FORECAST_URL,
            params=params,
            timeout=10
        )
    except requests.Timeout:
        raise WeatherError("Час очікування вичерпано.")
    except requests.RequestException:
        raise WeatherError("Не вдалося підключитися до сервісу.")

    if result.status_code >= 500:
        raise WeatherError("Сервіс погоди тимчасово недоступний.")

    if result.status_code >= 400:
        raise WeatherError("Помилка запиту до сервісу погоди.")

    data = result.json()

    if "current" not in data:
        raise WeatherError("У відповіді немає даних про погоду.")

    current = data["current"]

    if "temperature_2m" not in current:
        raise WeatherError("У відповіді немає температури.")

    if "wind_speed_10m" not in current:
        raise WeatherError("У відповіді немає швидкості вітру.")

    return {
        "city": location["name"],
        "temperature": current["temperature_2m"],
        "wind_speed": current["wind_speed_10m"]
    }
