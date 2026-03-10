"""
weather.py - Live weather using wttr.in JSON API (no API key required)
"""

import requests

DISTRICT_COORDS = {
    "Hyderabad": (17.385, 78.4867), "Warangal": (17.9784, 79.5941),
    "Karimnagar": (18.4386, 79.1288), "Nizamabad": (18.6725, 78.094),
    "Khammam": (17.2473, 80.1514), "Nalgonda": (17.0575, 79.2671),
    "Adilabad": (19.6641, 78.532), "Medak": (18.0504, 78.2635),
    "Guntur": (16.3, 80.45), "Vijayawada": (16.5082, 80.648),
    "Visakhapatnam": (17.6868, 83.2185), "Kurnool": (15.8281, 78.0373),
    "Kadapa": (14.4673, 78.8242), "Nellore": (14.4426, 79.9865),
    "Anantapur": (14.6819, 77.6006), "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882), "Nashik": (20.0059, 73.7997),
    "Aurangabad": (19.8762, 75.3433), "Solapur": (17.6862, 75.9064),
    "Kolhapur": (16.705, 74.2433), "Amravati": (20.9374, 77.7796),
    "Latur": (18.4088, 76.5604), "Ludhiana": (30.901, 75.8573),
    "Amritsar": (31.634, 74.8723), "Jalandhar": (31.326, 75.5762),
    "Patiala": (30.3398, 76.3869), "Bathinda": (30.211, 74.9455),
    "Chennai": (13.0827, 80.2707), "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198), "Salem": (11.6643, 78.146),
    "Bangalore": (12.9716, 77.5946), "Mysore": (12.2958, 76.6394),
    "Hubli": (15.3647, 75.124), "Belgaum": (15.8497, 74.4977),
    "Lucknow": (26.8467, 80.9462), "Kanpur": (26.4499, 80.3319),
    "Varanasi": (25.3176, 82.9739), "Agra": (27.1767, 78.0081),
    "Bhopal": (23.2599, 77.4126), "Indore": (22.7196, 75.8577),
    "Jabalpur": (23.1815, 79.9864), "Gwalior": (26.2183, 78.1828),
}

WEATHER_ICONS = {
    "sunny": "☀️", "clear": "🌤️", "cloud": "☁️", "rain": "🌧️",
    "drizzle": "🌦️", "thunder": "⛈️", "snow": "❄️",
    "mist": "🌫️", "fog": "🌫️", "haze": "🌫️",
    "overcast": "☁️", "partly": "⛅",
}

def _get_icon(description):
    d = description.lower()
    for key, icon in WEATHER_ICONS.items():
        if key in d:
            return icon
    return "🌡️"

def get_weather(district):
    coords = DISTRICT_COORDS.get(district)
    location_query = f"{coords[0]},{coords[1]}" if coords else district.replace(" ", "+") + "+India"
    try:
        resp = requests.get(
            f"https://wttr.in/{location_query}?format=j1",
            timeout=8, headers={"User-Agent": "KisanCallCentre/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        temp = int(current["temp_C"])
        feels_like = int(current["FeelsLikeC"])
        humidity = int(current["humidity"])
        wind_speed = int(current["windspeedKmph"])
        desc = current["weatherDesc"][0]["value"]
        cloud_cover = int(current.get("cloudcover", 0))
        hourly = data.get("weather", [{}])[0].get("hourly", [{}])
        rain_chance = int(hourly[0].get("chanceofrain", cloud_cover // 2)) if hourly else 30
        today = data.get("weather", [{}])[0]
        max_temp = int(today.get("maxtempC", temp + 2))
        min_temp = int(today.get("mintempC", temp - 3))
        icon = _get_icon(desc)
        return {
            "temp": temp, "feels_like": feels_like, "humidity": humidity,
            "description": f"{icon} {desc}", "rain_chance": rain_chance,
            "wind_speed": wind_speed, "max_temp": max_temp, "min_temp": min_temp,
            "icon": icon, "advisory": _generate_advisory(temp, humidity, rain_chance, desc, wind_speed),
            "is_live": True,
        }
    except Exception:
        return {
            "temp": 28, "feels_like": 30, "humidity": 68, "description": "⛅ Partly Cloudy",
            "rain_chance": 20, "wind_speed": 14, "max_temp": 33, "min_temp": 23,
            "icon": "⛅", "advisory": "☀️ Good conditions for field work. Spray before 10 AM.",
            "is_live": False,
        }

def _generate_advisory(temp, humidity, rain_chance, desc, wind_speed):
    parts = []
    if rain_chance > 70:
        parts.append("⚠️ Heavy rain likely — avoid spraying & harvesting today.")
    elif rain_chance > 40:
        parts.append("🌦️ Moderate rain chance — complete outdoor work in morning.")
    elif wind_speed > 30:
        parts.append("💨 Strong winds — avoid spraying, secure nets and covers.")
    else:
        parts.append("☀️ Good conditions for field work. Spray before 10 AM or after 4 PM.")
    if temp > 40:
        parts.append("🌡️ Extreme heat — irrigate in evening, apply mulch.")
    elif temp > 35:
        parts.append("🌡️ Hot day — morning irrigation recommended.")
    elif temp < 12:
        parts.append("❄️ Cold — protect nursery seedlings from frost.")
    if humidity > 85:
        parts.append("💧 Very high humidity — high risk of fungal diseases. Apply fungicide.")
    return " | ".join(parts[:2])