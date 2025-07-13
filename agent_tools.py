# agent_tools.py

import re
from langchain.tools import tool
from utils import (
    fetch_weather_data_by_city,
    fetch_weather_data_by_coords,
    format_weather_data,
    fetch_agriculture_prices,
    format_agriculture_data,
    get_current_month_and_year,
    extract_location_from_text,
    translate_location_name_to_english
)
from shared import llm  # use globally shared LLM

def is_non_english_text(text: str) -> bool:
    """Check if text contains non-English characters."""
    non_latin_pattern = r'[^\x00-\x7F\u00C0-\u017F\u0100-\u024F]'
    return bool(re.search(non_latin_pattern, text))

@tool
def get_weather_with_auto_translation(city: str = "", lat: float = None, lon: float = None) -> str:
    """Get weather for any city or coordinates. Automatically translates non-English city names to English."""
    try:
        if city:
            if is_non_english_text(city):
                city = translate_location_name_to_english(city)
            try:
                return format_weather_data(fetch_weather_data_by_city(city))
            except:
                pass  # Fallback if city fails

        if lat is not None and lon is not None:
            return format_weather_data(fetch_weather_data_by_coords(lat, lon))

        return "City not found and coordinates not provided."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

@tool
def get_weather_by_city(city: str) -> str:
    """Fetch weather for a city (English names only)."""
    try:
        data = fetch_weather_data_by_city(city)
        return format_weather_data(data)
    except Exception as e:
        return f"Error fetching weather for {city}: {str(e)}"

@tool
def get_weather_by_coordinates(lat: float, lon: float) -> str:
    """Fetch weather for given latitude and longitude."""
    try:
        data = fetch_weather_data_by_coords(lat, lon)
        return format_weather_data(data)
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

@tool
def get_agriculture_prices(city_or_text: str = "", lat: float = None, lon: float = None) -> str:
    """
    Fetch agriculture prices for a district.
    First tries to extract district from city_or_text.
    If not found, uses lat/lon to determine district name.
    """
    try:
        location = extract_location_from_text(city_or_text, llm) if city_or_text else None

        if not location and lat is not None and lon is not None:
            from utils import get_district_from_coords
            location = get_district_from_coords(lat, lon)

        if not location:
            return "Unable to detect your location from input or coordinates."

        if is_non_english_text(location):
            location = translate_location_name_to_english(location)

        data = fetch_agriculture_prices(location)
        return format_agriculture_data(data)
    except Exception as e:
        return f"Error fetching agriculture prices: {str(e)}"

@tool
def get_common_diseases(city_or_text: str) -> str:
    """Get common diseases for a location in current season."""
    try:
        time_info = get_current_month_and_year()
        month = time_info["month"]
        year = time_info["year"]

        location = extract_location_from_text(city_or_text, llm)
        if not location:
            return "Unable to detect location."

        if is_non_english_text(location):
            location = translate_location_name_to_english(location)

        prompt = f"What are the common diseases or health concerns in {location} during {month} {year}?"
        return llm.invoke(prompt).content.strip()

    except Exception as e:
        return f"Error detecting seasonal diseases: {str(e)}"

@tool
def get_current_season_crop_suggestion(city_or_text: str) -> str:
    """Suggest crops suitable for current season in a location."""
    try:
        time_info = get_current_month_and_year()
        month = time_info["month"]
        year = time_info["year"]

        location = extract_location_from_text(city_or_text, llm)
        if not location:
            return "Unable to detect location."

        if is_non_english_text(location):
            location = translate_location_name_to_english(location)

        prompt = f"Suggest suitable crops to plant in {location} during {month} {year}, based on the typical season and climate of the region."
        return llm.invoke(prompt).content.strip()

    except Exception as e:
        return f"Error generating crop suggestion: {str(e)}"

@tool
def extract_location(user_input: str) -> str:
    """Extract location name from text."""
    try:
        location = extract_location_from_text(user_input, llm)
        return location if location else "None"
    except Exception as e:
        return f"Error extracting location: {str(e)}"

@tool
def translate_location_to_english(location: str) -> str:
    """Translate location name to English."""
    try:
        return translate_location_name_to_english(location)
    except Exception as e:
        return f"Error translating location: {str(e)}"