"""
Utility functions for the Weather & Agriculture Chatbot API
"""

import os
import uuid
import requests
from datetime import datetime
from typing import Optional, Dict, Any
from sarvamai import SarvamAI
from sarvamai.play import save
from pydub import AudioSegment
from fastapi import HTTPException
from configs import api_config, app_config, weather_config, agriculture_config

# Initialize Sarvam AI client
sarvam_client = SarvamAI(api_subscription_key=api_config.sarvam_ai_api_key)

# Weather API Key
openweather_api_key = api_config.openweather_api_key

# Audio Processing Functions
def convert_speech_to_text(audio_file_path: str) -> str:
    """
    Convert audio file to text using Sarvam AI
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Transcribed text
        
    Raises:
        HTTPException: If transcription fails
    """
    try:
        with open(audio_file_path, 'rb') as audio_file:
            transcription_result = sarvam_client.speech_to_text.transcribe(file=audio_file)
            return transcription_result.transcript
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to transcribe audio: {str(e)}"
        )

def convert_text_to_speech(
    text: str, 
    language_code: str, 
    output_file_path: str,
    model: str = None,
    speaker: str = None
) -> str:
    """
    Convert text to speech using Sarvam AI
    
    Args:
        text: Text to convert
        language_code: Target language code
        output_file_path: Path to save the audio file
        model: TTS model to use (default from config)
        speaker: Speaker voice to use (default from config)
        
    Returns:
        Path to the generated audio file
        
    Raises:
        HTTPException: If TTS conversion fails
    """
    try:
        model = model or app_config.default_tts_model
        speaker = speaker or app_config.default_tts_speaker
        
        audio_response = sarvam_client.text_to_speech.convert(
            target_language_code=language_code,
            text=text,
            model=model,
            speaker=speaker
        )
        
        # Save as WAV first
        temp_wav_path = "temp_output.wav"
        save(audio_response, temp_wav_path)
        
        # Convert to MP3
        audio_segment = AudioSegment.from_wav(temp_wav_path)
        audio_segment.export(output_file_path, format="mp3")
        
        # Clean up temporary file
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            
        return output_file_path
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate speech: {str(e)}"
        )



# Language Processing Functions
def detect_text_language(text: str) -> str:
    """
    Detect the language of the given text
    
    Args:
        text: Text to analyze
        
    Returns:
        Language code
        
    Raises:
        HTTPException: If language detection fails
    """
    try:
        language_result = sarvam_client.text.identify_language(input=text)
        return language_result.language_code
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to detect language: {str(e)}"
        )

def translate_text(text: str, target_language_code: str, source_language_code: str = "auto") -> str:
    """
    Translate text to target language, safely chunked to support Sarvam's 1000-char limit.
    
    Args:
        text: Text to translate
        target_language_code: Target language code (e.g., "gu-IN")
        source_language_code: Source language code (default: auto)
        
    Returns:
        Translated text
        
    Raises:
        HTTPException: If translation fails
    """
    try:
        # Convert language codes to the format expected by Sarvam API
        def format_language_code(lang_code: str) -> str:
            if lang_code and not lang_code.endswith("-IN"):
                return f"{lang_code}-IN"
            return lang_code

        formatted_source = format_language_code(source_language_code)
        formatted_target = format_language_code(target_language_code)

        # If source and target are the same, skip translation
        if formatted_source == formatted_target:
            return text

        # Split text into chunks of <=1000 characters
        max_chars = 1000
        paragraphs = text.split("\n")
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 1 <= max_chars:
                current += para + "\n"
            else:
                chunks.append(current.strip())
                current = para + "\n"
        if current:
            chunks.append(current.strip())

        translated_chunks = []
        for chunk in chunks:
            result = sarvam_client.text.translate(
                input=chunk,
                source_language_code=formatted_source,
                target_language_code=formatted_target,
            )
            translated_chunks.append(result.translated_text.strip())

        return "\n".join(translated_chunks)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to translate text: {str(e)}"
        )
    
def translate_location_name_to_english(location_name: str) -> str:
    """
    Translate extracted location name to English and sanitize output
    
    Args:
        location_name: Extracted city/district name (in any language)
    
    Returns:
        English-translated and cleaned location name string
    
    Raises:
        HTTPException: If translation or language detection fails
    """
    if not location_name:
        return None

    try:
        lang_code = detect_text_language(location_name)
        translated = translate_text(
            text=location_name,
            target_language_code="en",
            source_language_code=lang_code
        )
        return translated.strip().rstrip(".")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to translate location name: {str(e)}")


# Location Extraction Function
def extract_location_from_text(text: str, llm) -> Optional[str]:
    """
    Extract location name from user input using an LLM
    
    Args:
        text: User input text
        llm: A LangChain-compatible LLM object with a `invoke` method
        
    Returns:
        Extracted location name as string or None if not found
        
    Raises:
        HTTPException: If LLM extraction fails
    """
    try:
        prompt = (
            "Extract the name of a city or district from the following text. "
            "If none is found, reply with 'None'.\n\n"
            f"Text: \"{text}\"\n\n"
            "Location:"
        )
        location = llm.invoke(prompt).content.strip()
        return None if location.lower() == "none" else location
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract location: {str(e)}")


# Weather Data Functions
def fetch_weather_data_by_city(city_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetch weather data for a given city
    
    Args:
        city_name: Name of the city
        
    Returns:
        Weather data dictionary or None if not found
        
    Raises:
        HTTPException: If API request fails
    """
    try:
        request_url = f"{weather_config.base_url}?q={city_name}&appid={api_config.openweather_api_key}&units={weather_config.units}"
        
        response = requests.get(request_url, timeout=weather_config.timeout_seconds)
        response.raise_for_status()
        
        weather_data = response.json()
        
        if weather_data.get("cod") != 200:
            return None
            
        return weather_data
        
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch weather data: {str(e)}"
        )

def fetch_weather_data_by_coords(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Fetch weather data for given coordinates
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        
    Returns:
        Weather data dictionary or None if not found
        
    Raises:
        HTTPException: If API request fails
    """
    try:
        request_url = f"{weather_config.base_url}?lat={lat}&lon={lon}&appid={api_config.openweather_api_key}&units={weather_config.units}"
        
        response = requests.get(request_url, timeout=weather_config.timeout_seconds)
        response.raise_for_status()
        
        weather_data = response.json()
        
        if weather_data.get("cod") != 200:
            return None
            
        return weather_data
        
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch weather data by coordinates: {str(e)}"
        )
    
def format_weather_data(weather_data: Dict[str, Any]) -> str:
    """
    Format weather data into a readable string
    
    Args:
        weather_data: Weather data dictionary
        
    Returns:
        Formatted weather string
    """
    if not weather_data:
        return "Weather data not available"
    
    try:
        city = weather_data.get("name", "Unknown")
        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
        
        formatted = f"Weather in {city}:\n"
        formatted += f"• Condition: {description.title()}\n"
        formatted += f"• Temperature: {temperature}°C (feels like {feels_like}°C)\n"
        formatted += f"• Humidity: {humidity}%\n"
        if wind_speed != "N/A":
            formatted += f"• Wind Speed: {wind_speed} m/s"
        
        return formatted
        
    except KeyError as e:
        return f"Weather data format error: {str(e)}"



# Agriculture Data Functions
def fetch_agriculture_prices(district: str) -> Dict[str, Any]:
    """
    Fetch agriculture price data for a given district
    
    Args:
        district: Name of the district
        
    Returns:
        Agriculture price data
        
    Raises:
        HTTPException: If API request fails
    """
    try:
        request_url = f"{agriculture_config.base_url}?api-key={api_config.data_gov_api_key}&format=json&filters[district]={district}&limit={agriculture_config.max_records}"
        
        response = requests.get(request_url, timeout=agriculture_config.timeout_seconds)
        response.raise_for_status()
        
        data = response.json()
        
        return {
            "status": "success",
            "district": district,
            "count": len(data.get("records", [])),
            "records": data.get("records", [])
        }
        
    except requests.RequestException as e:
        return {
            "status": "error",
            "message": str(e),
            "code": getattr(e.response, 'status_code', 500) if hasattr(e, 'response') else 500
        }

def format_agriculture_data(price_data: Dict[str, Any]) -> str:
    """
    Format agriculture price data into a readable string
    
    Args:
        price_data: Agriculture price data dictionary
        
    Returns:
        Formatted agriculture data string
    """
    if not price_data or price_data.get("status") != "success":
        return "Agriculture price data not available"
    
    try:
        district = price_data.get("district", "Unknown")
        records = price_data.get("records", [])
        
        if not records:
            return f"No agriculture price data found for {district}"
        
        formatted = f"Agriculture Prices in {district}:\n"
        formatted += f"Found {len(records)} records\n\n"
        
        # Show first 5 records
        for i, record in enumerate(records[:5], 1):
            commodity = record.get("commodity", "Unknown")
            variety = record.get("variety", "N/A")
            min_price = record.get("min_price", "N/A")
            max_price = record.get("max_price", "N/A")
            modal_price = record.get("modal_price", "N/A")
            market = record.get("market", "N/A")
            
            formatted += f"{i}. {commodity}"
            if variety != "N/A":
                formatted += f" ({variety})"
            formatted += f"\n   Market: {market}\n"
            formatted += f"   Min: ₹{min_price}, Max: ₹{max_price}, Modal: ₹{modal_price}\n\n"
        
        if len(records) > 5:
            formatted += f"... and {len(records) - 5} more records"
        
        return formatted
        
    except Exception as e:
        return f"Agriculture data format error: {str(e)}"



# File Handling Functions
def validate_audio_file(filename: str, file_size: int = None) -> bool:
    """
    Validate audio file format and size
    
    Args:
        filename: Name of the file
        file_size: Size of the file in bytes (optional)
        
    Returns:
        True if valid, False otherwise
    """
    # Check file extension
    file_extension = os.path.splitext(filename.lower())[1]
    if file_extension not in app_config.allowed_audio_formats:
        return False
    
    # Additional size check can be added here if needed
    return True

def validate_audio_duration(file_path: str) -> bool:
    """
    Validate audio file duration
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        True if duration is within limits, False otherwise
    """
    try:
        audio = AudioSegment.from_file(file_path)
        duration_seconds = audio.duration_seconds
        return duration_seconds <= app_config.max_audio_duration_seconds
    except Exception:
        return False

def generate_unique_filename(extension: str = ".mp3") -> str:
    """
    Generate a unique filename
    
    Args:
        extension: File extension (default: .mp3)
        
    Returns:
        Unique filename
    """
    return f"{str(uuid.uuid4())}{extension}"

def cleanup_temp_files(*file_paths: str) -> None:
    """
    Clean up temporary files
    
    Args:
        *file_paths: Paths to files to be deleted
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not delete {file_path}: {e}")



# Utility Functions
def create_error_response(message: str, status_code: int = 400) -> HTTPException:
    """
    Create a standardized error response
    
    Args:
        message: Error message
        status_code: HTTP status code
        
    Returns:
        HTTPException instance
    """
    return HTTPException(status_code=status_code, detail=message)

def sanitize_city_name(city_name: str) -> str:
    """
    Sanitize city name for API calls
    
    Args:
        city_name: Raw city name
        
    Returns:
        Sanitized city name
    """
    if not city_name or city_name.lower() == "no location found":
        return None
    
    # Remove extra spaces and convert to title case
    return city_name.strip().title()

def is_valid_language_code(language_code: str) -> bool:
    """
    Check if language code is supported
    
    Args:
        language_code: Language code to validate
        
    Returns:
        True if supported, False otherwise
    """
    try:
        from configs import language_config
        return language_code in language_config.supported_languages
    except ImportError:
        # Fallback if language_config is not available
        supported_languages = ["en", "hi", "gu", "ta", "te", "bn", "mr", "kn", "ml", "or", "pa", "as"]
        return language_code in supported_languages
    


# Utility function to get current month and year
def get_current_month_and_year() -> Dict[str, str]:
    """
    Get the current month and year as a dictionary

    Returns:
        A dictionary with 'month' and 'year' keys
    """
    now = datetime.now()
    return {
        "month": now.strftime("%B"),   # e.g., 'July'
        "year": str(now.year)          # e.g., '2025'
    }

# Utility function to get district or city name from coordinates
def get_district_from_coords(lat: float, lon: float) -> Optional[str]:
    """Get city name from coordinates by calling OpenWeatherMap weather API."""
    try:
        request_url = f"{weather_config.base_url}?lat={lat}&lon={lon}&appid={api_config.openweather_api_key}&units={weather_config.units}"
        response = requests.get(request_url, timeout=weather_config.timeout_seconds)
        response.raise_for_status()

        weather_data = response.json()

        if weather_data.get("cod") != 200:
            return None
        
        city_name = weather_data.get("name")
        return city_name

    except Exception as e:
        print(f"Failed to extract city from weather API: {e}")
        return None