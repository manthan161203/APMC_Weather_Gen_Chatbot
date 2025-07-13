"""
Configuration file for the Weather & Agriculture Chatbot API
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

# Load environment variables
load_dotenv()

@dataclass
class APIConfig:
    """Configuration for external API keys"""
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    google_gemini_api_key: str = os.getenv("GOOGLE_GEMINI_API_KEY")
    sarvam_ai_api_key: str = os.getenv("SARVAM_AI_API_KEY")
    data_gov_api_key: str = os.getenv("DATA_GOV_API_KEY")
    openweather_api_key: str = os.getenv("OPENWEATHERMAP_API_KEY")

@dataclass
class AppConfig:
    """Application configuration"""
    # File handling
    upload_directory: str = "uploads"
    output_directory: str = "outputs"
    allowed_audio_formats: list = None
    max_audio_duration_seconds: int = 20
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Weather & Agriculture Chatbot API"
    api_description: str = "Multi-language chatbot for weather and agriculture information"
    api_version: str = "1.0.0"
    
    # LLM settings
    default_llm_model: str = "gemini-1.5-flash"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # Audio settings
    default_tts_model: str = "bulbul:v2"
    default_tts_speaker: str = "anushka"
    
    # Session settings
    session_timeout_minutes: int = 30
    max_chat_history: int = 50
    
    def __post_init__(self):
        if self.allowed_audio_formats is None:
            self.allowed_audio_formats = ['.mp3', '.wav', '.m4a']

@dataclass
class WeatherAPIConfig:
    """Weather API configuration"""
    base_url: str = "https://api.openweathermap.org/data/2.5/weather"
    units: str = "metric"  # metric, imperial, kelvin
    language: str = "en"
    timeout_seconds: int = 10

@dataclass
class AgricultureAPIConfig:
    """Agriculture/Data.gov API configuration"""
    base_url: str = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    timeout_seconds: int = 10
    max_records: int = 1000

@dataclass
class LanguageConfig:
    """Language processing configuration"""
    supported_languages: list = None
    default_language: str = "en"
    fallback_language: str = "en"
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = [
                "en", "hi", "gu", "bn", "te", "ta", "kn", "ml", "mr", "pa"
            ]

# Initialize configuration instances
api_config = APIConfig()
app_config = AppConfig()
weather_config = WeatherAPIConfig()
agriculture_config = AgricultureAPIConfig()
language_config = LanguageConfig()

# Validation functions
def validate_api_keys():
    """Validate that required API keys are present"""
    missing_keys = []
    
    if not api_config.google_gemini_api_key:
        missing_keys.append("GOOGLE_GEMINI_API_KEY")
    if not api_config.sarvam_ai_api_key:
        missing_keys.append("SARVAM_AI_API_KEY")
    if not api_config.openweather_api_key:
        missing_keys.append("OPENWEATHERMAP_API_KEY")
    
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

def validate_directories():
    """Create required directories if they don't exist"""
    os.makedirs(app_config.upload_directory, exist_ok=True)
    os.makedirs(app_config.output_directory, exist_ok=True)

# Initialize on import
try:
    validate_api_keys()
    validate_directories()
    print("✅ Configuration loaded successfully")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    raise