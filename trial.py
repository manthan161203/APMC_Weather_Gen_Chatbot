# from utils import extract_location_from_text, translate_text, detect_text_language
# # from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI

# # Load your LLM (make sure your environment has the API key set)
# # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
# llm = ChatOpenAI(model="gpt-4o-mini")

# # List of test queries in different languages
# queries = [
#     "What’s the weather like in Ahmedabad?",
#     "અમદાવાદમાં હવામાન કેવું છે?",
#     "मुंबई में आज मौसम कैसा है?",
#     "What's the mandi rate in Rajkot?",
#     "રાજકોટમાં આજની માર્કેટ ભાવ શું છે?",
#     "Tell me the crop price in Nashik.",
#     "ફળોની બજાર કિંમત વિશે જણાવો.",
#     "What is the capital of Gujarat?",  # Should ideally return None
#     "હવામાન કેવી રીતે બદલાય છે?"  # No location
# ]

# for query in queries:
#     location = extract_location_from_text(query, llm)
#     english_location = None
    
#     if location:
#         # Detect language first
#         lang_code = detect_text_language(location)
#         if lang_code != "en":
#             english_location = translate_text(location, target_language_code="en-IN", source_language_code=lang_code)
#         else:
#             english_location = location

#     print(f"Query: {query}")
#     print(f"Extracted Location: {location}")
#     print(f"Translated to English: {english_location}")
#     print("-" * 40)


# from sarvamai import SarvamAI


# def speech_to_text(file):
#     client = SarvamAI(
#         api_subscription_key="10eb632f-0e99-4984-a9de-01e80123fa5a",
#     )
#     with open(file,'rb') as f:
#         x=client.speech_to_text.transcribe(file=f)
#         print(x.transcript)
#     return x.transcript

# print(speech_to_text("audio_for_testing/important_temple_in_gujarat_english.mp3"))

from utils import get_district_from_coords, fetch_weather_data_by_coords

print(fetch_weather_data_by_coords(21.1702, 72.8311))