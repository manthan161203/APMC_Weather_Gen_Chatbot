# agent.py

import os
from configs import api_config
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from agent_tools import (
    get_weather_by_city,
    get_weather_by_coordinates,
    get_agriculture_prices,
    get_common_diseases,
    get_current_season_crop_suggestion,
    extract_location,
    translate_location_to_english,
    get_weather_with_auto_translation,
)
from shared import llm


# List of available tools - prioritize the auto-translation tool
tools = [
    get_weather_with_auto_translation,
    get_weather_by_city,
    get_weather_by_coordinates,
    get_agriculture_prices,
    get_common_diseases,
    get_current_season_crop_suggestion,
    extract_location,
    translate_location_to_english,
]

# Simple and clear prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a smart, multilingual assistant that helps users with:

    - Weather forecasts
    - Agricultural market prices
    - Crop suggestions
    - Seasonal diseases

    ## Your logic:

    1. First, detect the **city or district name** from the user input, even if it's in a regional language like Gujarati.
    2. If a location is already clearly present (e.g., "Surat"), **use it directly** and do not call additional location tools.
    3. If no location is detected in the user input, and coordinates (lat, lon) are available, use them to find the location.
    4. Only call location tools if necessary. Avoid duplicate detection.
    5. Use the most relevant tool for the user’s question:
    - For **weather**, call `get_weather_with_auto_translation`.
    - For **agriculture prices**, call `get_agriculture_prices`.
    - For **diseases** or **crop suggestions**, ensure the location is known and translated if needed.
    6. For **general questions** (like "Who is PM of India?"), do not call any tools — just answer from your own knowledge.
    7. Combine tool outputs when needed, but don’t repeat or call multiple tools for the same data.
    """),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create LangChain agent with tools
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)