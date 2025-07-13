# agent.py

import os
from configs import api_config
from shared import llm

from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

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

# ✅ All tools
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

# ✅ Enhanced prompt with better context handling
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a smart, multilingual assistant that helps users with:

    - Weather forecasts and conditions
    - Agricultural market prices and rates
    - Crop suggestions based on season and location
    - Plant diseases and their management
    - General farming advice

    ## Your Context Awareness:
    You have access to the conversation history, so you can:
    - Remember what the user asked before
    - Refer to previously mentioned locations
    - Build on previous responses
    - Provide contextual follow-up information

    ## Your Logic:

    1. **Context Usage**: Always check the conversation history to understand what the user is referring to.
    2. **Location Memory**: If a user previously mentioned a location (like "Surat" or "Vadodara"), remember it for follow-up questions.
    3. **Follow-up Questions**: When users ask "what is temperature?" or "give me more", refer to the previous context.
    4. **Location Detection**: 
       - If a location is clearly present in current or previous messages, use it directly
       - If no location is detected and coordinates are available, use them
       - Only call location tools if absolutely necessary
    5. **Tool Selection**: Use the most relevant tool for the user's question:
       - For **weather**: call `get_weather_with_auto_translation`
       - For **agriculture prices**: call `get_agriculture_prices`
       - For **diseases/crop suggestions**: ensure location is known
    6. **General Questions**: For non-agricultural questions, answer from your knowledge without using tools.
    7. **Contextual Responses**: When asked for "more" information, expand on your previous response with additional relevant details.

    ## Response Style:
    - Be conversational and remember previous interactions
    - Use the language preferred by the user
    - Provide comprehensive answers that build on previous context
    - When asked for "more", provide additional relevant information about the last topic discussed
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# ✅ Create the agent
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# ✅ Session-based memory store
session_stores = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get or create a chat message history for a session"""
    if session_id not in session_stores:
        session_stores[session_id] = ChatMessageHistory()
    return session_stores[session_id]

# ✅ Create agent executor with proper message history
def get_agent_executor(session_id: str) -> AgentExecutor:
    """Get an agent executor with session-based memory"""
    
    # Create the base agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        max_execution_time=60
    )
    
    # Wrap with message history
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return agent_with_chat_history

# ✅ Enhanced invoke function with proper session handling
def invoke_agent(session_id: str, input_text: str, lat: float = None, lon: float = None):
    """Invoke the agent with session-based memory"""
    
    # Get the agent executor with message history
    agent_with_history = get_agent_executor(session_id)
    
    # Prepare the input with coordinates if provided
    enhanced_input = input_text
    if lat is not None and lon is not None:
        enhanced_input += f" (lat: {lat}, lon: {lon})"
    
    # Invoke the agent with session configuration
    result = agent_with_history.invoke(
        {"input": enhanced_input},
        config={"configurable": {"session_id": session_id}}
    )
    
    return result

# ✅ Optional: Function to get conversation history for debugging
def get_conversation_history(session_id: str) -> list:
    """Get the conversation history for a session"""
    if session_id in session_stores:
        return session_stores[session_id].messages
    return []

# ✅ Optional: Function to clear session history
def clear_session_history(session_id: str):
    """Clear the conversation history for a session"""
    if session_id in session_stores:
        session_stores[session_id].clear()

# ✅ Optional: Function to add manual message to history (for debugging)
def add_message_to_history(session_id: str, message: str, is_human: bool = True):
    """Manually add a message to the session history"""
    history = get_session_history(session_id)
    if is_human:
        history.add_user_message(message)
    else:
        history.add_ai_message(message)