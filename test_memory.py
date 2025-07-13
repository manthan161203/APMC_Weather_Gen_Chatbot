# test_memory.py

from agent import invoke_agent, get_conversation_history

def test_memory():
    """Test the memory functionality"""
    
    session_id = "test_session_123"
    
    print("=== Testing Memory Functionality ===\n")
    
    # Test 1: Ask about weather in Surat
    print("1. Asking about weather in Surat...")
    result1 = invoke_agent(session_id, "What is the weather in Surat?")
    print(f"Response: {result1['output'][:100]}...")
    
    # Test 2: Ask follow-up question about temperature
    print("\n2. Asking follow-up: What is the temperature?")
    result2 = invoke_agent(session_id, "What is the temperature?")
    print(f"Response: {result2['output'][:100]}...")
    
    # Test 3: Ask for wind speed
    print("\n3. Asking follow-up: What is the wind speed?")
    result3 = invoke_agent(session_id, "What is the wind speed?")
    print(f"Response: {result3['output'][:100]}...")
    
    # Test 4: Ask about market prices in Amreli
    print("\n4. Asking about market prices in Amreli...")
    result4 = invoke_agent(session_id, "What are the market rates in Amreli?")
    print(f"Response: {result4['output'][:100]}...")
    
    # Test 5: Ask for more information
    print("\n5. Asking for more information...")
    result5 = invoke_agent(session_id, "Give me more")
    print(f"Response: {result5['output'][:100]}...")
    
    # Check conversation history
    print("\n=== Conversation History ===")
    history = get_conversation_history(session_id)
    for i, msg in enumerate(history):
        print(f"{i+1}. {msg.type}: {msg.content[:50]}...")
    
    print(f"\nTotal messages in history: {len(history)}")

if __name__ == "__main__":
    test_memory()