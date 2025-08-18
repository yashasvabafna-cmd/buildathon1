# test_routing.py
import asyncio
from MCPClient import main

# Test cases to verify proper routing
test_queries = [
    "What pizzas do you have?",          # Should use menu_query
    "I want a pepperoni pizza",          # Should use extract_order  
    "What did I order?",                 # Should use order_summary
    "Show me the menu",                  # Should use menu_query
    "I'd like to order a root beer"      # Should use extract_order
]

async def test_routing():
    print("Testing tool routing...")
    for query in test_queries:
        print(f"\nTest: '{query}'")
        # You would run this manually to test
        
if __name__ == "__main__":
    asyncio.run(test_routing())
