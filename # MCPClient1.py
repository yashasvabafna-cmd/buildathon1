# MCPClient-1.py
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
import os

# Fix tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# Enhanced tool routing prompt
TOOL_ROUTING_PROMPT = """
You are a restaurant ordering assistant. Before using any tool, carefully analyze the user's intent:

🔍 USE menu_query for:
- "What pizzas do you have?"
- "Show me the menu"  
- "What's available?"
- Any menu browsing questions
- Questions about ingredients, prices, descriptions

📝 USE extract_order for:
- "I want a pizza"
- "I'd like to order"
- "Can I get..."
- "I'll have..."
- Actual ordering requests with specific items

📋 USE order_summary for:
- "What did I order?"
- "Show my order"  
- "Order summary"
- "What's in my cart?"
- Reviewing existing orders ONLY

🔍 USE classify_intent for:
- When you're unsure about user intent
- To determine the correct tool to use

CRITICAL RULES:
- NEVER use order_summary for menu questions!
- NEVER use extract_order for browsing questions!
- NEVER use menu_query for order reviews!
- Think before choosing a tool
- When in doubt, use classify_intent first
"""

# The state that our LangGraph agent will manage
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    order_history: Annotated[List[dict], operator.add]

async def main():
    # Fetch tools from the MCP server
    client = MultiServerMCPClient({
        "toolkit": {
            "command": "python",
            "args": ["MCP_BOT.py"],
            "transport": "stdio"
        }
    })
    tools = await client.get_tools()

    # Define the LLM and the agent
    llm = ChatGroq(model="llama-3.1-8b-instant")
    
    # Add memory for persistent conversations
    memory = MemorySaver()
    
    # Create agent with proper system message - THIS WAS THE KEY MISSING PIECE
    agent = create_react_agent(
        llm, 
        tools, 
        checkpointer=memory,
        messages_modifier=SystemMessage(content=TOOL_ROUTING_PROMPT)
    )

    # Define the LangGraph state machine
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("tool_node", ToolNode(tools))

    def should_continue(state):
        messages = state['messages']
        last_message = messages[-1]
        if 'tool_calls' in last_message and last_message['tool_calls']:
            return "tool_node"
        return "END"

    # Define the graph's edges
    workflow.add_edge("tool_node", "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.set_entry_point("agent")

    # Compile the graph with memory
    app = workflow.compile(checkpointer=memory)

    # Continuous conversation loop
    print("🤖 Restaurant Ordering Assistant")
    print("Type 'quit', 'exit', or 'bye' to end the conversation\n")
    
    # Thread configuration for memory persistence
    thread_config = {"configurable": {"thread_id": "conversation_1"}}
    
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("🤖 Thanks for using our ordering system! Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Create conversation state with system prompt reinforcement
            conversation_state = {
                "messages": [
                    {"role": "user", "content": user_input}
                ],
                "order_history": []
            }
            
            # Get response from agent
            response = await app.ainvoke(
                conversation_state, 
                config=thread_config
            )
            
            # Extract and display the assistant's response
            if response['messages']:
                assistant_message = response['messages'][-1]
                print(f"🤖 Assistant: {assistant_message.content}\n")
                
        except KeyboardInterrupt:
            print("\n🤖 Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again.\n")

if __name__ == "__main__":
    asyncio.run(main())
