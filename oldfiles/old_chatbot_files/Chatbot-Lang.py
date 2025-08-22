import operator
import pandas as pd
import json
import os
import traceback
from typing import TypedDict, Annotated, List, Tuple

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel # Using v1 for compatibility with existing code
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_huggingface.embeddings import HuggingFaceEmbeddings # Updated import for HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # Still needed for FAISS

# --- 1. Define Pydantic Models and Utility Functions ---
class Item(BaseModel):
    item: str
    quantity: int = 1
    modifiers: List[str] = []

class Order(BaseModel):
    items: List[Item]

# Custom combiner for order_history
def merge_orders(current_order: dict, new_order: dict) -> dict:
    """Combines a new order dictionary with the existing order dictionary."""
    if not current_order:
        return new_order
    
    # Simple merge logic: extend the items list.
    # More sophisticated merging (e.g., combining quantities of existing items)
    # would go here.
    if 'items' in new_order and isinstance(new_order['items'], list): # Changed to 'items' as per Order Pydantic model
        if 'items' not in current_order:
            current_order['items'] = []
        current_order['items'].extend(new_order['items'])
    
    return current_order

# Helper for LangGraph's messages list
def add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    """Combines messages for LangGraph state."""
    return left + right

# makeRetriever function using HuggingFaceEmbeddings
def makeRetriever(menu_df: pd.DataFrame, search_type="similarity", k=10):
    if menu_df.empty:
        print("Warning: Menu DataFrame is empty. Retriever will not function as expected.")
        # Ensure a default embedding model is provided even if empty
        return FAISS.from_texts(["No menu items available."], HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})).as_retriever()
    
    # DEBUG: Print columns found in the DataFrame
    print(f"DEBUG: Columns found in menu_df: {menu_df.columns.tolist()}")

    # Corrected: Use 'item_name' as per your CSV's actual column
    texts = menu_df.apply(lambda row: f"{row['item_name']}: ${row['price']}", axis=1).tolist()
    
    # Initialize HuggingFaceEmbeddings with an open-source model, explicitly set device to CPU
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Explicitly set device to CPU
    )
    
    # Create FAISS vector store
    vectorstore = FAISS.from_texts(texts, embeddings_model)
    return vectorstore.as_retriever(search_kwargs={"k": k})

def get_context(user_input: str, retriever) -> Tuple[List, str]:
    docs = retriever.invoke(user_input)
    context_str = "\n".join([d.page_content for d in docs])
    return docs, context_str

# --- 2. Define LangGraph State ---
class State(TypedDict):
    """Represents the state of our graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    order_history: Annotated[dict, merge_orders] # Stores the combined order JSON

# --- 3. Initialize LLM, Chains, and Tools ---
# Set GROQ_API_KEY directly in the script
# IMPORTANT: In a production environment, use os.environ.get("GROQ_API_KEY") for security.
GROQ_API_KEY="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"
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)


# Attempt to load menu, provide dummy if not found
try:
    menu = pd.read_csv("testmenu100.csv")
except FileNotFoundError:
    print("WARNING: 'testmenu100.csv' not found. Using dummy menu data for demonstration.")
    menu = pd.DataFrame({'item_name': ['Burger Combo', 'Soda', 'Fries', 'Espresso', 'Root Beer'], # Updated dummy columns
                         'price': [12.99, 2.49, 3.50, 3.00, 2.49]})

retriever = makeRetriever(menu, search_type="similarity", k=10)

# Prompts for specific chains (can be more detailed)
order_extraction_prompt = ChatPromptTemplate.from_template("Extract order details from this: {user_input}\n{format_instructions}")
conversation_chain_prompt = ChatPromptTemplate.from_template("Context: {context}\nUser: {user_input}\nChat History: {chat_history}")

parser = PydanticOutputParser(pydantic_object=Order)
orderChain = order_extraction_prompt | llm | parser
conversationChain = conversation_chain_prompt | llm

# Define the agent_prompt for the create_tool_calling_agent
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly restaurant assistant. Your job is to take customer orders, answer questions about the menu, and summarize the final order. You have access to three specialized tools to help you with these tasks."),
    ("system", """
    # Tool Usage Instructions
    * **extract_order**: Use this tool when the customer is explicitly placing an order, such as 'I want to order a burger' or 'I'll take the special combo'.
    * **menu_query**: Use this tool if the customer is asking a question about the menu, such as 'What's the price of a soda?' or 'Do you have vegetarian options?'.
    * **order_summary**: Use this tool when the customer asks to see their current order, for example, 'What did I order so far?' or 'Can you summarize my order?'.
    """),
    ("system", """
    # Behavioral Guidelines
    * **STRICTLY EXTRACT**: When using the `extract_order` tool, ONLY include items and modifiers the user EXPLICITLY mentions. DO NOT add any items like water or fries that were not requested. Be a strict extractor.
    * **ENDING THE CONVERSATION**: After you have called a tool and processed its result (e.g., extracted an order, found menu info), you **MUST** generate a final, natural language response to the user. This response should:
        1. Confirm the action taken (e.g., "Got it! I've added X to your order.").
        2. Provide the relevant information clearly (e.g., menu price, order summary).
        3. Politely ask a follow-up question to keep the conversation going (e.g., "Is there anything else I can get for you?", "Would you like to confirm this order?").
        **DO NOT** call another tool unless the user's *next* message explicitly requires it. Your goal is to provide a complete, conversational turn before waiting for the next user input.
    * **Handling Off-Topic Requests**: If the customer's request is not related to ordering, the menu, or their current order, politely let them know you can only assist with those tasks.
    """),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Import create_tool_calling_agent
from langchain.agents import create_tool_calling_agent



# Locate this section in your Chatbot-Lang.py file
@tool
def extract_order(user_input: str) -> str:
    """
    STRICTLY EXTRACT a structured order JSON from user input.
    ONLY include items and modifiers EXPLICITLY mentioned by the user.
    DO NOT add any items, modifiers, or details that were not requested.
    The function must be called only once per user turn.
    The order should be a single JSON object with the following structure:
    Example: {"items": [{"item": "Burger", "quantity": 1, "modifiers": ["extra cheese"]}]}
    """
    try:
        # ... (your existing successful extraction logic)
        result = orderChain.invoke({
            "user_input": user_input,
            "format_instructions": parser.get_format_instructions()
        })
        return result.model_dump_json()
    except Exception as e:
        # 🚨 CRITICAL CHANGE HERE: Always return a clean, parsable JSON string for errors.
        # The LLM will then interpret this JSON (in the agent_prompt) to craft a user-friendly message.
        return json.dumps({"status": "error", "message": f"Order extraction failed for: '{user_input}'. Details: {str(e)}"})

# Also, ensure your 'order_summary' tool returns clean conversational text, not raw JSON
# The current order_summary already seems to do this.
@tool
def menu_query(user_input: str) -> str:
    """
    Answer questions about the menu. Use this tool if the customer is asking about
    menu items, prices, availability, or ingredients.
    """
    # Note: In a real scenario, chat_history should come from the state for full context
    # For this simple tool, we're using an empty list, but for better conversation
    # the 'messages' from state should be passed.
    rel_docs, context = get_context(user_input, retriever)
    ai_response = conversationChain.invoke({
        "context": context,
        "user_input": user_input,
        "chat_history": [] # This should ideally come from the main graph state for full context
    })
    return ai_response.content

@tool
def order_summary(order_history_json: dict) -> str: # Tool now directly accepts the dict
    """
    Summarize the order. This tool summarizes the order by looking at the JSON
    representation of the current order history. Use this when the user asks
    'What did I order?' or 'Can you summarize my order?'
    """
    if order_history_json and 'items' in order_history_json and order_history_json['items']:
        summary_items = []
        for item_data in order_history_json['items']:
            item_name = item_data.get('item', 'unknown item')
            quantity = item_data.get('quantity', 1)
            modifiers = ", ".join(item_data.get('modifiers', []))
            
            item_desc = f"{quantity} x {item_name}"
            if modifiers:
                item_desc += f" (with {modifiers})"
            summary_items.append(item_desc)
        
        return "Your current order is: " + "; ".join(summary_items) + ". Would you like to confirm or add anything else?"
    else:
        return "You haven't ordered anything yet. What can I get for you?"

tools = [extract_order, menu_query, order_summary]

# Create the agent using the LLM, tools, and the agent_prompt
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# --- 4. Define Graph Nodes ---
# tool_calling_llm node
def tool_calling_llm(state: State):
    # Pass the entire messages list and intermediate_steps to the agent
    agent_output = agent.invoke({
        "messages": state['messages'],
        "intermediate_steps": [] # Crucial for the agent's scratchpad
    })
    if hasattr(agent_output, 'messages'):
        # assuming AgentFinish object
        return {"messages": agent_output.messages, "return_values": agent_output.return_values}
    else:
        # assuming list
        return {"messages": agent_output, "return_values": {}}

# update_order_history node
def update_order_history(state: State):
    last_message = state['messages'][-1]
    
    # Check if the last message is a FunctionMessage resulting from extract_order
    if isinstance(last_message, FunctionMessage) and last_message.name == 'extract_order':
        try:
            new_order_data = json.loads(last_message.content)
            # Return a dict that our `merge_orders` combiner will process
            return {"order_history": new_order_data}
        except json.JSONDecodeError:
            # Handle cases where the tool output is not valid JSON
            print(f"Error: extract_order returned invalid JSON: {last_message.content}")
            return {} # Return an empty update
            
    return {} # If not extract_order or not a FunctionMessage, do nothing to order_history

# --- 5. Build and Compile the Graph ---
builder = StateGraph(State)

builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("update_order_history", update_order_history)
builder.add_node("tools", ToolNode(tools)) # ToolNode automatically calls the selected tool

# Edges
builder.add_edge(START, "tool_calling_llm")

builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
    {
        "tools": "tools",  # If LLM wants to use a tool, go to 'tools' node
        "__end__": END,    # If LLM gives a final answer, end the graph
    }
)

# After a tool runs, if it was 'extract_order', update order_history.
# Otherwise (e.g., menu_query), go directly back to LLM to respond.
builder.add_conditional_edges(
    "tools",
    lambda state: state['messages'][-1].name == "extract_order",
    {
        True: "update_order_history",  # If extract_order was just run, update history
        False: "tool_calling_llm"     # Otherwise (e.g., menu_query), return to LLM
    }
)

# After updating order history, always return to the LLM for a response
builder.add_edge("update_order_history", "tool_calling_llm")

# Compile the graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --- 6. Interactive Terminal Chat Loop ---
def run_chatbot_conversation():
    print("Welcome to the Restaurant Chatbot! Type 'exit' to quit.")
    
    # Initialize chat history for the current thread
    thread_id = "user_session_1" # You can make this dynamic per user if needed
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        # Prepare the input message for the graph
        input_messages = [HumanMessage(content=user_input)]
        
        # Invoke the graph and stream responses
        print("Chatbot:", end=" ")
        full_response_content = ""
        try:
            for chunk in graph.stream({'messages': input_messages}, config=config, stream_mode="values"):
                # Print only the content of the AI's final message or tool output
                if 'messages' in chunk and chunk['messages']:
                    last_message = chunk['messages'][-1]
                    # if isinstance(last_message, AIMessage):
                    #     print(last_message.content, end="")
                    #     full_response_content += last_message.content
                    # elif isinstance(last_message, FunctionMessage):
                    #     # For FunctionMessages, we might want to print the tool's result
                    #     # but typically the LLM will summarize it.
                    #     pass # Don't print raw function output
                    print(type(last_message), last_message)
            print() # Newline after the streamed response
        except Exception as e:
            # Print the full traceback for better debugging
            print(f"\nAn error occurred during conversation: {e}")
            traceback.print_exc() # This will print the detailed stack trace
            print("Please try again.")

if __name__ == "__main__":
    # Removed os.environ.get check since API key is hardcoded.
    run_chatbot_conversation()
