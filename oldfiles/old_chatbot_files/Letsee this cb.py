import pandas as pd
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import BaseMessage, FunctionMessage # Used in add_messages if needed
import os
from dotenv import load_dotenv
load_dotenv()
# --- Dummy/Placeholder Definitions for Missing Components ---
# Define a simple Order Pydantic model
class Order(BaseModel):
    items: list[str]
    total_price: float
    notes: str = ""

# Dummy prompts for orderChain and conversationChain
orderPrompt = ChatPromptTemplate.from_template("Extract order details from this: {user_input}\n{format_instructions}")
conversationPrompt = ChatPromptTemplate.from_template("Context: {context}\nUser: {user_input}\nChat History: {chat_history}")

# Dummy makeRetriever and get_context functions
def makeRetriever(menu_df, search_type="similarity", k=10):
    # This is a basic in-memory retriever for demonstration.
    # In a real app, you'd load/create embeddings and use a proper vector store.
    if not isinstance(menu_df, pd.DataFrame) or menu_df.empty:
        return FAISS.from_texts(["No menu items available."], OpenAIEmbeddings()).as_retriever()
    texts = menu_df.apply(lambda row: f"{row['item']}: ${row['price']}", axis=1).tolist()
    return FAISS.from_texts(texts, OpenAIEmbeddings()).as_retriever()

def get_context(user_input: str, retriever) -> tuple[list, str]:
    # In a real scenario, this would perform a retrieval
    docs = retriever.invoke(user_input)
    context_str = "\n".join([d.page_content for d in docs])
    return docs, context_str

# Placeholder for add_messages if you were actually using LangGraph's TypedDict state
# For AgentExecutor, this isn't directly used in the same way.
def add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    """Combine messages."""
    return left + right

# --- Initialize LLM and Chains ---
llm = ChatGroq(model="llama-3.1-8b-instant")

# Assuming 'testmenu100.csv' exists and is readable, otherwise this will fail
try:
    menu = pd.read_csv("testmenu100.csv")
except FileNotFoundError:
    print("Warning: testmenu100.csv not found. Using dummy menu data.")
    menu = pd.DataFrame({'item': ['Burger Combo', 'Soda', 'Fries'], 'price': [12.99, 2.49, 3.50]})

retriever = makeRetriever(menu, search_type="similarity", k=10) # Initialize retriever once

orderChain = orderPrompt | llm | PydanticOutputParser(pydantic_object=Order) # Use the Order class directly
conversationChain = conversationPrompt | llm

# --- Define Tools with @tool decorator ---
@tool
def extract_order(user_input: str) -> str:
    """Extract structured order JSON from user input. Use this tool when the user is explicitly placing an order."""
    try:
        # In a real scenario, you'd pass the actual orderPrompt and parser
        result = orderChain.invoke({
            "user_input": user_input,
            "format_instructions": PydanticOutputParser(pydantic_object=Order).get_format_instructions()
        })
        return result.model_dump_json()
    except Exception as e:
        return f"Could not extract order: {e}"

@tool
def menu_query(user_input: str) -> str:
    """Answer questions about the menu. Use this tool when the user is asking about menu items, prices, or availability."""
    rel_docs, context = get_context(user_input, retriever)
    ai_response = conversationChain.invoke({
        "context": context,
        "user_input": user_input,
        "chat_history": [] # This should ideally be passed from the agent's state
    })
    return ai_response.content

tools = [extract_order, menu_query]

# --- Corrected Agent Prompt ---
# This is the crucial part that fixes the KeyError.
# It now uses 'messages' which LangGraph/AgentExecutor passes as a list of messages.
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly restaurant assistant. Your job is to take customer orders and answer questions about the menu. Your tools are designed to help you with these two specific tasks."),
    ("system", "If the customer is explicitly placing an order, such as 'I want a burger' or 'I'll take the special combo', you must use the `extract_order` tool to process their request."),
    ("system", "If the customer is asking a question about the menu, such as 'What's the price of a soda?' or 'Do you have vegetarian options?', you must use the `menu_query` tool to find the information."),
    ("system", "If the customer's request is not related to ordering or the menu, politely let them know you can only assist with those tasks."),
    MessagesPlaceholder(variable_name="messages"), # Corrected: uses 'messages'
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- Create the Agent ---
# Ensure 'agent' is created using the corrected 'agent_prompt'
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# --- Initialize AgentExecutor ---
# This line was missing or incorrectly placed in your previous attempts.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True # Keep verbose for debugging!
)

# --- Invoke the Agent ---
# Now this part should work correctly.
print("\n--- Invoking AgentExecutor ---")
response = agent_executor.invoke({
    "messages": [("human", "What is the price of the burger combo?")]
})

# Access the agent's response
print("Agent's final response:")
print(response["messages"][-1].content)

print("\n--- Invoking AgentExecutor for an order ---")
response = agent_executor.invoke({
    "messages": [("human", "I want to order a burger and fries.")]
})

print("Agent's final response:")
print(response["messages"][-1].content)