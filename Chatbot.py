from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
# --- Configuration ---
# Ensure Ollama is running and you have pulled the llama3 model.
# You can do this by running 'ollama run llama3' in your terminal.
OLLAMA_MODEL = "llama3"

# --- Define the Menu ---
# This list contains the items your chatbot can "sell".
menu = pd.read_csv('testmenu100.csv')

# --- Initialize LLM and Parser ---
# This connects to your local Ollama instance.
chat_model = ChatOllama(model=OLLAMA_MODEL)
output_parser = StrOutputParser()

# --- Define the Chat Prompt Template ---
# This template is crucial for maintaining conversation context.
# - The 'system' message sets the chatbot's persona and general instructions.
#   It now explicitly includes the menu and instructions for handling out-of-menu items.
# - 'MessagesPlaceholder' is where the ongoing chat history will be injected.
# - The final 'human' message is for the current user input.
chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are a helpful and friendly AI assistant for a restaurant. Your menu includes the following items: {', '.join(menu)}. "
            "You can only confirm orders for these items. "
            "If a customer asks for an item not on the menu, please politely inform them that it's not available and suggest something similar from our menu if possible, or list the available items."
            "Keep your responses concise and polite."
        ),
        MessagesPlaceholder(variable_name="chat_history"), # This is where the magic of memory happens!
        ("human", "{user_input}"),
    ]
)

# --- Create the LangChain Chain ---
# The chain pipes the user input through the prompt template, then to the LLM,
# and finally parses the LLM's raw output into a clean string.
conversation_chain = chat_prompt | chat_model | output_parser

# --- Conversation History Storage ---
# This list will store all messages (both human and AI) to maintain context.
# Each message is an object (HumanMessage or AIMessage) which includes its role and content.
chat_history = []

# --- Main Conversation Loop ---
print(f"Chatbot initialized with {OLLAMA_MODEL}. Type 'quit' or 'exit' to end the conversation.")
print("-" * 60)

while True:
    user_input = input("You: ")

    # Check for exit commands
    if user_input.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye! Have a great day!")
        break

    # Add user's message to history
    chat_history.append(HumanMessage(content=user_input))

    try:
        # Invoke the chain with the current user input and the entire chat history.
        # The 'chat_history' variable in the template will be populated by our list.
        ai_response_content = conversation_chain.invoke(
            {"user_input": user_input, "chat_history": chat_history}
        )

        # Add AI's response to history
        chat_history.append(AIMessage(content=ai_response_content))

        # Print the AI's response
        print(f"Chatbot: {ai_response_content}")

    except Exception as e:
        # Basic error handling for LLM communication
        print(f"Chatbot: An error occurred: {e}. Please try again.")
        # Optionally, remove the last human message if the AI couldn't respond to it
        if chat_history and isinstance(chat_history[-1], HumanMessage):
            chat_history.pop()

    print("-" * 60)
