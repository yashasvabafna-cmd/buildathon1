from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
# Ensure Ollama is running and you have pulled the llama3 model.
# You can do this by running 'ollama run llama3' in your terminal.
OLLAMA_MODEL = "llama3"

# --- Define the Menu ---
# This list contains the items your chatbot can "sell".
MENU_ITEMS = {
    "classic burger": {"price": 6.99, "category": "main"},
    "cheese burger": {"price": 7.49, "category": "main"},
    "bacon burger": {"price": 7.99, "category": "main"},
    "veggie burger": {"price": 6.49, "category": "main"},
    "mushroom swiss burger": {"price": 7.59, "category": "main"},
    "bbq burger": {"price": 7.79, "category": "main"},
    "spicy chicken burger": {"price": 7.29, "category": "main"},
    "grilled chicken burger": {"price": 7.49, "category": "main"},
    "double patty burger": {"price": 8.49, "category": "main"},
    "black bean burger": {"price": 6.99, "category": "main"},
    "french fries": {"price": 2.49, "category": "side"},
    "curly fries": {"price": 2.99, "category": "side"},
    "sweet potato fries": {"price": 3.29, "category": "side"},
    "cheese fries": {"price": 3.49, "category": "side"},
    "loaded fries": {"price": 4.49, "category": "side"},
    "garlic fries": {"price": 3.59, "category": "side"},
    "truffle fries": {"price": 4.99, "category": "side"},
    "waffle fries": {"price": 3.99, "category": "side"},
    "cajun fries": {"price": 3.79, "category": "side"},
    "chili cheese fries": {"price": 4.59, "category": "side"},
    "caesar salad": {"price": 5.99, "category": "main"},
    "greek salad": {"price": 6.49, "category": "main"},
    "garden salad": {"price": 5.49, "category": "main"},
    "cobb salad": {"price": 6.99, "category": "main"},
    "spinach salad": {"price": 6.29, "category": "main"},
    "kale salad": {"price": 6.49, "category": "main"},
    "southwest salad": {"price": 6.79, "category": "main"},
    "quinoa salad": {"price": 6.59, "category": "main"},
    "caprese salad": {"price": 5.99, "category": "main"},
    "asian sesame salad": {"price": 6.39, "category": "main"},
    "margherita pizza": {"price": 9.99, "category": "main"},
    "pepperoni pizza": {"price": 10.49, "category": "main"},
    "bbq chicken pizza": {"price": 10.99, "category": "main"},
    "veggie pizza": {"price": 9.49, "category": "main"},
    "hawaiian pizza": {"price": 10.29, "category": "main"},
    "meat lovers pizza": {"price": 11.49, "category": "main"},
    "four cheese pizza": {"price": 10.19, "category": "main"},
    "mushroom pizza": {"price": 9.99, "category": "main"},
    "white pizza": {"price": 10.69, "category": "main"},
    "buffalo chicken pizza": {"price": 11.29, "category": "main"},
    "chicken tenders": {"price": 5.49, "category": "appetizer"},
    "mozzarella sticks": {"price": 4.99, "category": "appetizer"},
    "onion rings": {"price": 4.79, "category": "appetizer"},
    "jalapeño poppers": {"price": 5.29, "category": "appetizer"},
    "garlic bread": {"price": 3.99, "category": "appetizer"},
    "spinach artichoke dip": {"price": 6.49, "category": "appetizer"},
    "stuffed mushrooms": {"price": 5.99, "category": "appetizer"},
    "bruschetta": {"price": 4.59, "category": "appetizer"},
    "nachos": {"price": 6.29, "category": "appetizer"},
    "deviled eggs": {"price": 4.49, "category": "appetizer"},
    "cola": {"price": 1.99, "category": "beverage"},
    "diet cola": {"price": 1.99, "category": "beverage"},
    "lemonade": {"price": 2.49, "category": "beverage"},
    "iced tea": {"price": 2.29, "category": "beverage"},
    "sweet tea": {"price": 2.29, "category": "beverage"},
    "orange juice": {"price": 2.79, "category": "beverage"},
    "apple juice": {"price": 2.79, "category": "beverage"},
    "bottled water": {"price": 1.49, "category": "beverage"},
    "sparkling water": {"price": 1.99, "category": "beverage"},
    "root beer": {"price": 2.49, "category": "beverage"},
    "chocolate cake": {"price": 4.99, "category": "dessert"},
    "cheesecake": {"price": 5.49, "category": "dessert"},
    "brownie": {"price": 4.49, "category": "dessert"},
    "ice cream": {"price": 3.99, "category": "dessert"},
    "apple pie": {"price": 4.99, "category": "dessert"},
    "banana split": {"price": 5.99, "category": "dessert"},
    "tiramisu": {"price": 6.29, "category": "dessert"},
    "pecan pie": {"price": 4.29, "category": "dessert"},
    "chocolate chip cookie": {"price": 2.99, "category": "dessert"},
    "strawberry shortcake": {"price": 5.49, "category": "dessert"},
    "mac & cheese": {"price": 3.99, "category": "side"},
    "baked beans": {"price": 2.99, "category": "side"},
    "mashed potatoes": {"price": 3.49, "category": "side"},
    "coleslaw": {"price": 2.49, "category": "side"},
    "cornbread": {"price": 2.99, "category": "side"},
    "potato wedges": {"price": 3.49, "category": "side"},
    "steamed broccoli": {"price": 3.29, "category": "side"},
    "grilled corn": {"price": 3.79, "category": "side"},
    "side salad": {"price": 3.99, "category": "side"},
    "fruit cup": {"price": 3.69, "category": "side"},
    "iced coffee": {"price": 2.99, "category": "beverage"},
    "hot coffee": {"price": 2.49, "category": "beverage"},
    "latte": {"price": 3.99, "category": "beverage"},
    "espresso": {"price": 2.29, "category": "beverage"},
    "cappuccino": {"price": 3.49, "category": "beverage"},
    "milkshake - vanilla": {"price": 4.99, "category": "beverage"},
    "milkshake - chocolate": {"price": 5.49, "category": "beverage"},
    "milkshake - strawberry": {"price": 5.49, "category": "beverage"},
    "smoothie - berry": {"price": 4.99, "category": "beverage"},
    "smoothie - mango": {"price": 4.99, "category": "beverage"},
    "chicken alfredo": {"price": 10.49, "category": "main"},
    "beef lasagna": {"price": 11.29, "category": "main"},
    "grilled salmon": {"price": 12.49, "category": "main"},
    "steak frites": {"price": 13.99, "category": "main"},
    "vegetable stir fry": {"price": 9.49, "category": "main"},
    "shrimp scampi": {"price": 12.99, "category": "main"},
    "tofu bowl": {"price": 9.99, "category": "main"},
    "pulled pork sandwich": {"price": 8.49, "category": "main"},
    "philly cheesesteak": {"price": 10.99, "category": "main"},
    "chicken parmesan": {"price": 11.49, "category": "main"},
}

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
            f"You are a helpful and friendly AI assistant for a restaurant. Your menu includes the following items: {', '.join(MENU_ITEMS)}. "
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
