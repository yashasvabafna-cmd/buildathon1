from typing import Annotated
from typing_extensions import TypedDict
import pandas as pd
import numpy as np
import pprint
import os
import csv
from datetime import datetime
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from operator import add

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher


# Removed: from langchain_core.tools import tool # No longer needed for processOrder as a direct node

from langgraph.prebuilt import create_react_agent

from promptstore import orderPrompt, conversationPrompt, agentPrompt, routerPrompt
from Classes import Item, Order # Assuming Item and Order are defined here
from utils import makeRetriever, get_context
import mysql.connector # Import mysql.connector for database operations
from inventory_depletion import deplete_inventory_from_order # NEW: Import depletion function

import warnings
warnings.filterwarnings("ignore")

# --- IMPORTANT: MySQL DB_CONFIG for basic_nodes_bot ---
# Ensure these details match your 'restaurant_new_db' setup
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',        # Your MySQL username
    'password': '12345678', # Your MySQL password
    'database': 'restaurant_new_db' # The database where 'Orders' table is
}
# ----------------------------------------------------

# Establish a single, persistent MySQL connection for the bot's session
try:
    mysql_conn = mysql.connector.connect(**DB_CONFIG)
    print("MySQL connection established for basic_nodes_bot.")
except mysql.connector.Error as err:
    print(f"Error connecting to MySQL for basic_nodes_bot: {err}")
    mysql_conn = None # Set to None if connection fails

# Check if DB connection failed
if mysql_conn is None:
    print("FATAL: Database connection failed. Bot will not be able to save orders or deplete inventory.")

parser = PydanticOutputParser(pydantic_object=Order)
menu = pd.read_csv("meals.csv")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    internals: Annotated[list, add]     # using this for internal info passing
    most_recent_order: object
    cart: list
    rejected_items: list[dict]
    menu_query: object

class MenuValidator:
    def __init__(self, menu_df):
        self.menu_df = menu_df
        # Create lowercase version for matching
        self.menu_df['item_name_lower'] = self.menu_df['item_name'].str.lower()
        self.menu_items = self.menu_df['item_name_lower'].tolist()
    
    def find_exact_match(self, item_name: str) -> dict:
        """Find exact match for item in menu"""
        item_lower = item_name.lower().strip()
        match = self.menu_df[self.menu_df['item_name_lower'] == item_lower]
        
        if not match.empty:
            return {
                'found': True,
                'item': match.iloc[0]['item_name'],
                'price': match.iloc[0]['price'],
                'match_type': 'exact'
            }
        return {'found': False}
    
    def find_partial_match(self, item_name: str) -> dict:
        """Find partial matches (contains)"""
        item_lower = item_name.lower().strip()
        
        # Check if item name contains menu item or vice versa
        for _, row in self.menu_df.iterrows():
            menu_item = row['item_name_lower']
            if (item_lower in menu_item) or (menu_item in item_lower):
                return {
                    'found': True,
                    'item': row['item_name'],
                    'price': row['price'],
                    'match_type': 'partial'
                }
        return {'found': False}
    
    def find_similar_items(self, item_name: str, threshold=0.6) -> list:
        """Find similar items using fuzzy matching"""
        item_lower = item_name.lower().strip()
        similar_items = []
        
        for _, row in self.menu_df.iterrows():
            menu_item = row['item_name_lower']
            similarity = SequenceMatcher(None, item_lower, menu_item).ratio()
            
            if similarity >= threshold:
                similar_items.append({
                    'item': row['item_name'],
                    'price': row['price'],
                    'similarity': similarity
                })
        
        # Sort by similarity (highest first)
        similar_items.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_items[:5] # Return top 5 matches
    
    def validate_item(self, item_name: str) -> dict:
        """Comprehensive item validation, """
        # Try exact match first
        exact = self.find_exact_match(item_name)
        if exact['found']:
            return exact
        
        # Try partial match
        partial = self.find_partial_match(item_name)
        if partial['found']:
            return partial
        
        # Find similar items
        similar = self.find_similar_items(item_name)
        if similar:
            return {
                'found': False,
                'similar_items': similar,
                'original_request': item_name
            }
        
        # No matches found
        return {
            'found': False,
            'original_request': item_name,
            'similar_items': []
        }

# ✅ Initialize menu validator
menu_validator = MenuValidator(menu)

# --- Helper function to get current inventory for debugging ---
def get_ingredient_current_inventory(ingredient_id, conn):
    """Fetches the current_inventory for a given ingredient_id."""
    if conn is None:
        return None
    try:
        with conn.cursor() as cursor:
            query = "SELECT ingredient_name, current_inventory, unit FROM Ingredients WHERE ingredient_id = %s;"
            cursor.execute(query, (ingredient_id,))
            result = cursor.fetchone()
            if result:
                return {"name": result[0], "inventory": result[1], "unit": result[2]}
            return None
    except mysql.connector.Error as err:
        print(f"Error fetching inventory for ingredient ID {ingredient_id}: {err}")
        return None

def insert_orders_from_bot(order_data, conn):
    """
    Saves order data from the bot's 'cart' list directly to the MySQL 'Orders' table.
    Then triggers inventory depletion and prints before/after inventory levels.
    """
    if conn is None:
        print("Error: MySQL connection not established. Cannot save order.")
        return

    try:
        with conn.cursor() as cursor:
            meal_name_to_id = {}
            try:
                cursor.execute("SELECT name, meal_id FROM Meals")
                meal_name_to_id = {name.lower(): meal_id for name, meal_id in cursor.fetchall()}
            except mysql.connector.Error as err:
                print(f"Error fetching meal_id mapping: {err}")
                return

            orders_to_insert = []
            for item in order_data:
                item_name = item.item_name
                quantity = item.quantity
                modifiers = json.dumps(item.modifiers) if item.modifiers else None
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                meal_id = meal_name_to_id.get(item_name.lower())
                
                if meal_id is not None:
                    orders_to_insert.append((meal_id, item_name, quantity, modifiers, timestamp))
                else:
                    print(f"Warning: Meal '{item_name}' not found in database. Skipping this order item.")
            
            if orders_to_insert:
                insert_query = """
                INSERT INTO Orders (meal_id, item_name, quantity, modifiers, timestamp)
                VALUES (%s, %s, %s, %s, %s);
                """
                cursor.executemany(insert_query, orders_to_insert)
                conn.commit()
                print(f"\nSuccessfully saved {len(orders_to_insert)} order items to the 'Orders' table.")
                
                # --- Pre-depletion Inventory Check ---
                print("\n--- Pre-depletion Inventory Check ---")
                ingredients_to_check = {} # {ingredient_id: (ingredient_name, unit)}
                
                # First, gather all unique ingredients involved in the *current* order from Recipe_Ingredients
                for order_item in order_data:
                    meal_id_for_item = meal_name_to_id.get(order_item.item_name.lower())
                    if meal_id_for_item:
                        recipe_query = """
                        SELECT ri.ingredient_id, i.ingredient_name, i.unit
                        FROM Recipe_Ingredients ri
                        JOIN Ingredients i ON ri.ingredient_id = i.ingredient_id
                        WHERE ri.meal_id = %s;
                        """
                        cursor.execute(recipe_query, (meal_id_for_item,))
                        for ing_id, ing_name, ing_unit in cursor.fetchall():
                            ingredients_to_check[ing_id] = {"name": ing_name, "unit": ing_unit}

                # Now fetch their current inventory levels
                ingredients_before_depletion = {} # {ingredient_id: {name, inventory, unit}}
                for ing_id, ing_data in ingredients_to_check.items():
                    inv = get_ingredient_current_inventory(ing_id, conn)
                    if inv:
                        ingredients_before_depletion[ing_id] = inv

                for ing_id, inv_data in ingredients_before_depletion.items():
                    print(f"  - BEFORE: {inv_data['name']} (ID: {ing_id}): {inv_data['inventory']} {inv_data['unit']}")


                # --- Call inventory depletion from the separate module ---
                # The deplete_inventory_from_order function itself will print detailed DEBUG messages
                deplete_inventory_from_order(order_data, conn)

                # --- Post-depletion Inventory Check ---
                print("\n--- Post-depletion Inventory Check ---")
                for ing_id, _ in ingredients_before_depletion.items(): # Use the same IDs checked before
                    inv = get_ingredient_current_inventory(ing_id, conn)
                    if inv:
                        print(f"  - AFTER: {inv['name']} (ID: {ing_id}): {inv['inventory']} {inv['unit']}")

            else:
                print("\nNo valid order items to save to the 'Orders' table.")

    except mysql.connector.Error as err:
        print(f"An error occurred while saving orders to MySQL: {err}")
    except Exception as e:
        print(f"An unexpected error occurred while saving orders: {e}")

llm = init_chat_model("ollama:llama3.1")
orderChain = orderPrompt | llm | parser
conversationChain = conversationPrompt | llm
routerChain = routerPrompt | llm
retriever = makeRetriever(menu, search_type="similarity", k=10)

def router_node(state: State):
    """Routes user input to either order extraction or menu query."""
    messages = state["messages"]
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_input = m.content
            break
    response = routerChain.invoke({"user_input": [user_input]})
    return {"internals": [response.content]}

def extract_order_node(state: State):
    """Extracts structured order JSON from user input."""
    messages = state["messages"]
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_input = m.content
            break
    
    try:
        result = orderChain.invoke({
            "user_input": user_input,
            "format_instructions": parser.get_format_instructions()
        })
        return {"internals": [AIMessage(content=result.model_dump_json(), name="extract")], "most_recent_order": result}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error parsing order: {str(e)}")]}
    
def menu_query_node(state: State):
    """Answers questions about the menu."""
    messages = state["messages"]
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_input = m.content
            break

    rel_docs, context = get_context(user_input, retriever)
    ai_response = conversationChain.invoke({
        "context": context,
        "user_input": user_input,
        "chat_history": messages
    })
    
    return {"messages": [AIMessage(content=ai_response.content)]}

def routeFunc(state: State):
    internals = state["internals"]
    last_m = internals[-1]

    if last_m.strip().lower() in ["extract", "conversation","menu_query"]:
        return last_m.strip().lower()
    else:
        print(f"unrecognized router output - {last_m}")
        return None

embedder = SentenceTransformer('all-MiniLM-L6-v2')

menuembeddings = embedder.encode(menu['item_name'].tolist())
menuembeddings = menuembeddings/np.linalg.norm(menuembeddings, axis=1, keepdims=True)


# Removed the @tool decorator for processOrder
def processOrder(state: State):
    """
    Processes the most recent order, adding, deleting, or modifying items in the cart.
    Handles item validation and suggests alternatives for unrecognized items.
    """
    mro = state["most_recent_order"]
    cart = state["cart"]
    rej_items = []

    for item in mro.items:
        if getattr(item, "delete", False):
            deleted = False
            for i, added_item in enumerate(cart):
                if item.item_name.lower().strip() == added_item.item_name.lower().strip() and item.modifiers == added_item.modifiers:
                    cart[i].quantity -= item.quantity
                    deleted = True
                    break
                elif item.item_name.lower().strip() == added_item.item_name.lower().strip():
                    cart[i].quantity -= item.quantity
                    deleted = True
                    break
            if deleted:
                continue
            else:
                continue

        elif getattr(item, "modify", False):
            modified = False
            for i, added_item in enumerate(cart):
                if item.item_name.lower().strip() == added_item.item_name.lower().strip():
                    if item.quantity is not None:
                        cart[i].quantity = item.quantity
                    if hasattr(item, "modifiers") and item.modifiers is not None:
                        cart[i].modifiers = item.modifiers
                    modified = True
                    break
            if modified:
                continue
            else:
                rej_items.append({"original_request": item.item_name, "reason": "modification_failed"})
                continue

        else:
            base_name = item.item_name.lower().strip()
            result = menu_validator.validate_item(base_name)

            if result.get('found', False):
                found_existing = False
                menu_item_name = result['item']
                for i, added_item in enumerate(cart):
                    if menu_item_name.lower().strip() == added_item.item_name.lower().strip() and item.modifiers == added_item.modifiers:
                        cart[i].quantity += item.quantity
                        found_existing = True
                        break
                if not found_existing:
                    cart.append(Item(item_name=menu_item_name, quantity=item.quantity, modifiers=item.modifiers))
            elif result.get('similar_items'):
                rej_items.append({
                    "original_request": item.item_name,
                    "similar_items": [s['item'] for s in result['similar_items']],
                    "reason": "similar_items"
                })
            else:
                rej_items.append({"original_request": item.item_name, "reason": "unrecognized"})

    cart = [c for c in cart if c.quantity > 0]

    print(f"Your cart is now {cart}")

    return {
        "cart": cart,
        "rejected_items": rej_items
    }

def summary_node(state: State):
    """Generates a summary of the current order cart."""
    cart = state.get("cart", [])
    if not cart:
        summary = "Your cart is empty."
    else:
        items = []
        for item in cart:
            item_desc = f"{item.quantity} x {item.item_name}"
            if hasattr(item, "modifiers") and item.modifiers:
                item_desc += f" ({', '.join(item.modifiers)})"
            items.append(item_desc)
        summary = "Your order:\n" + "\n".join(items)
    msg = AIMessage(content=summary, name="order_summary")
    return {"messages": [msg]}

def confirm_order(state: State):
    """Prepares a confirmation message for the user's order."""
    cart = state.get("cart", [])
    if not cart:
        summary = "Your cart is empty."
    else:
        items = []
        for item in cart:
            item_desc = f"{item.quantity} x {item.item_name}"
            if item.modifiers:
                item_desc += f" ({', '.join(item.modifiers)})"
            items.append(item_desc)
        summary = "Your order:\n" + "\n".join(items)
    msg = AIMessage(
        content=f"{summary}\n\nWould you like to confirm and place this order? (Type 'yes' to confirm)",
        name="confirm_order"
    )
    return {"messages": [msg]}

def checkRejected(state: State):
    """Checks for rejected items and routes accordingly."""
    rej_items = state.get("rejected_items", [])
    if not rej_items:
        return "summary_node"
   
    for item in rej_items:
        if item.get('reason') == 'similar_items':
            return "clarify_options"

    return "menu_query"

def display_rejected(state: State):
    """Displays information about rejected items and suggested alternatives."""
    rej_items = state.get("rejected_items", [])
    
    unavailable_items = [item['original_request'] for item in rej_items]
    alternatives = []
    for item in rej_items:
        if 'similar_items' in item:
            alternatives.extend(item['similar_items'])
    
    m = AIMessage(f"The following items - {unavailable_items} are unavailable. You can try these alternatives from our menu instead: {alternatives}", name="display_rejected")
    return {"messages": [m]}

def clarify_options_node(state: State):
    """Provides clarification options for rejected items."""
    rejected = state.get("rejected_items", [])
    if rejected:
        message = "I'm sorry, we don't have that exact item. Did you mean one of these?\n"
        for item in rejected:
            original = item.get('original_request', 'N/A')
            similar = item.get('similar_items', [])
            
            if similar:
                message += f"For '{original}', you can choose from: {', '.join(similar)}\n"
            else:
                message += f"I can't find '{original}'. Is there something similar you'd like?\n"
    else:
        message = "There are no rejected items to clarify."
    
    return {"messages": [AIMessage(content=message)]}

def makegraph():
    builder = StateGraph(State)
    builder.add_node("router", router_node)
    builder.add_node("extract_order", extract_order_node)
    builder.add_node("menu_query", menu_query_node)
    # Changed how processOrder is added to the graph to avoid Pydantic validation issues
    builder.add_node("process_order", lambda state: processOrder(state)) 
    builder.add_node("confirm_order", confirm_order)
    builder.add_node("summary_node", summary_node)
    builder.add_node("display_rejected", display_rejected)
    builder.add_node("clarify_options", clarify_options_node)
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        routeFunc, 
        {
            "extract": "extract_order",
            "conversation": "menu_query"
        }
    )

    builder.add_edge("clarify_options", END)
    builder.add_edge("extract_order", "process_order")
    builder.add_conditional_edges(
        "process_order",
        checkRejected, 
        {
            "summary_node": "summary_node",
            "clarify_options": "clarify_options",
            "menu_query": "menu_query"
        }
    )
    builder.add_edge("summary_node", "confirm_order")
    builder.add_edge("confirm_order", END)
    builder.add_edge("display_rejected", END)
    builder.add_edge("menu_query", END)

    memory = MemorySaver()

    graph = builder.compile(checkpointer=memory)
    return graph


if __name__ == "__main__":
    graph = makegraph()
    
    thread_id = "abc123"
    config = {"configurable": {"thread_id": thread_id}}
        
    graph.update_state(config, {
        "cart": [],
        "rejected_items": []
    })

    try:
        while True:
            user_input = input("You: ")

            if user_input.lower().strip() in {"checkout", "confirm", "yes", "y"}:
                current_cart = graph.get_state(config=config).values.get('cart', [])
                if current_cart:
                    insert_orders_from_bot(current_cart, mysql_conn) # Pass current_cart as order_data
                    print("\nChatbot: Order confirmed and will be sent to the Kitchen! Thank you.")
                    break
                else:
                    print("Chatbot: Your cart is empty, nothing to save.")
                    break
                        
            for update in graph.stream({"messages": [HumanMessage(user_input)]}, config=config):
                for step, output in update.items():
                    if "messages" in output:
                        for m in output["messages"]:
                            if isinstance(m, (AIMessage, ToolMessage)):
                                print(f"Chatbot: {m.content}")
            
    finally:
        if mysql_conn:
            mysql_conn.close()
            print("\nMySQL database connection closed.")
