from typing import Annotated
from typing_extensions import TypedDict
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime
import json
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from operator import add

from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from rank_bm25 import BM25Okapi
from searchers import MultiSearch
# Removed: from langchain_community.vectorstores import FAISS # No longer needed for processOrder as a direct node

from langchain_huggingface import HuggingFaceEmbeddings

from promptstore import orderPrompt, conversationPrompt, routerPrompt
from classes import Item, Order # Assuming Item and Order are defined here
from utils import makeRetriever, get_context

trace = True
load_dotenv("keys.env")
if trace:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = "default"
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
    rejected_items: list[tuple]

menu_searcher = MultiSearch(menu, bm_thresh= 0.01)

# Defining chains and tools
LLM_NAME="gpt-oss-120b-groq"

if LLM_NAME == "llama-local":
    llm = init_chat_model("ollama:llama3.1")
elif LLM_NAME == "gpt-oss-120b-groq":
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model='openai/gpt-oss-120b')

orderChain = orderPrompt | llm | parser
conversationChain = conversationPrompt | llm
routerChain = routerPrompt | llm

retriever = makeRetriever(menu, search_type="similarity", k=10)
corpus = list(menu["item_name"])
tcorpus = [c.lower().split() for c in corpus]
bm_searcher = BM25Okapi(tcorpus)
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.from_texts(corpus, embedder)
emb_thresh=0.5
seq_thresh=0.5

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
        print("order parsing error!")
        return {"messages": [AIMessage(content=f"Error parsing order: {str(e)}")]}
    
def menu_query_node(state: State):
    """Answers questions about the menu."""
    messages = state["messages"]
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_input = m.content
            # print(f"DEBUG - PROCESSING MESSAGE: ")
            break

    # print(f"DEBUG - PROCESSING THIS MESSAGE - {user_input}")
    
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

# save static version
# menuembeddings = embedder.encode(menu['item_name'].tolist())
# menuembeddings = menuembeddings/np.linalg.norm(menuembeddings, axis=1, keepdims=True)

import numpy as np
from difflib import SequenceMatcher

def cosine_similarity(query_emb, doc_embs):
    q = query_emb / np.linalg.norm(query_emb)
    d = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    return np.dot(d, q)

def deleteOrder(state: State, seq_thresh=0.6):
    mro = state["most_recent_order"]
    cart = state["cart"]
    rej_items = []

    # sequenceMatch function (pulled from MultiSearch)
    def sequenceMatch(item_name, seq_threshold, items):
        item_lower = item_name.lower().strip()
        res, scores = [], []
        for opt in items:
            similarity = SequenceMatcher(None, item_lower, opt.lower()).ratio()
            if similarity >= seq_threshold:
                res.append(opt)
                scores.append(similarity)
        if res:
            return {'found': True, 'items': res, 'scores': scores}
        return {'found': False, 'items': [], 'scores': []}

    for item in mro.delete:
        cart_names = [c.item_name for c in cart]
        if not cart_names:
            continue

        # 1. exact match
        if item.item_name.lower().strip() in [c.lower() for c in cart_names]:
            target_names = [item.item_name]
        else:
            # 2. sequence matching
            seq = sequenceMatch(item.item_name, seq_thresh, cart_names)

            # 3. embedding cosine similarity
            query_emb = np.array(embedder.embed_query(item.item_name))
            cart_embs = np.array(embedder.embed_documents(cart_names))
            sims = cosine_similarity(query_emb, cart_embs)

            # classify matches
            certain_match_seq = [n for n, s in zip(seq["items"], seq["scores"]) if s >= 0.8]
            certain_match_emb = [n for n, s in zip(cart_names, sims) if s >= 0.85]
            certain_set = list(set(certain_match_seq + certain_match_emb))

            if len(certain_set) == 1:
                target_names = certain_set
            elif len(certain_set) > 1:
                print(f"Multiple matches found for '{item.item_name}':")
                for i, opt in enumerate(certain_set, 1):
                    print(f"{i}. {opt}")
                try:
                    choice = int(input("Which one would you like to remove? ")) - 1
                    if 0 <= choice < len(certain_set):
                        target_names = [certain_set[choice]]
                    else:
                        print("Invalid choice. Skipping this item.")
                        continue
                except ValueError:
                    print("Invalid input. Skipping this item.")
                    continue
            else:
                good_match_seq = [n for n, s in zip(seq["items"], seq["scores"]) if s >= 0.6]
                good_match_emb = [n for n, s in zip(cart_names, sims) if s >= 0.5]
                good_set = list(set(good_match_seq + good_match_emb))

                if len(good_set) == 1:
                    target_names = good_set
                elif len(good_set) > 1:
                    print(f"Possible matches for '{item.item_name}':")
                    for i, opt in enumerate(good_set, 1):
                        print(f"{i}. {opt}")
                    try:
                        choice = int(input("Which one would you like to remove? ")) - 1
                        if 0 <= choice < len(good_set):
                            target_names = [good_set[choice]]
                        else:
                            print("Invalid choice. Skipping this item.")
                            continue
                    except ValueError:
                        print("Invalid input. Skipping this item.")
                        continue
                else:
                    # rejection case → suggest closest by embedding
                    maxidx = np.argmax(sims)
                    rej_items.append((item.item_name, cart_names[maxidx]))
                    continue

        # perform deletion
        for target in target_names:
            for i, added_item in enumerate(cart):
                if target.lower().strip() == added_item.item_name.lower().strip() and item.modifiers == added_item.modifiers:
                    cart[i].quantity -= item.quantity
                    deleted = True
                    break
                elif target.lower().strip() == added_item.item_name.lower().strip():
                    cart[i].quantity -= item.quantity
                    deleted = True
                    break

    cart = [c for c in cart if c.quantity > 0]
    return {"cart": cart, "rejected_items": rej_items}



def processOrder(state: State):
    mro = state["most_recent_order"]
    cart = state["cart"]
    rej_items = []
        
    new_messages = []
    print(f"mro items - {mro.items}")
    print(f"mro delete - {mro.delete}")
    for item in mro.items:
        # pass something to internal for each of the 3 scenarios so you can make conditional edges for all 3 later.
        # print(type(mro), type(item), type(mro.model_dump_json()))
        result = menu_searcher.unify(item.item_name, bm_searcher=bm_searcher, vectordb=vectordb, emb_thresh=emb_thresh, seq_thresh=seq_thresh)
        
        if result.get('exact', False):
            # Exact match
            cart.append(Item(item_name=result['item'], quantity=item.quantity, modifiers=item.modifiers))
        else:
            # no exact
            # 3 scenarios
            # 1. one very good match -> add directly to cart
            # 2. multiple good matches -> ask for clarification
            # 3. no good matches -> reject, and show best alternative (maybe use metadata based retriever which will be used by menu query)

            # certain match - 0.85 emb, 0.8 seq
            # good match - 0.5 emb, 0.6 seq
            # bad match - everything else

            # one certain match
            seq = result["seq"]
            emb = result["emb"]

            # print(seq)

            certain_match_seq = [item for item, score in zip(seq["items"], seq["scores"]) if score >= 0.8]
            certain_match_emb = [item for item, score in zip(emb["items"], emb["scores"]) if score >= 0.85]

            certain_set = set(certain_match_seq + certain_match_emb)
            if len(certain_set) == 1:
                # add to cart
                item_name = list(certain_set)[0]
                cart.append(Item(item_name=item_name, quantity=item.quantity, modifiers=item.modifiers))
                continue

            elif len(certain_set) > 1:
                # clarify -> go to clarify node to display options. or call a clarify function to do it here itself dont need another node?
                s = f"We have the following options related to {item.item_name} -\n" + "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(certain_set))
                new_messages.append(AIMessage(s))
                continue
            
            # no certains if code reaches here
            
            good_match_seq = [item for item, score in zip(seq["items"], seq["scores"]) if score >= 0.6]
            good_match_emb = [item for item, score in zip(emb["items"], emb["scores"]) if score >= 0.5]
            good_set = set(good_match_emb + good_match_seq)

            if len(good_set) != 0:
                # clarification
                s = f"We have the following options related to {item.item_name} -\n" + "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(good_set))
                new_messages.append(AIMessage(s))
                continue
            else:
                # bad. rejection logic.
                similars = menu_searcher.embeddingSearch(item.item_name, vectordb=vectordb, emb_thresh=0)
                # if not similars["found"]:
                #     new_messages.append(AIMessage())

                maxidx = np.argmax(similars["scores"])
                rej_items.append((item.item_name, similars["items"][maxidx]))

    
    print(f"Your cart is now {cart}")

    return {
        "messages": new_messages,
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
        content=f"{summary}\n\nWould you like anything else? To confirm and place your order, enter 'yes'.",
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
    builder.add_node("process_order", processOrder)
    builder.add_node("delete_order", deleteOrder)
    builder.add_node("confirm_order", confirm_order)
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
    
    builder.add_edge("extract_order", "delete_order")
    builder.add_edge("delete_order", "process_order")
    builder.add_conditional_edges(
        "process_order",
        checkRejected, 
        {
            "summary_node": "confirm_order",
            "display_rejected": "display_rejected"
        }
    )
    builder.add_edge("confirm_order", END)
    builder.add_edge("display_rejected", END)
    builder.add_edge("menu_query", END)

    memory = MemorySaver()

    graph = builder.compile(checkpointer=memory)
    return graph


if __name__ == "__main__":
    graph = makegraph()
    draw = True

    if draw:
        ascii_rep = graph.get_graph().draw_ascii()
        print(ascii_rep)
        graph.get_graph().draw_png("graph.png")
        os.system("open graph.png")

    thread_id = "abc1234"
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
