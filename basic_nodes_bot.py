from typing import Annotated
from typing_extensions import TypedDict
import pandas as pd
import numpy as np
import pprint
import os
import csv
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from operator import add

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher


from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from promptstore import orderPrompt, conversationPrompt, agentPrompt, routerPrompt
from Classes import Item, Order
from utils import makeRetriever, get_context
from dataclasses import field
import sqlite3
import SQLFILES

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

parser = PydanticOutputParser(pydantic_object=Order)
menu = pd.read_csv("datafiles/meals.csv")
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


# Defining chains and tools

llm = init_chat_model("ollama:llama3.1")
orderChain = orderPrompt | llm | parser
conversationChain = conversationPrompt | llm
routerChain = routerPrompt | llm
retriever = makeRetriever(menu, search_type="similarity", k=10)

def router_node(state: State):
    """
    Router node to classify user input as an order or a question about the menu.
    """
    
    messages = state["messages"]
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_input = m.content
            break
    # print(f"DEBUG - PROCESSING THIS MESSAGE - {user_input}")
    response = routerChain.invoke({"user_input": [user_input]})
    # print(f"raw router response - {response}")
    # print(f".content - {response.content}")
    return {"internals": [response.content]}

def extract_order_node(state: State):
    """
    Extract structured order JSON from user input.
    """

    # print(f"DEBUG - PROCESSING THIS MESSAGE - {user_input}")
    
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
    """
    Answer questions about the menu.
    """
    
    messages = state["messages"]
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_input = m.content
            print(f"DEBUG - PROCESSING MESSAGE: ")
            break

    print(f"DEBUG - PROCESSING THIS MESSAGE - {user_input}")
    
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

    # print(internals)

    if last_m.strip().lower() in ["extract", "conversation","menu_query"]:
        return last_m.strip().lower()
    else:
        print(f"unrecognized router output - {last_m}")
        return None

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# save static version
menuembeddings = embedder.encode(menu['item_name'].tolist())
menuembeddings = menuembeddings/np.linalg.norm(menuembeddings, axis=1, keepdims=True)



def processOrder(state: State):
    mro = state["most_recent_order"]
    cart = state["cart"]
    rej_items = []

    for item in mro.items:
        # Handle deletion
        if getattr(item, "delete", False):
            deleted = False
            for i, added_item in enumerate(cart):
                # Check for exact match with item name and modifiers
                if item.item_name.lower().strip() == added_item.item_name.lower().strip() and item.modifiers == added_item.modifiers:
                    cart[i].quantity -= item.quantity
                    deleted = True
                    break
                # Fallback to match by item name only
                elif item.item_name.lower().strip() == added_item.item_name.lower().strip():
                    cart[i].quantity -= item.quantity
                    deleted = True
                    break
            if deleted:
                continue
            else:
                continue

        # Handle modification
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

        # ADD or UPDATE item
        else:
            base_name = item.item_name.lower().strip()
            result = menu_validator.validate_item(base_name)

            if result.get('found', False):
                # Item exists in menu, add or update quantity with modifiers intact
                found_existing = False
                for i, added_item in enumerate(cart):
                    if base_name == added_item.item_name.lower().strip() and item.modifiers == added_item.modifiers:
                        cart[i].quantity += item.quantity
                        found_existing = True
                        break
                if not found_existing:
                    cart.append(Item(item_name=result['item'], quantity=item.quantity, modifiers=item.modifiers))
            elif result.get('similar_items'):
                # Suggest alternatives
                rej_items.append({
                    "original_request": item.item_name,
                    "similar_items": [s['item'] for s in result['similar_items']],
                    "reason": "similar_items"
                })
            else:
                # Completely unknown item
                rej_items.append({"original_request": item.item_name, "reason": "unrecognized"})

    # Remove zero or negative quantity items
    cart = [c for c in cart if c.quantity > 0]

    print(f"Your cart is now {cart}")

    return {
        "cart": cart,
        "rejected_items": rej_items
    }

def summary_node(state: State):
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
    """If the user asks to Confirm the Order call this function"""
    cart = state.get("cart", [])
    if not cart:
        summary = "Your cart is empty."
    else:
        items = []
        for item in cart:
            # Assuming Item has item_name, quantity, modifiers
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
    rej_items = state.get("rejected_items", [])
    if not rej_items:
        return "summary_node"
   
    for item in rej_items:
        if item.get('reason') == 'similar_items':
            return "clarify_options"

    # For a completely unknown item
    return "menu_query" # Corrected the routing name

def display_rejected(state: State):
    rej_items = state.get("rejected_items", [])
    
    # Corrected logic to handle dictionary items
    unavailable_items = [item['original_request'] for item in rej_items]
    alternatives = []
    for item in rej_items:
        if 'similar_items' in item:
            alternatives.extend(item['similar_items'])
    
    m = AIMessage(f"The following items - {unavailable_items} are unavailable. You can try these alternatives from our menu instead: {alternatives}", name="display_rejected")
    return {"messages": [m]}

def clarify_options_node(state: State):
    rejected = state.get("rejected_items", [])
    if rejected:
        message = "I'm sorry, we don't have that exact item. Did you mean one of these?\n"
        for item in rejected: # Iterating over dictionaries
            original = item.get('original_request', 'N/A')
            similar = item.get('similar_items', [])
            
            if similar:
                message += f"For '{original}', you can choose from: {', '.join(similar)}\n"
            else:
                message += f"I can't find '{original}'. Is there something similar you'd like?\n"
    else:
        message = "There are no rejected items to clarify."
    
    return {"messages": [AIMessage(content=message)]}

# graph
def makegraph():
    builder = StateGraph(State)
    builder.add_node("router", router_node)
    builder.add_node("extract_order", extract_order_node)
    builder.add_node("menu_query", menu_query_node)
    builder.add_node("process_order", processOrder)
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

    # Corrected Edge: This is the critical change.
    # It now ends the stream after clarifying options.
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
# draw
# ascii_rep = graph.get_graph().draw_ascii()
# print(ascii_rep)
# graph.get_graph().draw_png("graph.png")
# os.system("open graph.png")

# state = State(messages=[])



if __name__ == "__main__":
    graph = makegraph()
    
    thread_id = "abc123"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Establish a single database connection for the entire session.
    conn = sqlite3.connect('restaurant_.db')
    
    graph.update_state(config, {
        "cart": [],
        "rejected_items": []
    })

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in {"quit", "exit"}:
                print("Chatbot: Goodbye!")
                break
            
            if user_input.lower().strip() in {"checkout", "confirm", "yes","y"}:
                cart = graph.get_state(config=config).values.get("cart", [])
                if cart:
                    # Save order using the persistent connection
                    orders_data = [(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), item.item_name, item.quantity, str(item.modifiers)) for item in cart]
                    SQLFILES.insert_orders_from_bot(conn, orders_data)
                    print("Chatbot: Order confirmed and will be sent to the Kitchen! Thank you.")
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
        # Close the connection only when the program exits the loop.
        conn.close()
        print("\nDatabase connection closed.")
                # print(f"Your cart so far - {graph.get_state(config=config).values}")

        # cp = memory.get(config)       # MemorySaver.get(thread_id)
        # if cp is None:
        #     print("DEBUG: checkpointer returned None (no checkpoint).")
        # else:
        #     # cp is a dict like {"state": {...}, "metadata": {...}}
        #     saved_state = cp.get("state", {})
        #     saved_msgs = saved_state.get("messages", [])
        #     print(f"DEBUG: checkpoint saved {len(saved_msgs)} messages.")
        #     # show last few messages and their types
        #     for i, m in enumerate(saved_msgs[-6:], start=max(0, len(saved_msgs)-6)):
        #         print(f"  [{i}] type={type(m)} repr={getattr(m,'content',repr(m))[:120]}")