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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

parser = PydanticOutputParser(pydantic_object=Order)
menu = pd.read_csv("datafiles/testmenu100.csv")
class State(TypedDict):
    messages: Annotated[list, add_messages]
    internals: Annotated[list, add]     # using this for internal info passing
    most_recent_order: object
    cart: list
    rejected_items: list[tuple]
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
        return similar_items[:3]  # Return top 3 matches
    
    def validate_item(self, item_name: str) -> dict:
        """Comprehensive item validation"""
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

    if last_m.strip().lower() in ["extract", "conversation"]:
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
        # removal/deletion case first (keep your deletion logic as before)
        if item.delete:
            for i, added_item in enumerate(cart):
                if item.item_name.lower().strip() == added_item.item_name.lower().strip() and item.modifiers == added_item.modifiers:
                    cart[i].quantity -= item.quantity
                    break
                elif item.item_name.lower().strip() == added_item.item_name.lower().strip():
                    cart[i].quantity -= item.quantity
                    break
            continue

        
        result = menu_validator.validate_item(item.item_name)
        if result.get('found', False):
            # Exact or partial match, add to cart
            cart.append(Item(item_name=result['item'], quantity=item.quantity, modifiers=item.modifiers))
        elif result.get('similar_items'):
            for sim in result['similar_items']:
                rej_items.append((item.item_name, sim['item']))
        else:
            # No matches found at all
            rej_items.append((item.item_name, None))

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
    if not len(rej_items):
        return "summary_node"
    else:
        return "display_rejected"

    
def display_rejected(state: State):
    rej_items = state.get("rejected_items", [])

    m = AIMessage(f"The following items - {[n for (n, m) in rej_items]} are unavailable. You can try these alternatives from our menu instead: {[m for (n, m) in rej_items]}", name="display_rejected")

    return {"messages": [m]}

def save_order_to_csv(cart, filename="orders.csv"):
    # Create file with header if it doesn’t exist
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header only once
        if not file_exists:
            writer.writerow(["timestamp", "item_name", "quantity", "modifiers"])
        
        # Write each item in the order
        for item in cart:
            modifiers = ", ".join(item.modifiers) if item.modifiers else ""
            writer.writerow([datetime.now().isoformat(), item.item_name, item.quantity, modifiers])

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
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        routeFunc, 
        {
            "extract": "extract_order",
            "conversation": "menu_query"
        }
    )
    builder.add_edge("extract_order", "process_order")
    builder.add_conditional_edges(
        "process_order",
        checkRejected, 
        {
            "summary_node": "summary_node",
            "display_rejected": "display_rejected"
        }
    )

    builder.add_edge("summary_node", END)
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

    graph.update_state(config, {
        "cart": [],
        "rejected_items": []
    })

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"quit", "exit"}:
            print("Chatbot: Goodbye!")
            break
        # Add this check for confirmation:
        if user_input.lower().strip() in {"yes", "y", "confirm"}:
            cart = graph.get_state(config=config).values.get("cart", [])
            if cart:
                save_order_to_csv(cart)
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