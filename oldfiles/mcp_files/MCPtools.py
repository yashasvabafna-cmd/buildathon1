from typing import Annotated
from typing_extensions import TypedDict

import pandas as pd
import numpy as np
import os
import warnings
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from operator import add

from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

from promptstore import orderPrompt, conversationPrompt, routerPrompt
from Classes import Item, Order
from utils import makeRetriever, get_context


from dotenv import load_dotenv
import os

# Fix tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()



# -------------------
# State Definition
# -------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    internals: Annotated[list, add]  # internal info passing like router output
    most_recent_order: object
    cart: list
    rejected_items: list[tuple]

# -------------------
# Load Menu + Models
# -------------------
menu = pd.read_csv("datafiles/testmenu100.csv")

llm = ChatGroq(model="openai/gpt-oss-20b")   # Use Groq-powered LLM
parser = PydanticOutputParser(pydantic_object=Order)

# Chains
orderChain = orderPrompt | llm | parser
conversationChain = conversationPrompt | llm
routerChain = routerPrompt | llm

retriever = makeRetriever(menu, search_type="similarity", k=10)

# Vector embeddings for similarity matching
embedder = SentenceTransformer('all-MiniLM-L6-v2')
menuembeddings = embedder.encode(menu['item_name'].tolist())
menuembeddings = menuembeddings / np.linalg.norm(menuembeddings, axis=1, keepdims=True)

# -------------------
# Graph Nodes
# -------------------
def router_node(state: State):
    """Route user input → extract_order or conversation/menu query"""
    messages = state["messages"]
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_input = m.content
            break
    response = routerChain.invoke({"user_input": [user_input]})
    return {"internals": [response.content]}

def extract_order_node(state: State):
    """Extract structured order JSON."""
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
        return {"internals": [AIMessage(content=result.model_dump_json(), name="extract")],
                "most_recent_order": result}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error parsing order: {str(e)}")]}

def menu_query_node(state: State):
    """Answer menu queries / general conversation."""
    for m in state["messages"][::-1]:
        if isinstance(m, HumanMessage):
            user_input = m.content
            break

    rel_docs, context = get_context(user_input, retriever)
    ai_response = conversationChain.invoke({
        "context": context,
        "user_input": user_input,
        "chat_history": state["messages"]
    })
    print (f"DEBUG LLM response: {ai_response.content}")
    return {"messages": [AIMessage(content=ai_response.content)]}

def processOrder(state: State):
    """Validate and add items to the cart using embeddings similarity search."""
    mro = state["most_recent_order"]
    cart = state.get("cart", [])
    rej_items = []

    for item in mro.items:
        # Handle deletions (if user wants to remove items)
        if item.delete:
            for i, added_item in enumerate(cart):
                if item.item_name.lower().strip() == added_item.item_name.lower().strip():
                    cart[i].quantity -= item.quantity
                    break
            continue

        item_embedding = embedder.encode(item.item_name)
        item_embedding /= np.linalg.norm(item_embedding)

        similarities = np.dot(menuembeddings, item_embedding)
        best_match_index = np.argmax(similarities)
        best_match = menu.iloc[best_match_index]
        score = similarities[best_match_index]

        if score < 0.6:
            rej_items.append((item.item_name, best_match["item_name"]))
        else:
            cart.append(Item(item_name=best_match["item_name"],
                             quantity=item.quantity,
                             modifiers=item.modifiers))
    # Remove items with non-positive qty
    cart = [c for c in cart if c.quantity > 0]

    return {"cart": cart, "rejected_items": rej_items}

def display_rejected(state: State):
    rej_items = state.get("rejected_items", [])
    if not rej_items:
        return {"messages": [AIMessage("✅ All items successfully added to your cart!")]}
    m = AIMessage(
        f"❌ The following items are unavailable: {[n for (n, m) in rej_items]}. "
        f"Suggested alternatives: {[m for (n, m) in rej_items]}"
    )
    return {"messages": [m]}

# -------------------
# Routing Logic
# -------------------
def routeFunc(state: State):
    last_m = state["internals"][-1]
    if last_m.strip().lower() in ["extract", "conversation", "menu"]:
        return "extract_order" if last_m.strip().lower() == "extract" else "menu_query"
    return None

def checkRejected(state: State):
    rej_items = state.get("rejected_items", [])
    if not rej_items:
        return "stop"
    return "display_rejected"

# -------------------
# Graph Assembly
# -------------------
def makegraph():
    builder = StateGraph(State)

    builder.add_node("router", router_node)
    builder.add_node("extract_order", extract_order_node)
    builder.add_node("menu_query", menu_query_node)
    builder.add_node("process_order", processOrder)
    builder.add_node("display_rejected", display_rejected)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", routeFunc,
                                  {"extract_order": "extract_order",
                                   "menu_query": "menu_query"})

    builder.add_edge("extract_order", "process_order")
    builder.add_conditional_edges("process_order", checkRejected,
                                  {"stop": END, "display_rejected": "display_rejected"})

    builder.add_edge("display_rejected", END)
    builder.add_edge("menu_query", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# -------------------
# MCP Entry Point
# -------------------
graph = makegraph()
    
mcp = FastMCP("toolkit")

@mcp.tool()
def menu_query(user_input: str) -> str:
    # Initialize minimal state
    state = {"messages": [HumanMessage(user_input)],
             "internals": [],
             "most_recent_order": None,
             "cart": [],
             "rejected_items": []}
    result = graph.run(state)
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            return msg.content
    return "Sorry, I couldn't fetch the menu info."

@mcp.tool()
def extract_order(user_input: str) -> str:
    state = {"messages": [HumanMessage(user_input)],
             "internals": [],
             "most_recent_order": None,
             "cart": [],
             "rejected_items": []}
    # Run router to determine if extract_order or not
    router_out = router_node(state)
    state["internals"] = router_out.get("internals", [])
    # Force extraction node run
    extract_out = extract_order_node(state)
    state.update(extract_out)
    # Run process order
    process_out = processOrder(state)
    state.update(process_out)
    # Check rejected path
    if process_out["rejected_items"]:
        display_out = display_rejected(state)
        for msg in display_out["messages"]:
            return msg.content
    else:
        # Return the JSON string of matched order for simplicity
        matched_items = [{"item_name": i.item_name, "quantity": i.quantity, "modifiers": i.modifiers} for i in process_out["cart"]]
        return json.dumps({"status": "success", "items": matched_items})

@mcp.tool()
def order_summary(order_history_json: str) -> str:
    try:
        order_history = json.loads(order_history_json)
        if not order_history:
            return "You haven't placed any orders yet."
        # Summarize orders: sum quantities and price from menu
        total = 0
        lines = []
        for order in order_history:
            items = order.get("items", [])
            for item in items:
                name = item.get("item_name", "Unknown")
                qty = item.get("quantity", 1)
                price_row = menu[menu["item_name"] == name]
                price = float(price_row.iloc[0]["price"]) if not price_row.empty else 10.0
                lines.append(f"• {name} x{qty} - ${price*qty:.2f}")
                total += price * qty
        summary = "📋 Your Order Summary:\n" + "\n".join(lines) + f"\n\nTotal: ${total:.2f}"
        return summary
    except Exception as e:
        return f"Error generating order summary: {str(e)}"
    
if __name__ == "__main__":
    

    mcp.run("stdio")
