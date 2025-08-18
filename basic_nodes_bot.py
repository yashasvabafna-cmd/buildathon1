from typing import Annotated
from typing_extensions import TypedDict
import pandas as pd
import numpy as np
import pprint
import os

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from operator import add

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from promptstore import orderPrompt, conversationPrompt, agentPrompt, routerPrompt
from classes import Item, Order
from utils import makeRetriever, get_context
from dataclasses import field

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    internals: Annotated[list, add]     # using this for internal info passing
    most_recent_order: object
    cart: list
    rejected_items: list[tuple]

parser = PydanticOutputParser(pydantic_object=Order)
menu = pd.read_csv("datafiles/testmenu100.csv")

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
        # removal/deletion case first
        
        if item.delete:
            # check if this exact item (or item without modifiers if no exact match) is actually in the cart.
            for i, added_item in enumerate(cart):
                if item.item_name.lower().strip() == added_item.item_name.lower().strip() and item.modifiers == added_item.modifiers:
                    cart[i].quantity -= item.quantity
                    break
                elif item.item_name.lower().strip() == added_item.item_name.lower().strip():
                    # partial match. possible future change - list all partial matches and ask what to delete. could create new node "clarify_deletion"
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
            # print(f'No good match for {item.item_name}')
            # add (rejected item, most similar item in menu)
            
            rej_items.append((item.item_name, best_match["item_name"]))
        else:
            # print(f'Best match for {item.item_name}: {best_match["item_name"]}, score - {score:.4f}')
            cart.append(Item(item_name=best_match["item_name"], quantity=item.quantity, modifiers=item.modifiers))

    cart = [c for c in cart if c.quantity > 0]

    print(f"Your cart is now {cart}")
    return {
        "cart": cart,
        "rejected_items": rej_items
    }

def checkRejected(state:State):
    rej_items = state.get("rejected_items", [])

    if not len(rej_items): return "stop"
    else:
        # print(f"Chatbot: The following items - {[n for (n, m) in rej_items]} are unavailable. You can try these alternatives from our menu instead: {[m for (n, m) in rej_items]}")
        return "display_rejected"
    
def display_rejected(state: State):
    rej_items = state.get("rejected_items", [])

    m = AIMessage(f"The following items - {[n for (n, m) in rej_items]} are unavailable. You can try these alternatives from our menu instead: {[m for (n, m) in rej_items]}", name="display_rejected")

    return {"messages": [m]}

# graph
builder = StateGraph(State)
builder.add_node("router", router_node)
builder.add_node("extract_order", extract_order_node)
builder.add_node("menu_query", menu_query_node)
builder.add_node("process_order", processOrder)
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
        "stop": END,
        "display_rejected": "display_rejected"
    }
)
builder.add_edge("display_rejected", END)
builder.add_edge("menu_query", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)

# draw
ascii_rep = graph.get_graph().draw_ascii()
print(ascii_rep)
graph.get_graph().draw_png("graph.png")
os.system("open graph.png")

# state = State(messages=[])

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
    
    # state["messages"].append({"role": "human", "content": user_input})

    for update in graph.stream({"messages": [HumanMessage(user_input)]}, config=config):
        for step, output in update.items():
            if "messages" in output:
                for m in output["messages"]:
                    if isinstance(m, (AIMessage, ToolMessage)) and m.name not in ["router", "extract"]:
                        print(f"Chatbot: {m.content}")