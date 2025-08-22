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

from promptstore import (
    orderPrompt,
    conversationPrompt,
    agentPrompt,
    routerPrompt,
    PromptTemplate,
)

from Classes import Item, Order

from utils import makeRetriever, get_context

from dataclasses import field

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")

# Define State

class State(TypedDict):

    messages: Annotated[list, add_messages]

    internals: Annotated[list, add]  # used for internal routing signals

    most_recent_order: object

    cart: list

    rejected_items: list

# Load menu

menu = pd.read_csv("datafiles/testmenu100.csv")

# Initialize LLM and Chains

llm = init_chat_model("ollama:llama3.1")

# Intent classification prompt and chain

intentClassificationPrompt = PromptTemplate(

    input_variables=["user_input"],

    template=(

        "Classify if the following user input is an 'order' or 'conversation' about the menu.\n"

        "Respond ONLY with one word: 'order' or 'conversation'.\n"

        "User Input: {user_input}\n"

        "Answer:"

    ),

)
parser=PydanticOutputParser(pydantic_object=Order)
intentClassificationChain = intentClassificationPrompt | llm

orderChain = orderPrompt | llm | parser

conversationChain = conversationPrompt | llm

routerChain = routerPrompt | llm

retriever = makeRetriever(menu, search_type="similarity", k=10)

# Create embedding for menu matching

embedder = SentenceTransformer('all-MiniLM-L6-v2')

menu_embeddings = embedder.encode(menu['item_name'].tolist())

menu_embeddings = menu_embeddings / np.linalg.norm(menu_embeddings, axis=1, keepdims=True)

# Router node: classify the current input and return routing signal

def router_node(state: State):

    messages = state["messages"]

    for m in reversed(messages):

        if isinstance(m, HumanMessage):

            user_input = m.content.strip()

            break

    user_input_lc = user_input.lower()

    # Handle explicit commands first

    if user_input_lc in {"confirm", "done", "that's it", "place order"}:

        return {"internals": ["summary_node"]}

    if user_input_lc in {"yes", "y", "confirm"}:

        return {"internals": ["confirm_order"]}

    if user_input_lc in {"no", "cancel"}:

        return {"internals": ["cancel_order"]}

    # Use LLM intent classifier

    classification = intentClassificationChain.invoke({"user_input": user_input})

    intent = classification.content.strip().lower()

    if intent == "order":

        return {"internals": ["extract_order"]}

    if intent == "conversation":

        return {"internals": ["menu_query"]}

    # Fallback

    return {"internals": ["menu_query"]}

# Extract order node: parse order from user input

def extract_order_node(state: State):

    messages = state["messages"]
    
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_input = m.content
            break

    try:
        result = orderChain.invoke({
            "user_input": user_input,
            "format_instructions": parser.get_format_instructions()
        })
        return {"internals": ["extract_order_done"], "most_recent_order": result}
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error parsing order: {str(e)}")],
            "internals": ["extract_order"]
        }

# Menu query node: answer questions about menu

def menu_query_node(state: State):

    messages = state["messages"]

    for m in reversed(messages):

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

# Process order node: check and add items to cart

def process_order(state: State):

    mro = state.get("most_recent_order")
    
    if mro is None:
        return {
            "messages": [
                AIMessage(content="I couldn't understand your order. Please try rephrasing it or ask about the menu!")
            ],
            "internals": ["extract_order"]  # Route user back to try order again
        }

    cart = state.get("cart", [])

    rej_items = []

    for item in mro.items:

        if item.delete:

            for i, citem in enumerate(cart):

                if citem.item_name.lower() == item.item_name.lower() and citem.modifiers == item.modifiers:

                    citem.quantity -= item.quantity

                    if citem.quantity <= 0:

                        cart.pop(i)

                    break

        else:

            # Calculate best match

            embedding = embedder.encode([item.item_name])[0]

            embedding = embedding / np.linalg.norm(embedding)

            similarities = np.dot(menu_embeddings, embedding)

            best_idx = np.argmax(similarities)

            score = similarities[best_idx]

            if score < 0.6:

                rej_items.append((item.item_name, menu.iloc[best_idx].item_name))

            else:

                # Add or update cart

                found = False

                for citem in cart:

                    if citem.item_name == menu.iloc[best_idx].item_name and citem.modifiers == item.modifiers:

                        citem.quantity += item.quantity

                        found = True

                        break

                if not found:

                    cart.append(Item(

                        item_name=menu.iloc[best_idx].item_name,

                        quantity=item.quantity,

                        modifiers=item.modifiers,

                        delete=False

                    ))

    state["cart"] = [item for item in cart if item.quantity > 0]

    state["rejected_items"] = rej_items

    return {

        "cart": state["cart"],

        "rejected_items": rej_items,

        "internals": ["process_done"],

        "messages": [

            AIMessage(content="Your order has been updated. Would you like to add more, remove items, or confirm the order?")

        ]

    }

# Summary node: show cart summary and prompt for confirmation

def summary_node(state: State):

    cart = state.get("cart", [])

    if not cart:

        summary = "Your cart is empty."

    else:

        lines = []

        for item in cart:

            desc = f"{item.quantity} x {item.item_name}"

            if item.modifiers:

                desc += f" ({', '.join(item.modifiers)})"

            lines.append(desc)

        summary = "Your order:\n" + "\n".join(lines)

    return {"messages": [AIMessage(content=f"{summary}\nWould you like to place the order? (Type 'yes' to confirm)")]}

# Confirm order node

def confirm_order(state: State):

    cart = state.get("cart", [])

    if not cart:

        return {"messages": [AIMessage(content="Your cart is empty. No order to confirm.")]}

    lines = []

    for item in cart:

        desc = f"{item.quantity} x {item.item_name}"

        if item.modifiers:

            desc += f" ({', '.join(item.modifiers)})"

        lines.append(desc)

    summary = "Your order:\n" + "\n".join(lines)

    return {"messages": [AIMessage(content=f"{summary}\nThank you! Your order has been confirmed.")]}

# Check rejected items node

def checkRejected(state: State):

    rej_items = state.get("rejected_items", [])

    if not rej_items:

        return "summary_node"

    else:

        return "display_rejected"

# Display rejected items node

def display_rejected(state: State):

    rej_items = state.get("rejected_items", [])

    if not rej_items:

        return {"messages": [AIMessage(content="All items are available.")]}

    lines = [f"'{r[0]}' is not available. Suggested alternative: '{r[1]}'." for r in rej_items]

    msg = "Some items were not available.\n" + "\n".join(lines)

    return {"messages": [AIMessage(content=msg)]}

# Routing function

def routeFunc(state: State):

    internals = state.get("internals", [])

    if not internals:

        print("router: internals is empty, defaulting to menu_query")

        return "menu_query"

    last_intent = internals[-1]

    if last_intent in ["extract_order", "extract_order_done"]:

        return "extract_order"

    if last_intent == "process_done":

        return "summary_node"

    if last_intent in ["summary_node", "confirm_order", "cancel_order", "display_rejected"]:

        return last_intent

    if last_intent in ["menu_query", "conversation"]:

        return "menu_query"

    # Default fallback

    print(f"router: unrecognized internal '{last_intent}', defaulting to menu_query")

    return "menu_query"

# Build graph

def makegraph():

    builder = StateGraph(State)

    builder.add_node("router", router_node)

    builder.add_node("extract_order", extract_order_node)

    builder.add_node("menu_query", menu_query_node)

    builder.add_node("process_order", process_order)

    builder.add_node("summary_node", summary_node)

    builder.add_node("confirm_order", confirm_order)



    builder.add_node("display_rejected", display_rejected)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(

        "router",

        routeFunc,

        {

            "extract_order": "extract_order",

            "extract_order_done": "process_order",

            "process_done": "summary_node",

            "summary_node": "summary_node",

            "confirm_order": "confirm_order",

            "display_rejected": "display_rejected",

            "menu_query": "menu_query",

            "conversation": "menu_query",

        },

    )

    builder.add_edge("extract_order", "process_order")

    builder.add_conditional_edges(

        "process_order",

        checkRejected,

        {

            "summary_node": "summary_node",

            "display_rejected": "display_rejected",

        },

    )

    builder.add_edge("summary_node", "confirm_order")

    builder.add_edge("confirm_order", END)

    builder.add_edge("display_rejected", END)

    builder.add_edge("menu_query", END)

    return builder.compile(checkpointer=MemorySaver())

# Main run loop

if __name__ == "__main__":

    graph = makegraph()

    config = {"configurable": {"thread_id": "default"}}

    graph.update_state(config, {

        "cart": [],

        "rejected_items": []

    })

    while True:

        user_input = input("You: ")

        if user_input.lower() in {"quit", "exit"}:

            print("Chatbot: Goodbye!")

            break

        if user_input.lower().strip() in {"yes", "y", "confirm"}:

            print("Chatbot: Your order has been confirmed! Thank you.")

            break

        for update in graph.stream({"messages": [HumanMessage(user_input)]}, config=config):

            for v in update.values():

                if v is None:

                    continue

                if "messages" in v:

                    for message in v["messages"]:

                        if isinstance(message, (AIMessage, ToolMessage)):

                            print("Chatbot:", message.content)
