from typing import Annotated
from typing_extensions import TypedDict
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime
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
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from promptstore import orderPrompt, conversationPrompt, routerPrompt
from classes import Item, Order
from utils import makeRetriever, get_context

trace = True
load_dotenv("keys.env")
if trace:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = "default"

import warnings
warnings.filterwarnings("ignore")

parser = PydanticOutputParser(pydantic_object=Order)
# print(parser.get_format_instructions())
menu = pd.read_csv("datafiles/testmenu100.csv")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    internals: Annotated[list, add]     # using this for internal info passing
    most_recent_order: object
    cart: list
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
        print("order parsing error!")
        return {"messages": [AIMessage(content=f"Error parsing order: {str(e)}")]}
    
def menu_query_node(state: State):
    """
    Answer questions about the menu.
    """
    
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

    # print(internals)

    if last_m.strip().lower() in ["extract", "conversation"]:
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
                    break
                elif target.lower().strip() == added_item.item_name.lower().strip():
                    cart[i].quantity -= item.quantity
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
        content=f"{summary}\n\nWould you like anything else? To confirm and place your order, enter 'yes'.",
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
    builder.add_node("delete_order", deleteOrder)
    builder.add_node("confirm_order", confirm_order)
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
                print("Chatbot: Your cart was empty, no order has been placed. Goodbye.")
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