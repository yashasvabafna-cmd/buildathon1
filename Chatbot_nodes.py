import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from basic_nodes_bot import makegraph
import json


# ---- Streamlit Page Config ----
st.set_page_config(page_title="Menu Chatbot", page_icon="🍽️", layout="centered")
st.title("🍽️ Restaurant Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.messages = []
if "cart" not in st.session_state:
    st.cart = []
if "rejected_items" not in st.session_state:
    st.rejected_items = []

graph = makegraph()
thread_id = "abc123"
config = {"configurable": {"thread_id": thread_id}}

# Load graph state from session or initialize
graph.update_state(config, {
    "cart": st.cart,
    "rejected_items": st.rejected_items,
})

# Display chat history
for msg in st.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, (AIMessage, ToolMessage)):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input, processing user input through graph
if user_input := st.chat_input("What would you like to order or ask?"):
    user_msg = HumanMessage(user_input)
    st.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(user_input)
    for update in graph.stream({"messages": [user_msg]}, config=config):
        for step, output in update.items():
            if "messages" in output:
                for m in output["messages"]:
                    # Append messages to state
                    st.messages.append(m)
                    # Special handling for ToolMessage with JSON order content
                    if isinstance(m, ToolMessage):
                        try:
                            data = json.loads(m.content)
                            if "items" in data:
                                for item in data["items"]:
                                    # Avoid duplicates
                                    if not any(
                                        item.get('item_name') == x.get('item_name') and
                                        item.get('quantity') == x.get('quantity')
                                        for x in st.cart
                                    ):
                                        st.cart.append(item)
                        except Exception:
                            pass
                    # Display bot messages
                    if isinstance(m, (AIMessage, ToolMessage)):
                        with st.chat_message("assistant"):
                            # For ToolMessages containing JSON, show clean summary instead of raw JSON
                            try:
                                parsed = json.loads(m.content)
                                if "items" in parsed:
                                    items_str = ", ".join(f"{x['item_name']} x{x['quantity']}" for x in parsed["items"])
                                    st.markdown(f"Added to cart: {items_str}")
                                    continue  # Skip raw JSON display
                            except Exception:
                                # Not JSON, just print content
                                st.markdown(m.content)

# Sidebar: display cart
with st.sidebar:
    st.header("Your Cart")
    if st.cart:
        for item in st.cart:
            st.write(f"{item.get('item_name', 'Unknown')} x{item.get('quantity', 1)}")
    else:
        st.write("Cart is empty.")

    if st.rejected_items:
        st.warning(f"Unavailable: {', '.join(str(i) for i in st.rejected_items)}")

    if st.button("Clear Conversation"):
        st.messages.clear()
        st.cart.clear()
        st.rejected_items.clear()
        st.experimental_rerun()
