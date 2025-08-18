import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from basic_nodes_bot import makegraph
import json

# ---- Streamlit Page Config ----
st.set_page_config(page_title="Menu Order Chatbot", page_icon="🍽️", layout="centered")
st.title("🍽️ Restaurant Order Assistant")
st.caption("Ask questions about the menu or place an order.")

# ---- Initialize Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cart" not in st.session_state:
    st.session_state.cart = []
if "rejected_items" not in st.session_state:
    st.session_state.rejected_items = []

graph = makegraph()
thread_id = "abc123"
config = {"configurable": {"thread_id": thread_id}}
graph.update_state(config, {
    "cart": st.session_state.cart,
    "rejected_items": st.session_state.rejected_items
})

# ------ Show Chat History ------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, (AIMessage, ToolMessage)):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ---- Chat Input -----
if user_input := st.chat_input("What would you like to order or ask?"):
    user_msg = HumanMessage(user_input)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(user_input)

    # PROCESS AND UPDATE CART IN THIS RUN
    for update in graph.stream({"messages": [user_msg]}, config=config):
        for step, output in update.items():
            if "messages" in output:
                for m in output["messages"]:
                    if isinstance(m, ToolMessage):
                        try:
                            data = json.loads(m.content)
                            if "items" in data:
                                for item in data["items"]:
                                    already_in_cart = any(
                                        (i.get('item_name') == item.get('item_name') and i.get('quantity') == item.get('quantity'))
                                        for i in st.session_state.cart
                                    )
                                    if not already_in_cart:
                                        st.session_state.cart.append(item)
                                # Show summary to user instead of raw JSON
                                items_list = ", ".join(f"{i['item_name']} x{i['quantity']}" for i in data['items'])
                                with st.chat_message("assistant"):
                                    st.markdown(f"Added to cart: {items_list}")
                                continue  # Skip raw JSON display

                        except Exception:
                            pass
                    st.session_state.messages.append(m)
                    if isinstance(m, AIMessage):
                        with st.chat_message("assistant"):
                            st.markdown(m.content)
    # END FOR

# ---- Sidebar ----
with st.sidebar:
    st.subheader("🛒 Your Cart")
    if st.session_state.cart:
        for item in st.session_state.cart:
            name = item.get('item_name', 'Unknown Item')
            qty = item.get('quantity', 1)
            st.write(f"- {name} x {qty}")
    else:
        st.write("Cart is empty.")

    if st.session_state.rejected_items:
        st.warning(f"Unavailable: {st.session_state.rejected_items}")

    if st.button("Clear Conversation"):
        st.session_state.clear()
        st.rerun()