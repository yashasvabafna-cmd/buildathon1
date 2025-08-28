import streamlit as st
import mysql.connector
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# We need to import the core LangGraph setup functions or the graph itself
# Assuming 'makegraph' from basic_nodes_bot.py correctly sets up the graph
from basic_nodes_bot import makegraph, insert_orders_from_bot
from Classes import Item, Order # Assuming Item class is defined in Classes.py
from inventory_depletion import deplete_inventory_from_order
from db_utils import get_available_menu_meals, get_unavailable_meals # Import for displaying menu after order
from dotenv import load_dotenv
import os
load_dotenv("keys.env")
# --- IMPORTANT: MySQL DB_CONFIG for Streamlit App ---
# Ensure these details match your 'restaurant_new_db' setup
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',        # Your MySQL username
    'password': '12345678', # Your MySQL password
    'database': os.getenv('DB_NAME') # The database where 'Orders' table is
}
# ----------------------------------------------------

# Streamlit Page Config
st.set_page_config(page_title="Menu Order Chatbot", page_icon="🍽️", layout="centered")
st.title("🍽️ Restaurant Order Assistant")
st.caption("Ask questions about the menu or place an order.")

def get_item_price_from_db(item_name: str, conn):
    """
    Fetches the price of an item from the 'Meals' table in the database.
    Assumes 'Meals' table has 'item_name' and 'price' columns.
    """
    if conn is None:
        return None # Cannot get price without connection
    try:
        cursor = conn.cursor()
        query = "SELECT price FROM Meals WHERE name = %s"
        cursor.execute(query, (item_name,))
        result = cursor.fetchone()
        cursor.close()
        if result:
            return float(result[0])
        return None
    except mysql.connector.Error as err:
        st.error(f"Error fetching price for {item_name}: {err}")
        return None

def display_updated_menu_for_streamlit(current_mysql_conn):
    """
    Constructs the menu string for Streamlit display,
    including newly unavailable items.
    """
    if current_mysql_conn and current_mysql_conn.is_connected():
        available_meals = get_available_menu_meals(current_mysql_conn)
        unavailable_meals = get_unavailable_meals(current_mysql_conn)

        menu_display_str = ""
        if available_meals:
            menu_display_str += "Our current menu includes:\n"
            for meal in available_meals:
                menu_display_str += f"- {meal['meal_name']}\n"
        
        if unavailable_meals:
            if available_meals:
                menu_display_str += "\n"
            menu_display_str += "Please note, the following meals are currently unavailable due to insufficient ingredients:\n"
            for meal in unavailable_meals:
                menu_display_str += f"- {meal['meal_name']}\n"
        
        if not available_meals and not unavailable_meals:
            menu_display_str = "I'm sorry, I can't retrieve the menu right now. Please try again later."
        elif not available_meals and unavailable_meals:
            menu_display_str += "\nIs there anything else I can help you with?"
        else:
            menu_display_str += "\nWhat would you like to order next?"
        
        return menu_display_str
    else:
        return "I'm sorry, I can't display the updated menu. Database connection is not available."


def initialize_session_state():
    """Initializes all necessary session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "cart" not in st.session_state:
        st.session_state.cart = [] # This will store Item objects
    if "rejected_items" not in st.session_state:
        st.session_state.rejected_items = [] # This will store dicts for rejected items
    if "mysql_conn" not in st.session_state:
        try:
            st.session_state.mysql_conn = mysql.connector.connect(**DB_CONFIG)
            st.success("✅ MySQL connection established!")
        except mysql.connector.Error as err:
            st.error(f"❌ Error connecting to MySQL: {err}. Order saving and price display will not work.")
            st.session_state.mysql_conn = None

    # Initialize LangGraph graph
    if "graph" not in st.session_state:
        if st.session_state.mysql_conn:
            # makegraph is expected to return the compiled LangGraph object
            st.session_state.graph = makegraph()
            st.session_state.thread_id = "streamlit_user_thread" # A fixed thread ID for the Streamlit user
            st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Initialize graph state for the new session
            st.session_state.graph.update_state(st.session_state.config, {
                "messages": [],
                "cart": [],
                "rejected_items": [],
                "most_recent_order": None,
                "internals": []
            })
            st.success("✅ Restaurant assistant initialized!")
        else:
            st.warning("⚠️ Cannot initialize bot without database connection.")
            st.session_state.graph = None


def process_message(user_input: str):
    """Processes user input through the LangGraph bot and updates session state."""
    if st.session_state.graph is None:
        return "The bot is not initialized. Please check the database connection."

    # Prepend HumanMessage for the stream, but don't append to session_state.messages yet
    # LangGraph will return the full message history
    current_graph_messages = st.session_state.graph.get_state(config=st.session_state.config).values.get("messages", [])
    updated_graph_messages = current_graph_messages + [HumanMessage(content=user_input)]
    
    st.session_state.messages.append(HumanMessage(content=user_input)) # Append to Streamlit's display messages


    full_response_content = ""
    ai_messages_for_display = []

    # --- Handle checkout logic with detailed feedback ---
    if user_input.lower().strip() in {"checkout", "confirm", "yes", "y"}:
        if st.session_state.cart:
            if st.session_state.mysql_conn:
                order_process_result = insert_orders_from_bot(st.session_state.cart, st.session_state.mysql_conn, deplete_inventory_from_order)
                
                if order_process_result and order_process_result["success"]:
                    confirmation_message = "Order confirmed and will be sent to the Kitchen! Thank you."
                    if order_process_result["unavailable_meals"]:
                        unavailable_names = ", ".join([m['meal_name'] for m in order_process_result["unavailable_meals"]])
                        confirmation_message += f"\nNote: The following meals are now unavailable due to ingredient shortages: {unavailable_names}."
                    
                    ai_messages_for_display.append(AIMessage(content=confirmation_message))
                    full_response_content += confirmation_message + "\n"

                    # Reset the cart and related state variables in Streamlit's session and the graph
                    st.session_state.cart = []
                    st.session_state.rejected_items = []
                    st.session_state.graph.update_state(st.session_state.config, {
                        "cart": [], 
                        "rejected_items": [], 
                        "most_recent_order": None,
                        "messages": [], # Clear graph's messages for a fresh start after order
                        "internals": []
                    })
                    
                    # Immediately display the updated menu
                    menu_str = display_updated_menu_for_streamlit(st.session_state.mysql_conn)
                    ai_messages_for_display.append(AIMessage(content=menu_str))
                    full_response_content += menu_str + "\n"

                else:
                    error_msg = f"There was an issue processing your order: {order_process_result.get('error', 'Unknown error') if order_process_result else 'Order processing failed without specific error info'}. Please try again."
                    ai_messages_for_display.append(AIMessage(content=error_msg))
                    full_response_content += error_msg + "\n"
            else:
                full_response_content = "Cannot confirm order. Database connection not established."
                ai_messages_for_display.append(AIMessage(content=full_response_content))
        else:
            full_response_content = "Your cart is empty, nothing to save or confirm."
            ai_messages_for_display.append(AIMessage(content=full_response_content))
        
        st.session_state.messages.extend(ai_messages_for_display)
        return full_response_content.strip()

    # --- Stream updates from the graph for non-checkout inputs ---
    # The graph expects the full message history in the input `messages`
    for update in st.session_state.graph.stream({"messages": updated_graph_messages}, config=st.session_state.config):
        for step, output in update.items():
            if "messages" in output:
                for m in output["messages"]:
                    if isinstance(m, (AIMessage, ToolMessage)):
                        full_response_content += m.content + "\n"
                        ai_messages_for_display.append(m)
            
    # Get the final state after the stream to update session_state.cart and rejected_items
    final_state_values = st.session_state.graph.get_state(config=st.session_state.config).values
    st.session_state.cart = final_state_values.get('cart', [])
    st.session_state.rejected_items = final_state_values.get('rejected_items', [])

    st.session_state.messages.extend(ai_messages_for_display)
    return full_response_content.strip()


def display_chat_messages():
    """Display chat messages from session state."""
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, (AIMessage, ToolMessage)):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

def display_order_summary():
    """Display current order summary in sidebar using st.session_state.cart, including prices."""
    st.sidebar.markdown("### 📋 Current Order")

    if st.session_state.cart:
        total_order_price = 0.0
        st.sidebar.markdown("---")
        for item in st.session_state.cart:
            name = item.item_name
            qty = item.quantity
            modifiers = item.modifiers if hasattr(item, 'modifiers') and item.modifiers else []
            mod_text = f" ({', '.join(modifiers)})" if modifiers else ""

            item_price = get_item_price_from_db(name, st.session_state.mysql_conn)
            if item_price is not None:
                item_total = item_price * qty
                total_order_price += item_total
                st.sidebar.markdown(f"""
                <div class="menu-item">
                    <strong>{name}</strong> x{qty}{mod_text} <span style="float:right;">${item_total:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.sidebar.markdown(f"""
                <div class="menu-item">
                    <strong>{name}</strong> x{qty}{mod_text} <span style="float:right;">Price N/A</span>
                </div>
                """, unsafe_allow_html=True)

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 0.75rem; border-radius: 5px; text-align: center;">
            <strong>Total:</strong> <span style="float:right; color: #28a745; font-weight: bold;">${total_order_price:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        st.sidebar.markdown("---")
        st.sidebar.info("Type 'checkout' or 'confirm' to place your order.")

        # Clear order button
        if st.sidebar.button("🗑️ Clear Cart", type="secondary", use_container_width=True):
            st.session_state.cart = []
            st.session_state.rejected_items = []
            if st.session_state.graph:
                st.session_state.graph.update_state(st.session_state.config, {
                    "cart": [], 
                    "rejected_items": [], 
                    "most_recent_order": None,
                    "messages": [], # Clear graph's messages
                    "internals": []
                })
            st.rerun()

    elif st.session_state.rejected_items:
        st.sidebar.warning("Some items were not recognized or are unavailable.")
        for item in st.session_state.rejected_items:
            # Assuming rejected_items are dictionaries here as set up in db_utils
            original_req = item.get('original_request', 'Unknown Item')
            similar_items = item.get('similar_items', [])
            
            st.sidebar.markdown(f"- **'{original_req}'**")
            if similar_items:
                st.sidebar.markdown(f"  *Did you mean*: {', '.join(similar_items)}")
        if st.sidebar.button("Clear Rejected Items", type="secondary", use_container_width=True):
            st.session_state.rejected_items = []
            if st.session_state.graph:
                st.session_state.graph.update_state(st.session_state.config, {"rejected_items": [], "messages": [], "internals": []}) # Clear graph's messages and internals too
            st.rerun()
    else:
        st.sidebar.info("No items in your cart yet.")


def main():
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📜 Show Menu", use_container_width=True):
                with st.spinner("Getting menu..."):
                    # Trigger the menu_query_node via a message to the graph
                    response = process_message("Show me the menu") 
                st.rerun()

        with col2:
            if st.button("📋 Order Summary", use_container_width=True):
                with st.spinner("Getting order summary..."):
                    # Trigger the summary_node via a message to the graph
                    response = process_message("Show my order summary")
                st.rerun()

        st.markdown("---")
        display_order_summary()

        st.markdown("---")
        st.markdown("### 💡 Sample Orders")
        sample_orders = [ "What paneer items do you have?", " What are the Beverages you offer? ", "I want a large pizza", "Two burgers and a coke", "I want to checkout"]

        for sample in sample_orders:
            if st.button(f"💬 {sample}\U0001F4AC", key=sample, use_container_width=True): # Added emoji
                with st.spinner("Processing..."):
                    response = process_message(sample)
                st.rerun()

    # Main chat interface
    st.markdown("### 💬 Chat with our Assistant")

    # Display chat messages
    display_chat_messages()

    # Chat input
    if prompt := st.chat_input("Type your order or question here..."):
        with st.chat_message("assistant"):
            with st.spinner("Processing your request...\U0001F916"): # Added emoji
                _ = process_message(prompt)
        st.rerun()


    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🤖 Built with Streamlit and LangGraph
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

