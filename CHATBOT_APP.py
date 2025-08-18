import streamlit as st
import asyncio
import pandas as pd
import json
from datetime import datetime
import sys
import os

# Add the current directory to Python path to import your modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MCPClient import RestaurantClient

# Configure Streamlit page
st.set_page_config(
    page_title="Restaurant Ordering Assistant",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "client" not in st.session_state:
    st.session_state.client = None
if "client_initialized" not in st.session_state:
    st.session_state.client_initialized = False
if "order_history" not in st.session_state:
    st.session_state.order_history = []

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #ff6b35;
    margin-bottom: 2rem;
}

.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.user-message {
    background-color: #e8f4f8;
    border-left: 4px solid #2196F3;
}

.assistant-message {
    background-color: #f0f8f0;
    border-left: 4px solid #4CAF50;
}

.order-summary {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #ffeaa7;
    margin: 1rem 0;
}

.menu-item {
    padding: 0.5rem;
    margin: 0.2rem 0;
    background-color: #f8f9fa;
    border-radius: 5px;
    border-left: 3px solid #28a745;
}
</style>
""", unsafe_allow_html=True)

async def initialize_client():
    """Initialize the restaurant client"""
    try:
        client = RestaurantClient()
        success = await client.initialize()
        if success:
            st.session_state.client = client
            st.session_state.client_initialized = True
            return True
        return False
    except Exception as e:
        st.error(f"Failed to initialize client: {str(e)}")
        return False

async def process_message(user_input: str):
    """Process user message and get response"""
    try:
        if not st.session_state.client:
            return " Client not initialized. Please refresh the page."
        
        response = await st.session_state.client.process_user_input(user_input)
        
        # Update order history from client
        if hasattr(st.session_state.client, 'order_history'):
            st.session_state.order_history = st.session_state.client.order_history
        
        return response
    except Exception as e:
        return f" Error processing message: {str(e)}"

def display_chat_messages():
    """Display chat messages"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_order_summary():
    """Display current order summary in sidebar"""
    if st.session_state.order_history:
        st.sidebar.markdown("### 📋 Current Order")
        
        all_items = []
        total_price = 0
        
        try:
            for order in st.session_state.order_history:
                items = order.get("items", [])
                for item in items:
                    name = item.get("item_name", "Unknown")
                    qty = item.get("quantity", 1)
                    
                    # Get price from client if available
                    if st.session_state.client:
                        item_price = st.session_state.client.get_item_price_from_menu(name)  
                    
                    total_item_price = item_price * qty
                    total_price += total_item_price
                    
                    modifiers = item.get("modifiers", [])
                    mod_text = f" ({', '.join(modifiers)})" if modifiers else ""
                    
                    all_items.append({
                        "name": name,
                        "quantity": qty,
                        "modifiers": mod_text,
                        "price": total_item_price
                    })
            
            # Display items
            for item in all_items:
                st.sidebar.markdown(f"""
                <div class="menu-item">
                    <strong>{item['name']}</strong> x{item['quantity']}{item['modifiers']}<br>
                    <span style="color: #28a745; font-weight: bold;">${item['price']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.sidebar.markdown(f"""
            <div style="background-color: #28a745; color: yellow; padding: 1rem; border-radius: 8px; text-align: center; margin-top: 1rem;">
                <strong>Total: ${total_price:.2f}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Clear order button
            if st.sidebar.button("🗑️ Clear Order", type="secondary", use_container_width=True):
                st.session_state.order_history = []
                if st.session_state.client:
                    st.session_state.client.order_history = []
                st.rerun()
                
        except Exception as e:
            st.sidebar.error(f"Error displaying order: {str(e)}")
    else:
        st.sidebar.markdown("### 📋 Current Order")
        st.sidebar.info("No items in your order yet.")

def main():
    # Header
    st.markdown('<h1 class="main-header">🍕 Restaurant Ordering Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize client if not done
    if not st.session_state.client_initialized:
        with st.spinner("🔧 Initializing restaurant assistant..."):
            success = asyncio.run(initialize_client())
            if not success:
                st.error(" Failed to initialize the restaurant assistant. Please check your setup and refresh the page.")
                st.stop()
        st.success("✅ Restaurant assistant initialized successfully!")
        st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📜 Show Menu", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Show me the menu"})
                with st.spinner("Getting menu..."):
                    response = asyncio.run(process_message("Show me the menu"))
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        with col2:
            if st.button("📋 Order Summary", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Show my order summary"})
                with st.spinner("Getting order summary..."):
                    response = asyncio.run(process_message("Show my order summary"))
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.markdown("---")
        display_order_summary()
        
        st.markdown("---")
        st.markdown("### 💡 Sample Orders")
        sample_orders = [
            "I want a large pizza",
            "Two burgers and a coke",
            "Show me your drinks",
            "Add fries to my order"
        ]
        
        for sample in sample_orders:
            if st.button(f"💬 {sample}", key=sample, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": sample})
                with st.spinner("Processing..."):
                    response = asyncio.run(process_message(sample))
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Main chat interface
    st.markdown("### 💬 Chat with our Assistant")
    
    # Display chat messages
    display_chat_messages()
    
    # Chat input
    if prompt := st.chat_input("Type your order or question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                response = asyncio.run(process_message(prompt))
            st.markdown(response)
        
        # Add assistant response to messages
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the order summary
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🤖 Powered by MCP (Model Context Protocol) | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
