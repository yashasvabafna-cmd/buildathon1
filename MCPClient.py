
import asyncio
import pandas as pd
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

class RestaurantClient:
    def __init__(self):
        self.client = None
        self.tools = None
        self.llm = None
        self.order_history = []
        self.menu_df = None
        self.name_column = None
        self.price_column = None
        
    async def initialize(self):
        """Initialize the MCP client and tools"""
        try:
            print("🔧 Initializing MCP client...")
            
            # Load and validate CSV menu
            try:
                self.menu_df = pd.read_csv("datafiles/testmenu100.csv")
                print(f"🔧 DEBUG: CSV columns: {list(self.menu_df.columns)}")
                print(f"🔧 DEBUG: First few rows:")
                print(self.menu_df.head())
                
                # Detect column names dynamically
                columns = list(self.menu_df.columns)
                for col in columns:
                    col_lower = col.lower()
                    if 'name' in col_lower or 'item' in col_lower or 'product' in col_lower:
                        self.name_column = col
                    elif 'price' in col_lower or 'cost' in col_lower or 'amount' in col_lower:
                        self.price_column = col
                
                if self.name_column and self.price_column:
                    print(f"✅ Found name column: '{self.name_column}', price column: '{self.price_column}'")
                    self.menu_df['name_lower'] = self.menu_df[self.name_column].str.lower()
                else:
                    print(f"❌ Could not identify name/price columns in: {columns}")
                    return False
                    
            except FileNotFoundError:
                print("❌ testmenu100.csv file not found!")
                return False
            except Exception as csv_error:
                print(f"❌ Error loading CSV: {str(csv_error)}")
                return False
            
            print(f"✅ Loaded menu with {len(self.menu_df)} items")
            
            # Initialize MCP client
            self.client = MultiServerMCPClient({
                "toolkit": {
                    "command": "python",
                    "args": ["MCP_BOT.py"],
                    "transport": "stdio"
                }
            })
            
            print("🔧 Loading tools...")
            self.tools = await self.client.get_tools()
            print(f"✅ Loaded {len(self.tools)} tools")
            
            self.llm = ChatGroq(model="openai/gpt-oss-20b")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            return False
    
    def get_item_price_from_menu(self, item_name: str) -> float:
        """Look up price using detected column names"""
        try:
            if self.menu_df is None or not self.name_column or not self.price_column:
                print("🔧 DEBUG: Menu not loaded or columns not detected")
                return 10.00
            
            item_name_lower = item_name.lower()
            
            # Try exact match first
            exact_match = self.menu_df[self.menu_df['name_lower'] == item_name_lower]
            if not exact_match.empty:
                return float(exact_match.iloc[0][self.price_column])
            
            # Try partial match
            partial_match = self.menu_df[self.menu_df['name_lower'].str.contains(item_name_lower, na=False)]
            if not partial_match.empty:
                return float(partial_match.iloc[0][self.price_column])
            
            # Enhanced matching for variations
            normalized_name = item_name_lower.replace(" ", "").replace("-", "")
            for _, row in self.menu_df.iterrows():
                menu_name_normalized = str(row[self.name_column]).lower().replace(" ", "").replace("-", "")
                if normalized_name in menu_name_normalized or menu_name_normalized in normalized_name:
                    return float(row[self.price_column])
            
            print(f"🔧 DEBUG: Item '{item_name}' not found in menu, using default price")
            return 10.00
            
        except Exception as e:
            print(f"🔧 DEBUG: Error looking up price for '{item_name}': {str(e)}")
            return 10.00

    async def process_user_input(self, user_input: str) -> str:
        """Process user input and return response"""
        try:
            tool_name = self.classify_user_intent(user_input)
            print(f"🔧 Selected tool: {tool_name}")
            
            if tool_name == "menu_query":
                response = await self.call_menu_query(user_input)
            elif tool_name == "extract_order":
                response = await self.call_extract_order(user_input)
            elif tool_name == "order_summary":
                response = await self.call_order_summary()
            else:
                response = "I'm not sure how to help with that. Could you ask about our menu or place an order?"
            
            return response
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}. Please try again."
    
    def classify_user_intent(self, user_input: str) -> str:
        """Classify user intent using keyword analysis"""
        user_input_lower = user_input.lower()
        
        # Order review detection (highest priority)
        review_keywords = [
            "order", "summary", "summarise", "summarize", "confirm", "check",
            "what did i", "my order", "cart", "review", "show my", "total", "so far"
        ]
        review_phrases = [
            "what is my order", "summarise my order", "check order", 
            "confirm my order", "show order", "order summary", "what is my order so far"
        ]
        
        if (any(keyword in user_input_lower for keyword in review_keywords) or
            any(phrase in user_input_lower for phrase in review_phrases)):
            return "order_summary"
        
        # Order placement detection
        order_indicators = [
            "want", "like", "get", "buy", "purchase", "i'll have", "add", 
            "order", "give me", "i need", "can i have"
        ]
        quantity_patterns = [
            "one ", "two ", "three ", "four ", "five ", "a ", "an ",
            "1 ", "2 ", "3 ", "4 ", "5 "
        ]
        food_items = [
            "pizza", "burger", "sandwich", "drink", "coke", "water", "fries", 
            "salad", "wings", "chicken", "beef", "cheese"
        ]
        
        has_quantity = any(pattern in user_input_lower for pattern in quantity_patterns)
        has_food_item = any(item in user_input_lower for item in food_items)
        
        # Order placement logic
        if has_quantity and has_food_item:
            return "extract_order"
        if any(indicator in user_input_lower for indicator in order_indicators):
            return "extract_order"
        
        # Menu browsing (default for non-ordering queries)
        menu_keywords = ["what", "show", "have", "available", "menu", "list", "see", "types"]
        if (any(keyword in user_input_lower for keyword in menu_keywords) and
            not has_quantity and not has_food_item):
            return "menu_query"
        
        return "menu_query"  # Default fallback

    async def call_menu_query(self, user_input: str) -> str:
        """Call menu_query tool"""
        try:
            menu_tool = self._find_tool("menu_query")
            if not menu_tool:
                return "Menu query tool not available"
            
            result = await menu_tool.ainvoke({"user_input": user_input})
            return result
            
        except Exception as e:
            return "I can help you with our menu. What would you like to know?"

    async def call_extract_order(self, user_input: str) -> str:
        """✅ Handle menu validation responses"""
        try:
            order_tool = None
            for tool in self.tools:
                if tool.name == "extract_order":
                    order_tool = tool
                    break
            
            if not order_tool:
                return "Order extraction tool not available"
            
            result = await order_tool.ainvoke({"user_input": user_input})
            print(f"🔧 DEBUG - JSON Output: {result}")
            
            # Parse validation response
            try:
                import json
                response_data = json.loads(result)
                
                if response_data.get("status") == "success":
                    # Items validated successfully
                    items = response_data.get("items", [])
                    self.order_history.append(response_data)
                    
                    item_list = []
                    for item in items:
                        name = item.get("item_name", "Unknown")
                        qty = item.get("quantity", 1)
                        price = self.get_item_price_from_menu(name)
                        modifiers = item.get("modifiers", [])
                        mod_text = f" ({', '.join(modifiers)})" if modifiers else ""
                        item_list.append(f"{name} (x{qty}){mod_text} - ${price:.2f}")
                    
                    return f"✅ Added to your order:\n" + "\n".join(f"• {item}" for item in item_list) + "\n\nWould you like anything else?"
                
                elif response_data.get("status") == "items_not_found":
                    # Items not available
                    unavailable = response_data.get("unavailable_items", [])
                    suggestions = response_data.get("suggestions", [])
                    
                    response_parts = [f"❌ **Sorry, these items are not available on our menu:**"]
                    for item in unavailable:
                        response_parts.append(f"• {item}")
                    
                    if suggestions:
                        response_parts.append(f"\n💡 **Did you mean one of these?**")
                        for suggestion in suggestions:
                            response_parts.append(f"• {suggestion['item']} - ${suggestion['price']:.2f}")
                        response_parts.append("\nJust say the name of any item you'd like to add!")
                    
                    response_parts.append("\nPlease choose from our available menu items.")
                    return "\n".join(response_parts)
                
            except json.JSONDecodeError:
                return f"Error: Could not parse order response"
            
            return "Error processing your order. Please try again."
            
        except Exception as e:
            return f"Error processing order: {str(e)}"

    async def call_order_summary(self) -> str:
        """Generate order summary with real menu prices"""
        try:
            if not self.order_history:
                return "You haven't placed any orders yet. Would you like to see our menu?"
            
            all_items = []
            total_price = 0
            
            for order in self.order_history:
                items = order.get("items", [])
                for item in items:
                    name = item.get("item_name", "Unknown")
                    qty = item.get("quantity", 1)
                    item_price = self.get_item_price_from_menu(name)
                    total_item_price = item_price * qty
                    total_price += total_item_price
                    
                    modifiers = item.get("modifiers", [])
                    mod_text = f" ({', '.join(modifiers)})" if modifiers else ""
                    all_items.append(f"• {name} x{qty}{mod_text} - ${total_item_price:.2f}")
            
            summary = "📋 **Your Order Summary:**\n\n"
            summary += "\n".join(all_items)
            summary += f"\n\n**Total: ${total_price:.2f}**"
            summary += "\n\nWould you like to confirm this order or make changes?"
            
            return summary
            
        except Exception as e:
            return f"Error generating order summary: {str(e)}"

    def _find_tool(self, tool_name: str):
        """Helper method to find a tool by name"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

async def main():
    """Main conversation loop"""
    client = RestaurantClient()
    if not await client.initialize():
        return
    
    print("\n🤖 Restaurant Ordering Assistant")
    print("Type 'quit', 'exit', or 'bye' to end the conversation\n")
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("🤖 Thanks for using our ordering system! Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("🔧 Processing your request...")
            
            response = await client.process_user_input(user_input)
            print(f"🤖 Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n🤖 Conversation interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again.\n")

if __name__ == "__main__":
    asyncio.run(main())
