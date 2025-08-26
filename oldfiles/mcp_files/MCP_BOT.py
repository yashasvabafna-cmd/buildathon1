### ALL the IMPORTS
from mcp.server.fastmcp import FastMCP
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict,List
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from promptstore import orderPrompt, conversationPrompt, agentPrompt
from Classes import Item, Order
from utils import makeRetriever, get_context
import warnings
import operator
warnings.filterwarnings("ignore")
import os
from dotenv import load_dotenv
from difflib import SequenceMatcher  # ✅ For fuzzy matching
import json
import re

# Fix tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display

memory = MemorySaver()

mcp = FastMCP("toolkit")

parser = PydanticOutputParser(pydantic_object=Order)
menu = pd.read_csv("datafiles/testmenu100.csv")
llm = ChatGroq(model="llama-3.1-8b-instant")
orderChain = orderPrompt | llm | parser
conversationChain = conversationPrompt | llm
retriever = makeRetriever(menu, search_type="similarity",k=len(menu))

# ✅ NEW: Menu item finder with fuzzy matching
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

@mcp.tool()
def extract_order(user_input: str) -> str:
    """
    Extract structured order data from natural language ordering requests.
    Always returns valid JSON with a status field.
    """
    try:
        # Ask LLM to parse the order
        result = orderChain.invoke({
            "user_input": user_input,
            "format_instructions": parser.get_format_instructions()
        })

        # Convert to dict (Pydantic model → JSON → dict)
        try:
            order_data = json.loads(result.model_dump_json())
        except Exception as e:
            # As a fallback, try cleaning the raw JSON string
            try:
                cleaned = clean_json_output(str(result))
                order_data = json.loads(cleaned)
            except Exception as e2:
                return json.dumps({
                    "status": "items_not_found",
                    "unavailable_items": [user_input],
                    "suggestions": []
                })

        validated_items = []
        unavailable_items = []
        suggestions = []

        # Validate each detected item against menu
        for item in order_data.get("items", []):
            item_name = item.get("item_name", "").strip()
            qty = item.get("quantity", 1)
            modifiers = item.get("modifiers", [])

            if not item_name:
                continue

            validation = menu_validator.validate_item(item_name)
            if validation.get("found"):
                validated_items.append({
                    "item_name": validation["item"],  # normalized matched name
                    "quantity": qty,
                    "modifiers": modifiers
                })
            else:
                unavailable_items.append(item_name)
                for s in validation.get("similar_items", []):
                    suggestions.append({
                        "item": s["item"],
                        "price": s["price"]
                    })

        # SUCCESS case
        if validated_items:
            return json.dumps({
                "status": "success",
                "items": validated_items
            })

        # NOTHING matched → items_not_found
        return json.dumps({
            "status": "items_not_found",
            "unavailable_items": unavailable_items,
            "suggestions": suggestions
        })

    except Exception as e:
        # Failsafe output
        return json.dumps({
            "status": "items_not_found",
            "unavailable_items": [user_input],
            "suggestions": []
        })

def generate_validation_response(validated_items, unavailable_items, suggested_items) -> str:
    """Generate user-friendly response based on menu validation"""
    
    response_parts = []
    
    # ✅ Show successfully added items
    if validated_items:
        valid_json = {"items": [
            {
                "item_name": item["item_name"],
                "quantity": item["quantity"], 
                "modifiers": item["modifiers"]
            } for item in validated_items
        ]}
        
        added_items = []
        for item in validated_items:
            match_indicator = " ✓" if item["match_type"] == "exact" else " ≈"
            added_items.append(f"{item['item_name']} (x{item['quantity']}) - ${item['price']:.2f}{match_indicator}")
        
        response_parts.append(f"✅ **Added to your order:**\n" + "\n".join(f"• {item}" for item in added_items))
        response_parts.append(f"\n🔧 **JSON:** {json.dumps(valid_json)}")
    
    # ✅ Show unavailable items
    if unavailable_items:
        response_parts.append(f"\n❌ **Not available on our menu:**\n" + 
                            "\n".join(f"• {item}" for item in unavailable_items))
    
    # ✅ Show suggestions for unavailable items
    if suggested_items:
        response_parts.append(f"\n💡 **Did you mean one of these?**")
        unique_suggestions = {item['item']: item for item in suggested_items}.values()
        for suggestion in list(unique_suggestions)[:3]:
            response_parts.append(f"• {suggestion['item']} - ${suggestion['price']:.2f}")
        response_parts.append("\nJust say the name of any item you'd like to add!")
    
    # ✅ Add helpful closing
    if validated_items and not unavailable_items:
        response_parts.append("\n\nWould you like anything else?")
    elif unavailable_items and not validated_items:
        response_parts.append("\n\nPlease choose from our available menu items or ask to see our menu.")
    else:
        response_parts.append("\n\nWould you like to add any of the suggested items or something else?")
    
    return "\n".join(response_parts)

def clean_json_output(json_str: str) -> str:
    """✅ Clean up common JSON formatting issues"""
    try:
        import re
        import json
        
        # Remove trailing commas before closing brackets/braces
        cleaned = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Test if the cleaned version is valid
        json.loads(cleaned)
        return cleaned
        
    except Exception:
        # If cleaning fails, return a fallback structure
        return '{"items": [{"item_name": "Unknown Item", "quantity": 1, "modifiers": []}]}'

# ✅ Keep your existing tools
@mcp.tool()
def menu_query(user_input: str) -> str:
    """Answers questions about the restaurant's menu"""
    try:
        rel_docs, context = get_context(user_input, retriever)
        ai_response = conversationChain.invoke({
            "context": context,
            "user_input": user_input,
            "chat_history": []
        })
        return ai_response.content
        
    except Exception as e:
        return f"Error querying menu: {str(e)}"

@mcp.tool()
def order_summary(order_history: Annotated[list, list.__add__]) -> str:
    """Show order summary"""
    if order_history:
        most_recent_order = order_history[-1]
        return f"Your current order is: {most_recent_order}. Would you like to confirm this order?"
    else:
        return "No order found. You haven't placed any orders yet. Would you like to see our menu?"

if __name__ == "__main__":
    mcp.run(transport="stdio")
