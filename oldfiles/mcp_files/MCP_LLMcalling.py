# MCPClient.py - COMPLETE FIXED VERSION
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from promptstore import orderPrompt, conversationPrompt, agentPrompt
from langchain_core.prompts import ChatPromptTemplate
class RestaurantClient:
    async def process_user_input(self, user_input: str) -> str:
        """✅ IMPROVED: LLM-powered tool selection"""
        try:
            # Let LLM decide which tool to use
            tool_name = await self.classify_intent_with_llm(user_input)
            print(f"🔧 Selected tool: {tool_name}")
            
            # Route to appropriate tool
            if tool_name == "menu_query":
                return await self.call_menu_query(user_input)
            elif tool_name == "extract_order":
                return await self.call_extract_order(user_input)
            elif tool_name == "order_summary":
                return await self.call_order_summary()
            else:
                return await self.handle_unclear_intent(user_input)
                
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}. Please try again."
    
    async def classify_intent_with_llm(self, user_input: str) -> str:
        """✅ NEW: Use LLM for intelligent tool selection"""
        
        tool_selection_prompt = f"""
        You are a tool selector for a restaurant ordering system. Based on the user's request, choose the most appropriate tool:

        AVAILABLE TOOLS:
        - menu_query: For questions about menu items, prices, availability, ingredients
        - extract_order: For placing orders, adding items to cart
        - order_summary: For reviewing current order, checking what was ordered

        USER REQUEST: "{user_input}"

        Consider the intent carefully. Examples:
        - "What pizzas do you have?" → menu_query
        - "I want a burger" → extract_order  
        - "What did I order?" → order_summary
        - "I want to know the price of pizza" → menu_query
        - "Can I get a coke?" → extract_order

        Respond with ONLY the tool name: menu_query, extract_order, or order_summary
        """
        
        try:
            response = await self.llm.ainvoke(tool_selection_prompt)
            tool_name = response.content.strip().lower()
            
            # Validate response
            valid_tools = ["menu_query", "extract_order", "order_summary"]
            if tool_name in valid_tools:
                return tool_name
            else:
                # Fallback to keyword matching if LLM gives invalid response
                return self.fallback_intent_classification(user_input)
                
        except Exception as e:
            print(f"🔧 DEBUG: LLM tool selection failed: {e}")
            return self.fallback_intent_classification(user_input)
    
    def fallback_intent_classification(self, user_input: str) -> str:
        """Simplified fallback for when LLM fails"""
        user_input_lower = user_input.lower()
        
        # Order review (most specific)
        if any(word in user_input_lower for word in ["my order", "summary", "ordered", "cart"]):
            return "order_summary"
        
        # Order placement (explicit intent)
        if any(word in user_input_lower for word in ["want", "get", "order", "buy", "i'll have"]):
            return "extract_order"
            
        # Default to menu query
        return "menu_query"
