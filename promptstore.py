from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

orderPrompt = ChatPromptTemplate.from_messages([
    ("system", """
                1. You are an order-taking assistant. Extract the customer's order from the input. 
                2. Pay attention to modifiers like "no sugar" or "extra cheese" that will usually be written after item names.
                3. Only return valid JSON output in this format: {format_instructions}
                4. NEVER include any additional text or explanations, preceding or following the JSON.
                """),
    ("human", "{user_input}")
])

conversationPrompt = ChatPromptTemplate(
    [
    ("system", """
    You are a helpful assistant providing information and help regarding the restaurant's menu.
    Only answer based on the menu context provided below.

    CONTEXT:
    {context}
    END CONTEXT
     """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
    ]
)

agentPrompt = ChatPromptTemplate.from_messages([
    ("system", "You are a restaurant assistant that takes orders or answers menu questions. You have access to two tools for these functions."),
    MessagesPlaceholder("messages"),
    MessagesPlaceholder("agent_scratchpad")
])