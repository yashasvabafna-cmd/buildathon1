import numpy as np
import pandas as pd
from langchain_ollama import OllamaLLM, ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from typing import List
from langchain.output_parsers import PydanticOutputParser
from sentence_transformers import SentenceTransformer

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from warnings import filterwarnings
from langchain_core.exceptions import OutputParserException
filterwarnings("ignore")


# class defs
class Item(BaseModel):
    item_name: str
    quantity: int
    modifiers: List[str] = []

class Order(BaseModel):
    items: List[Item]

parser = PydanticOutputParser(pydantic_object=Order)

menu = pd.read_csv("testmenu100.csv")


llm = ChatOllama(model="llama3")

prompt = ChatPromptTemplate.from_messages([
    ("system", """
                1. You are an order-taking assistant. Extract the customer's order from the input. 
                2. Pay attention to modifiers like "no sugar" or "extra cheese" that will usually be written after item names.
                3. Only return valid JSON output in this format: {format_instructions}
                4. NEVER include any additional text or explanations, preceding or following the JSON.
                """),
    ("human", "{user_input}")
])

chain = prompt | llm | parser


docs = [
    Document(page_content=f"{menu.iloc[i]['item_name']} - ${menu.iloc[i]['price']} ({menu.iloc[i]['category']})", metadata=menu.iloc[i].to_dict())
    for i in range(len(menu))
]

embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedder)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

def get_context(query):
    rel_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in rel_docs])
    return rel_docs, context


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

conversationChain = conversationPrompt | llm


# instead of using this we could train a simple classification model based on embeddings of the input.

routerbot = OllamaLLM(model="llama3")

routerPrompt = PromptTemplate(
    template="""
You are a router bot that classifies a user input as either "order" or "conversation".

Rules:
- If the user is **trying to place an order**, return: "order".
  This includes requests using words like: "want", "get me", "I'd like", "can I have", "I'll take", "give me", etc. They may also order by responding "yes" to questions like "Would you like to order something?". Use chat history for these cases.
- If the user is **asking a question**, making small talk, or seeking info, return: "conversation".
- Return only one word: either "order" or "conversation". Do not explain or return anything else.

Examples:
Input: "I want a burger and fries"  
Output: order

Input: "What types of burgers do you have?"  
Output: conversation

Input: "Can I get a coke and a salad?"  
Output: order

Input: "Tell me about your menu"  
Output: conversation

Input: "I'll take a pepperoni pizza"  
Output: order

Now classify the following:

Recent chat history: {chat_history}
User input: {user_input}  
Output:
"""
)

routerChain = routerPrompt | llm

chat_history = []

activeOrder = Order(items=[])


# this seems like the balance between speed and accuracy in the SentenceTransformer models
embedder = SentenceTransformer('all-MiniLM-L6-v2')

menuembeddings = embedder.encode(menu['item_name'].tolist())

def ordertake(chain, user_input, parser):
    order = []
    print('ORDER DETECTED')
    try:
        ai_response = chain.invoke({
            "user_input": user_input,
            "format_instructions": parser.get_format_instructions()
        })

    except OutputParserException:
        print("PARSING ERROR - REROUTING TO CONVERSATION")
        return None, None


    for item in ai_response.items:
        print(item.item_name, item.quantity, item.modifiers)

        # print(f'menu embeddings shape - {menuembeddings.shape}')
    for item in ai_response.items:
        item_embedding = embedder.encode(item.item_name)
        item_embedding /= np.linalg.norm(item_embedding)
        # print(f'item embedding shape - {item_embedding.shape}')
        similarities = np.dot(menuembeddings/np.linalg.norm(menuembeddings, axis=1, keepdims=True), item_embedding)

        best_match_index = np.argmax(similarities)
        best_match = menu.iloc[best_match_index]
        score = similarities[best_match_index]
        if score < 0.6:
            print(f'No good match for {item.item_name}')
        else:
            print(f'Best match for {item.item_name}: {best_match["item_name"]}, score - {score:.4f}')
            temp_item = Item(item_name=best_match["item_name"], quantity=item.quantity, modifiers=item.modifiers)
            order.append(temp_item)

        return order, ai_response

while True:

    user_input = input("You: ")

    if user_input.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye! Have a great day!")
        break
    
    # extend chat history
    chat_history.append(HumanMessage(content=user_input))

    routerResponse = routerChain.invoke({"user_input": user_input, "chat_history": chat_history[-2:]})

    convToken = False

    if routerResponse.content.strip() == "order":
        order, ai_response = ordertake(chain, user_input, parser)
        if order or ai_response:
            activeOrder.items += order
            chat_history.append(AIMessage(content=ai_response.model_dump_json()))
        else:
            convToken = True


    elif routerResponse.content.strip() == "conversation" or convToken:
        print('CONVERSATION DETECTED')
        print(user_input)
        rel_docs, context = get_context(user_input)
        if len(rel_docs) == 0:
            # no relevant context
            print("NO CONTEXT USED/FOUND")
            context = ""
        ai_response = conversationChain.invoke({
            "context": context,
            "user_input": user_input,
            "chat_history": chat_history
        })
        print(f"Chatbot: {ai_response.content}")
        chat_history.append(AIMessage(content=str(ai_response.content)))
    else:
        print(f"Router output not recognized - {routerResponse}")
  
print(f"final order - {[item for item in activeOrder]}")