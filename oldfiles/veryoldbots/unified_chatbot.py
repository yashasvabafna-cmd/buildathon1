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
filterwarnings("ignore")


# class defs
class Item(BaseModel):
    item_name: str
    quantity: int
    customisations: List[str] = []

class Order(BaseModel):
    items: List[Item]

parser = PydanticOutputParser(pydantic_object=Order)


menu = pd.read_csv("testmenu100.csv")

llm = ChatOllama(model="llama3")

unified_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a helpful restaurant assistant. Your job is to extract orders from customer inputs and help with information regarding the menu.

        CASE 1 - If the user is placing an order, extract it and return a JSON in this format:
        ```json
        {{
        "items": [
            {{
            "item_name": "string",
            "quantity": integer,
            "customisations": ["string", ...]
            }}
        ]
        }}

        If you are returning a JSON, do not include any additional text or explanations. The output in this case should be 100% valid JSON.

        CASE 2 - If the user is asking a question, answer it normally using the context below. Do NOT use JSON or any other structured output.

        CONTEXT:
        {context}
        END CONTEXT
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

chain = unified_prompt | llm

# context for RAG
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


chat_history = []

activeOrder = Order(items=[])


# this seems like the balance between speed and accuracy in the SentenceTransformer models
embedder = SentenceTransformer('all-MiniLM-L6-v2')

menuembeddings = embedder.encode(menu['item_name'].tolist())

while True:

    user_input = input("You: ")

    if user_input.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye! Have a great day!")
        break

    chat_history.append(HumanMessage(content=user_input))

    rel_docs, context = get_context(user_input)
    if len(rel_docs) == 0:
        # no relevant context
        print("NO CONTEXT USED/FOUND")
        context = ""

    ai_response = chain.invoke({
        "user_input":user_input,
        "chat_history":chat_history,
        "context":context
    })

    try:
        parsed = parser.parse(ai_response.content)
    except:
        print(ai_response.content)