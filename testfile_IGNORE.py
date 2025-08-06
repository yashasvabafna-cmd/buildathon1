import numpy as np
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List
from langchain.output_parsers import PydanticOutputParser
from sentence_transformers import SentenceTransformer

from warnings import filterwarnings
filterwarnings("ignore")

class Item(BaseModel):
    item_name: str
    quantity: int
    modifiers: List[str] = []

class Order(BaseModel):
    items: List[Item]

# make test menu
menu = pd.DataFrame({'itemid': [1, 2, 3, 4, 5, 6, 7, 8], 
                     'name': ['burger', 'fries', 'soda', 'salad', 
                             'pizza', 'pasta', 'ice cream', 'coffee'],
                     'price': [5.99, 2.49, 1.99, 3.99, 7.99, 6.49, 4.49, 2.99],
                     'category': ['main', 'side', 'drink', 'side', 
                                  'main', 'main', 'dessert', 'drink']})

llm = OllamaLLM(model="llama3")
parser = PydanticOutputParser(pydantic_object=Order)


# apparently using Pydantic based on the two classes above automaticaly generates the format instructions.
prompt = PromptTemplate(
    template="""
You are an order-taking assistant. Extract the customer's order from the input. Pay attention to modifiers like "no sugar" or "extra cheese" that will usually be written after item names.
Only return valid output in this format: {format_instructions}

Do not include any additional text or explanations.

Customer: {user_input}
""",
    input_variables=["user_input"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

orderstr = "I want 1 salad, 3 peperoni pizzas, 1 diet coke, 1 coffee with extra sugar, and an ice tea."
result = chain.invoke(orderstr)

print('############# Raw Order ###############\n')
print(orderstr)
print('\n\n############# Parsed Order ###############\n')
for item in result.items:
    print(item.item_name, item.quantity, item.modifiers)


# this seems like the balance between speed and accuracy in the SentenceTransformer models
embedder = SentenceTransformer('all-MiniLM-L6-v2')

menuembeddings = embedder.encode(menu['name'].tolist())
# print(f'menu embeddings shape - {menuembeddings.shape}')
print('\n\n############# Menu Embedding Matches ###############\n')
for item in result.items:
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
        print(f'Best match for {item.item_name}: {best_match["name"]}, score - {score:.4f}')
