import numpy as np
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from Classes import Item, Order, OrderUpdate
from langchain.output_parsers import PydanticOutputParser
from sentence_transformers import SentenceTransformer

from warnings import filterwarnings
filterwarnings("ignore")


# make test menu
# Load menu from CSV file
menu = pd.read_csv('testmenu100.csv')

llm = OllamaLLM(model="llama3")
parser = PydanticOutputParser(pydantic_object=Order)


# apparently using Pydantic based on the two classes above automaticaly generates the format instructions.
prompt = PromptTemplate(
    template="""
You are an order-taking assistant. Extract the customer's order from the input. Pay attention to modifiers like "no sugar" or "extra cheese" that will usually be written after item names. If any order is a custom order put that as another order with the modifier
Only return valid output in this format: {format_instructions}

Do not include any additional text or explanations.

Customer: {user_input}
""",
    input_variables=["user_input"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

orderstr = "I root beer and a vanilla icecream" \
""
result = chain.invoke(orderstr)

print('############# Raw Order ###############\n')
print(orderstr)
print('\n\n############# Parsed Order ###############\n')
for item in result.items:
    print(item.item_name, item.quantity, item.modifiers)


# this seems like the balance between speed and accuracy in the SentenceTransformer models
embedder = SentenceTransformer('all-MiniLM-L6-v2')

menuembeddings = embedder.encode(menu['item_name'].tolist())
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
        print(f'Best match for {item.item_name}: {best_match["item_name"]}, score - {score:.4f}')

