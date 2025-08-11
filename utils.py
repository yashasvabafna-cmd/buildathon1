from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def makeRetriever(menu, search_type="similarity", k=10):    
    docs = [
        Document(page_content=f"{menu.iloc[i]['item_name']} - ${menu.iloc[i]['price']} ({menu.iloc[i]['category']})", metadata=menu.iloc[i].to_dict())
        for i in range(len(menu))
    ]

    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embedder)
    retriever = db.as_retriever(search_type=search_type, search_kwargs={"k": k})
    return retriever

def get_context(query, retriever):
    rel_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in rel_docs])
    return rel_docs, context