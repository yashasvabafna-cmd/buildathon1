from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def makeRetriever(menu, search_type="similarity", k=10):    
    docs = [
        Document(page_content=f"{menu.iloc[i]['item_name']} - ₹{menu.iloc[i]['price']} ({menu.iloc[i]['category']} | {menu.iloc[i]['vegetarian']} | {menu.iloc[i]['description']} | {menu.iloc[i]['type']} | {menu.iloc[i]['cuisine']} | {menu.iloc[i]['ingredients']})", metadata={"index":i})
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

def threshold_search(query, vectorstore, emb_thresh):
    temp = vectorstore.similarity_search_with_relevance_scores(query=query, k=10, score_threshold=emb_thresh)
    res = []
    scores = []
    for doc, score in temp:
        res.append(doc.page_content)
        scores.append(score)
    return res, scores