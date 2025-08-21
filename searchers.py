import numpy as np
from utils import threshold_search
from difflib import SequenceMatcher

class MultiSearch:

    # lexical matching -> exact + partial (sequence or BM25)
    # semantic matching -> vector db
    # rerank with sentence transformer

    def __init__(self, df, bm_thresh):
        self.menu_df = df
        self.menu_df['item_name_lower'] = self.menu_df['item_name'].str.lower()
        self.menu_items = self.menu_df['item_name_lower'].tolist()
        self.bm_thresh = bm_thresh
    
    def find_exact_match(self, item_name: str) -> dict:
        """Find exact match for item in menu"""
        item_lower = item_name.lower().strip()
        match = self.menu_df[self.menu_df['item_name_lower'] == item_lower]
        
        if not match.empty:
            return {
                'found': True,
                'items': match.iloc[0]['item_name'],
                'match_type': 'exact'
            }
        return {'found': False}
    
    def bm25_search(self, item_name, bm_searcher):
        """Use BM25 Okapi algorithm and return results above some threshold (post-softmax)."""

        tkns = item_name.lower().split(" ")
        doc_scores = bm_searcher.get_scores(tkns)
        
        # softmax
        e = np.exp(doc_scores)
        doc_scores = e/sum(e)

        # threshold
        res = self.menu_df.iloc[np.where(doc_scores > self.bm_thresh)]
        scores = doc_scores[np.where(doc_scores > self.bm_thresh)]

        assert len(res) == len(scores)

        if not res.empty:
            return {
                'found': True,
                'items': res["item_name"],
                'scores': scores,
                'match_type': "bm"
            }
        else:
            return {"found": False,
                    "items": [],
                    "scores": [],
                    "match_type": "bm"
            }
        
    def sequenceMatch(self, item_name, seq_threshold):
        item_lower = item_name.lower().strip()
        res = []
        scores = []
        
        for _, row in self.menu_df.iterrows():
            menu_item = row['item_name_lower']
            similarity = SequenceMatcher(None, item_lower, menu_item).ratio()
            
            if similarity >= seq_threshold:
                res.append(row['item_name'])
                scores.append(similarity)

        if res:
            return {
                'found': True,
                'items': res,
                'scores': scores,
                'match_type': "seq"
            }
        else:
            return {"found": False,
                    "items": [],
                    "scores": [],
                    "match_type": "seq"
            }
        
    
    def embeddingSearch(self, item_name, vectordb, emb_thresh):
        # using vectorstore (FAISS)
        res, scores = threshold_search(item_name.lower(), vectorstore=vectordb, emb_thresh=emb_thresh)
        if res:
            return {
                'found': True,
                'items': res,
                'scores': scores,
                'match_type': "emb"
            }
        else:
            return {"found": False,
                    "items": [],
                    "scores": [],
                    "match_type": "emb"
            }

    def unify(self, query, bm_searcher, vectordb, emb_thresh, seq_thresh):
        exact = self.find_exact_match(query)
        # bm = self.bm25_search(query, bm_searcher)
        emb = self.embeddingSearch(query, vectordb, emb_thresh)
        seq = self.sequenceMatch(query, seq_thresh)

        # check for exact first and exit if found
        if exact["found"]:
            return {"exact": True, "item": exact["items"]}
        else:
            # if seq["found"]:
            #     print(seq["items"])
            #     print(seq["scores"])
            # else:
            #     print("no seq")
            # if emb["found"]:
            #     print(emb["items"])
            #     print(emb["scores"])
            # else:
            #     print("no emb")

            # print("no exact")
            return {"exact": False, "seq": seq, "emb": emb}
        
        # 3 scenarios
        # 1. one very good match -> add directly to cart
        # 2. multiple good matches -> ask for clarification
        # 3. no good matches -> reject, and show best alternative (maybe use metadata based retriever which will be used by menu query)
        

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