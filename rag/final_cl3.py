import requests
import os
import uuid
from dotenv import load_dotenv
from litellm import embedding

# Load configuration
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def rag_pipeline(operation, collection_name, data=None, query=None, vector_dim=1536, embedding_model="text-embedding-3-small"):
    """Unified RAG pipeline for indexing and retrieval operations."""
    # Fixed constants
    C = {"metric": "Cosine", "chunk_size": 500, "overlap": 50, "batch": 50, "limit": 3, "threshold": 0.35}
    
    # Helper function for API requests
    def _req(method, path, json=None, params=None):
        if not QDRANT_URL or not QDRANT_API_KEY:
            return None
        try:
            resp = requests.request(
                method, 
                f"{QDRANT_URL}{path}", 
                headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
                json=json,
                params=params,
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Qdrant API Error: {str(e)}")
            return None
    
    # LiteLLM embedding function
    def _embed(texts):
        try:
            response = embedding(model=embedding_model, input=texts)
            return [item['embedding'] for item in response.data]
        except Exception as e:
            print(f"Embedding Error: {str(e)}")
            return None
    
    # INDEXING OPERATION
    if operation == "index" and data:
        # Ensure collection exists
        exists = _req("GET", f"/collections/{collection_name}/exists")
        if not exists or exists.get("status") != "ok":
            return False
            
        if not exists.get("result", {}).get("exists", False):
            if not _req("PUT", f"/collections/{collection_name}", 
                      json={"vectors": {"size": vector_dim, "distance": C["metric"]}}):
                return False
        
        # Process documents into chunks
        chunks = []
        for doc_idx, content in enumerate([d for d in data if isinstance(d, str) and d.strip()]):
            start, chunk_idx = 0, 0
            while start < len(content):
                end = min(start + C["chunk_size"], len(content))
                chunk_text = content[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": chunk_text,
                        "meta": {"doc_idx": doc_idx, "chunk_idx": chunk_idx}
                    })
                    chunk_idx += 1
                start = end - C["overlap"] if end - C["overlap"] > start else start + 1
        
        if not chunks:
            return True
            
        # Embed and upsert in batches
        success = True
        for i in range(0, len(chunks), C["batch"]):
            batch = chunks[i:i + C["batch"]]
            batch_texts = [item['text'] for item in batch]
            embeddings = _embed(batch_texts)
            
            if not embeddings or len(embeddings) != len(batch):
                success = False
                continue
                
            points = [{
                "id": item["id"],
                "vector": emb,
                "payload": {"content": item["text"], "metadata": item["meta"]}
            } for item, emb in zip(batch, embeddings)]
            
            resp = _req("PUT", f"/collections/{collection_name}/points", 
                      json={"points": points}, params={"wait": "true"})
            if not resp or resp.get("result", {}).get("status") != "completed":
                success = False
                
        return success
        
    # RETRIEVAL OPERATION
    elif operation == "retrieve" and query:
        query_emb = _embed([query])[0]
        resp = _req("POST", f"/collections/{collection_name}/points/query", json={
            "query": query_emb,
            "limit": C["limit"],
            "with_payload": True,
            "with_vector": False,
            "score_threshold": C["threshold"]
        })
        
        if not resp or resp.get("status") != "ok":
            return None, None
            
        results = resp.get("result", {}).get("points", [])
        if not results:
            return "No relevant context found.", []
            
        context = []
        sources = []
        for i, doc in enumerate(results):
            text = doc.get("payload", {}).get("content", "[Missing]")
            meta = doc.get("payload", {}).get("metadata", {})
            score = doc.get("score", 0.0)
            
            context.append(f"Context {i+1} (Score: {score:.4f}):\n{text}")
            meta['score'] = score
            sources.append(meta)
            
        return "\n\n---\n\n".join(context), sources
        
    return None

# Usage:
# success = rag_pipeline("index", "collection", data=["Document 1", "Document 2"])
# context, sources = rag_pipeline("retrieve", "collection", query="What is...")

# from dotenv import load_dotenv
# from openai import OpenAI

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# with open("data/markmanson.md", "r", encoding="utf-8") as file:
#     content = file.read()
#     print("Successfully read markmanson.md file.")
# # Sample documents for testing

# user_input = input("Enter your question: ")
# rag_pipeline("index", "selfrag", data=[content])
# final_context, sources = rag_pipeline("retrieve", "selfrag", query=user_input)

# prompt = f"""
# You are a helpful AI assistant. Answer the following question based on the provided context.

# ### Context:
# {final_context}
# {sources}
# ### Question:
# {user_input}

# ### Answer:
# """

# completion = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]
# )

# print("\n--- Final Answer ---")
# print(completion.choices[0].message.content)

