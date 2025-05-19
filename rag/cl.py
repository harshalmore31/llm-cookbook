import requests
import os
import json
import uuid
import time
from dotenv import load_dotenv

# Load configuration
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def rag_pipeline(operation, collection_name, data=None, query=None, vector_dim=1536):
    """
    Unified RAG pipeline handling both indexing and retrieval operations.
    
    Args:
        operation: Either "index" or "retrieve"
        collection_name: Qdrant collection name
        data: List of document strings (for indexing)
        query: Query string (for retrieval)
        vector_dim: Vector dimension for embeddings
        
    Returns:
        For indexing: bool indicating success
        For retrieval: tuple of (context, sources)
    """
    # Fixed constants for the entire pipeline
    CONSTANTS = {
        "distance_metric": "Cosine",
        "chunk_size": 500, 
        "chunk_overlap": 50,
        "batch_size": 50,
        "retrieval_limit": 3,
        "score_threshold": 0.35
    }
    
    # Helper function for Qdrant API requests
    def _qdrant_request(method, endpoint, payload=None, params=None):
        if not QDRANT_URL or not QDRANT_API_KEY:
            print("Error: QDRANT_URL and QDRANT_API_KEY must be configured.")
            return None
            
        headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
        full_url = f"{QDRANT_URL}{endpoint}"
        
        try:
            response = requests.request(method, full_url, headers=headers, json=payload, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            error_message = f"Qdrant API Error ({method} {endpoint}): {str(e)}"
            print(error_message)
            return None
    
    # Placeholder embedding function (replace with actual implementation)
    def _get_embeddings(texts):
        print(f"--- SWARMS: Embedding {len(texts)} texts (using placeholder)... ---")
        # TODO: Replace with actual embedding model
        return [[float(i % 100) / 100.0] * vector_dim for i, _ in enumerate(texts)]
    
    # INDEXING OPERATION
    if operation == "index" and data:
        # Ensure collection exists
        exists_info = _qdrant_request("GET", f"/collections/{collection_name}/exists")
        if not exists_info or exists_info.get("status") != "ok":
            return False
            
        if not exists_info.get("result", {}).get("exists", False):
            create_payload = {"vectors": {"size": vector_dim, "distance": CONSTANTS["distance_metric"]}}
            create_resp = _qdrant_request("PUT", f"/collections/{collection_name}", payload=create_payload)
            if not create_resp or create_resp.get("result") is not True:
                return False
            time.sleep(1)
        
        # Process and chunk documents
        chunks = []
        for doc_idx, content in enumerate(data):
            if not isinstance(content, str) or not content.strip():
                continue
                
            start = 0
            text_len = len(content)
            chunk_idx = 0
            
            while start < text_len:
                end = min(start + CONSTANTS["chunk_size"], text_len)
                chunk_text = content[start:end].strip()
                
                if chunk_text:
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": chunk_text,
                        "metadata": {"original_doc_index": doc_idx, "chunk_index": chunk_idx}
                    })
                    chunk_idx += 1
                
                # Calculate next position with overlap
                next_start = end - CONSTANTS["chunk_overlap"]
                start = next_start if next_start > start else start + 1
                if start >= text_len:
                    break
        
        # Nothing to index
        if not chunks:
            return True
            
        # Embed and upsert in batches
        success = True
        for i in range(0, len(chunks), CONSTANTS["batch_size"]):
            batch = chunks[i:i + CONSTANTS["batch_size"]]
            texts = [item['text'] for item in batch]
            embeddings = _get_embeddings(texts)
            
            if not embeddings or len(embeddings) != len(batch):
                success = False
                continue
                
            points = [
                {
                    "id": item["id"],
                    "vector": emb,
                    "payload": {"page_content": item["text"], "metadata": item["metadata"]}
                } for item, emb in zip(batch, embeddings)
            ]
            
            upsert_resp = _qdrant_request(
                "PUT", 
                f"/collections/{collection_name}/points", 
                payload={"points": points}, 
                params={"wait": "true"}
            )
            
            if not upsert_resp or upsert_resp.get("result", {}).get("status") != "completed":
                success = False
                
        return success
        
    # RETRIEVAL OPERATION
    elif operation == "retrieve" and query:
        # Embed query
        query_embedding = _get_embeddings([query])[0]
        
        # Search Qdrant
        query_payload = {
            "query": query_embedding,
            "limit": CONSTANTS["retrieval_limit"],
            "with_payload": True,
            "with_vector": False,
            "score_threshold": CONSTANTS["score_threshold"]
        }
        
        response = _qdrant_request(
            "POST", 
            f"/collections/{collection_name}/points/query", 
            payload=query_payload
        )
        
        if not response or response.get("status") != "ok":
            return None, None
            
        results = response.get("result", {}).get("points", [])
        
        if not results:
            return "No relevant context found.", []
            
        # Format results
        context_parts = []
        sources = []
        
        for i, doc in enumerate(results):
            text = doc.get("payload", {}).get("page_content", "[Content Missing]")
            metadata = doc.get("payload", {}).get("metadata", {})
            score = doc.get("score", 0.0)
            
            context_parts.append(f"Context Chunk {i+1} (Score: {score:.4f}):\n{text}")
            metadata['qdrant_score'] = score
            sources.append(metadata)
            
        return "\n\n---\n\n".join(context_parts), sources
        
    return None

# Example usage:
# 1. Index documents:
success = rag_pipeline("index", "my_collection", data=["Document 1", "Document 2"])
# 
# 2. Retrieve context:
# context, sources = rag_pipeline("retrieve", "my_collection", query="What is...")