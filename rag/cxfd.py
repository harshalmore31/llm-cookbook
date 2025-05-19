import requests
import os
import json
import uuid
import time
import math
from dotenv import load_dotenv

# --- Load Configuration ---
# Load environment variables once, e.g., when this module is imported
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- Low-Level Qdrant API Helper ---

def _make_qdrant_request(method, endpoint, payload=None, params=None):
    """Internal helper to make requests to Qdrant API."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("Error: QDRANT_URL and QDRANT_API_KEY must be configured.")
        # In a real API, raise a specific configuration error
        return None

    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
    full_url = f"{QDRANT_URL}{endpoint}"

    try:
        # Increased timeout for potentially long operations like collection creation/upserts
        response = requests.request(method, full_url, headers=headers, json=payload, params=params, timeout=120)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.Timeout:
        print(f"Qdrant API Error: Request timed out ({method} {full_url})")
        return None
    except requests.exceptions.RequestException as e:
        error_message = f"Qdrant API Error ({method} {full_url}): {e}"
        if e.response is not None:
            try:
                error_message += f" | Response: {e.response.json()}"
            except json.JSONDecodeError:
                error_message += f" | Response: {e.response.text}"
        print(error_message) # Use proper logging in your API
        return None
    except json.JSONDecodeError:
        print(f"Qdrant JSON Decode Error ({method} {full_url}). Response: {response.text}")
        return None

# --- Placeholder for Your Custom Embedding Function ---

def get_custom_embeddings(texts: list[str], vector_dim: int) -> list[list[float] | None] | None:
    """
    Placeholder function to get embeddings using your custom model.
    ** IMPORTANT: Replace this with your actual SWARMS model embedding logic. **

    Args:
        texts: A list of text strings to embed.
        vector_dim: The expected dimension of the embeddings.

    Returns:
        A list of embeddings (list of floats) or None if embedding fails for any text.
        Length must match input texts list. Returns None if the overall call fails.
    """
    print(f"--- SWARMS: Embedding {len(texts)} texts (using placeholder)... ---")
    # ---------------------------------------------------------------------
    # TODO: Replace placeholder logic with your actual embedding call.
    # Example:
    # try:
    #     embeddings = your_embedding_model.encode(texts) # Your model's batch encoding
    #     if not embeddings or len(embeddings) != len(texts):
    #          print("Error: Embedding count mismatch.")
    #          return None # Indicate failure
    #     # Optional validation: Check dimensions and types
    #     if not all(isinstance(e, list) and len(e) == vector_dim for e in embeddings):
    #         print(f"Error: Embedding dimension mismatch or invalid type. Expected {vector_dim}")
    #         return None # Indicate failure
    #     return embeddings
    # except Exception as e:
    #     print(f"Error during custom embedding: {e}")
    #     return None # Indicate failure
    # ---------------------------------------------------------------------

    # Placeholder implementation:
    try:
        # Ensure vector_dim is positive
        if vector_dim <= 0:
            print("Error: vector_dim must be positive for placeholder.")
            return None

        dummy_embeddings = []
        for i, _ in enumerate(texts):
            # Simple dummy vector generation
            dummy_vector = [(float(j % 100) / 100.0) + (i * 0.01) for j in range(vector_dim)]
            dummy_embeddings.append(dummy_vector)

        if len(dummy_embeddings) != len(texts):
             print("Error: Placeholder embedding count mismatch.") # Should not happen with this logic
             return None

        return dummy_embeddings
    except Exception as e:
         print(f"Error in placeholder embedding generation: {e}")
         return None


# --- Core RAG Indexing Function ---

def index_documents_to_qdrant(collection_name: str, documents: list[str], vector_dim: int) -> bool:
    """
    Ensures collection exists, processes, embeds (placeholder), and indexes documents.

    Args:
        collection_name: Name of the Qdrant collection.
        documents: A list of document strings.
        vector_dim: The dimension of vectors from your embedding model.

    Returns:
        True if indexing completes without critical errors, False otherwise.
    """
    # Fixed Constants for Indexing/Chunking
    DISTANCE_METRIC = "Cosine"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    UPSERT_BATCH_SIZE = 50

    # 1. Ensure Collection Exists
    print(f"Checking/Creating collection '{collection_name}'...")
    collection_exists_info = _make_qdrant_request("GET", f"/collections/{collection_name}/exists")
    # Handle case where the request itself fails
    if collection_exists_info is None:
        print(f"Failed to check existence for collection '{collection_name}'. Cannot proceed.")
        return False
    if collection_exists_info.get("status") != "ok":
        print(f"Received non-ok status when checking existence for collection '{collection_name}'.")
        return False # Cannot proceed without knowing if collection exists

    if not collection_exists_info.get("result", {}).get("exists", False):
        print(f"Collection '{collection_name}' does not exist. Creating...")
        create_payload = {"vectors": {"size": vector_dim, "distance": DISTANCE_METRIC}}
        create_response = _make_qdrant_request("PUT", f"/collections/{collection_name}", payload=create_payload, params={"timeout": 60})
        if not create_response or create_response.get("result") is not True:
            print(f"Fatal: Failed to create collection '{collection_name}'. Aborting index.")
            return False
        print(f"Collection '{collection_name}' created.")
        time.sleep(1.5) # Brief pause after creation, might help ensure readiness
    else:
        print(f"Collection '{collection_name}' already exists.")

    # 2. Process Documents: Chunking
    all_chunks_data = []
    print(f"Processing {len(documents)} documents into chunks...")
    for doc_index, doc_content in enumerate(documents):
        if not isinstance(doc_content, str) or not doc_content.strip():
            # print(f"Skipping empty/invalid document index {doc_index}.")
            continue
        # Basic chunking logic
        start = 0
        text_len = len(doc_content)
        chunk_index = 0
        while start < text_len:
            end = min(start + CHUNK_SIZE, text_len)
            chunk_text = doc_content[start:end].strip()
            if chunk_text:
                all_chunks_data.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "metadata": {"original_doc_index": doc_index, "chunk_index": chunk_index}
                })
                chunk_index += 1
            next_start = end - CHUNK_OVERLAP
            start = max(next_start, start + 1) # Ensure progress
            if start >= text_len: break

    if not all_chunks_data:
        print("No text chunks generated. Nothing to index.")
        return True # Not an error

    print(f"Split into {len(all_chunks_data)} chunks. Starting indexing...")
    overall_success = True

    # 3. Embed and Upsert in Batches
    total_chunks = len(all_chunks_data)
    num_batches = math.ceil(total_chunks / UPSERT_BATCH_SIZE)
    for i in range(0, total_chunks, UPSERT_BATCH_SIZE):
        batch_data = all_chunks_data[i:i + UPSERT_BATCH_SIZE]
        batch_num = (i // UPSERT_BATCH_SIZE) + 1
        print(f"  Processing batch {batch_num}/{num_batches}...")

        texts_in_batch = [item['text'] for item in batch_data]
        embeddings = get_custom_embeddings(texts_in_batch, vector_dim) # Call placeholder/custom function

        if embeddings is None: # Handle overall embedding failure for the batch
            print(f"  Error: Embedding function failed for batch {batch_num}. Skipping.")
            overall_success = False
            continue

        points_to_upsert = []
        valid_embedding_count = 0
        for item_data, embedding in zip(batch_data, embeddings):
            if embedding is not None and len(embedding) == vector_dim: # Check individual embedding and dimension
                points_to_upsert.append({
                    "id": item_data["id"],
                    "vector": embedding,
                    "payload": {"page_content": item_data["text"], "metadata": item_data["metadata"]}
                })
                valid_embedding_count += 1
            else:
                print(f"    Warning: Skipping chunk due to invalid embedding (ID: {item_data['id']}, Original Index: {item_data['metadata']['original_doc_index']}, Chunk Index: {item_data['metadata']['chunk_index']})")
                overall_success = False # Mark partial failure

        if not points_to_upsert:
            print(f"  Warning: No valid points generated for batch {batch_num}. Check embedding generation.")
            continue # Skip upsert if no valid points

        print(f"  Upserting {len(points_to_upsert)} points...")
        upsert_payload = {"points": points_to_upsert}
        # Use the more specific upsert endpoint which is generally preferred for this operation
        upsert_resp = _make_qdrant_request("PUT", f"/collections/{collection_name}/points", payload=upsert_payload, params={"wait": "true"})

        if not upsert_resp or upsert_resp.get("result", {}).get("status") != "completed":
            print(f"  Error: Failed to upsert batch {batch_num}. Response: {upsert_resp}")
            overall_success = False # Mark failure for the batch
        else:
            print(f"  Batch {batch_num} upsert completed.")

    print(f"Document indexing finished. Overall success indicative: {overall_success}")
    return overall_success

# --- Core RAG Retrieval Function ---

def retrieve_context_from_qdrant(collection_name: str, query_text: str, vector_dim: int) -> tuple[str | None, list | None]:
    """
    Embeds (placeholder) query, retrieves context from Qdrant, returns formatted context & sources.

    Args:
        collection_name: The Qdrant collection name.
        query_text: The user's query.
        vector_dim: The dimension of vectors from your embedding model.

    Returns:
        Tuple (formatted_context: str | None, sources_metadata: list | None).
    """
    # Fixed Constants for Retrieval
    RETRIEVAL_LIMIT = 3
    SCORE_THRESHOLD = 0.35 # Minimum similarity

    print(f"\n--- SWARMS RAG: Retrieving context for query: '{query_text[:50]}...' ---")

    # 1. Embed Query (Placeholder)
    query_embedding_list = get_custom_embeddings([query_text], vector_dim)
    if not query_embedding_list or not query_embedding_list[0]: # Check if list is valid and first element is valid
        print("  Error: Could not embed query.")
        return None, None
    query_vector = query_embedding_list[0]

    # 2. Query Qdrant
    query_payload = {
        "query": query_vector,
        "limit": RETRIEVAL_LIMIT,
        "with_payload": True,
        "with_vector": False,
        "score_threshold": SCORE_THRESHOLD
    }
    print(f"  Querying Qdrant collection '{collection_name}'...")
    response_data = _make_qdrant_request("POST", f"/collections/{collection_name}/points/query", payload=query_payload)

    if not response_data or response_data.get("status") != "ok":
        print(f"  Error querying collection '{collection_name}'.")
        return None, None

    search_results = response_data.get("result", {}).get("points", [])

    # 3. Format Results
    if not search_results:
        print("  No relevant documents found matching threshold.")
        return "No relevant context found.", []

    print(f"  Retrieved {len(search_results)} results. Formatting...")
    context_parts = []
    source_metadata_list = []
    for i, doc in enumerate(search_results):
        text = doc.get("payload", {}).get("page_content", "[Content Missing]")
        metadata = doc.get("payload", {}).get("metadata", {})
        score = doc.get("score", 0.0)
        context_parts.append(f"Context Chunk {i+1} (Score: {score:.4f}):\n{text}")
        metadata['qdrant_score'] = score
        source_metadata_list.append(metadata)

    final_context = "\n\n---\n\n".join(context_parts)
    print(f"--- SWARMS RAG: Context retrieval complete. ---")
    return final_context, source_metadata_list


# --- Main Execution Block for Testing ---
if __name__ == "__main__":
    print("--- Running Direct Qdrant RAG Test ---")

    # --- Configuration for Test ---
    TEST_COLLECTION_NAME = "swarms_rag_direct_test_v3"
    # !!! IMPORTANT: Set this to your actual model's dimension !!!
    TEST_VECTOR_DIM = 768 # Example: Use 768 for many BERT-like models

    if TEST_VECTOR_DIM <= 0:
         raise ValueError("TEST_VECTOR_DIM must be a positive integer.")

    TEST_DOCUMENTS = [
        "Photosynthesis is the process used by plants, algae and cyanobacteria to convert light energy into chemical energy.",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "Artificial intelligence research focuses on creating systems capable of performing tasks that typically require human intelligence.",
        "The main ingredients for pizza dough are flour, water, yeast, salt, and olive oil.",
        "Quantum entanglement is a physical phenomenon that occurs when pairs or groups of particles are generated, interact, or share spatial proximity in ways such that the quantum state of each particle cannot be described independently of the state of the other(s).",
        "The primary colors in additive color models (like RGB used in screens) are red, green, and blue.",
        "Renewable energy sources include solar, wind, hydroelectric, geothermal, and biomass.",
        "Claude Monet was a founder of French Impressionist painting.",
        "The chemical formula for water is H2O."
    ]

    # --- Step 1: Index Documents ---
    print(f"\n[TEST] Indexing {len(TEST_DOCUMENTS)} documents into '{TEST_COLLECTION_NAME}' (Dim: {TEST_VECTOR_DIM})...")
    indexing_successful = index_documents_to_qdrant(TEST_COLLECTION_NAME, TEST_DOCUMENTS, TEST_VECTOR_DIM)

    if not indexing_successful:
        print("\n[TEST] Indexing process reported errors. Test may not proceed correctly.")
        # Decide if you want to exit or continue despite potential partial indexing
        # exit() # Uncomment to stop if indexing fails

    print("\n[TEST] Indexing step finished (check logs for details/errors). Waiting a bit...")
    time.sleep(2) # Allow potential background operations in Qdrant to settle

    # --- Step 2: Retrieve Context for a Query ---
    test_query = "What is photosynthesis?"
    print(f"\n[TEST] Retrieving context for query: '{test_query}'...")
    retrieved_context, sources = retrieve_context_from_qdrant(TEST_COLLECTION_NAME, test_query, TEST_VECTOR_DIM)

    # --- Step 3: Display Results and Simulate LLM Call ---
    if retrieved_context is not None:
        print("\n--------- Retrieved Context ---------")
        print(retrieved_context)
        print("------------------------------------")
        print("\n--------- Source Metadata ---------")
        if sources:
             print(json.dumps(sources, indent=2))
        else:
             print("No sources metadata returned (might happen if context was 'No relevant context found.').")
        print("------------------------------------")

        print("\n[TEST] >>> Simulation: Now pass the 'retrieved_context' and 'test_query' to your SWARMS LLM for answer generation. <<<")
        # Example:
        # swarms_llm_input_prompt = f"Context:\n{retrieved_context}\n\nQuestion:{test_query}\n\nAnswer:"
        # final_answer = your_swarms_llm.process(swarms_llm_input_prompt)
        # print(f"\n[TEST] Simulated LLM Answer: {final_answer}")

    else:
        print("\n[TEST] Context retrieval failed (e.g., embedding or Qdrant query error).")

    print("\n--- Direct Qdrant RAG Test Finished ---")