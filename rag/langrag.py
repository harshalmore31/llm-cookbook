from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Qdrant client (using in-memory storage for this example)
client = QdrantClient(
    url="xxx",
    api_key="xx",
)


# # Create a new collection for storing document vectors
# client.create_collection(
#     collection_name="swarms",
#     vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
# )

# Initialize the vector store with the Qdrant client
vector_store = QdrantVectorStore(
    client=client,
    collection_name="markmanson_collection",
    embedding=embeddings,
)

# Load documents from the data folder
# loader = DirectoryLoader("./data", glob="**/*.*")
# documents = loader.load()

# print(f"Loaded {len(documents)} documents")

# # Split documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
# )
# splits = text_splitter.split_documents(documents)

# print(f"Split into {len(splits)} chunks")

# # Add documents to the vector store in batches
# def batch_upload(vector_store, documents, batch_size=10):
#     for i in range(0, len(documents), batch_size):
#         batch = documents[i:i + batch_size]
#         vector_store.add_documents(batch)
#         print(f"Uploaded batch {i // batch_size + 1} of {len(documents) // batch_size + 1}")

# batch_upload(vector_store, splits, batch_size=10)

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-4o")

# Create a conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

# Example of using the chain
def ask_question(question, chat_history=[]):
    result = qa_chain.invoke({"question": question, "chat_history": chat_history})
    return result["answer"], result["source_documents"]

# Example usage
if __name__ == "__main__":
    # Initialize an empty chat history
    chat_history = []
    
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
            
        answer, sources = ask_question(question, chat_history)
        
        print("\nAnswer:", answer)
        print("\nSources:")
        for i, source in enumerate(sources):
            print(f"Source {i+1}: {source.metadata.get('source', 'Unknown')}")
            
        # Update chat history
        chat_history.append((question, answer))
