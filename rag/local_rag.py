import os
import unittest
from dotenv import load_dotenv
import openai
from swarms.structs import Agent
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.vector_stores import SimpleVectorStore  # Import from core package
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from loguru import logger

from pathlib import Path
from typing import Optional

load_dotenv()

# Get OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure API key is set
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Configure OpenAI
openai_client = openai.OpenAI(api_key=openai_api_key)
openai_embed_model = OpenAIEmbedding(api_key=openai_api_key)

# Still use LlamaIndex Settings for Embedding model, but LLM will be handled directly
Settings.embed_model = openai_embed_model


class LocalVectorDatabase:  # Renamed class to LocalVectorDatabase
    """
    A class to manage document indexing and querying using LlamaIndex with a local, in-memory vector store.
    This class uses SimpleVectorStore for local RAG testing.
    """

    def __init__(self, data_dir: str = "docs", collection_name="local_rag_collection", **kwargs) -> None: # Renamed collection_name for clarity
        """Initialize the LocalVectorDatabase with a SimpleVectorStore index."""
        self.data_dir = data_dir
        self.collection_name = collection_name # Not really used for SimpleVectorStore but kept for consistency
        self.vector_store = SimpleVectorStore() # Initialize SimpleVectorStore (in-memory)
        self.index: Optional[VectorStoreIndex] = None

        logger.info("Initialized LocalVectorDatabase with SimpleVectorStore") # Using loguru logger

        data_path = Path(self.data_dir)
        if not data_path.exists():
            logger.error(f"Directory not found: {self.data_dir}")
            raise FileNotFoundError(
                f"Directory {self.data_dir} does not exist"
            )

        try:
            documents = SimpleDirectoryReader(
                self.data_dir, **kwargs # Pass kwargs to SimpleDirectoryReader
            ).load_data()
            self.index = VectorStoreIndex.from_documents( # Create VectorStoreIndex
                documents,
                vector_store=self.vector_store, # Use SimpleVectorStore
                embed_model=openai_embed_model, # Explicitly set embed_model
                **kwargs # Pass other kwargs to VectorStoreIndex if needed
            )
            logger.success(
                f"Successfully indexed documents from {self.data_dir} into local VectorStore" # Updated log message
            )
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise

    def query(self, query: str, **kwargs) -> str:
        """Query the indexed documents in the local VectorStore."""
        if self.index is None:
            logger.error("No documents have been indexed yet")
            raise ValueError("Must add documents before querying")

        try:
            query_engine = self.index.as_query_engine(**kwargs) # Create query engine
            response = query_engine.query(query) # Perform query
            print(response) # Keep printing response for debugging
            logger.info(f"Successfully queried: {query}")
            return str(response)
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise


class TestRAGAgentWithLocalDB(unittest.TestCase):  # Renamed Test Class
    def setUp(self):
        # 1. Create a test document with Luck text content
        self.docs_dir = "test_rag_docs_local" # Using a different docs directory for local test
        os.makedirs(self.docs_dir, exist_ok=True)
        self.doc_path = os.path.join(self.docs_dir, "test_doc_local.txt") # Different doc file name
        with open(self.doc_path, "w") as f:
            f.write(
                """Why do you say, “Get rich without getting lucky”?

In 1,000 parallel universes, you want to be wealthy in 999 of them. You don’t want to be wealthy in the fifty of them where you got lucky, so we want to factor luck out of it.

But getting lucky would help, right?

Just recently, Babak Nivi, my co-founder, and I were talking on Twitter about how one gets lucky, and there are really four kinds of luck we were talking about.

The first kind of luck is blind luck where one just gets lucky because something completely out of their control happened. This includes fortune, fate, etc.

Then, there’s luck through persistence, hard work, hustle, and motion. This is when you’re running around creating opportunities. You’re generating a lot of energy, you’re doing a lot to stir things up. It’s almost like mixing a petri dish or mixing a bunch of reagents and seeing what combines. You’re just generating enough force, hustle, and energy for luck to find you.

A third way is you become very good at spotting luck. If you are very skilled in a field, you will notice when a lucky break happens in your field, and other people who aren’t attuned to it won’t notice. So, you become sensitive to luck.

The last kind of luck is the weirdest, hardest kind, where you build a unique character, a unique brand, a unique mindset, which causes luck to find you.

For example, let’s say you’re the best person in the world at deep-sea diving. You’re known to take on deep-sea dives nobody else will even dare to attempt. By sheer luck, somebody finds a sunken treasure ship off the coast they can’t get to. Well, their luck just became your luck, because they’re going to come to you to get to the treasure, and you’re going to get paid for it.

This is an extreme example, but it shows how one person had blind luck finding the treasure. Them coming to you to extract it and give you half is not blind luck. You created your own luck. You put yourself in a position to capitalize on luck or to attract luck when nobody else created the opportunity for themselves. To get rich without getting lucky, we want to be deterministic. We don’t want to leave it to chance. [78]

Ways to get lucky:
• Hope luck finds you.
• Hustle until you stumble into it.
• Prepare the mind and be sensitive to chances others miss.
• Become the best at what you do. Refine what you do until this is true. Opportunity will seek you out. Luck becomes your destiny.
It starts becoming so deterministic, it stops being luck. The definition starts fading from luck to destiny. To summarize the fourth type: build your character in a certain way, then your character becomes your destiny.

One of the things I think is important to make money is having a reputation that makes people do deals through you. Remember the example of being a great diver where treasure hunters will come and give you a piece of the treasure for your diving skills.

If you are a trusted, reliable, high-integrity, long-term-thinking dealmaker, when other people want to do deals but don’t know how to do them in a trustworthy manner with strangers, they will literally approach you and give you a cut of the deal just because of the integrity and reputation you’ve built up.

Warren Buffett gets offered deals to buy companies, buy warrants, bail out banks, and do things other people can’t do because of his reputation. Of course, he has accountability on the line, and he has a strong brand on the line.

Your character and your reputation are things you can build, which will let you take advantage of opportunities other people may characterize as lucky, but you know it wasn’t luck. [78] My co-founder Nivi said, “In a long-term game, it seems that everybody is making each other rich. And in a short-term game, it seems like everybody is making themselves rich.”

I think that is a brilliant formulation. In a long-term game, it’s positive sum. We’re all baking the pie together. We’re trying to make it as big as possible. And in a short-term game, we’re cutting up the pie. [78]

How important is networking?

I think business networking is a complete waste of time. And I know there are people and companies popularizing this concept because it serves them and their business model well, but the reality is if you’re building something interesting, you will always have more people who will want to know you. Trying to build business relationships well in advance of doing business is a complete waste of time. I have a much more comfortable philosophy: “Be a maker who makes something interesting people want. Show your craft, practice your craft, and the right people will eventually find you.” [14]

And once you’ve met someone, how do you determine if you can trust someone? What signals do you pay attention to?

If someone is talking a lot about how honest they are, they’re probably dishonest. That is just a little telltale indicator I’ve learned. When someone spends too much time talking about their own values or they’re talking themselves up, they’re covering for something. [4]

Sharks eat well but live a life surrounded by sharks.
I have great people in my life who are extremely successful, very desirable (like everybody wants to be their friend), very smart. Yet, I’ve seen them do one or two things slightly not great to other people. The first time, I’ll say, “Hey, I don’t think you should do this to that other person. Not because you won’t get away with it. You will get away with it, but because it will hurt you in the end.”

Not in some cosmic, karma kind of way, but I believe deep down we all know who we are. You cannot hide anything from yourself. Your own failures are written within your psyche, and they are obvious to you. If you have too many of these moral shortcomings, you will not respect yourself. The worst outcome in this world is not having self-esteem. If you don’t love yourself, who will?

I think you just have to be very careful about doing things you are fundamentally not going to be proud of, because they will damage you. The first time someone acts this way, I will warn them. By the way, nobody changes. Then I just distance myself from them. I cut them out of my life. I just have this saying inside my head: “The closer you want to get to me, the better your values have to be.” [4]
                """
            )

        # 2. Initialize LocalVectorDatabase wrapper (using LocalVectorDatabase now)
        self.local_db = LocalVectorDatabase(data_dir=self.docs_dir, collection_name="local_rag_collection") # Initialize LocalVectorDatabase

        # --- Simple OpenAI API Test --- (No change)
        print("\n--- Simple OpenAI API Test ---")
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello, OpenAI! Please respond with a short greeting."}],
            )
            test_api_response_content = response.choices[0].message.content
            print("Simple OpenAI API Test Response:", test_api_response_content)
            self.assertIsInstance(test_api_response_content, str)
            self.assertTrue(len(test_api_response_content) > 0, "API test response should not be empty")

        except Exception as e:
            print("Simple OpenAI API Test Error:", e)
            self.fail(f"Simple OpenAI API test failed: {e}")
        print("--- End Simple OpenAI API Test ---\n")

        # 3. Initialize Swarms Agent (using local_db)
        self.agent = Agent(
            llm=openai_client,
            agent_name="RAG-Test-Agent-Local", # Different agent name
            agent_description="Agent for testing RAG with local VectorDB", # Updated description
            system_prompt="You are a helpful expert on luck and wealth. Use the provided documents to answer questions accurately and concisely.", # Updated system prompt
            long_term_memory=self.local_db.query, # Use local_db.query
            rag_every_loop=True,
            verbose=True,
            model_name="4o",
        )

    def tearDown(self):
        # Clean up test documents directory (Different directory for local test)
        import shutil
        shutil.rmtree(self.docs_dir)
        pass

    def test_local_db_direct_query(self): # Renamed test method
        query = "Who was the co-founder?" # Updated query
        retrieved_context = self.local_db.query(query) # Use local_db.query
        print("\nDirect Local DB Query Result:", retrieved_context) # Updated print message
        self.assertIsInstance(retrieved_context, str)
        self.assertIn("blind luck", retrieved_context.lower()) # Updated assertion
        self.assertIn("luck through persistence", retrieved_context.lower()) # Updated assertion
        self.assertIn("become very good at spotting luck", retrieved_context.lower()) # Updated assertion
        self.assertIn("build a unique character", retrieved_context.lower()) # Updated assertion


    def test_rag_query_local_db(self): # Renamed test method
        query = "What are the four kinds of luck mentioned?" # Updated query
        response = self.agent.run(query)

        print(f"\nQuery: {query}")
        print(f"Agent Response (Local DB): {response}") # Updated print message

        self.assertIsInstance(response, str)
        self.assertIn("blind luck", response.lower(), "Response should mention blind luck") # Updated assertion
        self.assertIn("persistence", response.lower(), "Response should mention persistence luck") # Updated assertion
        self.assertIn("spotting luck", response.lower(), "Response should mention spotting luck") # Updated assertion
        self.assertIn("character", response.lower(), "Response should mention character luck") # Updated assertion
        # You can add more specific assertions based on the expected content and quality of the RAG response


if __name__ == "__main__":
    unittest.main()