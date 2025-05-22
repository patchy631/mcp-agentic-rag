import nest_asyncio
import os
from typing import Optional
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core.workflow import Event, Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.schema import NodeWithScore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response_synthesizers import CompactAndRefine

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]

class RAGWorkflow(Workflow):
    def __init__(
        self, 
        model_name: str = None,
        embedding_model: str = None,
        top_k: int = None
    ):
        super().__init__()
        # Get configuration from environment variables or use defaults
        self.model_name = model_name or os.getenv("RAG_MODEL_NAME", "llama3.2")
        self.embedding_model = embedding_model or os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.top_k = top_k or int(os.getenv("RAG_TOP_K", "2"))
        
        # Initialize LLM and embedding model
        try:
            self.llm = Ollama(model=self.model_name)
            self.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
            
            # Configure global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            
            self.index = None
        except Exception as e:
            print(f"Error initializing models: {e}")
            print("Make sure Ollama is running and the specified models are available.")
            raise

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest documents from a directory."""
        dirname = ev.get("dirname")
        if not dirname:
            print("No directory specified for ingestion.")
            return None

        try:
            if not os.path.exists(dirname):
                print(f"Directory {dirname} does not exist.")
                return None
                
            documents = SimpleDirectoryReader(dirname).load_data()
            if not documents:
                print(f"No documents found in {dirname}.")
                return None
                
            print(f"Ingested {len(documents)} documents from {dirname}.")
            self.index = VectorStoreIndex.from_documents(documents=documents)
            return StopEvent(result=self.index)
        except Exception as e:
            print(f"Error during document ingestion: {e}")
            return None

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        """Entry point for RAG retrieval."""
        query = ev.get("query")
        index = ev.get("index") or self.index
        top_k = ev.get("top_k") or self.top_k

        if not query:
            print("No query provided.")
            return None

        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        try:
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes = await retriever.aretrieve(query)
            await ctx.set("query", query)
            print(f"Retrieved {len(nodes)} documents for query: {query}")
            return RetrieverEvent(nodes=nodes)
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return None

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Generate a response using retrieved nodes."""
        try:
            summarizer = CompactAndRefine(streaming=True, verbose=True)
            query = await ctx.get("query", default=None)
            response = await summarizer.asynthesize(query, nodes=ev.nodes)
            return StopEvent(result=response)
        except Exception as e:
            print(f"Error during response synthesis: {e}")
            return StopEvent(result=f"Error generating response: {str(e)}")

    async def query(self, query_text: str, top_k: Optional[int] = None):
        """Helper method to perform a complete RAG query."""
        if self.index is None:
            raise ValueError("No documents have been ingested. Call ingest_documents first.")
        
        result = await self.run(query=query_text, index=self.index, top_k=top_k or self.top_k)
        return result

    async def ingest_documents(self, directory: str):
        """Helper method to ingest documents."""
        result = await self.run(dirname=directory)
        self.index = result
        return result

# Example usage
async def main():
    # Initialize the workflow
    workflow = RAGWorkflow()
    
    try:
        # Ingest documents
        await workflow.ingest_documents("data")
        
        # Perform a query
        result = await workflow.query("How was DeepSeekR1 trained?")
        
        # Print the response
        async for chunk in result.async_response_gen():
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"Error in main workflow: {e}")
        print("Please check that:")
        print("1. Ollama is running and the specified model is available")
        print("2. The data directory exists and contains documents")
        print("3. Environment variables are properly set")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
