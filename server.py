import asyncio
import os
from dotenv import load_dotenv
from linkup import LinkupClient
from rag import RAGWorkflow
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Initialize MCP server
server_name = os.getenv("MCP_SERVER_NAME", "linkup-server")
mcp = FastMCP(server_name)

# Initialize clients
try:
    client = LinkupClient()
    print("Linkup client initialized successfully")
except Exception as e:
    print(f"Error initializing Linkup client: {e}")
    print("Make sure your LINKUP_API_KEY is set correctly in the .env file")
    client = None

# Initialize RAG workflow
try:
    rag_workflow = RAGWorkflow()
    print(f"RAG workflow initialized with model: {rag_workflow.model_name}")
    print(f"Using embedding model: {rag_workflow.embedding_model}")
    print(f"Top-k documents for retrieval: {rag_workflow.top_k}")
except Exception as e:
    print(f"Error initializing RAG workflow: {e}")
    rag_workflow = None

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for the given query."""
    if client is None:
        return "Error: Linkup client is not initialized. Check your API key."
    
    try:
        search_response = client.search(
            query=query,
            depth=os.getenv("LINKUP_SEARCH_DEPTH", "standard"),  # "standard" or "deep"
            output_type=os.getenv("LINKUP_OUTPUT_TYPE", "sourcedAnswer"),  # "searchResults" or "sourcedAnswer" or "structured"
            structured_output_schema=None,  # must be filled if output_type is "structured"
        )
        return search_response
    except Exception as e:
        return f"Error performing web search: {str(e)}"

@mcp.tool()
async def rag(query: str) -> str:
    """
    Use a RAG workflow to answer queries using documents from the data directory.
    
    This tool uses a local document store containing information about DeepSeek AI
    and other topics. The documents should be placed in the 'data' directory.
    """
    if rag_workflow is None:
        return "Error: RAG workflow is not initialized."
    
    try:
        # Check if index exists, if not try to ingest documents
        if rag_workflow.index is None:
            print("Index not found, attempting to ingest documents...")
            try:
                await rag_workflow.ingest_documents("data")
                if rag_workflow.index is None:
                    return "Error: Failed to ingest documents. Make sure the data directory exists and contains documents."
            except Exception as e:
                return f"Error ingesting documents: {str(e)}"
        
        # Perform the query
        response = await rag_workflow.query(query)
        return str(response)
    except Exception as e:
        return f"Error processing RAG query: {str(e)}"

if __name__ == "__main__":
    print("Starting MCP server...")
    print(f"Server name: {server_name}")
    
    # Ingest documents on startup
    if rag_workflow is not None:
        try:
            print("Ingesting documents from data directory...")
            asyncio.run(rag_workflow.ingest_documents("data"))
            print("Document ingestion complete")
        except Exception as e:
            print(f"Error during document ingestion: {e}")
            print("The RAG tool may not work correctly until documents are ingested")
    
    # Run the MCP server
    print("MCP server is running. Use an MCP client to connect.")
    mcp.run(transport="stdio")
