import asyncio
from dotenv import load_dotenv
from linkup import LinkupClient
from rag import RAGWorkflow
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP('linkup-server')
client = LinkupClient()
rag_workflow = RAGWorkflow()

@mcp.tool()
def web_search(query: str, depth: str = "standard") -> str:
    """Search the web for the given query.
    
    Args:
        query: The search query string
        depth: The search depth, either "standard" or "deep"
    """
    search_response = client.search(
        query=query,
        depth=depth,  # User can now specify "standard" or "deep"
        output_type="sourcedAnswer",  # "searchResults" or "sourcedAnswer" or "structured"
        structured_output_schema=None,  # must be filled if output_type is "structured"
    )
    return search_response

@mcp.tool()
async def rag(query: str) -> str:
    """Use a simple RAG workflow to answer queries using documents from data directory about Deep Seek"""
    response = await rag_workflow.query(query)
    return str(response)

if __name__ == "__main__":
    asyncio.run(rag_workflow.ingest_documents("data"))
    mcp.run(transport="stdio")
