from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

@asynccontextmanager
async def mcp_server_session():
    """
    Context manager that starts the MCP server process, initializes the session,
    and yields the available tools. Ensures the process is properly cleaned up.
    """
    server_params = StdioServerParameters(
        command="npx",
        args=["mcp-remote", "https://mcp.kite.trade/mcp"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Load the MCP tools using langchain's adapter
            tools = await load_mcp_tools(session)
            
            yield tools
