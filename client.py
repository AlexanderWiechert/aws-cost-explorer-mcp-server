from mcp import types
from mcp import ClientSession
from mcp.client.sse import sse_client

# Optional: create a sampling callback
async def handle_sampling_message(message: types.CreateMessageRequestParams) -> types.CreateMessageResult:
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text="Hello, world! from model",
        ),
        model="gpt-3.5-turbo",
        stopReason="endTurn",
    )

async def run():
    server_url = "http://localhost:8000/sse"
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write, sampling_callback=handle_sampling_message) as session:
            # Initialize the connection
            await session.initialize()

            # List available prompts
            prompts = await session.list_prompts()
            print(prompts)

            # Get a prompt
            #prompt = await session.get_prompt("example-prompt", arguments={"arg1": "value"})

            # List available resources
            resources = await session.list_resources()
            print(resources)

            # List available tools
            tools = await session.list_tools()
            print(tools)

            # Read a resource
            #content, mime_type = await session.read_resource("file://some/path")

            # Call a tool
            result = await session.call_tool("get_bedrock_daily_usage_stats", arguments={"params": dict(days=7, region="us-east-1")})
            #print(result)
            for r in result.content:
                print(r.text)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())