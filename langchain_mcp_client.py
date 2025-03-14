import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrock

model = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")  # Update model ID if needed

async def main():
    async with MultiServerMCPClient(
        {
            "weather": {
                "url": "http://ec2-44-192-72-20.compute-1.amazonaws.com:8000/sse",  # Ensure the weather server is running
                "transport": "sse",
            }
        }
    ) as client:
        agent = create_react_agent(model, client.get_tools())
        
        response = await agent.ainvoke({"messages": "my bedrock usage in last 7 days?"})
        print(response)

if __name__ == "__main__":
    asyncio.run(main())
