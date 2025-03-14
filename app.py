import chainlit as cl
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrock

model = ChatBedrock(model="us.anthropic.claude-3-5-haiku-20241022-v1:0")


async def initialize_agent():
    client = MultiServerMCPClient(
            {
                "aws_cost_explorer": {
                    "url": "http://localhost:8000/sse",
                    "transport": "sse",
                }
            }
        )
    print(f"tools={client.get_tools()}")    
    return create_react_agent(model, client.get_tools())

@cl.on_chat_start
async def start():
    cl.user_session.set("agent", await initialize_agent())
    await cl.Message("Chatbot is ready! Ask me anything.").send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    if agent:
        response = await agent.ainvoke({"messages": message.content})
        await cl.Message(content=str(response)).send()
    else:
        await cl.Message(content="Agent not initialized").send()


