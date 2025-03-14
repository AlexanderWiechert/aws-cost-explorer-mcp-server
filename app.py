import chainlit as cl
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrock
import asyncio
import re
from typing import Dict, Any, List

# Initialize the model
model = ChatBedrock(model="us.anthropic.claude-3-5-haiku-20241022-v1:0")

# Helper function to parse and format response content
def format_response(response: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Format the LangChain response to correctly handle markdown and code blocks.
    
    Args:
        response: The raw response from the LangChain agent
        
    Returns:
        A dictionary with properly formatted content
    """
    if not response or "messages" not in response:
        return {"content": "No valid response received"}
    
    # Extract the content from the last message
    if isinstance(response["messages"], list) and response["messages"]:
        last_message = response["messages"][-1]
        
        if isinstance(last_message, dict) and "content" in last_message:
            return last_message
        else:
            # Handle string content
            content = str(last_message)
    else:
        content = str(response)
    
    return {"content": content}

# Extract and format tool steps if present
def extract_tool_steps(response: Dict[Any, Any]) -> List[Dict[str, Any]]:
    """
    Extract tool steps from the response for display in the UI.
    
    Args:
        response: The raw response from the LangChain agent
        
    Returns:
        A list of tool steps to display
    """
    steps = []
    
    # Check if there are tool calls in the response
    if "intermediate_steps" in response:
        for step in response["intermediate_steps"]:
            if len(step) >= 2:
                # Extract tool call and result
                tool_call = step[0]
                tool_result = step[1]
                
                steps.append({
                    "name": tool_call.tool if hasattr(tool_call, "tool") else "Tool",
                    "input": tool_call.tool_input if hasattr(tool_call, "tool_input") else str(tool_call),
                    "output": str(tool_result)
                })
    
    return steps

@cl.on_chat_start
async def start():
    await cl.Message("Chatbot is ready! Ask me anything.").send()

@cl.on_message
async def main(message: cl.Message):
    # Create a message with a thinking effect
    msg = cl.Message(content="")
    await msg.send()
    

    
    try:
        async with MultiServerMCPClient(
            {
                "weather": {
                    "url": "http://ec2-44-192-72-20.compute-1.amazonaws.com:8000/sse",
                    "transport": "sse",
                }
            }
        ) as client:
            agent = create_react_agent(
                model, 
                client.get_tools(), 
                prompt="You are an AI assistant, use your knowledge and tools provided to you to answer user questions"
            )
        
            # Invoke the agent
            response = await agent.ainvoke({"messages": [{"role": "user", "content": message.content}]})
            
            # Format the response
            formatted_response = format_response(response)
            
            # Extract any tool steps
            steps = extract_tool_steps(response)
            
            # Update the message with the formatted content
            msg = cl.Message(content=formatted_response["content"], elements=steps)
            await msg.send()
            
    except Exception as e:
        # Handle any errors that occur
        msg = cl.Message(content=f"Error: {str(e)}")
        await msg.send()

if __name__ == "__main__":
    cl.run()