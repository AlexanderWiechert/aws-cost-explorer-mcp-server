import chainlit as cl
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrock
import asyncio

# Initialize the model
model = ChatBedrock(model="us.anthropic.claude-3-5-haiku-20241022-v1:0")

@cl.on_chat_start
async def start():
    welcome_message = """
# 👋 Welcome to your AWS cost explorer assistant.
    
I'm ready to help you with your questions related to your AWS spend. How can I help you save today?
    """
    await cl.Message(content=welcome_message).send()
    
    # Initialize conversation history with a system message at the beginning
    cl.user_session.set(
        "message_history",
        []  # Start with an empty history - we'll add the system message when formatting for the agent
    )

@cl.on_message
async def main(message: cl.Message):
    # Get the conversation history
    message_history = cl.user_session.get("message_history")
    
    # Add the current user message to history
    #if len(message_history) > 0:
    message_history.append({"role": "user", "content": message.content})
    
    # Show a thinking message
    thinking_msg = cl.Message(content="Thinking...")
    await thinking_msg.send()
    
    try:
        async with MultiServerMCPClient(
            {
                "weather": {
                    "url": "http://ec2-44-192-72-20.compute-1.amazonaws.com:8000/sse",
                    "transport": "sse",
                }
            }
        ) as client:
            # Create the agent
            agent = create_react_agent(
                model, 
                client.get_tools(), 
                #prompt="You are an AI assistant, use your knowledge and tools provided to you to answer user questions"
            )
            
            # Format messages for the agent - ensure system message is first
            formatted_messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Answer the user's questions accurately and concisely."}
            ]
            # Add the rest of the conversation history
            formatted_messages.extend(message_history)
        
            # Invoke the agent with properly formatted message history
            print(f"formatted_messages={formatted_messages}")
            response = await agent.ainvoke({"messages": formatted_messages})
            
            # Remove the thinking message
            await thinking_msg.remove()
            
            # Extract the content from the response
            if response and "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                
                if isinstance(last_message, dict) and "content" in last_message:
                    content = last_message["content"]
                else:
                    content = str(last_message.content)
                
                # Add the assistant's response to the conversation history
                message_history.append({"role": "assistant", "content": content})
                
                # Save the updated history (without system message)
                cl.user_session.set("message_history", message_history)
                
                # Send the message
                await cl.Message(content=content).send()
            else:
                await cl.Message(content="No valid response received").send()
                
    except Exception as e:
        # Remove the thinking message
        await thinking_msg.remove()
        
        # Send error message
        error_message = f"""
## ❌ Error Occurred

```
{str(e)}
```

Please try again or check your query.
        """
        await cl.Message(content=error_message, author="System").send()
        
        # Print error to console for debugging
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    cl.run(
        title="Claude Assistant",
        description="A simple interface for Claude"
    )