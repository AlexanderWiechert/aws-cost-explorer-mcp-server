# AWS Cost Explorer and Amazon Bedrock Model Invocation Logs MCP Server & Client

An MCP server for getting AWS spend data via Cost Explorer and Amazon Bedrock usage data via [`Model invocation logs`](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html) in Amazon Cloud Watch through [Anthropic's MCP (Model Control Protocol)](https://www.anthropic.com/news/model-context-protocol).

```mermaid
flowchart LR
    User([User]) --> UserApp[User Application]
    UserApp --> |Queries| Host[Host]
    
    subgraph "Claude Desktop"
        Host --> MCPClient[MCP Client]
    end
    
    MCPClient --> |MCP Protocol| MCPServer[AWS Cost Explorer MCP Server]
    
    subgraph "AWS Services"
        MCPServer --> |API Calls| CostExplorer[(AWS Cost Explorer)]
        MCPServer --> |API Calls| CloudWatchLogs[(AWS CloudWatch Logs)]
    end
```

You can run the MCP server locally and access it via the Claude Desktop or you could also run a Remote MCP server on Amazon EC2 and access it via a MCP client built into a LangGraph Agent.

### Demo video

[![AWS Cost Explorer MCP Server Deep Dive](https://img.youtube.com/vi/WuVOmYLRFmI/maxresdefault.jpg)](https://youtu.be/WuVOmYLRFmI)

## Overview

This tool provides a convenient way to analyze and visualize AWS cloud spending data using Anthropic's Claude model as an interactive interface. It functions as an MCP server that exposes AWS Cost Explorer API functionality to Claude, allowing you to ask questions about your AWS costs in natural language.

## Features

- **Amazon EC2 Spend Analysis**: View detailed breakdowns of EC2 spending for the last day
- **Amazon Bedrock Spend Analysis**: View breakdown by region, users and models over the last 30 days
- **Service Spend Reports**: Analyze spending across all AWS services for the last 30 days
- **Detailed Cost Breakdown**: Get granular cost data by day, region, service, and instance type
- **Interactive Interface**: Use Claude to query your cost data through natural language

## Requirements

- Python 3.12
- AWS credentials with Cost Explorer access
- Anthropic API access (for Claude integration)
- [Optional] Amazon Bedrock access (for LangGraph Agent)
- [Optional] Amazon EC2 for running a remote MCP server

## Installation

1. Install `uv`:
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   
   ```powershell
   # On Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Clone this repository:
   ```
   git clone https://github.com/aarora79/aws-cost-explorer-mcp.git
   cd aws-cost-explorer-mcp
   ```

3. Set up the Python virtual environment and install dependencies:
   ```
   uv venv --python 3.12 && source .venv/bin/activate && uv pip install --requirement pyproject.toml
   ```
   
4. Configure your AWS credentials:
   ```
   mkdir -p ~/.aws
   # Set up your credentials in ~/.aws/credentials and ~/.aws/config
   ```

## Usage

### Prerequisites

1. Setup [model invocation logs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html#setup-cloudwatch-logs-destination) in Amazon CloudWatch.
1. Ensure that the IAM user/role being used has full read-only access to Amazon Cost Explorer and Amazon CloudWatch, this is required for the MCP server to retrieve data from these services.

### Local setup

Uses `stdio` as a transport for MCP, both the MCP server and client are running on your local machine.

#### Starting the Server (local)

Run the server using:

```
export MCP_TRANSPORT=stdio
export BEDROCK_LOG_GROUP_NAME=YOUR_BEDROCK_CW_LOG_GROUP_NAME
python server.py
```

#### Claude Desktop Configuration

There are two ways to configure this tool with Claude Desktop:

##### Option 1: Using Docker

Add the following to your Claude Desktop configuration file. The file can be found out these paths depending upon you operating system.

- macOS: ~/Library/Application Support/Claude/claude_desktop_config.json.
- Windows: %APPDATA%\Claude\claude_desktop_config.json.
- Linux: ~/.config/Claude/claude_desktop_config.json.

```json
{
  "mcpServers": {
    "aws-cost-explorer": {
      "command": "docker",
      "args": [ "run", "-i", "--rm", "-e", "AWS_ACCESS_KEY_ID", "-e", "AWS_SECRET_ACCESS_KEY", "-e", "AWS_REGION", "-e", "BEDROCK_LOG_GROUP_NAME", "-e", "MCP_TRANSPORT", "aws-cost-explorer-mcp:latest" ],
      "env": {
        "AWS_ACCESS_KEY_ID": "YOUR_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY": "YOUR_SECRET_ACCESS_KEY",
        "AWS_REGION": "us-east-1",
        "BEDROCK_LOG_GROUP_NAME": "YOUR_CLOUDWATCH_BEDROCK_MODEL_INVOCATION_LOG_GROUP_NAME",
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

> **IMPORTANT**: Replace `YOUR_ACCESS_KEY_ID` and `YOUR_SECRET_ACCESS_KEY` with your actual AWS credentials. Never commit actual credentials to version control.

##### Option 2: Using UV (without Docker)

If you prefer to run the server directly without Docker, you can use UV:

```json
{
  "mcpServers": {
    "aws_cost_explorer": {
      "command": "uv",
      "args": [
          "--directory",
          "/path/to/aws-cost-explorer-mcp-server",
          "run",
          "server.py"
      ],
      "env": {
        "AWS_ACCESS_KEY_ID": "YOUR_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY": "YOUR_SECRET_ACCESS_KEY",
        "AWS_REGION": "us-east-1",
        "BEDROCK_LOG_GROUP_NAME": "YOUR_CLOUDWATCH_BEDROCK_MODEL_INVOCATION_LOG_GROUP_NAME",
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

Make sure to replace the directory path with the actual path to your repository on your system.

### Remote setup

Uses `sse` as a transport for MCP, the MCP servers on EC2 and the client is running on your local machine. Note that Claude Desktop does not support remote MCP servers at this time (see [this](https://github.com/orgs/modelcontextprotocol/discussions/16) GitHub issue).

#### Starting the Server (remote)

You can start a remote MCP server on Amazon EC2 by following the same instructions as above. Make sure to set the `MCP_TRANSPORT` as `sse` (server side events) as shown below. **Note that the MCP uses JSON-RPC 2.0 as its wire format, therefore the protocol itself does not include authorization and authentication (see [this GitHub issue](https://github.com/modelcontextprotocol/specification/discussions/102)), do not send or receive sensitive data over MCP**.

nsports - Model Context Protocol

modelcontextprotocol.io
https://modelcontextprotocol.io › docs › concepts › tran...
MCP uses JSON-RPC 2.0 as its wire format

Run the server using:

```
export MCP_TRANSPORT=sse
export BEDROCK_LOG_GROUP_NAME=YOUR_BEDROCK_CW_LOG_GROUP_NAME
python server.py
```

1. The MCP server will start listening on TCP port 8000.
1. Configure an ingress rule in the security group associated with your EC2 instance to allow access to TCP port 8000 from your local machine (where you are running the MCP client/LangGraph based app) to your EC2 instance.

#### Testing with a CLI MCP client

You can test your remote MCP server with the `mcp_sse_client.py` script. Running this script will print the list of tools available from the MCP server and an output for the `get_bedrock_daily_usage_stats` tool.

```{.bashrc}
MCP_SERVER_HOSTNAME=YOUR_MCP_SERVER_EC2_HOSTNAME
python mcp_sse_client.py --host $MCP_SERVER_HOSTNAME
```


#### Testing with Chainlit app

The `app.py` file in this repo provides a Chainlit app (chatbot) which creates a LangGraph agent that uses the [`LangChain MCP Adapter`](https://github.com/langchain-ai/langchain-mcp-adapters) to import the tools provided by the MCP server as tools in a LangGraph Agent. The Agent is then able to use an LLM to respond to user questions and use the tools available to it as needed. Thus if the user asks a question such as "_What was my Bedrock usage like in the last one week?_" then the Agent will use the tools available to it via the remote MCP server to answer that question. We use Claude 3.5 Haiku model available via Amazon Bedrock to power this agent.

Run the Chainlit app using:

```{.bashrc}
chainlit run app.py --port 8080 
```

A browser window should open up on `localhost:8080` and you should be able to use the chatbot to get details about your AWS spend.

### Available Tools

The server exposes the following tools that Claude can use:

1. **`get_ec2_spend_last_day()`**: Retrieves EC2 spending data for the previous day
1. **`get_detailed_breakdown_by_day(days=7)`**: Delivers a comprehensive analysis of costs by region, service, and instance type
1. **`get_bedrock_daily_usage_stats(days=7, region='us-east-1', log_group_name='BedrockModelInvocationLogGroup')`**: Delivers a per-day breakdown of model usage by region and users.
1. **`get_bedrock_hourly_usage_stats(days=7, region='us-east-1', log_group_name='BedrockModelInvocationLogGroup')`**: Delivers a per-day per-hour breakdown of model usage by region and users.

### Example Queries

Once connected to Claude through an MCP-enabled interface, you can ask questions like:

- "Help me understand my Bedrock spend over the last few weeks"
- "What was my EC2 spend yesterday?"
- "Show me my top 5 AWS services by cost for the last month"
- "Analyze my spending by region for the past 14 days"
- "Which instance types are costing me the most money?"

## Docker Support

A Dockerfile is included for containerized deployment:

```
docker build -t aws-cost-explorer-mcp .
docker run -v ~/.aws:/root/.aws aws-cost-explorer-mcp
```

## Development

### Project Structure

- `server.py`: Main server implementation with MCP tools
- `pyproject.toml`: Project dependencies and metadata
- `Dockerfile`: Container definition for deployments

### Adding New Cost Analysis Tools

To extend the functionality:

1. Add new functions to `server.py`
2. Annotate them with `@mcp.tool()`
3. Implement the AWS Cost Explorer API calls
4. Format the results for easy readability

## License

[MIT License](LICENSE)

## Acknowledgments

- This tool uses Anthropic's MCP framework
- Powered by AWS Cost Explorer API
- Built with [FastMCP](https://github.com/jlowin/fastmcp) for server implementation
- README was generated by providing a text dump of the repo via [GitIngest](https://gitingest.com/) to Claude
