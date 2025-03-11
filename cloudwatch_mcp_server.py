"""
AWS CloudWatch Logs MCP Server.

This server provides MCP tools to interact with AWS CloudWatch Logs API.
"""

import boto3
from datetime import datetime, timedelta
import json
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


def get_bedrock_logs(days_back=7):
    """
    Retrieve Bedrock invocation logs from CloudWatch for the past n days.

    Args:
        days_back (int): Number of days of logs to retrieve (default: 7)

    Returns:
        list: List of dictionaries containing filtered log data
    """
    # Initialize CloudWatch Logs client
    client = boto3.client("logs")

    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)

    # Convert to milliseconds since epoch
    start_time_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)

    filtered_logs = []

    try:
        paginator = client.get_paginator("filter_log_events")

        # Parameters for the log query
        params = {
            "logGroupName": "/aws/bedrock",  # The main log group
            "logStreamNames": [
                "aws/bedrock/modelinvocations"
            ],  # The specific log stream
            "startTime": start_time_ms,
            "endTime": end_time_ms,
        }

        # Paginate through results
        for page in paginator.paginate(**params):
            for event in page.get("events", []):
                try:
                    # Parse the message as JSON
                    message = json.loads(event["message"])

                    # Get user prompt from the input messages
                    prompt = ""
                    if (
                        message.get("input", {})
                        .get("inputBodyJson", {})
                        .get("messages")
                    ):
                        for msg in message["input"]["inputBodyJson"]["messages"]:
                            if msg.get("role") == "user" and msg.get("content"):
                                for content in msg["content"]:
                                    if content.get("text"):
                                        prompt += content["text"] + " "
                        prompt = prompt.strip()

                    # Extract only the required fields
                    filtered_event = {
                        "timestamp": message.get("timestamp"),
                        "region": message.get("region"),
                        "modelId": message.get("modelId"),
                        "userId": message.get("identity", {}).get("arn"),
                        "inputTokens": message.get("input", {}).get("inputTokenCount"),
                        "completionTokens": message.get("output", {}).get(
                            "outputTokenCount"
                        ),
                        "totalTokens": (
                            message.get("input", {}).get("inputTokenCount", 0)
                            + message.get("output", {}).get("outputTokenCount", 0)
                        ),
                    }

                    filtered_logs.append(filtered_event)
                except json.JSONDecodeError:
                    continue  # Skip non-JSON messages
                except KeyError:
                    continue  # Skip messages missing required fields

        return filtered_logs

    except client.exceptions.ResourceNotFoundException:
        print(
            f"Log group '/aws/bedrock' or stream 'aws/bedrock/modelinvocations' not found"
        )
        return []
    except Exception as e:
        print(f"Error retrieving logs: {str(e)}")
        return []


class DaysParam(BaseModel):
    """Parameters for specifying the number of days to look back."""

    days: int = Field(default=7, description="Number of days to look back for logs")


# Initialize FastMCP server
mcp = FastMCP("aws_cloudwatch_logs")


@mcp.tool()
async def get_bedrock_logs_df(params: DaysParam) -> pd.DataFrame:
    """
    Retrieve Bedrock invocation logs as a pandas DataFrame.

    Args:
        params: Parameters specifying the number of days to look back

    Returns:
        pd.DataFrame: DataFrame containing the log data with columns:
            - timestamp: Timestamp of the invocation
            - region: AWS region
            - modelId: Bedrock model ID
            - userId: User ARN
            - inputTokens: Number of input tokens
            - completionTokens: Number of completion tokens
            - totalTokens: Total tokens used
    """
    # Get logs using existing function
    logs = get_bedrock_logs(days_back=params.days)

    # Convert to DataFrame
    df = pd.DataFrame(logs)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


@mcp.tool()
async def get_model_usage_stats(params: DaysParam) -> pd.DataFrame:
    """
    Get usage statistics grouped by model.

    Args:
        params: Parameters specifying the number of days to look back

    Returns:
        pd.DataFrame: DataFrame with model usage statistics
    """
    df = await get_bedrock_logs_df(params)

    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby("modelId")
        .agg(
            {
                "inputTokens": ["count", "sum", "mean"],
                "completionTokens": ["sum", "mean"],
                "totalTokens": ["sum", "mean"],
            }
        )
        .round(2)
    )


@mcp.tool()
async def get_user_usage_stats(params: DaysParam) -> pd.DataFrame:
    """
    Get usage statistics grouped by user.

    Args:
        params: Parameters specifying the number of days to look back

    Returns:
        pd.DataFrame: DataFrame with user usage statistics
    """
    df = await get_bedrock_logs_df(params)

    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby("userId")
        .agg(
            {
                "inputTokens": ["count", "sum", "mean"],
                "completionTokens": ["sum", "mean"],
                "totalTokens": ["sum", "mean"],
                "modelId": lambda x: list(set(x)),  # List of unique models used
            }
        )
        .round(2)
    )


@mcp.tool()
async def get_daily_usage_stats(params: DaysParam) -> pd.DataFrame:
    """
    Get daily usage statistics.

    Args:
        params: Parameters specifying the number of days to look back

    Returns:
        pd.DataFrame: DataFrame with daily usage statistics
    """
    df = await get_bedrock_logs_df(params)

    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby(df["timestamp"].dt.date)
        .agg(
            {
                "inputTokens": ["count", "sum", "mean"],
                "completionTokens": ["sum", "mean"],
                "totalTokens": ["sum", "mean"],
                "modelId": lambda x: list(set(x)),  # List of unique models used per day
            }
        )
        .round(2)
    )


def main():
    # Run the server with SSE transport
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
