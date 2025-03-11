"""
AWS Cost Explorer MCP Server.

This server provides MCP tools to interact with AWS Cost Explorer API.
"""
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import boto3
import pandas as pd
import json
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from tabulate import tabulate


class DaysParam(BaseModel):
    """Parameters for specifying the number of days to look back."""
    
    days: int = Field(
        default=7,
        description="Number of days to look back for cost data"
    )

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

@mcp.tool()
async def get_ec2_spend_last_day() -> Dict[str, Any]:
    """
    Retrieve EC2 spend for the last day using standard AWS Cost Explorer API.
    
    Returns:
        Dict[str, Any]: The raw response from the AWS Cost Explorer API, or None if an error occurs.
    """
    # Initialize the Cost Explorer client
    ce_client = boto3.client('ce')
    
    # Calculate the time period - last day
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        # Make the API call using get_cost_and_usage (standard API)
        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Filter={
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': [
                        'Amazon Elastic Compute Cloud - Compute'
                    ]
                }
            },
            Metrics=[
                'UnblendedCost',
                'UsageQuantity'
            ],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'INSTANCE_TYPE'
                }
            ]
        )
        
        # Process and print the results
        print(f"EC2 Spend from {start_date} to {end_date}:")
        print("-" * 50)
        
        total_cost = 0.0
        
        if 'ResultsByTime' in response and response['ResultsByTime']:
            time_period_data = response['ResultsByTime'][0]
            
            if 'Groups' in time_period_data:
                for group in time_period_data['Groups']:
                    instance_type = group['Keys'][0]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    currency = group['Metrics']['UnblendedCost']['Unit']
                    usage = float(group['Metrics']['UsageQuantity']['Amount'])
                    
                    print(f"Instance Type: {instance_type}")
                    print(f"Cost: {cost:.4f} {currency}")
                    print(f"Usage: {usage:.2f}")
                    print("-" * 30)
                    
                    total_cost += cost
            
            # If no instance-level breakdown, show total
            if not time_period_data.get('Groups'):
                if 'Total' in time_period_data:
                    total = time_period_data['Total']
                    cost = float(total['UnblendedCost']['Amount'])
                    currency = total['UnblendedCost']['Unit']
                    print(f"Total EC2 Cost: {cost:.4f} {currency}")
                else:
                    print("No EC2 costs found for this period")
            else:
                print(f"Total EC2 Cost: {total_cost:.4f} {currency if 'currency' in locals() else 'USD'}")
                
            # Check if results are estimated
            if 'Estimated' in time_period_data:
                print(f"Note: These results are {'estimated' if time_period_data['Estimated'] else 'final'}")
        
        return response
        
    except Exception as e:
        print(f"Error retrieving EC2 cost data: {str(e)}")
        return None


@mcp.tool()
async def get_detailed_breakdown_by_day(params: DaysParam) -> str: #Dict[str, Any]:
    """
    Retrieve daily spend breakdown by region, service, and instance type.
    
    Args:
        params: Parameters specifying the number of days to look back
    
    Returns:
        Dict[str, Any]: A tuple containing:
            - A nested dictionary with cost data organized by date, region, and service
            - A string containing the formatted output report
        or (None, error_message) if an error occurs.
    """
    # Initialize the Cost Explorer client
    ce_client = boto3.client('ce')
    
    # Get the days parameter
    days = params.days
    
    # Calculate the time period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Initialize output buffer
    output_buffer = []
    
    try:
        output_buffer.append(f"\nDetailed Cost Breakdown by Region, Service, and Instance Type ({days} days):")
        output_buffer.append("-" * 75)
        
        # First get the daily costs by region and service
        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'REGION'
                },
                {
                    'Type': 'DIMENSION',
                    'Key': 'SERVICE'
                }
            ]
        )
        
        # Create data structure to hold the results
        all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Process the results
        for time_data in response['ResultsByTime']:
            date = time_data['TimePeriod']['Start']
            
            output_buffer.append(f"\nDate: {date}")
            output_buffer.append("=" * 50)
            
            if 'Groups' in time_data and time_data['Groups']:
                # Create data structure for this date
                region_services = defaultdict(lambda: defaultdict(float))
                
                # Process groups
                for group in time_data['Groups']:
                    region, service = group['Keys']
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    currency = group['Metrics']['UnblendedCost']['Unit']
                    
                    region_services[region][service] = cost
                    all_data[date][region][service] = cost
                
                # Add the results for this date to the buffer
                for region in sorted(region_services.keys()):
                    output_buffer.append(f"\nRegion: {region}")
                    output_buffer.append("-" * 40)
                    
                    # Create a DataFrame for this region's services
                    services_df = pd.DataFrame({
                        'Service': list(region_services[region].keys()),
                        'Cost': list(region_services[region].values())
                    })
                    
                    # Sort by cost descending
                    services_df = services_df.sort_values('Cost', ascending=False)
                    
                    # Get top services by cost
                    top_services = services_df.head(5)
                    
                    # Add region's services table to buffer
                    output_buffer.append(tabulate(top_services.round(2), headers='keys', tablefmt='pretty', showindex=False))
                    
                    # If there are more services, indicate the total for other services
                    if len(services_df) > 5:
                        other_cost = services_df.iloc[5:]['Cost'].sum()
                        output_buffer.append(f"... and {len(services_df) - 5} more services totaling {other_cost:.2f} {currency}")
                    
                    # For EC2, get instance type breakdown
                    if any(s.startswith('Amazon Elastic Compute') for s in region_services[region].keys()):
                        try:
                            instance_response = get_instance_type_breakdown(
                                ce_client, 
                                date, 
                                region, 
                                'Amazon Elastic Compute Cloud - Compute', 
                                'INSTANCE_TYPE'
                            )
                            
                            if instance_response:
                                output_buffer.append("\n  EC2 Instance Type Breakdown:")
                                output_buffer.append("  " + "-" * 38)
                                
                                # Get table with indentation
                                instance_table = tabulate(instance_response.round(2), headers='keys', tablefmt='pretty', showindex=False)
                                for line in instance_table.split('\n'):
                                    output_buffer.append(f"  {line}")
                        
                        except Exception as e:
                            output_buffer.append(f"  Note: Could not retrieve EC2 instance type breakdown: {str(e)}")
                    
                    # For SageMaker, get instance type breakdown
                    if any(s == 'Amazon SageMaker' for s in region_services[region].keys()):
                        try:
                            sagemaker_instance_response = get_instance_type_breakdown(
                                ce_client,
                                date,
                                region,
                                'Amazon SageMaker',
                                'INSTANCE_TYPE'
                            )
                            
                            if sagemaker_instance_response is not None and not sagemaker_instance_response.empty:
                                output_buffer.append("\n  SageMaker Instance Type Breakdown:")
                                output_buffer.append("  " + "-" * 38)
                                
                                # Get table with indentation
                                sagemaker_table = tabulate(sagemaker_instance_response.round(2), headers='keys', tablefmt='pretty', showindex=False)
                                for line in sagemaker_table.split('\n'):
                                    output_buffer.append(f"  {line}")
                            
                            # Also try to get usage type breakdown for SageMaker (notebooks, endpoints, etc.)
                            sagemaker_usage_response = get_instance_type_breakdown(
                                ce_client,
                                date,
                                region,
                                'Amazon SageMaker',
                                'USAGE_TYPE'
                            )
                            
                            if sagemaker_usage_response is not None and not sagemaker_usage_response.empty:
                                output_buffer.append("\n  SageMaker Usage Type Breakdown:")
                                output_buffer.append("  " + "-" * 38)
                                
                                # Get table with indentation
                                usage_table = tabulate(sagemaker_usage_response.round(2), headers='keys', tablefmt='pretty', showindex=False)
                                for line in usage_table.split('\n'):
                                    output_buffer.append(f"  {line}")
                        
                        except Exception as e:
                            output_buffer.append(f"  Note: Could not retrieve SageMaker breakdown: {str(e)}")
            else:
                output_buffer.append("No data found for this date")
            
            output_buffer.append("\n" + "-" * 75)
        
        # Join the buffer into a single string
        formatted_output = "\n".join(output_buffer)
        
        # Return both the raw data and the formatted output
        #return {"data": all_data, "formatted_output": formatted_output}
        return formatted_output
    
    except Exception as e:
        error_message = f"Error retrieving detailed breakdown: {str(e)}"
        #return {"data": None, "formatted_output": error_message}
        return error_message

def get_instance_type_breakdown(ce_client, date, region, service, dimension_key):
    """
    Helper function to get instance type or usage type breakdown for a specific service.
    
    Args:
        ce_client: The Cost Explorer client
        date: The date to query
        region: The AWS region
        service: The AWS service name
        dimension_key: The dimension to group by (e.g., 'INSTANCE_TYPE' or 'USAGE_TYPE')
    
    Returns:
        DataFrame containing the breakdown or None if no data
    """
    tomorrow = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    
    instance_response = ce_client.get_cost_and_usage(
        TimePeriod={
            'Start': date,
            'End': tomorrow
        },
        Granularity='DAILY',
        Filter={
            'And': [
                {
                    'Dimensions': {
                        'Key': 'REGION',
                        'Values': [region]
                    }
                },
                {
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': [service]
                    }
                }
            ]
        },
        Metrics=['UnblendedCost'],
        GroupBy=[
            {
                'Type': 'DIMENSION',
                'Key': dimension_key
            }
        ]
    )
    
    if ('ResultsByTime' in instance_response and 
        instance_response['ResultsByTime'] and 
        'Groups' in instance_response['ResultsByTime'][0] and 
        instance_response['ResultsByTime'][0]['Groups']):
        
        instance_data = instance_response['ResultsByTime'][0]
        instance_costs = []
        
        for instance_group in instance_data['Groups']:
            type_value = instance_group['Keys'][0]
            cost_value = float(instance_group['Metrics']['UnblendedCost']['Amount'])
            
            # Add a better label for the dimension used
            column_name = 'Instance Type' if dimension_key == 'INSTANCE_TYPE' else 'Usage Type'
            
            instance_costs.append({
                column_name: type_value,
                'Cost': cost_value
            })
        
        # Create DataFrame and sort by cost
        result_df = pd.DataFrame(instance_costs)
        if not result_df.empty:
            result_df = result_df.sort_values('Cost', ascending=False)
            return result_df
    
    return None

def main():
    # Run the server with SSE transport
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
    