#!/usr/bin/env python3
"""
MCP Server for LangGraph Agent

This MCP server wraps a LangGraph agent and exposes its capabilities
through the Model Context Protocol, allowing other agents to interact
with the LangGraph agent's tools and conversation capabilities.
"""

import os
import sys
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import InitializationOptions
from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types

# Import the LangGraph agent
from langgraph_agent import Agent, AgentState, tools, model, prompt
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mcp_langgraph_server')


class LangGraphAgentManager:
    """Manager for the LangGraph agent with conversation history."""
    
    def __init__(self):
        self.agent = Agent(model, tools, system=prompt)
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_id = None
        
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new conversation session."""
        self.session_id = session_id
        self.conversation_history = []
        
        return {
            "session_id": session_id,
            "status": "created",
            "available_tools": list(self.agent.tools.keys()),
            "system_prompt": self.agent.system
        }
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history."""
        return self.conversation_history[-limit:] if limit > 0 else self.conversation_history
    
    def process_message(self, message: str, include_tools: bool = True) -> Dict[str, Any]:
        """Process a message through the LangGraph agent."""
        try:
            # Create message for the agent
            human_message = HumanMessage(content=message)
            
            # Get the current state from conversation history
            messages = []
            for entry in self.conversation_history:
                if entry["role"] == "human":
                    messages.append(HumanMessage(content=entry["content"]))
                elif entry["role"] == "assistant":
                    messages.append(AIMessage(content=entry["content"]))
                elif entry["role"] == "system":
                    messages.append(SystemMessage(content=entry["content"]))
            
            # Add current message
            messages.append(human_message)
            
            # Invoke the agent
            result = self.agent.graph.invoke({"messages": messages})
            
            # Extract the response
            response_messages = result.get("messages", [])
            
            # Get the last AI message
            ai_response = None
            tool_calls = []
            
            for msg in response_messages:
                if hasattr(msg, 'content') and msg.content:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_calls.extend(msg.tool_calls)
                    ai_response = msg.content
            
            # Store in conversation history
            self.conversation_history.append({
                "role": "human",
                "content": message,
                "timestamp": self._get_timestamp()
            })
            
            if ai_response:
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": ai_response,
                    "tool_calls": tool_calls if include_tools else [],
                    "timestamp": self._get_timestamp()
                })
            
            return {
                "response": ai_response or "No response generated",
                "tool_calls": tool_calls if include_tools else [],
                "conversation_length": len(self.conversation_history),
                "session_id": self.session_id
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "error": str(e),
                "response": "I encountered an error processing your message.",
                "tool_calls": [],
                "conversation_length": len(self.conversation_history)
            }
    
    def execute_tool_directly(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool directly."""
        try:
            if tool_name not in self.agent.tools:
                return {
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": list(self.agent.tools.keys())
                }
            
            tool = self.agent.tools[tool_name]
            result = tool.invoke(parameters)
            
            return {
                "tool_name": tool_name,
                "result": result,
                "parameters": parameters
            }
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {
                "error": str(e),
                "tool_name": tool_name,
                "parameters": parameters
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and capabilities."""
        return {
            "session_id": self.session_id,
            "conversation_length": len(self.conversation_history),
            "available_tools": [
                {
                    "name": name,
                    "description": tool.description if hasattr(tool, 'description') else "No description",
                    "schema": tool.args_schema.schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}
                }
                for name, tool in self.agent.tools.items()
            ],
            "system_prompt": self.agent.system,
            "model_info": {
                "model_name": getattr(self.agent.model, 'model_name', 'gpt-3.5-turbo'),
                "temperature": getattr(self.agent.model, 'temperature', 0.7)
            }
        }
    
    def reset_conversation(self) -> Dict[str, Any]:
        """Reset the conversation history."""
        history_length = len(self.conversation_history)
        self.conversation_history = []
        
        return {
            "status": "reset",
            "previous_length": history_length,
            "session_id": self.session_id
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()


async def main():
    """Main function to run the MCP LangGraph server."""
    logger.info("Starting MCP LangGraph Agent Server")
    
    # Initialize the agent manager
    agent_manager = LangGraphAgentManager()
    
    # Create MCP server
    server = Server("langgraph-agent")
    
    @server.list_resources()
    async def handle_list_resources() -> List[types.Resource]:
        """List available resources."""
        return [
            types.Resource(
                uri="conversation://history",
                name="Conversation History",
                description="Chat conversation history with the LangGraph agent",
                mimeType="application/json",
            ),
            types.Resource(
                uri="agent://status",
                name="Agent Status",
                description="Current status and capabilities of the LangGraph agent",
                mimeType="application/json",
            )
        ]
    
    @server.read_resource()
    async def handle_read_resource(uri: str) -> str:
        """Read resource content."""
        if uri == "conversation://history":
            history = agent_manager.get_conversation_history()
            return json.dumps(history, indent=2)
        elif uri == "agent://status":
            status = agent_manager.get_agent_status()
            return json.dumps(status, indent=2)
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    @server.list_prompts()
    async def handle_list_prompts() -> List[types.Prompt]:
        """List available prompts."""
        return [
            types.Prompt(
                name="research-assistant",
                description="Intelligent research assistant prompt for the LangGraph agent",
                arguments=[
                    types.PromptArgument(
                        name="query",
                        description="Research query or question to ask the agent",
                        required=True,
                    ),
                    types.PromptArgument(
                        name="context",
                        description="Additional context for the query",
                        required=False,
                    )
                ],
            ),
            types.Prompt(
                name="conversation-starter",
                description="Start a new conversation with the LangGraph agent",
                arguments=[
                    types.PromptArgument(
                        name="session_id",
                        description="Session ID for the conversation",
                        required=False,
                    )
                ],
            )
        ]
    
    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: Dict[str, str] = None) -> types.GetPromptResult:
        """Get prompt content."""
        if name == "research-assistant":
            query = arguments.get("query", "") if arguments else ""
            context = arguments.get("context", "") if arguments else ""
            
            prompt_text = f"""You are working with a LangGraph agent that has access to search tools and calculation capabilities.

Query: {query}

{f"Context: {context}" if context else ""}

The agent can:
1. Search the web for information using Tavily search
2. Perform mathematical calculations
3. Chain multiple operations together
4. Provide detailed, well-researched responses

Please process this query using the available tools as needed."""
            
            return types.GetPromptResult(
                description=f"Research assistant prompt for query: {query}",
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(type="text", text=prompt_text),
                    )
                ],
            )
        
        elif name == "conversation-starter":
            session_id = arguments.get("session_id", f"session_{agent_manager._get_timestamp()}") if arguments else f"session_{agent_manager._get_timestamp()}"
            
            session_info = agent_manager.create_session(session_id)
            
            prompt_text = f"""Starting new conversation with LangGraph Agent

Session ID: {session_id}
Available Tools: {', '.join(session_info['available_tools'])}

You can now chat with the intelligent research assistant. The agent has access to:
- Web search capabilities
- Mathematical calculations
- Multi-step reasoning

What would you like to explore or ask about?"""
            
            return types.GetPromptResult(
                description=f"Conversation starter for session {session_id}",
                messages=[
                    types.PromptMessage(
                        role="assistant",
                        content=types.TextContent(type="text", text=prompt_text),
                    )
                ],
            )
        
        else:
            raise ValueError(f"Unknown prompt: {name}")
    
    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        """List available tools."""
        return [
            types.Tool(
                name="chat_with_agent",
                description="Send a message to the LangGraph agent and get a response",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to send to the agent"
                        },
                        "include_tools": {
                            "type": "boolean",
                            "description": "Whether to include tool call information in response",
                            "default": True
                        }
                    },
                    "required": ["message"],
                },
            ),
            types.Tool(
                name="execute_tool",
                description="Execute a specific tool directly",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool to execute",
                            "enum": ["calculate_operators", "tavily_search_results_json"]
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Parameters for the tool"
                        }
                    },
                    "required": ["tool_name", "parameters"],
                },
            ),
            types.Tool(
                name="get_conversation_history",
                description="Get the conversation history",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of messages to return",
                            "default": 10
                        }
                    },
                },
            ),
            types.Tool(
                name="create_session",
                description="Create a new conversation session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID"
                        }
                    },
                },
            ),
            types.Tool(
                name="get_agent_status",
                description="Get current agent status and capabilities",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="reset_conversation",
                description="Reset the conversation history",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]
    
    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: Dict[str, Any] = None
    ) -> List[types.TextContent]:
        """Handle tool execution requests."""
        try:
            arguments = arguments or {}
            
            if name == "chat_with_agent":
                message = arguments.get("message", "")
                include_tools = arguments.get("include_tools", True)
                
                if not message:
                    raise ValueError("Message is required")
                
                result = agent_manager.process_message(message, include_tools)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "execute_tool":
                tool_name = arguments.get("tool_name", "")
                parameters = arguments.get("parameters", {})
                
                if not tool_name:
                    raise ValueError("Tool name is required")
                
                result = agent_manager.execute_tool_directly(tool_name, parameters)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "get_conversation_history":
                limit = arguments.get("limit", 10)
                history = agent_manager.get_conversation_history(limit)
                return [types.TextContent(type="text", text=json.dumps(history, indent=2))]
            
            elif name == "create_session":
                session_id = arguments.get("session_id")
                if not session_id:
                    session_id = f"session_{agent_manager._get_timestamp()}"
                
                result = agent_manager.create_session(session_id)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "get_agent_status":
                result = agent_manager.get_agent_status()
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "reset_conversation":
                result = agent_manager.reset_conversation()
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        except Exception as e:
            logger.error(f"Error executing tool {name}: {str(e)}")
            error_result = {
                "error": str(e),
                "tool": name,
                "arguments": arguments
            }
            return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("LangGraph MCP Server running with stdio transport")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="langgraph-agent",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


# Run the server
asyncio.run(main()) 