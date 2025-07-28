"""
MCP Client for Zwigato Customer Support Tools

This module handles the connection to the MCP server and loads tools
for customer support operations like order management and wiki search.
"""

import os
import asyncio
from typing import List, Optional
from loguru import logger
from langchain_mcp_adapters.tools import load_mcp_tools


class MCPClient:
    """
    Client for connecting to and loading tools from MCP servers.
    
    This class manages the connection to the Zwigato customer support MCP server
    and provides methods to load and use the available tools.
    """
    
    def __init__(self):
        """Initialize the MCP client."""
        self.mcp_tools = None
        self.server_config = self._get_server_config()
        self._load_tools()
    
    def _get_server_config(self) -> dict:
        """
        Get the MCP server configuration.
        
        Returns:
            Dictionary containing server configuration
        """
        return {
            "CustomerSupportAssistantTools": {
                "command": "python",
                "args": ["./mcp_server_remote.py"],
                "transport": "stdio"
            }
        }
    
    def _load_tools(self):
        """Load tools from the MCP server."""
        try:
            logger.info("Loading MCP tools...")
            
            # Load tools from the MCP server
            self.mcp_tools = load_mcp_tools(
                server_configs=self.server_config,
                # Optional: specify which tools to load
                # tool_names=["search_wiki", "read_order_status", "update_order_status"]
            )
            
            logger.success(f"Successfully loaded {len(self.mcp_tools)} MCP tools")
            
            # Log available tools
            for tool in self.mcp_tools:
                logger.info(f"Available MCP tool: {tool.name}")
                
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {str(e)}")
            self.mcp_tools = []
    
    def get_tools(self) -> List:
        """
        Get the loaded MCP tools.
        
        Returns:
            List of MCP tools ready for use with LangGraph
        """
        if self.mcp_tools is None:
            logger.warning("MCP tools not loaded, attempting to reload...")
            self._load_tools()
        
        return self.mcp_tools or []
    
    def get_tool_by_name(self, tool_name: str) -> Optional[object]:
        """
        Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            The tool if found, None otherwise
        """
        tools = self.get_tools()
        for tool in tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def is_available(self) -> bool:
        """
        Check if MCP tools are available.
        
        Returns:
            True if tools are loaded, False otherwise
        """
        return bool(self.mcp_tools)
    
    def get_available_tool_names(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List of tool names
        """
        tools = self.get_tools()
        return [tool.name for tool in tools]
    
    def describe_tools(self) -> str:
        """
        Get a description of all available tools.
        
        Returns:
            String description of tools
        """
        tools = self.get_tools()
        if not tools:
            return "No MCP tools available"
        
        descriptions = []
        for tool in tools:
            descriptions.append(f"- {tool.name}: {getattr(tool, 'description', 'No description available')}")
        
        return "Available MCP Tools:\n" + "\n".join(descriptions)


# Global instance for use throughout the application
mcp_client = MCPClient()