"""
MCP Client for Zwigato Customer Support Tools

This module handles the connection to the MCP server and loads tools
for customer support operations like order management and wiki search.
"""

import os
import asyncio
from typing import List, Optional
from loguru import logger


class MCPClient:
    """
    Client for connecting to and loading tools from MCP servers.
    
    This class manages the connection to the Zwigato customer support MCP server
    and provides methods to load and use the available tools.
    
    Note: This is a simplified version that gracefully handles MCP loading failures.
    """
    
    def __init__(self):
        """Initialize the MCP client."""
        self.mcp_tools = []
        self._load_tools()
    
    def _load_tools(self):
        """Load tools from the MCP server."""
        try:
            logger.info("Starting MCP tools loading process...")
            
            # Try to import and load MCP tools
            try:
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client
                from langchain_mcp_adapters.tools import load_mcp_tools
                
                logger.info("✅ MCP libraries imported successfully")
                
                # Create server parameters for the Zwigato MCP server
                server_params = StdioServerParameters(
                    command="python",
                    args=["./mcp_server_remote.py"]
                )
                
                logger.info(f"📝 MCP server parameters created: {server_params.command} {' '.join(server_params.args)}")
                
                # Load MCP tools synchronously using the adapter
                try:
                    logger.info("🔄 Attempting to load MCP tools...")
                    self.mcp_tools = load_mcp_tools(server_params)
                    
                    if self.mcp_tools:
                        logger.success(f"✅ Successfully loaded {len(self.mcp_tools)} MCP tools")
                        for i, tool in enumerate(self.mcp_tools, 1):
                            logger.info(f"   {i}. {tool.name} - {getattr(tool, 'description', 'No description')}")
                    else:
                        logger.warning("⚠️  MCP tools loaded but list is empty")
                        
                except Exception as load_error:
                    logger.error(f"❌ Failed to load MCP tools from server: {str(load_error)}")
                    logger.info("🔄 Falling back to no MCP tools")
                    self.mcp_tools = []
                
            except ImportError as e:
                logger.warning(f"❌ MCP libraries not available: {str(e)}")
                logger.info("📋 Required packages: mcp, langchain-mcp-adapters")
                self.mcp_tools = []
                
        except Exception as e:
            logger.error(f"❌ Critical error in MCP tools loading: {str(e)}")
            logger.exception("Full error details:")
            self.mcp_tools = []
            
        # Final status report
        if self.mcp_tools:
            logger.success(f"🎉 MCP Client initialized with {len(self.mcp_tools)} tools")
        else:
            logger.warning("⚠️  MCP Client initialized with NO tools - running in fallback mode")
    
    def get_tools(self) -> List:
        """
        Get the loaded MCP tools.
        
        Returns:
            List of MCP tools ready for use with LangGraph
        """
        tool_count = len(self.mcp_tools)
        logger.info(f"📊 MCP Client - Returning {tool_count} tools")
        
        if not self.mcp_tools:
            logger.warning("⚠️  No MCP tools available - agent will run without external tools")
        else:
            logger.info(f"🔧 Available tools: {[tool.name for tool in self.mcp_tools]}")

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
        available = bool(self.mcp_tools)
        logger.debug(f"🔍 MCP Client availability check: {available} (tools: {len(self.mcp_tools)})")
        return available
    
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