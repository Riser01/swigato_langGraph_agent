#!/usr/bin/env python3
"""
Test script for the integrated MCP functionality in ConversationGraph.
"""

import asyncio
import sys
import os
from loguru import logger

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.conversation_graph import ConversationGraph


async def test_mcp_integration():
    """Test the MCP integration in ConversationGraph."""
    logger.info("🧪 Starting MCP Integration test...")
    
    try:
        # Create ConversationGraph instance (this will initialize MCP)
        graph = ConversationGraph()
        
        # Check if MCP tools are available
        is_available = graph.is_mcp_available()
        logger.info(f"📊 MCP Available: {is_available}")
        
        # Get tool count
        tool_count = len(graph.mcp_tools)
        logger.info(f"🛠️  Tool count: {tool_count}")
        
        # Get available tool names
        tool_names = graph.get_available_tools()
        logger.info(f"📋 Available tools: {tool_names}")
        
        # Get detailed status
        status = graph.get_mcp_status()
        logger.info(f"📊 Detailed status: {status}")
        
        # Test tool descriptions
        descriptions = graph.describe_tools()
        #!/usr/bin/env python3
"""
Test script for the integrated MCP functionality in ConversationGraph.
"""

import asyncio
import sys
import os
from loguru import logger

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.conversation_graph import ConversationGraph


async def test_mcp_integration():
    """Test the MCP integration in ConversationGraph."""
    logger.info("🧪 Starting MCP Integration test...")
    
    try:
        # Create ConversationGraph instance (this will initialize MCP)
        graph = ConversationGraph()
        
        # Check if MCP tools are available
        is_available = graph.is_mcp_available()
        logger.info(f"📊 MCP Available: {is_available}")
        
        # Get tool count
        tool_count = len(graph.mcp_tools)
        logger.info(f"🛠️  Tool count: {tool_count}")
        
        # Get available tool names
        tool_names = graph.get_available_tools()
        logger.info(f"� Available tools: {tool_names}")
        
        # Get detailed status
        status = graph.get_mcp_status()
        logger.info(f"📊 Detailed status: {status}")
        
        # Test tool descriptions
        descriptions = graph.describe_tools()
        logger.info(f"�📖 Tool descriptions:
{descriptions}")
        
        if tool_count > 0:
            logger.success(f"✅ MCP Integration test PASSED - {tool_count} tools loaded")
            
            # Test a simple chat interaction
            logger.info("🧪 Testing chat functionality...")
            response = graph.chat("Hello, can you help me?", "test-session")
            logger.info(f"💬 Chat response: {response['response'][:100]}...")
            
            return True
        else:
            logger.warning("⚠️  MCP Integration test completed with 0 tools (fallback mode)")
            return False
            
    except Exception as e:
        logger.error(f"❌ MCP Integration test FAILED: {str(e)}")
        logger.exception("Full error details:")
        return False
    finally:
        # Cleanup
        try:
            await graph.cleanup()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {str(e)}")


async def main():
    """Main test function."""
    logger.info("🚀 Starting MCP Integration tests...")
    
    success = await test_mcp_integration()
    
    if success:
        logger.success("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    # Run the async test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
        
        if tool_count > 0:
            logger.success(f"✅ MCP Integration test PASSED - {tool_count} tools loaded")
            
            # Test a simple chat interaction
            logger.info("🧪 Testing chat functionality...")
            response = graph.chat("Hello, can you help me?", "test-session")
            logger.info(f"💬 Chat response: {response['response'][:100]}...")
            
            return True
        else:
            logger.warning("⚠️  MCP Integration test completed with 0 tools (fallback mode)")
            return False
            
    except Exception as e:
        logger.error(f"❌ MCP Integration test FAILED: {str(e)}")
        logger.exception("Full error details:")
        return False
    finally:
        # Cleanup
        try:
            await graph.cleanup()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {str(e)}")


async def main():
    """Main test function."""
    logger.info("🚀 Starting MCP Integration tests...")
    
    success = await test_mcp_integration()
            logger.info(f"📊 MCP Client available: {is_available}")
            
            # Get tools
            tools = initialized_client.get_tools()
            logger.info(f"🛠️  Number of tools loaded: {len(tools)}")
            
            # List tool names
            tool_names = initialized_client.get_available_tool_names()
            logger.info(f"📋 Available tools: {tool_names}")
            
            # Get detailed status
            status = initialized_client.get_status()
            logger.info(f"📊 Detailed status: {status}")
            
            # Describe tools
            description = initialized_client.describe_tools()
            logger.info(f"📝 Tools description:\n{description}")
            
        logger.success("🎉 MCP Client test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ MCP Client test failed: {str(e)}")
        logger.exception("Full error details:")
        return False


async def main():
    """Main test function."""
    logger.info("🚀 Starting MCP Client integration test...")
    
    success = await test_mcp_client()
    
    if success:
        logger.success("✅ All tests passed!")
        return 0
    else:
        logger.error("❌ Tests failed!")
        return 1


if __name__ == "__main__":
    # Run the test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
