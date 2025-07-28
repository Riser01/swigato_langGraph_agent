import uuid
import os
import warnings
import asyncio
import nest_asyncio
from typing import Dict, Any, List, Union, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from .state import ChatState

# Apply nest_asyncio to allow running async code from sync contexts
nest_asyncio.apply()


class UnifiedChatbotService:
    """
    Unified chatbot service that combines simple OpenAI chat with advanced 
    LangGraph ReAct agent capabilities and MCP tool integration.
    
    This class provides both basic chat functionality and advanced tool-based
    conversation management in a single interface.
    """
    
    def __init__(self):
        """Initialize the unified chatbot service."""
        # Memory for conversation state
        self.memory = MemorySaver()
        
        # Initialize LLM with fallback support
        self.llm = self._initialize_llm()
        
        # MCP integration
        self._mcp_client = None
        self.mcp_tools = []
        self._initialized = False
        
        # Default MCP configuration
        self.mcp_config = {
            "zwigato-support": {
                "command": "python",
                "args": ["./mcp_server_remote.py"],
                "transport": "stdio"
            }
        }
        
        # System prompt template for simple chat mode
        self.simple_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """<ROLE>
You are a smart agent with an ability to use tools. 
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question. 
If you are failed to answer the question, try different tools to get context.
Your answer should be very polite and professional.
</ROLE>
----
<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question is consist of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- If you are failed to answer the question, try different tools to get context.

Step 3: Answer the question
- Answer the question in the same language as the question.
- Your answer should be very polite and professional.

Step 4: Provide the source of the answer(if applicable)
- If you've used the tool, provide the source of the answer.
- Valid sources are either a website(URL) or a document(PDF, etc).

Guidelines:
- If you've used the tool, your answer should be based on the tool's output(tool's output is more important than your own knowledge).
- If you've used the tool, and the source is valid URL, provide the source(URL) of the answer.
- Skip providing the source if the source is not URL.
- Answer in the same language as the question.
- Answer should be concise and to the point.
- Avoid response your output with any other information than the answer and the source.  
</INSTRUCTIONS>
----
<OUTPUT_FORMAT>
(concise answer to the question)


</OUTPUT_FORMAT>
            
            Current conversation turn: {turn_count}
            Context: {context}
            Tool Results: {tool_results}"""),
            ("human", "{user_input}")
        ])
        
        # System message for ReAct agent mode
        self.react_system_message = """<ROLE>
You are a smart Zwigato customer support agent with an ability to use tools. 
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question. 
If you fail to answer the question, try different tools to get context.
Your answer should be very polite and professional.
</ROLE>
----
<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question consists of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- **If the query is a simple greeting or does not require a tool, answer directly in a polite and professional manner.**
- If you fail to answer the question, try different tools to get context.

Step 3: Answer the question
- Answer the question in the same language as the question.
- Your answer should be very polite and professional.

Step 4: Provide the source of the answer (if applicable)
- If you've used a tool, provide the source of the answer.
- Valid sources are either a website (URL) or a document (PDF, etc).

Guidelines:
- If you've used a tool, your answer should be based on the tool's output (tool's output is more important than your own knowledge).
- If you've used a tool and the source is a valid URL, provide the source (URL) of the answer.
- Skip providing the source if the source is not a URL.
- Answer in the same language as the question.
- Answer should be concise and to the point.
- Always think step by step and explain your reasoning process.
- Show intermediate steps when using tools.
</INSTRUCTIONS>
----
<OUTPUT_FORMAT>
(concise answer to the question)

Source: (if applicable)
</OUTPUT_FORMAT>"""
        
        # Initialize MCP tools and ReAct agent
        try:
            logger.info("Starting unified chatbot initialization...")
            asyncio.run(self._async_initialize_mcp_tools())
        except Exception as e:
            logger.error(f"❌ Critical error during MCP initialization: {e}")
        
        # Create the ReAct agent
        self.agent = self._create_react_agent()
    
    def _initialize_llm(self) -> Union[ChatOpenAI, ChatGoogleGenerativeAI]:
        """Initialize the LLM with preference for OpenAI, fallback to Google Gemini."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        provider_preference = os.getenv("PROVIDER_PREFERENCE", "openai").lower()
        
        # Prefer OpenAI if available and preference is set to openai
        if provider_preference == "openai" and openai_api_key:
            try:
                model_name = os.getenv("MODEL", "gpt-4o")
                logger.info(f"Using OpenAI: {model_name}")
                return ChatOpenAI(
                    model=model_name,
                    temperature=0.7,
                    api_key=openai_api_key,
                    streaming=True
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {str(e)}")
        
        # Fallback to Google Gemini
        if google_api_key:
            try:
                model_name = os.getenv("FALLBACK_MODEL", "gemini-1.5-flash")
                logger.info(f"Using Google Gemini: {model_name}")
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.7,
                    google_api_key=google_api_key,
                    streaming=True
                )
            except Exception as e:
                logger.error(f"Failed to initialize Google Gemini: {str(e)}")
        
        # Try OpenAI as last resort if Google Gemini failed
        if openai_api_key:
            try:
                model_name = os.getenv("MODEL", "gpt-4o")
                logger.info(f"Using OpenAI as fallback: {model_name}")
                return ChatOpenAI(
                    model=model_name,
                    temperature=0.7,
                    api_key=openai_api_key,
                    streaming=True
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI as fallback: {str(e)}")
        
        # If neither works, raise an error
        raise ValueError(
            "No valid API key found. Please set either OPENAI_API_KEY or GOOGLE_API_KEY environment variable."
        )

    async def _async_initialize_mcp_tools(self):
        """Asynchronously starts the MCP client and loads the tools."""
        try:
            logger.info("🔄 Starting MCP client initialization...")
            
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient
                logger.info("✅ MCP libraries imported successfully")
                
                self._mcp_client = MultiServerMCPClient(self.mcp_config)
                logger.info("✅ MCP client created successfully")
                
                self.mcp_tools = await self._mcp_client.get_tools()
                logger.info("✅ MCP tools retrieved successfully")
                
                if self.mcp_tools:
                    logger.success(f"✅ Successfully loaded {len(self.mcp_tools)} MCP tools")
                    for i, tool in enumerate(self.mcp_tools, 1):
                        logger.info(f"   {i}. {tool.name} - {getattr(tool, 'description', 'No description')}")
                else:
                    logger.warning("⚠️  MCP tools loaded but list is empty")
                
                self._initialized = True
                
            except ImportError as e:
                logger.warning(f"❌ MCP libraries not available: {str(e)}")
                logger.info("📋 Required packages: langchain-mcp-adapters")
                self.mcp_tools = []
                
        except Exception as e:
            logger.error(f"❌ Critical error in MCP client initialization: {str(e)}")
            logger.exception("Full error details:")
            self.mcp_tools = []
            await self._cleanup_mcp()
        
        # Final status report
        if self.mcp_tools:
            logger.success(f"🎉 MCP Client initialized with {len(self.mcp_tools)} tools")
        else:
            logger.warning("⚠️  MCP Client initialized with NO tools - running in fallback mode")

    def _create_react_agent(self):
        """Create a ReAct agent with MCP tools."""
        tool_count = len(self.mcp_tools)
        logger.info(f"🤖 Creating ReAct agent with {tool_count} tools")
        
        if self.mcp_tools:
            tool_names = [getattr(tool, 'name', 'Unknown') for tool in self.mcp_tools]
            logger.info(f"🛠️  Agent will use tools: {tool_names}")
        else:
            logger.warning("⚠️  Agent will be created WITHOUT tools (basic responses only)")

        try:
            agent = create_react_agent(
                model=self.llm,
                tools=self.mcp_tools,
                checkpointer=self.memory
            )
            
            logger.success(f"✅ ReAct agent created successfully with {tool_count} tools")
            return agent
            
        except Exception as e:
            logger.error(f"❌ Failed to create ReAct agent: {str(e)}")
            logger.exception("Full error details:")
            raise

    async def _cleanup_mcp(self):
        """Cleanup MCP client resources."""
        if self._mcp_client:
            try:
                await self._mcp_client.close()
                logger.info("✅ MCP client closed successfully")
            except Exception as e:
                logger.error(f"❌ Error closing MCP client: {str(e)}")

    async def cleanup(self):
        """Gracefully shuts down the MCP client and its subprocesses."""
        await self._cleanup_mcp()
    
    # ========== SIMPLE CHAT MODE METHODS (from ChatbotService) ==========
    
    def set_mcp_tools(self, tools: List):
        """
        Set the MCP tools for this service.
        
        Args:
            tools: List of MCP tools
        """
        self.mcp_tools = tools
        logger.info(f"Set {len(tools)} MCP tools for chatbot service")
    
    def process_message(self, state: ChatState) -> Dict[str, Any]:
        """
        Process a user message using simple chat mode (without ReAct agent).
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with bot response
        """
        try:
            logger.info(f"Processing message for session {state.get('session_id', 'unknown')}")
            
            # Prepare the context from recent messages
            context = self._build_context(state.get("messages", []))
            
            # Format the prompt
            formatted_prompt = self.simple_prompt_template.format_messages(
                user_input=state["user_input"],
                context=context,
                turn_count=state.get("turn_count", 0),
                tool_results=""  # No tools in simple mode
            )
            
            # Generate response using LLM
            response = self.llm.invoke(formatted_prompt)
            
            logger.success("Successfully generated response")
            
            return {
                "bot_response": response.content,
                "context": context,
                "error": "",
                "turn_count": state.get("turn_count", 0) + 1,
                "messages": [
                    HumanMessage(content=state["user_input"]),
                    AIMessage(content=response.content)
                ]
            }
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return {
                "bot_response": "I apologize, but I encountered an error while processing your message. Please try again.",
                "error": error_msg,
                "turn_count": state.get("turn_count", 0),
                "messages": [
                    HumanMessage(content=state["user_input"]),
                    AIMessage(content="I apologize, but I encountered an error while processing your message. Please try again.")
                ]
            }
    
    async def stream_response(self, state: ChatState):
        """
        Stream response for real-time chat experience (simple mode).
        
        Args:
            state: Current conversation state
            
        Yields:
            Streamed response chunks
        """
        try:
            context = self._build_context(state.get("messages", []))
            
            formatted_prompt = self.simple_prompt_template.format_messages(
                user_input=state["user_input"],
                context=context,
                turn_count=state.get("turn_count", 0),
                tool_results=""
            )
            
            # Stream the response
            async for chunk in self.llm.astream(formatted_prompt):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield "I apologize, but I encountered an error while processing your message."
    
    def _build_context(self, messages: list) -> str:
        """
        Build context string from recent conversation messages.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Context string summarizing recent conversation
        """
        if not messages:
            return "Starting a new conversation."
        
        # Get last few messages for context (max 5 exchanges)
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        context_parts = []
        for msg in recent_messages:
            if hasattr(msg, 'content'):
                msg_type = "User" if isinstance(msg, HumanMessage) else "Assistant"
                # Truncate very long messages
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                context_parts.append(f"{msg_type}: {content}")
        
        return " | ".join(context_parts) if context_parts else "New conversation."
    
    # ========== REACT AGENT MODE METHODS (from ConversationGraph) ==========
    
    def chat(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """
        Process user input through the ReAct agent (advanced mode with tools).
        
        Args:
            user_input: The user's message
            session_id: Session identifier for conversation continuity
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            logger.info(f"💬 Processing message through ReAct agent: {user_input[:50]}...")
            
            # Log current setup
            tool_count = len(self.mcp_tools)
            logger.info(f"🛠️  Agent has {tool_count} tools available")
            
            # Fallback to direct LLM if no tools are available
            if tool_count == 0:
                logger.warning("⚠️  No tools available, using direct LLM response")
                try:
                    response = self.llm.invoke([HumanMessage(content=user_input)])
                    return {
                        "response": response.content,
                        "intermediate_steps": ["Direct LLM response (no tools available)"],
                        "error": "",
                        "tools_used": 0
                    }
                except Exception as e:
                    logger.error(f"❌ Direct LLM response failed: {str(e)}")
                    return {
                        "response": "I apologize, but I'm having technical difficulties. Please try again later.",
                        "intermediate_steps": [f"❌ Error: {str(e)}"],
                        "error": str(e),
                        "tools_used": 0
                    }
            
            # Create the thread config for session management
            config = {"configurable": {"thread_id": session_id}}
            
            # Prepare input messages with system message first
            messages = [
                SystemMessage(content=self.react_system_message),
                HumanMessage(content=user_input)
            ]
            
            # Track intermediate steps and final response
            intermediate_steps = []
            final_response = ""
            
            # Stream the agent's response to capture intermediate steps
            logger.info("🤖 ReAct Agent: Starting to process query...")
            
            # Add warning suppression for Google Gemini FinishReason enum issue
            warnings.filterwarnings("ignore", message="Unrecognized FinishReason enum value")
            
            for chunk in self.agent.stream({"messages": messages}, config, stream_mode="values"):
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if hasattr(message, 'content') and message.content:
                            if isinstance(message, AIMessage):
                                # Extract tool usage information
                                if hasattr(message, 'tool_calls') and message.tool_calls:
                                    for tool_call in message.tool_calls:
                                        tool_name = tool_call.get('name', 'Unknown')
                                        intermediate_steps.append(f"🛠️  Using tool: {tool_name}")
                                        logger.info(f"🛠️  ReAct Agent: Using tool {tool_name}")
                                
                                # Capture the final response
                                if not hasattr(message, 'tool_calls') or not message.tool_calls:
                                    final_response = message.content
                                    logger.info(f"💬 ReAct Agent: Final response captured")
            
            # If no final response was captured, get the last AI message
            if not final_response:
                try:
                    state = self.agent.get_state(config)
                    for msg in reversed(state.values.get("messages", [])):
                        if isinstance(msg, AIMessage) and msg.content and (not hasattr(msg, 'tool_calls') or not msg.tool_calls):
                            final_response = msg.content
                            break
                except Exception as e:
                    logger.warning(f"⚠️  Could not retrieve state: {str(e)}")
            
            # Clean up warnings
            warnings.filterwarnings("default")
            
            if not final_response:
                final_response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                logger.warning("⚠️  No final response captured, using fallback")
            
            # Calculate tools used
            tools_used_count = len([step for step in intermediate_steps if "Using tool:" in step])
            
            logger.success(f"✅ ReAct agent completed processing (Tools used: {tools_used_count})")
            
            return {
                "response": final_response,
                "intermediate_steps": intermediate_steps,
                "error": "",
                "tools_used": tools_used_count
            }
            
        except Exception as e:
            error_msg = f"Error in ReAct agent processing: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.exception("Full error trace:")
            
            # Provide more helpful error message based on error type
            if "'int' object has no attribute 'name'" in str(e):
                error_explanation = "Tool configuration error - this usually means the MCP tools weren't properly loaded"
                fallback_response = "I'm having trouble accessing my tools right now. Let me try to help you directly."
            else:
                error_explanation = f"Agent processing error: {str(e)}"
                fallback_response = "I apologize, but I encountered an error while processing your request. Please try again."
            
            logger.info(f"🔧 Error explanation: {error_explanation}")
            
            return {
                "response": fallback_response,
                "intermediate_steps": [f"❌ Error: {error_explanation}"],
                "error": error_msg,
                "tools_used": 0
            }
    
    # ========== UTILITY METHODS ==========
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        tool_names = []
        for tool in self.mcp_tools:
            tool_name = getattr(tool, 'name', 'Unknown')
            tool_names.append(tool_name)
        
        logger.info(f"📋 Available tool names: {tool_names}")
        return tool_names

    def get_mcp_status(self) -> Dict[str, Any]:
        """Get detailed MCP connection status and tool information."""
        try:
            mcp_available = self.is_mcp_available()
            tool_count = len(self.mcp_tools)
            tool_names = self.get_available_tools()
            
            status = {
                "mcp_available": mcp_available,
                "tool_count": tool_count,
                "tool_names": tool_names,
                "mcp_client_tools": len(self.mcp_tools) if self._mcp_client else 0,
                "connection_status": "Connected" if mcp_available else "Disconnected"
            }
            
            logger.info(f"📊 MCP Status: {status}")
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting MCP status: {str(e)}")
            return {
                "mcp_available": False,
                "tool_count": 0,
                "tool_names": [],
                "mcp_client_tools": 0,
                "connection_status": "Error",
                "error": str(e)
            }

    def is_mcp_available(self) -> bool:
        """Check if MCP tools are available."""
        available = bool(self.mcp_tools and self._initialized)
        logger.debug(f"🔍 MCP availability check: {available} (tools: {len(self.mcp_tools)}, initialized: {self._initialized})")
        return available

    def get_tool_by_name(self, tool_name: str) -> Optional[object]:
        """Get a specific tool by name."""
        for tool in self.mcp_tools:
            if tool.name == tool_name:
                logger.info(f"🔍 Found tool: {tool_name}")
                return tool
        return None

    def describe_tools(self) -> str:
        """Get a description of all available tools."""
        if not self.mcp_tools:
            return "No MCP tools available"
        
        descriptions = []
        for tool in self.mcp_tools:
            descriptions.append(f"- {tool.name}: {getattr(tool, 'description', 'No description available')}")
        
        return "Available MCP Tools:\n" + "\n".join(descriptions)

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.agent.get_state(config)
            
            messages = []
            for msg in state.values.get("messages", []):
                if isinstance(msg, HumanMessage):
                    messages.append({"type": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"type": "ai", "content": msg.content})
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    def get_current_provider(self) -> Dict[str, str]:
        """Get information about the current LLM provider being used."""
        try:
            if hasattr(self.llm, 'model_name'):
                return {"provider": "OpenAI", "model": self.llm.model_name}
            elif hasattr(self.llm, 'model'):
                if 'gemini' in str(self.llm.model).lower():
                    return {"provider": "Google", "model": str(self.llm.model)}
                else:
                    return {"provider": "OpenAI", "model": str(self.llm.model)}
            else:
                return {"provider": "Unknown", "model": "Unknown"}
        except Exception as e:
            logger.error(f"Error getting provider info: {str(e)}")
            return {"provider": "Error", "model": "Error"}
