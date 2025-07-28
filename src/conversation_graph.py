import uuid
import os
from typing import Dict, Any, List
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from .mcp_client import mcp_client


class ConversationGraph:
    """
    LangGraph ReAct agent implementation for managing conversation flow.
    
    This class creates and manages a ReAct agent that can intelligently
    select and use MCP tools to answer customer support questions.
    """
    
    def __init__(self):
        """Initialize the ReAct agent."""
        self.memory = MemorySaver()  # In-memory state persistence
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize MCP tools
        self.mcp_tools = []
        self._initialize_mcp_tools()
        
        # Create the ReAct agent
        self.agent = self._create_react_agent()
    
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the OpenAI LLM."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Get model from environment variable or use default
        model_name = os.getenv("MODEL", "gpt-3.5-turbo")
        
        return ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=api_key,
            streaming=True
        )
    
    def _initialize_mcp_tools(self):
        """Initialize MCP tools for customer support."""
        try:
            if mcp_client.is_available():
                self.mcp_tools = mcp_client.get_tools()
                logger.success(f"Initialized {len(self.mcp_tools)} MCP tools")
                
                # Log available tools
                for tool in self.mcp_tools:
                    logger.info(f"Available MCP tool: {tool.name}")
            else:
                logger.warning("MCP client not available, continuing without MCP tools")
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {str(e)}")
            self.mcp_tools = []
    
    def _create_react_agent(self):
        """Create a ReAct agent with MCP tools."""
        # Define system message for the agent
        system_message = """<ROLE>
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
        
        # Create the ReAct agent with tools and system message
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.mcp_tools,
            system_message=system_message,
            checkpointer=self.memory
        )
        
        logger.success("ReAct agent created successfully")
        return self.agent
        """
        Build the LangGraph conversation flow.
        
        Returns:
            Compiled StateGraph for conversation management
        """
        # Create the state graph
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("process_input", self._process_user_input)
        workflow.add_node("check_mcp_need", self._check_mcp_tool_need)
        workflow.add_node("use_mcp_tools", self._use_mcp_tools)
        workflow.add_node("generate_response", self._generate_bot_response)
        workflow.add_node("finalize", self._finalize_conversation)
        
        # Define the conversation flow
        workflow.add_edge(START, "process_input")
        workflow.add_edge("process_input", "check_mcp_need")
        
        # Conditional routing based on MCP tool need
        workflow.add_conditional_edges(
            "check_mcp_need",
            self._should_use_mcp_tools,
            {
                "use_mcp": "use_mcp_tools",
                "generate": "generate_response"
            }
        )
        
        workflow.add_edge("use_mcp_tools", "generate_response")
        workflow.add_edge("generate_response", "finalize")
        workflow.add_edge("finalize", END)
        
        # Compile the graph with memory for state persistence
        return workflow.compile(checkpointer=self.memory)
    
    def _process_user_input(self, state: ChatState) -> Dict[str, Any]:
        """
        Process and validate user input.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with processed input
        """
        logger.info("Processing user input...")
        
        user_input = state.get("user_input", "").strip()
        
        if not user_input:
            return {
                "error": "Empty input received",
                "bot_response": "I didn't receive any message. Could you please try again?"
            }
        
        # Validate input length
        if len(user_input) > 1000:
            return {
                "error": "Input too long",
                "bot_response": "Your message is too long. Please keep it under 1000 characters."
            }
        
        logger.success(f"User input processed: {user_input[:50]}...")
        
        return {
            "user_input": user_input,
            "error": ""
        }
    
    def _check_mcp_tool_need(self, state: ChatState) -> Dict[str, Any]:
        """
        Determine if MCP tools are needed for the user's query.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with MCP need flag
        """
        logger.info("Checking if MCP tools are needed...")
        
        if state.get("error"):
            return {"needs_mcp": False}
        
        user_input = state.get("user_input", "").lower()
        
        # Keywords that indicate MCP tool usage
        mcp_keywords = [
            "order", "status", "cancel", "refund", "delivery", 
            "zwigato", "gold", "membership", "policy", "restaurant",
            "track", "update", "help", "support"
        ]
        
        needs_mcp = any(keyword in user_input for keyword in mcp_keywords)
        
        logger.info(f"MCP tools needed: {needs_mcp}")
        
        return {"needs_mcp": needs_mcp}
    
    def _should_use_mcp_tools(self, state: ChatState) -> str:
        """
        Conditional routing function to determine next node.
        
        Args:
            state: Current conversation state
            
        Returns:
            Next node name
        """
        return "use_mcp" if state.get("needs_mcp", False) else "generate"
    
    def _use_mcp_tools(self, state: ChatState) -> Dict[str, Any]:
        """
        Use MCP tools to gather information for the user query.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with MCP tool results
        """
        logger.info("Using MCP tools...")
        
        try:
            # Use the chatbot service to determine and execute appropriate tools
            mcp_results = self.chatbot_service.use_mcp_tools(state)
            
            logger.success("MCP tools executed successfully")
            
            return {
                "mcp_results": mcp_results,
                "context": mcp_results.get("context", "")
            }
            
        except Exception as e:
            logger.error(f"Error using MCP tools: {str(e)}")
            return {
                "mcp_results": {"error": str(e)},
                "context": ""
            }
    
    def _generate_bot_response(self, state: ChatState) -> Dict[str, Any]:
        """
        Generate bot response using the chatbot service.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with bot response
        """
        logger.info("Generating bot response...")
        
        # Skip response generation if there's an error
        if state.get("error"):
            return {}
        
        # Use the chatbot service to generate response
        response_data = self.chatbot_service.process_message(state)
        
        logger.success("Bot response generated successfully")
        
        return response_data
    
    def _finalize_conversation(self, state: ChatState) -> Dict[str, Any]:
        """
        Finalize the conversation turn and prepare for next interaction.
        
        Args:
            state: Current conversation state
            
        Returns:
            Finalized state
        """
        logger.info("Finalizing conversation turn...")
        
        # Ensure we have a session ID
        if not state.get("session_id"):
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session ID: {session_id}")
            return {"session_id": session_id}
        
        # Clear temporary fields for next turn
        return {
            "user_input": "",  # Clear for next input
        }
    
    def chat(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """
        Process user input through the ReAct agent.
        
        Args:
            user_input: The user's message
            session_id: Session identifier for conversation continuity
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            logger.info(f"Processing message through ReAct agent: {user_input[:50]}...")
            
            # Create the thread config for session management
            config = {"configurable": {"thread_id": session_id}}
            
            # Prepare input messages
            messages = [HumanMessage(content=user_input)]
            
            # Track intermediate steps and final response
            intermediate_steps = []
            final_response = ""
            
            # Stream the agent's response to capture intermediate steps
            logger.info("🤖 ReAct Agent: Starting to process query...")
            
            for chunk in self.agent.stream({"messages": messages}, config, stream_mode="values"):
                if "messages" in chunk:
                    last_message = chunk["messages"][-1]
                    
                    if hasattr(last_message, 'content') and last_message.content:
                        # Check if this is a tool call or reasoning step
                        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                            for tool_call in last_message.tool_calls:
                                tool_step = f"🔧 Using tool: {tool_call['name']} with args: {tool_call['args']}"
                                logger.info(tool_step)
                                intermediate_steps.append(tool_step)
                        
                        # Check if this is the final response
                        elif last_message.type == "ai" and not hasattr(last_message, 'tool_calls'):
                            final_response = last_message.content
                            logger.info(f"✅ Final response generated: {final_response[:100]}...")
                        
                        # Log reasoning steps
                        elif last_message.type == "ai" and "Thought:" in last_message.content:
                            reasoning_step = f"💭 Reasoning: {last_message.content}"
                            logger.info(reasoning_step)
                            intermediate_steps.append(reasoning_step)
            
            # If no final response was captured, get the last AI message
            if not final_response:
                # Get the conversation state to extract the final response
                state = self.agent.get_state(config)
                if state.values.get("messages"):
                    for msg in reversed(state.values["messages"]):
                        if msg.type == "ai" and not hasattr(msg, 'tool_calls'):
                            final_response = msg.content
                            break
            
            if not final_response:
                final_response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            logger.success("ReAct agent completed processing")
            
            return {
                "response": final_response,
                "intermediate_steps": intermediate_steps,
                "error": "",
                "tools_used": len([step for step in intermediate_steps if "Using tool:" in step])
            }
            
        except Exception as e:
            error_msg = f"Error in ReAct agent processing: {str(e)}"
            logger.error(error_msg)
            
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please try again.",
                "intermediate_steps": [f"❌ Error: {error_msg}"],
                "error": error_msg,
                "tools_used": 0
            }
    
    def get_available_tools(self) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List of tool names
        """
        return [tool.name for tool in self.mcp_tools]

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages in the conversation
        """
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.agent.get_state(config)
            
            messages = []
            for msg in state.values.get("messages", []):
                messages.append({
                    "type": msg.type,
                    "content": msg.content,
                    "timestamp": getattr(msg, "timestamp", None)
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
