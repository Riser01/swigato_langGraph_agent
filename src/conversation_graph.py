import uuid
from typing import Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

from .state import ChatState
from .chatbot_service import ChatbotService


class ConversationGraph:
    """
    LangGraph implementation for managing conversation flow.
    
    This class creates and manages the conversation graph that handles
    the flow of messages between user input and bot responses.
    """
    
    def __init__(self):
        """Initialize the conversation graph."""
        self.chatbot_service = ChatbotService()
        self.memory = MemorySaver()  # In-memory state persistence
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph conversation flow.
        
        Returns:
            Compiled StateGraph for conversation management
        """
        # Create the state graph
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("process_input", self._process_user_input)
        workflow.add_node("generate_response", self._generate_bot_response)
        workflow.add_node("finalize", self._finalize_conversation)
        
        # Define the conversation flow
        workflow.add_edge(START, "process_input")
        workflow.add_edge("process_input", "generate_response")
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
    
    def chat(self, user_input: str, session_id: str = None) -> Dict[str, Any]:
        """
        Main chat interface for processing user messages.
        
        Args:
            user_input: User's message
            session_id: Optional session identifier for conversation continuity
            
        Returns:
            Bot response and updated state
        """
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Configuration for the graph execution
            config = {"configurable": {"thread_id": session_id}}
            
            # Initial state
            initial_state = {
                "user_input": user_input,
                "session_id": session_id,
                "messages": [],
                "context": "",
                "bot_response": "",
                "error": "",
                "turn_count": 0
            }
            
            logger.info(f"Starting chat for session {session_id}")
            
            # Execute the graph
            final_state = self.graph.invoke(initial_state, config)
            
            return {
                "response": final_state.get("bot_response", "I'm sorry, I couldn't generate a response."),
                "session_id": session_id,
                "error": final_state.get("error", ""),
                "turn_count": final_state.get("turn_count", 0)
            }
            
        except Exception as e:
            error_msg = f"Error in conversation graph: {str(e)}"
            logger.error(error_msg)
            return {
                "response": "I apologize, but I encountered an error. Please try again.",
                "session_id": session_id or str(uuid.uuid4()),
                "error": error_msg,
                "turn_count": 0
            }
    
    def get_conversation_history(self, session_id: str) -> list:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation messages
        """
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.graph.get_state(config)
            return state.values.get("messages", [])
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
