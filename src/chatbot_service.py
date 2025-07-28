import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from .state import ChatState


class ChatbotService:
    """
    Service class that handles the core chatbot logic using OpenAI API.
    """
    
    def __init__(self):
        """Initialize the chatbot service with OpenAI client."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize the OpenAI chat model
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=self.api_key,
            streaming=True  # Enable streaming for better user experience
        )
        
        # Define the system prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. You are engaging, friendly, and knowledgeable.
            
            Guidelines:
            - Be conversational and natural
            - Provide helpful and accurate information
            - If you don't know something, admit it honestly
            - Keep responses concise but informative
            - Maintain context from previous messages in the conversation
            
            Current conversation turn: {turn_count}
            Context: {context}"""),
            ("human", "{user_input}")
        ])
    
    def process_message(self, state: ChatState) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
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
            formatted_prompt = self.prompt_template.format_messages(
                user_input=state["user_input"],
                context=context,
                turn_count=state.get("turn_count", 0)
            )
            
            # Generate response using OpenAI
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
    
    async def stream_response(self, state: ChatState):
        """
        Stream response for real-time chat experience.
        
        Args:
            state: Current conversation state
            
        Yields:
            Streamed response chunks
        """
        try:
            context = self._build_context(state.get("messages", []))
            
            formatted_prompt = self.prompt_template.format_messages(
                user_input=state["user_input"],
                context=context,
                turn_count=state.get("turn_count", 0)
            )
            
            # Stream the response
            async for chunk in self.llm.astream(formatted_prompt):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield "I apologize, but I encountered an error while processing your message."
