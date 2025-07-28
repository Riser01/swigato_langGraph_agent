"""
Example script demonstrating how to use the chatbot programmatically.
This can be useful for testing, automation, or integration with other systems.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.conversation_graph import ConversationGraph
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


class ChatbotExample:
    """Example class showing how to interact with the chatbot."""
    
    def __init__(self):
        """Initialize the chatbot."""
        print("🤖 Initializing AI Chatbot...")
        
        # Check for API key
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ OpenAI API key not found!")
            print("📝 Please set OPENAI_API_KEY in your .env file")
            sys.exit(1)
        
        try:
            self.conversation_graph = ConversationGraph()
            print("✅ Chatbot initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {e}")
            sys.exit(1)
    
    def single_message_example(self):
        """Example of sending a single message to the chatbot."""
        print("\n🔹 Single Message Example")
        print("=" * 50)
        
        user_message = "Hello! Can you tell me a joke?"
        print(f"👤 User: {user_message}")
        
        try:
            response = self.conversation_graph.chat(user_message)
            print(f"🤖 Bot: {response['response']}")
            print(f"📊 Session ID: {response['session_id']}")
            
            if response.get('error'):
                print(f"⚠️  Error: {response['error']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def conversation_example(self):
        """Example of a multi-turn conversation."""
        print("\n🔹 Multi-turn Conversation Example")
        print("=" * 50)
        
        # Start a conversation with a session
        session_id = None
        
        messages = [
            "Hi there! What's your name?",
            "Can you help me write a short poem about technology?",
            "That's great! Now can you make it rhyme?",
            "Perfect! What other topics can you help with?"
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"\n👤 Turn {i}: {message}")
            
            try:
                response = self.conversation_graph.chat(message, session_id)
                print(f"🤖 Bot: {response['response']}")
                
                # Use the same session for conversation continuity
                session_id = response['session_id']
                
                if response.get('error'):
                    print(f"⚠️  Error: {response['error']}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                break
    
    def interactive_chat(self):
        """Interactive chat mode."""
        print("\n🔹 Interactive Chat Mode")
        print("=" * 50)
        print("💬 Type your messages (type 'quit' to exit)")
        print()
        
        session_id = None
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = self.conversation_graph.chat(user_input, session_id)
                print(f"🤖 Bot: {response['response']}")
                
                # Maintain session
                session_id = response['session_id']
                
                if response.get('error'):
                    print(f"⚠️  Error: {response['error']}")
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def conversation_history_example(self):
        """Example of retrieving conversation history."""
        print("\n🔹 Conversation History Example")
        print("=" * 50)
        
        # Send a few messages first
        messages = ["Hello!", "How are you?", "Tell me about AI."]
        session_id = None
        
        for msg in messages:
            response = self.conversation_graph.chat(msg, session_id)
            session_id = response['session_id']
            print(f"👤 {msg}")
            print(f"🤖 {response['response']}")
            print()
        
        # Retrieve conversation history
        try:
            history = self.conversation_graph.get_conversation_history(session_id)
            
            print("📋 Conversation History:")
            for i, message in enumerate(history):
                role = "👤" if hasattr(message, 'content') and "user" in str(type(message)).lower() else "🤖"
                content = message.content if hasattr(message, 'content') else str(message)
                print(f"   {i+1}. {role} {content[:100]}...")
                
        except Exception as e:
            print(f"❌ Error retrieving history: {e}")


def main():
    """Main function to run examples."""
    print("🚀 AI Chatbot MVP - Programming Examples")
    print("=========================================")
    
    # Initialize chatbot
    chatbot = ChatbotExample()
    
    # Choose which example to run
    while True:
        print("\n📋 Available Examples:")
        print("1. Single Message Example")
        print("2. Multi-turn Conversation Example")
        print("3. Interactive Chat Mode")
        print("4. Conversation History Example")
        print("5. Exit")
        
        try:
            choice = input("\n🔢 Choose an example (1-5): ").strip()
            
            if choice == '1':
                chatbot.single_message_example()
            elif choice == '2':
                chatbot.conversation_example()
            elif choice == '3':
                chatbot.interactive_chat()
            elif choice == '4':
                chatbot.conversation_history_example()
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
