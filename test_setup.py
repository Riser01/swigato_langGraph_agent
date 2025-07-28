"""
Test script to validate the chatbot setup and functionality.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.state import ChatState
    from src.chatbot_service import ChatbotService
    from src.conversation_graph import ConversationGraph
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


class TestChatbotSetup(unittest.TestCase):
    """Test cases for chatbot setup and basic functionality."""
    
    def test_environment_variables(self):
        """Test that required environment variables are present."""
        print("🔧 Testing environment variables...")
        
        # Check if .env file exists
        self.assertTrue(os.path.exists('.env'), ".env file should exist")
        
        # Check for OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        self.assertIsNotNone(api_key, "OPENAI_API_KEY should be set")
        self.assertTrue(api_key.startswith('sk-'), "OPENAI_API_KEY should start with 'sk-'")
        
        print("✅ Environment variables test passed")
    
    def test_state_schema(self):
        """Test the ChatState schema."""
        print("📋 Testing state schema...")
        
        # Test state creation
        state = {
            "messages": [],
            "session_id": "test-session",
            "user_input": "Hello",
            "bot_response": "",
            "context": "",
            "error": "",
            "turn_count": 0
        }
        
        # Validate state structure
        required_keys = ["messages", "session_id", "user_input", "bot_response", "context", "error", "turn_count"]
        for key in required_keys:
            self.assertIn(key, state, f"State should contain '{key}' field")
        
        print("✅ State schema test passed")
    
    @patch('src.chatbot_service.ChatOpenAI')
    def test_chatbot_service_initialization(self, mock_openai):
        """Test chatbot service initialization."""
        print("🤖 Testing chatbot service initialization...")
        
        # Mock OpenAI client
        mock_openai.return_value = MagicMock()
        
        try:
            service = ChatbotService()
            self.assertIsNotNone(service.llm, "LLM should be initialized")
            self.assertIsNotNone(service.prompt_template, "Prompt template should be initialized")
            print("✅ Chatbot service initialization test passed")
        except Exception as e:
            self.fail(f"Chatbot service initialization failed: {e}")
    
    @patch('src.conversation_graph.ChatbotService')
    def test_conversation_graph_initialization(self, mock_service):
        """Test conversation graph initialization."""
        print("🔄 Testing conversation graph initialization...")
        
        # Mock chatbot service
        mock_service.return_value = MagicMock()
        
        try:
            graph = ConversationGraph()
            self.assertIsNotNone(graph.graph, "Graph should be initialized")
            self.assertIsNotNone(graph.memory, "Memory should be initialized")
            print("✅ Conversation graph initialization test passed")
        except Exception as e:
            self.fail(f"Conversation graph initialization failed: {e}")
    
    def test_docker_files(self):
        """Test that Docker files exist and have basic content."""
        print("🐳 Testing Docker configuration...")
        
        # Check Dockerfile exists
        self.assertTrue(os.path.exists('Dockerfile'), "Dockerfile should exist")
        
        # Check docker-compose.yml exists
        self.assertTrue(os.path.exists('docker-compose.yml'), "docker-compose.yml should exist")
        
        # Check Dockerfile content
        with open('Dockerfile', 'r') as f:
            dockerfile_content = f.read()
            self.assertIn('FROM python', dockerfile_content, "Dockerfile should use Python base image")
            self.assertIn('streamlit', dockerfile_content, "Dockerfile should reference Streamlit")
        
        # Check docker-compose content
        with open('docker-compose.yml', 'r') as f:
            compose_content = f.read()
            self.assertIn('8501:8501', compose_content, "docker-compose should expose port 8501")
            self.assertIn('OPENAI_API_KEY', compose_content, "docker-compose should reference OpenAI API key")
        
        print("✅ Docker configuration test passed")
    
    def test_project_structure(self):
        """Test that all required files exist."""
        print("📁 Testing project structure...")
        
        required_files = [
            'app.py',
            'requirements.txt',
            'README.md',
            '.env.example',
            '.gitignore',
            'src/__init__.py',
            'src/state.py',
            'src/chatbot_service.py',
            'src/conversation_graph.py'
        ]
        
        for file_path in required_files:
            self.assertTrue(os.path.exists(file_path), f"{file_path} should exist")
        
        print("✅ Project structure test passed")


def run_tests():
    """Run all tests and display results."""
    print("🧪 AI Chatbot MVP - Setup Validation")
    print("=====================================")
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestChatbotSetup)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    print()
    print("📊 Test Results:")
    print("================")
    
    if result.wasSuccessful():
        print("✅ All tests passed! Your chatbot setup is ready.")
        print("🚀 You can now run the application using:")
        print("   Local: python app.py or streamlit run app.py")
        print("   Docker: docker-compose up --build")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print(f"📈 Tests run: {result.testsRun}")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"⚠️  Errors: {len(result.errors)}")
        
        if result.failures:
            print("\n📋 Failures:")
            for test, traceback in result.failures:
                print(f"   - {test}: {traceback}")
        
        if result.errors:
            print("\n🚨 Errors:")
            for test, traceback in result.errors:
                print(f"   - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
