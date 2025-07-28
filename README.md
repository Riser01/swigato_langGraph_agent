# 🍕 Zwigato Customer Support Agent

A production-ready customer support chatbot built with **Streamlit**, **LangGraph**, and **dual LLM support** (OpenAI/Google Gemini), featuring MCP (Model Context Protocol) tool integration for order management and knowledge base search.

## ✨ Key Features

- **🎨 Interactive Chat Interface**: Modern Streamlit UI with ReAct agent visualization
- **🔄 Dual LLM Support**: OpenAI GPT + Google Gemini with automatic failover
- **🛠️ MCP Tools Integration**: Order management and knowledge base search via MCP server
- **🐳 Docker Ready**: Containerized deployment with health checks
- **📱 Session Management**: Persistent conversation history with unique session IDs
- **🔍 ReAct Agent**: Visible thinking process with intermediate steps display
- **⚡ Real-time Processing**: Live tool usage tracking and response streaming

## 🚀 Quick Start

### 🐳 Docker Deployment (Recommended)

1. **Clone Repository**
```bash
git clone https://github.com/Riser01/swigato_langGraph_agent.git
cd swigato_langGraph_agent
```

2. **Setup Environment**
```bash
cp .env.example .env
```

3. **Configure API Keys**
Edit `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
PROVIDER_PREFERENCE=openai
MODEL=gpt-4o
FALLBACK_MODEL=gemini-1.5-flash
```

4. **Deploy with Docker**
```bash
# Linux/Mac
./deploy_docker.sh

# Windows
./deploy_docker.bat

# Or manually
docker-compose up -d
```

5. **Access Application**: Open http://localhost:8501

### 💻 Local Development

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac  
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Run application
streamlit run app.py

# For local testing (optional)
# Windows
./run_local.bat
# Linux/Mac
./run_local.sh
```

## 🧪 Testing with Example Scenarios

Once the application is running, try these demo conversations to test the MCP tools:

### **Order Management**
```
User: "Can I get the status of my order ORDZW011?"
```
*Tests: Order lookup tool with MCP server integration*

```
User: "I need to cancel order ORDZW011"
```
*Tests: Order status update functionality*

```
User: "What's the status of ORDZW011 now?"
```
*Tests: Verification of order changes*

### **Knowledge Base Search**
```
User: "Tell me about Zwigato Gold membership benefits"
```
*Tests: Wiki search for membership information*

```
User: "How do I list my restaurant on Zwigato?"
```
*Tests: Wiki search for business information*

```
User: "What are your delivery fees?"
```
*Tests: Policy information retrieval*

### **General Conversation**
```
User: "Hi there! How can you help me?"
```
*Tests: Natural conversation flow without tools*

**Expected Behavior:**
- ReAct agent process visualization in expandable sections
- Tool usage counter and step-by-step reasoning display
- Appropriate MCP tool selection for different query types
- Graceful fallback when tools aren't needed
- Real-time response streaming with thinking indicators
## ⚙️ Configuration

### Environment Variables
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | One of OpenAI/Google |
| `GOOGLE_API_KEY` | Google Gemini API key | - | One of OpenAI/Google |
| `PROVIDER_PREFERENCE` | Preferred LLM provider | `openai` | No |
| `MODEL` | Primary model name | `gpt-4o` | No |
| `FALLBACK_MODEL` | Fallback model name | `gemini-1.5-flash` | No |
| `APP_TITLE` | Application title | `Zwigato Customer Support Agent` | No |

*At least one API key (OpenAI or Google) is required for the application to function*

## 📁 Project Structure

```
swigato_langGraph_agent/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── mcp_config.json           # MCP tools configuration
├── mcp_server_remote.py      # MCP server implementation
├── data.py                   # Sample data and mock responses
├── deploy_docker.sh          # Docker deployment script (Linux/Mac)
├── deploy_docker.bat         # Docker deployment script (Windows)
├── run_local.sh             # Local development script (Linux/Mac)
├── run_local.bat            # Local development script (Windows)
├── health_check.sh          # Docker health check script
├── src/                     # Core application modules
│   ├── __init__.py         # Package initialization
│   ├── state.py           # LangGraph state definitions
│   └── unified_chatbot_service.py  # Main chatbot service
└── logs/                   # Application logs directory
    └── chatbot.log        # Main application log file
```

## 🏗️ Architecture

The application follows a modular architecture:

- **Frontend**: Streamlit provides the web interface
- **Backend**: LangGraph ReAct agent handles conversation flow
- **LLM Integration**: Dual provider support with automatic failover
- **Tools**: MCP server provides order management and knowledge base
- **State Management**: Session-based conversation persistence
- **Logging**: Comprehensive logging with rotation
## 🚨 Troubleshooting

### Common Issues

**API Key Problems**
```bash
# Check environment variables
docker-compose exec chatbot env | grep API

# Verify API key format
echo $OPENAI_API_KEY | wc -c  # Should be around 51 characters
```

**Docker Issues**
```bash
# Check container logs
docker-compose logs chatbot

# Check container status
docker-compose ps

# Rebuild container if needed
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

**MCP Connection Problems**
```bash
# Test MCP server separately
python mcp_server_remote.py

# Check MCP configuration
cat mcp_config.json
```

**Streamlit Application Issues**
```bash
# Check application health
curl http://localhost:8501/_stcore/health

# View real-time logs
docker-compose logs -f chatbot

# Restart application
docker-compose restart chatbot
```

### Performance Tips

- Use `gpt-3.5-turbo` for faster responses during development
- Monitor token usage in the sidebar
- Clear chat history periodically for better performance
- Check Docker memory allocation if experiencing slowdowns

## 📊 Features Overview

| Feature | Status | Description |
|---------|--------|-------------|
| ✅ Chat Interface | Ready | Modern Streamlit UI with message history |
| ✅ ReAct Agent | Ready | LangGraph agent with tool integration |
| ✅ Dual LLM Support | Ready | OpenAI + Google Gemini with failover |
| ✅ MCP Tools | Ready | Order management and wiki search |
| ✅ Docker Support | Ready | Full containerization with health checks |
| ✅ Session Management | Ready | Persistent conversation state |
| ✅ Real-time Logging | Ready | Comprehensive logging with rotation |
| ⚠️ Authentication | Planned | User authentication system |
| ⚠️ Database Integration | Planned | Persistent data storage |

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Streamlit](https://streamlit.io/)** - Web app framework
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Conversation flow management
- **[OpenAI](https://openai.com/)** & **[Google Gemini](https://ai.google.dev/)** - LLM APIs
- **[Docker](https://www.docker.com/)** - Containerization

---

**🎉 Ready to deploy? Choose your method above and start chatting! 🚀**
