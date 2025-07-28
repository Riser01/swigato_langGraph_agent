# 🍕 Zwigato Customer Support Agent

A production-ready customer support chatbot built with **Streamlit**, **LangGraph**, and **dual LLM support** (OpenAI/Google Gemini), featuring MCP tool integration for order management and knowledge base search.

## ✨ Key Features

- **🎨 Interactive Chat Interface**: Modern Streamlit UI with real-time streaming
- **🔄 Dual LLM Support**: OpenAI GPT + Google Gemini with automatic failover
- **🛠️ MCP Tools**: Order management and knowledge base search
- **� Docker Ready**: Containerized deployment with health checks
- **� Session Management**: Persistent conversation history

## 🚀 Quick Start

### 🐳 Docker Deployment (Recommended)

1. **Setup Environment**
```bash
git clone <your-repo-url>
cd swigato_docker
cp .env.example .env
```

2. **Configure API Keys**
Edit `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
PROVIDER_PREFERENCE=openai
```

3. **Deploy**
```bash
./deploy_docker.sh
# Or manually: docker-compose up -d
```

4. **Access**: Open http://localhost:8501

### 💻 Local Development

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env  # Edit with your API keys
streamlit run app.py
```

## � Testing with Example Scenarios

Once the application is running, try these demo conversations to test the MCP tools:

### **Membership Information**
```
Hi, can you tell me about membership
```
*Tests: Knowledge base search functionality*

### **Restaurant Listing**
```
How do I list my restaurant on Zwigato?
```
*Tests: Wiki search for business information*

### **Order Status Check**
```
Can I get the status of my order ORDZW011?
```
*Tests: Order lookup tool*

### **Order Cancellation**
```
Actually, I need to cancel order ORDZW011.
```
*Tests: Order status update functionality*

### **Status Verification**
```
What's the status of ORDZW011 now?
```
*Tests: Verification of order changes*

### **General Greeting**
```
Hi, can you tell me about membership
```
*Tests: Natural conversation flow without tools*

**Expected Behavior:**
- The bot should use appropriate MCP tools for order-related queries
- Knowledge base searches should return relevant information
- Order updates should be reflected in subsequent status checks
- The interface should show which tools are being used
## ⚙️ Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `GOOGLE_API_KEY` | Google Gemini API key | - |
| `PROVIDER_PREFERENCE` | Preferred LLM provider | `openai` |
| `MODEL` | Primary model name | `gpt-4o` |
| `FALLBACK_MODEL` | Fallback model name | `gemini-1.5-flash` |

*At least one API key is required*

## 📁 Project Structure

```
swigato_docker/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── .env.example            # Environment template
├── mcp_config.json         # MCP tools configuration
├── mcp_server_remote.py    # MCP server implementation
├── data.py                 # Sample data
├── src/                    # Core modules
│   ├── state.py           # LangGraph state definitions
│   └── unified_chatbot_service.py  # Main service
└── logs/                   # Application logs
```
## 🚨 Troubleshooting

### Common Issues

**API Key Problems**
```bash
# Check environment variables
docker-compose exec chatbot env | grep API

# Test API connectivity
python test_llm_config.py
```

**Docker Issues**
```bash
# Check logs
docker-compose logs chatbot

# Rebuild container
docker-compose build --no-cache
docker-compose up -d
```

**Connection Problems**
```bash
# Verify container is running
docker-compose ps

# Test health endpoint
curl http://localhost:8501/health
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Streamlit](https://streamlit.io/)** - Web app framework
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Conversation flow management
- **[OpenAI](https://openai.com/)** & **[Google Gemini](https://ai.google.dev/)** - LLM APIs
- **[Docker](https://www.docker.com/)** - Containerization

---

**🎉 Ready to deploy? Choose your method above and start chatting! 🚀**
