# Zwigato Customer Support Agent

A modern customer support chatbot application built with **Streamlit**, **LangGraph**, and **OpenAI API**, fully containerized with Docker.

## 🚀 Features

- **Interactive Chat Interface**: Clean and responsive Streamlit UI
- **Advanced Conversation Management**: LangGraph for state management and conversation flow
- **OpenAI Integration**: Powered by configurable OpenAI models for intelligent responses
- **MCP Tools Integration**: Customer support tools for order management and wiki search
- **Session Management**: Persistent conversation history per session
- **Real-time Streaming**: Smooth chat experience with streaming responses
- **Docker Containerization**: Easy deployment anywhere
- **Comprehensive Logging**: Built-in logging with Loguru
- **Error Handling**: Robust error handling and recovery

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │ ── │   LangGraph      │ ── │   OpenAI API    │
│   (Frontend)    │    │   (Backend)      │    │   (LLM)         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        │              ┌─────────────────┐              │
        └──────────────│ State Management │──────────────┘
                       │ & Conversation   │
                       │     Memory       │
                       └─────────────────┘
```

## 📋 Prerequisites

- Docker and Docker Compose
- OpenAI API key

## 🛠️ Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd swigato_docker
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your OpenAI API key
# OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 4. Access the Application

Open your browser and navigate to: `http://localhost:8501`

## 🔧 Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key

# Run the application
streamlit run app.py
```

## 📁 Project Structure

```
swigato_docker/
├── src/
│   ├── __init__.py
│   ├── state.py              # LangGraph state definitions
│   ├── chatbot_service.py    # OpenAI integration service
│   └── conversation_graph.py # LangGraph conversation flow
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container configuration
├── docker-compose.yml      # Docker Compose configuration
├── .env.example            # Environment variables template
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

## 🧩 Core Components

### 1. State Management (`src/state.py`)
- Defines the conversation state schema using TypedDict
- Manages message history, session data, and context
- Ensures type safety across the application

### 2. Chatbot Service (`src/chatbot_service.py`)
- Handles OpenAI API integration
- Processes user messages and generates responses
- Implements streaming for real-time chat experience
- Manages conversation context and memory

### 3. Conversation Graph (`src/conversation_graph.py`)
- LangGraph implementation for conversation flow
- Manages state transitions and conversation logic
- Provides session persistence and memory management
- Handles error recovery and validation

### 4. Streamlit UI (`app.py`)
- Interactive chat interface
- Session management and chat history
- Real-time message display
- Configuration and settings management

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `STREAMLIT_SERVER_PORT` | Port for Streamlit app | 8501 |
| `STREAMLIT_SERVER_ADDRESS` | Server address | 0.0.0.0 |
| `APP_TITLE` | Application title | "AI Chatbot MVP" |
| `MAX_MESSAGE_HISTORY` | Max messages to keep | 50 |

### Streamlit Configuration

The app automatically configures Streamlit with optimal settings:
- Wide layout for better chat experience
- Custom page title and icon
- Disabled usage statistics
- Headless mode for containerization

## 🐳 Docker Configuration

### Building the Image

```bash
# Build the Docker image
docker build -t chatbot-mvp .

# Run the container
docker run -p 8501:8501 --env-file .env chatbot-mvp
```

### Docker Compose Features

- **Health Checks**: Automatic health monitoring
- **Volume Mounting**: Persistent logs storage
- **Environment Integration**: Seamless .env file support
- **Auto-restart**: Automatic container restart on failure

## 📊 Monitoring and Logging

### Logging
- Comprehensive logging with Loguru
- Rotating log files (1MB rotation, 7-day retention)
- Structured logging with timestamps and levels
- Application performance monitoring

### Health Checks
- Built-in Docker health checks
- Streamlit health endpoint monitoring
- Automatic container restart on failure

## 🔧 Customization

### Adding New Features

1. **Extend State Schema**: Modify `src/state.py` to add new state fields
2. **Add Graph Nodes**: Create new processing nodes in `conversation_graph.py`
3. **Enhance UI**: Extend the Streamlit interface in `app.py`
4. **Integrate APIs**: Add new services in `src/`

### LangGraph Customization

```python
# Example: Add a new processing node
def custom_processing_node(state: ChatState) -> Dict[str, Any]:
    # Your custom logic here
    return {"custom_field": "processed_value"}

# Add to graph
workflow.add_node("custom_process", custom_processing_node)
workflow.add_edge("process_input", "custom_process")
```

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Key Not Found**
   - Ensure `.env` file exists and contains valid `OPENAI_API_KEY`
   - Check that the key has sufficient credits

2. **Docker Build Fails**
   - Ensure Docker is running
   - Check internet connectivity for dependency downloads
   - Verify Python 3.11 compatibility

3. **Streamlit Connection Issues**
   - Check if port 8501 is available
   - Verify firewall settings
   - Ensure container is running: `docker ps`

4. **Memory Issues**
   - Monitor container memory usage: `docker stats`
   - Increase Docker memory allocation if needed
   - Check conversation history limits

### Debugging

```bash
# View application logs
docker-compose logs -f chatbot

# Access container shell
docker-compose exec chatbot /bin/bash

# Check container health
docker-compose ps
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit changes: `git commit -am 'Add new feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web app framework
- **LangGraph** - For conversation flow management
- **OpenAI** - For the powerful language models
- **Docker** - For containerization technology

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the Docker and Streamlit documentation

---

**Built with ❤️ using Streamlit, LangGraph, and OpenAI**
