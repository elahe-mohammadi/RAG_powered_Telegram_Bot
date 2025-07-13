# RAG-Powered Telegram Support Bot

This project implements a Retrieval-Augmented Generation (RAG) system accessible via a Telegram bot. It uses FastAPI for the backend, FAISS for vector search, and integrates with openrouter APIs for language generation.

## Features

- **Document Ingestion**: Loads and processes `.txt` documents from a local folder
- **Vector Search**: Uses FAISS to index and retrieve relevant context
- **LLM Integration**: Uses openrouter API for text generation
- **Telegram Bot**: Provides a chat interface for users
- **Caching**: Implements in-memory caching to reduce latency and API usage
- **Docker Support**: Containerized deployment for easy scaling

## Architecture



The system follows a modular architecture:

1. **Document Processing**: Text documents are loaded, split into chunks, and embedded
2. **Vector Store**: Embeddings are indexed using FAISS for efficient similarity search
3. **API Layer**: FastAPI provides HTTP endpoints for question answering
4. **LLM Service**: openrouter API generates answers based on retrieved context
5. **Telegram Integration**: The Bot connects users with the backend system

## Setup Instructions

### Prerequisites

- Python 3.8+
- Docker (optional)

### Local Setup

1. Clone the repository:
   ```bash
   https://github.com/elahe-mohammadi/RAG_powered_Telegram_Bot.git
   cd RAG_powered_Telegram_Bot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```

4. Create a `.env` file, `data` folder for txt documents, and `index` folder for faiss index :
   ```bash
   touch .env app/data
   ```

5. Configure your environment variables in the `.env` file similar to .env.example:
   - Add your Hugging Face API key (optional but recommended) or openrouter API token
   - Add your Telegram Bot Token (get one from @BotFather on Telegram)
   - Adjust other settings as needed

6. Add your text documents to the `app/data` directory (`.txt` files only)

7. Run the application:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t rag-telegram-bot .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env -v $(pwd)/app/data:/app/app/data -v $(pwd)/app/faiss_index:/app/app/faiss_index rag-telegram-bot
   ```

## Usage

### API Endpoints

- `POST /api/ask`: Submit questions to the RAG system
  ```json
  {
    "question": "What is RAG?"
  }
  ```

### Telegram Bot

1. Start a chat with your bot on Telegram (the one you configured with @BotFather)
2. Send the bot a message with your question
3. The bot will respond with an answer based on your documents

## Testing

Run the tests using pytest:

```bash
pytest tests/test_rag_api.py
```

## Performance Optimizations

### Minimizing LLM API Usage
- Implemented in-memory caching for both document retrieval and LLM responses
- Pre-filters empty questions to avoid unnecessary API calls
- Retrieves only the most relevant documents (top-k selection)

### Reducing Latency
- Uses FAISS for efficient vector similarity search
- Initializes the vector store at application startup
- Implements response caching for repeated questions

### Scaling Considerations
- Modular design allows for easy component replacement
- Docker containerization supports deployment to any environment
- Separated concerns (vector storage, LLM, API, bot) for independent scaling

## Future Improvements

- Replace in-memory cache with Redis for distributed deployments
- Add support for PDF and other document formats
- Implement user feedback mechanism to improve responses
- Add authentication for API endpoints
- Improve error handling and logging
