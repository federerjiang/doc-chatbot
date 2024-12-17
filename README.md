# Dockerized Doc ChatBot System

## Features

- **Backend**: FastAPI with WebSocket support for real-time communication.
- **Frontend**: React application using Framework7 for UI components.
- **Local Language Model**: Utilizes Ollama with the __ELYZA-japanese-Llama-2-7b-instruct-gguf__ model, which can be configured to use other models from the Ollama library.
- **Embedding Model**: Also served by Ollama or OpenAI


## Getting Started

To get the application up and running, follow these steps:

1. Clone the repository:
   ```bash
   git clone 
   cd doc-chatbot
   ```
2. Set OPENAI Key in the following py file.
```bash
backend/app/llm_util.py 

api_key = "" # TODO: Add your OpenAI API key here
```

3. Start the services using Docker Compose:
   ```bash
   docker compose up --build
   ```

This command will build the Docker images and start all the services defined in the `docker-compose.yml`. The backend will be accessible on port 80, and the frontend will be available on port 3010.

## Usage

Open your web browser and navigate to `http://localhost:3010` to access the frontend. 

## Note
Need to wait for finishing all the pre-processing logic before starting to chat with the bot


## Optimzied parts from last month
- Index: Better pdf parser && chunking
  - Use marker to convert pdf to markdown && preprocess
  - Chunk with more seperators
- Search: Hybrid search with vector search & keyword search
  - vector search: still with qdrant
  - keyword: with elasticsearch (bm25)
- Post-process: Reciprocal Rank Fusion (combine results form vector search & keyword search)
- Rerank with Cross-Encoder to filter resutls again.