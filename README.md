# Shreya's GitHub RAG Assistant

A LangGraph-powered Retrieval Augmented Generation (RAG) application that answers questions about Shreya Ramraika's( mine :) ) GitHub projects using AI-driven search and analysis.

## 🚀 Features

- **Smart Repository Analysis**: Automatically loads and processes content from multiple GitHub repositories
- **Vector Search**: Uses semantic search to find relevant information across projects
- **Conversational Interface**: Interactive command-line chat for asking questions
- **Tool Integration**: LangGraph-powered agent that intelligently uses retrieval tools
- **Multi-Model Support**: Combines Google Gemini for reasoning and Ollama for embeddings

## 📋 Prerequisites

- Python 3.8+
- Google API key for Gemini
- Ollama installed locally with `nomic-embed-text` model

## 🎯 Usage

Run the application:
python main.py

**Example interactions:**

- "Does Shreya have a RAG project?"
- "What AI projects has she worked on?"
- "Tell me about her stock analysis project"
- Type `exit` or `quit` to end

## 🏗️ Architecture

### Components

- **Document Loader**: Fetches content from GitHub repositories
- **Vector Store**: Chroma database with semantic embeddings
- **LangGraph Agent**: Manages conversation flow and tool usage
- **Retrieval Tool**: Searches relevant project information

### Monitored Repositories

- stockanalysisAI
- AIreserarchagentb2b
- autoorganizeAI
- AIautoimagecap
- secmsg

## 🔧 Configuration

The application uses a recursion limit of 10 to prevent infinite loops and includes proper error handling for robust operation.



