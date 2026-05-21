# LinkedIn Post Generator

Just a Generator

## 🚀 Prerequisites

Make sure you have [uv](https://github.com) installed.

```bash
# Install uv if you haven't already
curl -LsSf https://astral-sh/uv/install.sh | sh
```

## 🛠️ Setup Instruction

Follow these steps to set up the project locally:

```bash
# 1. Clone the repository
git clone https://github.com/Shubhamhingu/LinkedInPost.git
cd LinkedInPost

# 2. Create a virtual environment and install dependencies
uv sync
```
## pre-requisites : 
create a .env with the following api keys
OPENAI_API_KEY=
TAVILY_API_KEY=


LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com/
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
PINECONE_API_KEY=your-key-here
PINECONE_INDEX_NAME=linkedin-posts   # optional, this is the default

SERVER_URL = "http://127.0.0.1:8000/mcp"


## 💻 How to Run

Use the following commands to execute the Python scripts:

```bash
# Run the main script using uv
uv run Post_generation/main.py

```
