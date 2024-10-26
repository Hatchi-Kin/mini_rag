# Mini RAG

Mini RAG is a simple application that uses Milvus for vector storage and retrieval, and Llama3.2 for answering queries about the documentation of the Megapi api.

## Prerequisites

1. **Download and Install Ollama**:
   - Visit the [Ollama Download Page](https://ollama.com/library/download) to download the Ollama software.


2. **Install Llama3.2 Model**:
   - After downloading Ollama, run the following command to install the Llama3.2 model:
     ```sh
     ollama run llama3.2
     ```

## Setup

1. **Clone the repository**:
   ```sh
   git clone https://github.com/Hatchi-Kin/mini_rag.git
   cd mini_rag
   ```

2. **Install dependencies in a .venv**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the FastAPI server**:
   ```sh
   python api.py
   ```

4. **Run the FastHTML app**:
   ```sh
   python app.py
   ```
## Usage

Send a POST request to `/query` with a JSON payload containing your query:
```json
{
  "query": "Your question here"
}
```

## Or go to web UI

Go to `localhost:8002` and ask your question in the input box.