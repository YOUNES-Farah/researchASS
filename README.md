# 🔬 ResearchASS — Multi-Agent Research Assistant

> A fast, free, production-ready RAG agent built with LangGraph, Groq, ChromaDB and Tavily.

## 🏗️ Architecture

```
researchASS/
├── graph/
│   ├── state.py       ← Shared TypedDict state across all nodes
│   ├── nodes.py       ← Agent functions (router, retriever, generator...)
│   ├── edges.py       ← Conditional routing logic
│   └── graph.py       ← LangGraph compilation
├── tools/
│   ├── retriever.py   ← ChromaDB + HuggingFace embeddings + reranker
│   ├── web_search.py  ← Tavily (+ DuckDuckGo fallback)
│   └── cache.py       ← Semantic cache (cosine similarity)
├── prompts.py         ← All LLM prompts
├── main.py            ← FastAPI server
├── run.py             ← Terminal mode
├── requirements.txt
└── .env
```

### Graph Flow

```
[Question]
    │
    ▼
[cache_check] ──── cache hit ──────────────────────────► [END]
    │
  miss
    │
    ▼
[router] (keywords, no LLM)
    │
    ├── vectorstore ──► [retriever] ──► reranker score OK ──► [generator] ──► [END]
    │                        │
    │                   score too low
    │                        │
    └── web_search ◄─────────┘
            │
            ▼
        [generator] ──► [END]
```

---

## 🛠️ Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq — `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` |
| Orchestration | LangGraph |
| Vector Store | ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Web Search | Tavily + DuckDuckGo fallback |
| API | FastAPI + uvicorn |
| Observability | LangSmith |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUNES-Farah/researchASS.git
cd researchASS
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file at the root :
```env
GROQ_API_KEY=gsk_...

TAVILY_API_KEY=tvly-...         # optional — falls back to DuckDuckGo

LANGCHAIN_TRACING_V2=true       # optional — LangSmith tracing
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=researchASS
```

Get your free API keys :
- **Groq** → [console.groq.com](https://console.groq.com)
- **Tavily** → [app.tavily.com](https://app.tavily.com) (1000 free requests/month)
- **LangSmith** → [smith.langchain.com](https://smith.langchain.com)

---

## 💻 Usage

### Terminal mode
```bash
python run.py
```
```
❓ Ta question : qui a inventé LangChain ?
⏳ Recherche en cours...
💡 RÉPONSE :
LangChain a été créé par Harrison Chase en octobre 2022...
Source : vectorstore  |  ⏱️ 2.31s
```

### API mode
```bash
uvicorn main:app --reload
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed status |
| `POST` | `/research` | Ask a question (JSON response) |
| `POST` | `/research/stream` | Ask a question (SSE streaming) |
| `POST` | `/add-documents` | Ingest plain text |
| `POST` | `/add-pdf` | Ingest a PDF file |
| `POST` | `/add-url` | Ingest a web page |
| `DELETE` | `/cache` | Clear semantic cache |

### Example request
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangChain?"}'
```

```json
{
  "question": "What is LangChain?",
  "answer": "LangChain is a framework for building LLM-powered applications...",
  "source": "vectorstore",
  "cache_hit": false,
  "session_id": "abc-123"
}
```

---

## ⚙️ Configuration

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `RELEVANCE_THRESHOLD` | `retriever.py` | `0.0` | Min reranker score before web fallback |
| `_SIMILARITY_THRESHOLD` | `cache.py` | `0.92` | Min cosine similarity for cache hit |
| `MAX_CHARS` | `nodes.py` | `150` | Max characters per document chunk sent to LLM |
| `WEB_KEYWORDS` | `nodes.py` | see file | Keywords that trigger web search |

---

## 📄 License

MIT