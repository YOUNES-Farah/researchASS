# ResearchASS — Multi-Agent Research Assistant
## Stack 100% gratuite : Groq + LLaMA 3.3 + ChromaDB + DuckDuckGo

---

## Structure
```
researchASS/
├── graph/
│   ├── state.py       ← TypedDict partagé
│   ├── nodes.py       ← Fonctions des agents
│   ├── edges.py       ← Transitions conditionnelles
│   └── graph.py       ← Compilation LangGraph
├── tools/
│   ├── retriever.py   ← ChromaDB + HuggingFace
│   └── web_search.py  ← DuckDuckGo
├── prompts.py         ← Tous les prompts
├── main.py            ← API FastAPI
├── run.py             ← Mode terminal
├── requirements.txt
└── .env
```

---

## Installation
```bash
pip install -r requirements.txt
```

## Lancer en mode terminal
```bash
python run.py
```

## Lancer l'API
```bash
uvicorn main:app --reload
```
