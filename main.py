from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph.graph import graph   # ✅ graph + embeddings chargés au démarrage
import uuid

app = FastAPI(title="ResearchASS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    session_id: str = None


@app.get("/")
def root():
    return {"status": "ResearchASS is running 🚀"}


@app.post("/research")
async def research(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    result = await graph.ainvoke(
        {"question": req.question},
        config=config,
    )

    return {
        "question": result["question"],
        "answer": result["generation"],
        "source": result["source"],
        "session_id": session_id,
    }


@app.post("/add-documents")
def add_documents(texts: list[str]):
    from tools.retriever import add_documents_to_db
    add_documents_to_db(texts)
    return {"message": f"{len(texts)} documents ajoutés"}