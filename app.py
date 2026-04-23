"""
Step 4: FastAPI Web服务
用法: python app.py
访问: http://localhost:8000/docs 查看API文档
"""
from fastapi import FastAPI
from pydantic import BaseModel
from chain import RAGChain

app = FastAPI(title="Dify文档RAG助手", version="1.0")
rag_chain = RAGChain()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved_chunks: int


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = rag_chain.query(req.question)
    return QueryResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
