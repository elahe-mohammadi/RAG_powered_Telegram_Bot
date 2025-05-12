from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from functools import lru_cache

from ..services.index_creator import VectorStore, VSSettings
from ..services.llm import LLMService, LLMSettings

router = APIRouter()

# Shared singletons
_vs = VectorStore(VSSettings())
_ls = LLMService(LLMSettings())

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None

@lru_cache(maxsize=1024)
def _get_response(question: str) -> AnswerResponse:
    docs = _vs.similarity_search(question, k=5)
    if not docs:
        return AnswerResponse(
            answer="I don't have enough information to answer that question.",
            sources=[],
        )

    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        "You are a helpful assistant answering questions based on the provided context. "
        "If the context doesn't contain information to answer the question, say that You don't have enough information to answer that question. "
        "Don't make up information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    answer = _ls.generate(prompt)
    sources = [d.metadata.get("source", "") for d in docs]
    return AnswerResponse(answer=answer, sources=sources)

# FastAPI endpoint
@router.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        # Now a cache lookup under the hood
        return _get_response(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
