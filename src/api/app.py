from fastapi import FastAPI
from pydantic import BaseModel
from src.models.retrieval import retrieve_documents
from src.models.generation import generate_answer

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    documents = retrieve_documents(query.question)
    answer = generate_answer(query.question, documents)
    return {"answer": answer}