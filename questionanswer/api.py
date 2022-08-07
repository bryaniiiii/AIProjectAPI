from typing import Dict

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .modelfolder.model import Model, get_model

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionAnsweringRequest(BaseModel):
    question: str
    answer_text: str  

class QuestionAnsweringResponse(BaseModel):
    answer: str
    confidence: float 



@app.post("/predict", response_model=QuestionAnsweringResponse)
def predict(request: QuestionAnsweringRequest, model: Model = Depends(get_model)):
    answer, confidence = model.predict(request.question,request.answer_text)
    return QuestionAnsweringResponse(
        answer=answer,
        confidence = confidence
    )
