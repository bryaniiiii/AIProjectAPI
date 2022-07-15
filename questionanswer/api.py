from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .modelfolder.model import Model, get_model

app = FastAPI()


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
