import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ML.predmodel import Model, get_model


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class SentimentResponse(BaseModel):
    sentiment: str


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=SentimentResponse)
def predict(input: SentimentRequest, model: Model = Depends(get_model)):
    try:
        prediction = model.predict(input.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to run sentiment prediction") from exc
    return SentimentResponse(sentiment=prediction)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
