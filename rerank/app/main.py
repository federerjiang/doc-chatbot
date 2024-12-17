from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# Initialize the model
MODEL_NAME = "hotchpotch/japanese-reranker-cross-encoder-large-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder(MODEL_NAME, max_length=512, device=device)
if device == "cuda":
    model.model.half()

app = FastAPI()

class Query(BaseModel):
    sentence1: str
    sentence2: str

@app.post("/score/")
def calculate_score(query: Query):
    try:
        logging.info(f"Received request: {query}")
        score = model.predict([(query.sentence1, query.sentence2)])[0]
        logging.info(f"Returning score: {score}")
        # Convert numpy.float32 to Python float, if necessary
        if isinstance(score, np.float32) or isinstance(score, np.float64):
            score = float(score)
        return {"score": score}
    except Exception as e:
        logging.error("Error in calculate_score", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Run the application
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
