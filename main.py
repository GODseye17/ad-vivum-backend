from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import subprocess
from retriever.retrieve_and_respond import answer_query

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.vivum.app"],  # Replace with ["https://your-frontend.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "API is running!"}

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.post("/query")
def get_query_response(request: QueryRequest):
    try:
        # Call the same answer_query function from your script
        response = answer_query(request.query, "pubmed")  # If selected isn't used, keep it None
        return {"response": response}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error during subprocess: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

