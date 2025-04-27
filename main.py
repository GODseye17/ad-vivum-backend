# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# import subprocess
# from retriever.retrieve_and_respond import answer_query

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["https://www.vivum.app"],  # Replace with ["https://your-frontend.vercel.app"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QueryRequest(BaseModel):
#     query: str

# @app.get("/")
# def root():
#     return {"message": "API is running!"}

# @app.get("/ping")
# def ping():
#     return {"status": "alive"}

# @app.post("/query")
# def get_query_response(request: QueryRequest):
#     try:
#         # Call the same answer_query function from your script
#         response = answer_query(request.query, "pubmed")  # If selected isn't used, keep it None
#         return {"response": response}
#     except subprocess.CalledProcessError as e:
#         raise HTTPException(status_code=500, detail=f"Error during subprocess: {e}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import subprocess
from retriever.retrieve_and_respond import answer_query

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.vivum.app"],  # Replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

responses_cache = {}  # TEMP cache to store results

@app.get("/")
def root():
    return {"message": "API is running!"}

@app.get("/ping")
def ping():
    return {"status": "alive"}

def process_query_background(query_text: str):
    try:
        response = answer_query(query_text, "pubmed")
        responses_cache[query_text] = response  # Save it
    except Exception as e:
        responses_cache[query_text] = f"Error: {str(e)}"

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)

