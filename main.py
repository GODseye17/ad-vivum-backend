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
# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# import subprocess
# from retriever.retrieve_and_respond import answer_query

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["https://www.vivum.app"],  # Replace with your frontend domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QueryRequest(BaseModel):
#     query: str

# responses_cache = {}  # TEMP cache to store results

# @app.get("/")
# def root():
#     return {"message": "API is running!"}

# @app.get("/ping")
# def ping():
#     return {"status": "alive"}

# def process_query_background(query_text: str):
#     try:
#         response = answer_query(query_text, "pubmed")
#         responses_cache[query_text] = response  # Save it
#     except Exception as e:
#         responses_cache[query_text] = f"Error: {str(e)}"

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
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=port)
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Dict, List, Optional
import uuid
import asyncio
from contextlib import asynccontextmanager
import logging
import time
from supabase import create_client, Client
import datetime
import ssl
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track active background tasks
background_tasks_status = {}

# Supabase setup
supabase_url = "https://emefyicilkiaaqkbjsjy.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWZ5aWNpbGtpYWFxa2Jqc2p5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NTMzMzMxOCwiZXhwIjoyMDYwOTA5MzE4fQ.oQv782SBbK0VQPy6wuQS0oh1sfF9mcBE8dcR1J4W0SA"

if not supabase_url or not supabase_key:
    logger.error("Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY environment variables.")

# Supabase client as a global variable
supabase: Optional[Client] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Supabase client
    global supabase
    logger.info("Application startup: Initializing Supabase connection")
    try:
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase connection established")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {str(e)}")
    
    yield
    
    # Shutdown: No specific cleanup needed for Supabase client
    logger.info("Application shutdown: Cleaning up resources")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.vivum.app", "http://localhost:8081"],  # Add your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TopicRequest(BaseModel):
    topic: str
    max_results: Optional[int] = 20  # Limit the number of results to prevent overload

class QueryRequest(BaseModel):
    query: str
    topic_id: str
    conversation_id: Optional[str] = None

class TopicResponse(BaseModel):
    topic_id: str
    message: str
    status: str

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

# Memory-efficient conversation cache with auto-expiry
conversation_cache: Dict[str, Dict] = {}
MAX_CONVERSATIONS = 1000  # Prevent memory leak by limiting total conversations

@app.get("/")
def root():
    return {"message": "API is running!"}

@app.get("/supabase-status")
async def check_supabase_status():
    if supabase:
        try:
            # Try a simple query to confirm connection works
            result = supabase.table("topics").select("count").execute()
            return {"status": "connected", "message": "Supabase connection working"}
        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}
    else:
        return {"status": "disconnected", "message": "Supabase client not initialized"}
    

@app.get("/ping")
def ping():
    return {"status": "alive", "active_tasks": len(background_tasks_status)}

async def fetch_pubmed_data(topic: str, topic_id: str, max_results: int):
    """Fetch data from PubMed and store in Supabase"""
    try:
        logger.info(f"Fetching PubMed data for topic: {topic}, topic_id: {topic_id}")
        
        # PubMed API Endpoints
        SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        DETAILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        
        # Step 1: Get Article IDs
        search_params = {
            "db": "pubmed",
            "term": topic,  # Use the provided topic as search term
            "retmode": "json",
            "retmax": max_results,  # Use the user-specified max_results
        }
        
        # Use aiohttp or httpx for async requests
        import aiohttp
        
        articles = []
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            # Search for article IDs
            async with session.get(SEARCH_URL, params=search_params) as response:
                if response.status != 200:
                    logger.error(f"Error searching PubMed: {response.status}")
                    raise Exception(f"PubMed search API returned status {response.status}")
                
                data = await response.json()
                article_ids = data.get("esearchresult", {}).get("idlist", [])
                logger.info(f"Found {len(article_ids)} articles for topic: {topic}")
                
                # Step 2: Fetch details for each article
                for article_id in article_ids:
                    details_params = {
                        "db": "pubmed",
                        "id": article_id,
                        "retmode": "xml",
                        "rettype": "abstract",
                    }
                    
                    # Add a small delay to avoid overwhelming the API
                    await asyncio.sleep(0.2)
                    
                    async with session.get(DETAILS_URL, params=details_params) as details_response:
                        if details_response.status != 200:
                            logger.warning(f"Error fetching details for article {article_id}: {details_response.status}")
                            continue
                        
                        # Parse the XML response to extract title, abstract, and authors
                        import xml.etree.ElementTree as ET
                        
                        text_content = await details_response.text()
                        
                        try:
                            # Parse XML
                            root = ET.fromstring(text_content)
                            
                            # Extract title (may need adjustment based on actual XML structure)
                            title_elem = root.find(".//ArticleTitle")
                            title = title_elem.text if title_elem is not None else f"Article {article_id}"
                            
                            # Extract abstract
                            abstract_text = ""
                            abstract_elems = root.findall(".//AbstractText")
                            if abstract_elems:
                                for elem in abstract_elems:
                                    # Check if there's a label
                                    label = elem.get("Label")
                                    if label:
                                        abstract_text += f"{label}: {elem.text}\n"
                                    else:
                                        abstract_text += f"{elem.text}\n"
                            else:
                                abstract_text = "Abstract not available"
                            
                            # Extract authors
                            authors = []
                            author_elems = root.findall(".//Author")
                            for author in author_elems:
                                last_name = author.find("LastName")
                                fore_name = author.find("ForeName")
                                if last_name is not None and fore_name is not None:
                                    authors.append(f"{last_name.text} {fore_name.text}")
                                elif last_name is not None:
                                    authors.append(last_name.text)
                            
                            authors_text = ", ".join(authors) if authors else "Unknown"
                            
                            # Create article object
                            article = {
                                "id": article_id,
                                "title": title,
                                "abstract": abstract_text,
                                "authors": authors_text,
                                "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"
                            }
                            
                            articles.append(article)
                            
                        except Exception as e:
                            logger.error(f"Error parsing article {article_id}: {str(e)}")
                            # Add basic info even if parsing failed
                            articles.append({
                                "id": article_id,
                                "title": f"Article {article_id}",
                                "abstract": "Error retrieving full article details",
                                "authors": "Unknown",
                                "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"
                            })
        # Store data in Supabase
        if supabase:
            # Update the existing topic record instead of inserting a new one
            supabase.table("topics").update({
                "status": "completed"
            }).eq("id", topic_id).execute()
            
            # Then store articles
            for article in articles:
                supabase.table("articles").insert({
                    "topic_id": topic_id,
                    "pubmed_id": article["id"],
                    "title": article["title"],
                    "abstract": article["abstract"],
                    "authors": article["authors"],
                    "url": article["url"]
                }).execute()
                
            logger.info(f"Stored {len(articles)} articles for topic_id: {topic_id}")
            return True
        else:
            logger.error("Supabase client not initialized")
            return False
            
    except Exception as e:
        logger.error(f"Error fetching PubMed data: {str(e)}")
        
        # Update status in Supabase if possible
        if supabase:
            supabase.table("topics").update({"status": f"error: {str(e)}", "article_count": 0}).eq("id", topic_id).execute()
        
        return False

def check_topic_fetch_status(topic_id: str):
    """Check if data fetching is complete for a topic"""
    # First check our internal background task status
    if topic_id in background_tasks_status:
        return background_tasks_status[topic_id]
    
    # Then check in Supabase
    if supabase:
        try:
            result = supabase.table("topics").select("status").eq("id", topic_id).execute()
            if result.data and len(result.data) > 0:
                return result.data[0]["status"]
            return "not_found"
        except Exception as e:
            logger.error(f"Error checking topic status: {str(e)}")
            return f"error: {str(e)}"
    else:
        logger.error("Supabase client not initialized")
        return "database_error"

async def answer_from_stored_data(query: str, topic_id: str, conversation_context=None):
    """Retrieve data from Supabase and generate an answer using Gemini"""
    try:
        logger.info(f"Generating answer for query about topic_id: {topic_id}")
        
        if not supabase:
            return "Unable to connect to the database."
        
        # Fetch relevant articles from Supabase
        result = supabase.table("articles").select("*").eq("topic_id", topic_id).execute()
        
        if not result.data or len(result.data) == 0:
            return "No data found for this topic."
        
        articles = result.data
        logger.info(f"Retrieved {len(articles)} articles for topic_id: {topic_id}")
        
        # Prepare context for answering
        context = ""
        # Add all articles (up to 10 to keep context size reasonable)
        for i, article in enumerate(articles[:10]):
            context += f"Article {i+1}: {article['title']}\n"
            context += f"Abstract: {article['abstract']}\n"
            context += f"Authors: {article['authors']}\n"
            context += f"URL: {article.get('url', '')}\n\n"
        
        # Import Gemini libraries
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        # Set up the API
        
        genai.configure(api_key="AIzaSyA-AfbLuDw6cJbWkU3w8ADhNfXj6DGEQ0Y")
        
        # Configure the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Create a Gemini model instance
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # First-time query vs follow-up determination
        is_first_query = not conversation_context or conversation_context == ""
        
        # Construct the prompt
        if is_first_query:
            prompt = """You are a medical research assistant that creates literature surveys based on PubMed articles.
            Analyze the articles provided and create a comprehensive literature survey on the topic.
            Your response should:
            1. Summarize the key findings across all articles
            2. Identify common themes and research directions
            3. Highlight any contradictions or gaps in the research
            4. Include relevant citations to the specific articles
            Be concise but thorough, and format your response in a structured way with clear headings.
            
            Based on these PubMed articles, create a literature survey:
            
            """
            prompt += f"{context}\n\nTopic: {query}"
        else:
            prompt = """You are a medical research assistant that answers questions based on PubMed articles.
            Use only the information provided in the articles to answer questions.
            If the answer cannot be found in the articles, acknowledge this limitation.
            Cite specific articles when providing information.
            Be concise, direct, and helpful.
            
            Articles:
            """
            
            prompt += f"{context}\n\nPrevious conversation:\n{conversation_context}\n\nQuestion: {query}"
        
        # Generate the response using the combined prompt
        response = model.generate_content(prompt)
        
        # Extract the text from the response
        answer = response.text if hasattr(response, "text") else str(response)
        
        # Log the query to Supabase for analysis
        supabase.table("queries").insert({
            "topic_id": topic_id,
            "query": query,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }).execute()
        
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"I'm sorry, there was an error processing your query: {str(e)}"

async def fetch_data_background(topic: str, topic_id: str, max_results: int):
    """Background task to fetch data from PubMed and store in Supabase"""
    try:
        background_tasks_status[topic_id] = "processing"
        
        # Set timeout for the fetch operation to prevent hanging
        fetch_timeout = 60  # seconds
        try:
            # Run with timeout
            success = await asyncio.wait_for(
                fetch_pubmed_data(topic, topic_id, max_results),
                timeout=fetch_timeout
            )
            
            if success:
                background_tasks_status[topic_id] = "completed"
            else:
                background_tasks_status[topic_id] = "failed"
        except asyncio.TimeoutError:
            background_tasks_status[topic_id] = "timeout"
            logger.error(f"Fetch operation timed out for topic_id: {topic_id}")
            
            # Update status in Supabase
            if supabase:
                supabase.table("topics").update({"status": "timeout"}).eq("id", topic_id).execute()
                
        except Exception as e:
            background_tasks_status[topic_id] = f"error: {str(e)}"
            logger.error(f"Error in fetch operation for topic_id {topic_id}: {str(e)}")
            
            # Update status in Supabase
            if supabase:
                supabase.table("topics").update({"status": f"error: {str(e)}"}).eq("id", topic_id).execute()
    finally:
        # Keep status for a while but eventually clean up
        await asyncio.sleep(3600)  # Keep status for 1 hour
        if topic_id in background_tasks_status:
            del background_tasks_status[topic_id]

@app.post("/fetch-topic-data", response_model=TopicResponse)
async def fetch_topic_data(request: TopicRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to fetch data from PubMed for a topic and store in Supabase
    Returns a topic_id that can be used for querying later
    """
    try:
        # Check if Supabase is connected
        if not supabase:
            raise HTTPException(
                status_code=503,
                detail="Database connection not available"
            )
        
        # Limit concurrent tasks to prevent overload
        if len(background_tasks_status) >= 5:  # Adjust based on Railway resources
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many processing tasks. Please try again later."
            )
        
        # Generate a unique topic ID
        topic_id = str(uuid.uuid4())
        
        # Create initial record in Supabase
        supabase.table("topics").insert({
            "id": topic_id,
            "topic": request.topic,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "processing"
        }).execute()
        
        # Start background task to fetch and store data
        background_tasks.add_task(
            fetch_data_background, 
            request.topic, 
            topic_id, 
            request.max_results
        )
        
        return {
            "topic_id": topic_id,
            "message": f"Started fetching data for topic: {request.topic} (limited to {request.max_results} results)",
            "status": "processing"
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error initiating fetch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_conversation_history(conversation_id: str) -> tuple:
    """Get conversation history or create a new one"""
    # Clean up old conversations if needed
    if len(conversation_cache) > MAX_CONVERSATIONS:
        # Remove oldest conversations (simple approach)
        oldest_keys = sorted(conversation_cache.keys(), 
                            key=lambda k: conversation_cache[k].get("last_access", 0))[:100]
        for key in oldest_keys:
            del conversation_cache[key]
    
    current_time = time.time()
    
    if not conversation_id or conversation_id not in conversation_cache:
        # Create a new conversation history
        new_id = str(uuid.uuid4())
        conversation_cache[new_id] = {
            "messages": [],
            "last_access": current_time
        }
        return new_id, []
    
    # Update last access time
    conversation_cache[conversation_id]["last_access"] = current_time
    return conversation_id, conversation_cache[conversation_id]["messages"]

@app.post("/query", response_model=ChatResponse)
async def answer_query_from_stored(request: QueryRequest):
    """
    Endpoint to answer questions based on previously fetched data
    Uses the topic_id to retrieve relevant data from Supabase
    """
    try:
        # Check if Supabase is connected
        if not supabase:
            raise HTTPException(
                status_code=503,
                detail="Database connection not available"
            )
            
        # Check if data fetching is complete
        status = check_topic_fetch_status(request.topic_id)
        if status != "completed":
            if status == "processing":
                message = "Data is still being fetched. Please try again in a moment."
            elif status == "not_found":
                message = "No data found for this topic. Please fetch the topic data first."
            else:
                message = f"Cannot process query. Data fetch status: {status}"
            
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=message)
        
        # Get or create conversation history
        conversation_id, history = get_conversation_history(request.conversation_id)
        
        # Add the user's query to the history
        history.append({"role": "user", "content": request.query})
        
        # Format history for context
        context = ""
        if history:
            # Only use last few messages to keep context size manageable
            for message in history[-3:]:  # Reduced from 5 to 3 for efficiency
                if message["role"] == "assistant":
                    context += f"Assistant: {message['content']}\n"
                else:
                    context += f"User: {message['content']}\n"
        
        # Set a timeout for answer generation
        try:
            # Get the answer using the stored data for the topic
            response = await asyncio.wait_for(
                answer_from_stored_data(
                    query=request.query,
                    topic_id=request.topic_id,
                    conversation_context=context if len(history) > 1 else None
                ),
                timeout=30  # 30 seconds timeout
            )
        except asyncio.TimeoutError:
            response = "I'm sorry, the response is taking too long to generate. Please try a simpler query."
        
        # Add the assistant's response to history
        history.append({"role": "assistant", "content": response})
        
        # Update the cache - make sure we're limiting memory usage
        if len(history) > 20:  # Limit conversation length
            history = history[-20:]  # Keep only the most recent messages
        
        conversation_cache[conversation_id]["messages"] = history
        
        # Optionally store the conversation in Supabase for persistence
        # supabase.table("conversations").upsert({
        #     "id": conversation_id,
        #     "topic_id": request.topic_id,
        #     "messages": history,
        #     "last_updated": time.time()
        # }).execute()
        
        return {"response": response, "conversation_id": conversation_id}
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error in query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topic/{topic_id}/articles")
async def get_topic_articles(topic_id: str, limit: int = 100, offset: int = 0):
    """
    Fetch all articles for a specific topic
    
    Args:
        topic_id: The UUID of the topic
        limit: Maximum number of articles to return (default: 100)
        offset: Number of articles to skip (for pagination)
        
    Returns:
        List of articles with their metadata and content
    """
    try:
        # Check if Supabase is connected
        if not supabase:
            raise HTTPException(
                status_code=503,
                detail="Database connection not available"
            )
            
        # First verify the topic exists
        topic_result = supabase.table("topics").select("*").eq("id", topic_id).execute()
        
        if not topic_result.data:
            raise HTTPException(
                status_code=404,
                detail="Topic not found"
            )
            
        # Check if data fetching is complete
        status = check_topic_fetch_status(topic_id)
        if status != "completed":
            return {
                "topic_id": topic_id,
                "status": status,
                "articles": [],
                "message": "Data is still being processed or had an error"
            }
        
        # Fetch articles with pagination
        articles_result = supabase.table("articles") \
            .select("*") \
            .eq("topic_id", topic_id) \
            .range(offset, offset + limit - 1) \
            .execute()
            
        # Get the total count (for pagination info)
        count_result = supabase.table("articles") \
            .select("id", count="exact") \
            .eq("topic_id", topic_id) \
            .execute()
            
        total_count = count_result.count if hasattr(count_result, "count") else len(articles_result.data)
        
        return {
            "topic_id": topic_id,
            "status": "completed",
            "articles": articles_result.data,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count
            }
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error fetching articles for topic {topic_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topic/{topic_id}/status")
async def check_topic_status(topic_id: str):
    """
    Check the status of data fetching for a topic
    """
    # First check our internal background task status
    if topic_id in background_tasks_status:
        return {"topic_id": topic_id, "status": background_tasks_status[topic_id]}
    
    # If not in our internal tracking, check in the database
    status = check_topic_fetch_status(topic_id)
    return {"topic_id": topic_id, "status": status}

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Endpoint to retrieve conversation history"""
    if conversation_id not in conversation_cache:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conversation_cache[conversation_id]["messages"]}

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Endpoint to delete a conversation"""
    if conversation_id in conversation_cache:
        del conversation_cache[conversation_id]
    return {"status": "deleted"}

# Health check endpoint for Railway
@app.get("/health")
def health_check():
    return {"status": "healthy", "database": "connected" if supabase else "disconnected"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Use uvicorn with optimized settings for Railway
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        workers=1,  # Single worker to avoid memory issues
        log_level="info",
        timeout_keep_alive=65  # Railway closes idle connections after 75s
    )

