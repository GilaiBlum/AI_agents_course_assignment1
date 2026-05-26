from fastapi import FastAPI
from pydantic import BaseModel
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from langchain_core.messages import SystemMessage, HumanMessage

from KEYS import API_KEY, PINECONE_KEY

# --- 1. Credentials ---
LLMOD_API_KEY = API_KEY
LLMOD_BASE_URL = "https://api.llmod.ai/v1"
PINECONE_API_KEY = PINECONE_KEY
INDEX_NAME = "assignment1"

# --- 2. RAG Hyperparameters ---
# Endpoint /api/stats must reflect these exactly
CHUNK_SIZE = 512
OVERLAP_RATIO = 0.2
TOP_K = 7

# --- 3. Initialize Clients ---
emb = OpenAIEmbeddings(
    api_key=LLMOD_API_KEY,
    base_url=LLMOD_BASE_URL,
    model="4UHRUIN-text-embedding-3-small"
)

llm = ChatOpenAI(
    api_key=LLMOD_API_KEY,
    base_url=LLMOD_BASE_URL,
    model="4UHRUIN-gpt-5-mini" 
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

app = FastAPI()

# --- 4. Data Models for API ---
class PromptRequest(BaseModel):
    question: str

# --- 5. Mandatory System Prompt ---
SYSTEM_PROMPT_TEMPLATE = """You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). 
You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided Medium articles data."
Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.

Context provided:
{context_str}
"""

# --- 6. Endpoints ---

@app.get("/api/stats")
def get_stats():
    """Returns the strict JSON configuration chosen for the RAG system."""
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }

@app.post("/api/prompt")
def generate_prompt(request: PromptRequest):
    """Handles the core RAG logic."""
    question = request.question
    
    # 1. Embed the user's question
    question_vector = emb.embed_query(question)
    
    # 2. Retrieve top-k chunks from Pinecone
    search_results = index.query(
        vector=question_vector,
        top_k=TOP_K,
        include_metadata=True
    )
    
    # 3. Format the retrieved context for the output and the LLM
    context_list = []
    context_text_for_llm = ""
    
    for match in search_results['matches']:
        metadata = match['metadata']
        score = match['score']
        
        # Build the array requested in the assignment output format
        context_list.append({
            "article_id": metadata.get("article_id", "unknown"),
            "title": metadata.get("title", "unknown"),
            "chunk": metadata.get("chunk", ""),
            "score": score
        })
        
        # Build the readable string for the LLM's system prompt
        context_text_for_llm += f"\n--- Article ID: {metadata.get('article_id')} | Title: {metadata.get('title')} ---\n{metadata.get('chunk')}\n"
    
    # 4. Construct the Prompts
    final_system_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{context_str}", context_text_for_llm)
    
    # 5. Call the Chat Model
    messages = [
        SystemMessage(content=final_system_prompt),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    
    # 6. Return the exact structured JSON format
    return {
        "response": response.content,
        "context": context_list,
        "Augmented_prompt": {
            "System": final_system_prompt,
            "User": question
        }
    }