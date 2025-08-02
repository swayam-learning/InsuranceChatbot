import os
import shutil
import json
import uuid
import faiss
import numpy as np
import re
import concurrent.futures
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import nest_asyncio
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Configuration - Using smaller L5 model
BASE_PATH = "/app"
PARSED_TEXT_OUTPUT_FOLDER = os.path.join(BASE_PATH, "Parsed_text")
FAISS_INDEX_OUTPUT_FOLDER = os.path.join(BASE_PATH, "faiss_index")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L5-v2'  # Changed to smaller L5 model
TOP_K_RETRIEVAL = 5

# Ensure directories exist
os.makedirs(PARSED_TEXT_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_OUTPUT_FOLDER, exist_ok=True)

# Initialize FastAPI
nest_asyncio.apply()
app = FastAPI(title="Bajaj PDF QnA Pipeline API")

# Security setup
bearer_scheme = HTTPBearer(auto_error=False)
VALID_TOKEN = os.getenv(
    "VALID_TOKEN", 
    "ssbakscstobcb3609e845e387e9f7ac988ea36090473eefbe6dae9cfe880c35c6b67d87a7757"
)

# Initialize models with memory optimization
try:
    print("Loading smaller SentenceTransformer model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/app/cache")
    print(f"Model loaded successfully: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    print(f"Failed to load SentenceTransformer: {e}")
    raise

def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    if token != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid token",
        )
    return token

def cleanup_temp_files():
    """Clean up temporary files while preserving directory structure"""
    try:
        for root, dirs, files in os.walk(PARSED_TEXT_OUTPUT_FOLDER):
            for f in files:
                os.unlink(os.path.join(root, f))
        for root, dirs, files in os.walk(FAISS_INDEX_OUTPUT_FOLDER):
            for f in files:
                os.unlink(os.path.join(root, f))
    except Exception as e:
        print(f"Cleanup error: {e}")

def parse_and_chunk_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract and chunk PDF with smaller chunks"""
    all_page_texts = []
    try:
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    all_page_texts.append({"text": page_text, "page_number": i + 1})
    except Exception as e:
        raise RuntimeError(f"PDF reading failed: {e}")

    if not all_page_texts:
        return []

    # Using smaller chunks for memory optimization
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Reduced from 500
        chunk_overlap=30,  # Reduced from 50
        length_function=len,
        is_separator_regex=False,
    )

    processed_chunks = []
    for page_info in all_page_texts:
        chunks_from_page = text_splitter.split_text(page_info["text"])
        for chunk_content in chunks_from_page:
            clean_chunk = " ".join(chunk_content.split()).strip()
            if clean_chunk:
                processed_chunks.append({
                    "content": clean_chunk,
                    "metadata": {
                        "page_number": page_info["page_number"],
                        "source_file": os.path.basename(file_path),
                    }
                })
    return processed_chunks

def clean_query(query: str) -> str:
    """Clean and normalize user query"""
    cleaned_text = "".join(char for char in query if char.isalnum() or char.isspace()).strip()
    return re.sub(r"\s+", " ", cleaned_text)

def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    """Normalize vectors for similarity comparison"""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vecs / norms

async def call_llm_api(prompt_messages: List[Dict[str, str]], json_output: bool = False) -> str:
    """Call the Together API for LLM completion"""
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    if not TOGETHER_API_KEY:
        raise RuntimeError("TOGETHER_API_KEY not set in environment.")
    
    client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1/")
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": prompt_messages,
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    if json_output:
        payload["response_format"] = {"type": "json_object"}
    
    try:
        response = client.chat.completions.create(**payload)
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {e}")

async def enhance_and_extract_query_with_llm(raw_query: str) -> Dict[str, Any]:
    """Use LLM to improve and extract entities from query"""
    prompt = f"""You are a smart assistant that improves insurance-related queries and extracts important entities.
Given:
"{raw_query}"
Respond with a JSON like:
{{
    "corrected_and_rephrased_query": "...",
    "extracted_entities": {{
        "age": "...",
        "location": "...",
        ...
    }}
}}
Only include non-null entities. Omit any missing or irrelevant ones.
"""
    try:
        messages = [{"role": "user", "content": prompt}]
        llm_output_str = await call_llm_api(messages, json_output=True)
        parsed = json.loads(llm_output_str)
        parsed["extracted_entities"] = {
            k: v for k, v in parsed.get("extracted_entities", {}).items() if v is not None
        }
        return parsed
    except Exception as e:
        print(f"Query enhancement failed, using fallback: {e}")
        return {
            "corrected_and_rephrased_query": clean_query(raw_query),
            "extracted_entities": {}
        }

def construct_rag_prompt(user_query: str, context_chunks: List[Dict[str, Any]], extracted_entities: Dict[str, Any]) -> str:
    """Construct the final prompt for the LLM"""
    if not context_chunks:
        return f"""You are a helpful assistant. No document context was found. Try to answer this:
USER QUERY:
{user_query}"""
    
    context_text = "\n\n".join([
        f"--- Document: {chunk.get('source_file', 'Unknown')}, Page: {chunk.get('page_number', 'N/A')} ---\n"
        f"{chunk.get('content', '') or chunk.get('text', '')}"
        for chunk in context_chunks
    ])
    
    entity_lines = "\n".join([f"- {k}: {v}" for k, v in extracted_entities.items()]) if extracted_entities else "None"
    
    return f"""You are an expert insurance assistant.
USER QUERY:
{user_query}
EXTRACTED ENTITIES:
{entity_lines}
DOCUMENT CONTEXT:
{context_text}
Answer clearly using the document context above. If no answer is found, say: "I could not find the answer in the provided documents."
"""

def retrieve_top_k(query_for_embedding: str, k: int, faiss_index, metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Retrieve top k most relevant chunks"""
    if faiss_index.ntotal == 0:
        return []
    
    query_vector = embedding_model.encode([query_for_embedding]).astype("float32")
    query_vector = normalize_vectors(query_vector)
    D, I = faiss_index.search(query_vector, k)
    
    results = []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            chunk = metadata[idx].copy()
            score_index = (I[0] == idx).nonzero()[0][0]
            chunk["retrieval_score"] = float(D[0, score_index])
            results.append(chunk)
    return results

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": EMBEDDING_MODEL_NAME,
        "memory_optimized": True,
        "service": "Bajaj PDF QnA Pipeline"
    }

@app.post("/api/v1/hackrx/run")
async def hackrx_run_api(
    pdf: UploadFile = File(...),
    query: str = Form(...),
    token: str = Depends(verify_token),
):
    """Main API endpoint for processing PDF queries with memory optimizations"""
    try:
        cleanup_temp_files()

        # Save PDF file
        unique_name = f"{uuid.uuid4()}_{pdf.filename}"
        pdf_path = os.path.join(PARSED_TEXT_OUTPUT_FOLDER, unique_name)
        with open(pdf_path, "wb") as f:
            f.write(await pdf.read())

        # Parse and chunk PDF with smaller chunks
        all_chunks = parse_and_chunk_pdf(pdf_path)
        if not all_chunks:
            return {"error": "Parsing failed or PDF contains no extractable text."}

        # Vectorize chunks with limited concurrency
        chunk_contents = [chunk["content"] for chunk in all_chunks]
        chunk_metadatas = [chunk["metadata"] for chunk in all_chunks]

        # Process embeddings in batches with limited workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            embeddings = list(executor.map(
                lambda x: embedding_model.encode(x, show_progress_bar=False),
                [chunk_contents]
            ))[0]

        embeddings = np.array(embeddings).astype("float32")
        embedding_dim = embeddings.shape[1]
        
        # Use more memory-efficient FAISS index
        quantizer = faiss.IndexFlatL2(embedding_dim)
        faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, min(100, len(embeddings)))
        faiss_index.train(embeddings)
        faiss_index.add(embeddings)

        # Process query
        processed = await enhance_and_extract_query_with_llm(query)
        query_for_embedding = clean_query(processed["corrected_and_rephrased_query"])
        extracted_entities = processed["extracted_entities"]

        # Retrieve relevant chunks
        top_chunks = retrieve_top_k(
            query_for_embedding, 
            TOP_K_RETRIEVAL, 
            faiss_index, 
            chunk_metadatas
        )

        # Construct and send to LLM
        rag_prompt = construct_rag_prompt(query, top_chunks, extracted_entities)
        answer = await call_llm_api([
            {"role": "system", "content": "You are a helpful insurance assistant."},
            {"role": "user", "content": rag_prompt}
        ], json_output=False)

        cleanup_temp_files()
        return {"answer": answer.strip()}

    except Exception as e:
        cleanup_temp_files()
        return {"error": str(e), "model": EMBEDDING_MODEL_NAME}
