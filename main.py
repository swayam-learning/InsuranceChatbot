# # import os
# # import shutil
# # import json
# # from typing import List, Dict, Any
# # from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Security
# # from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# # import uuid
# # import faiss
# # import numpy as np
# # from sentence_transformers import SentenceTransformer
# # import pypdf
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # import nest_asyncio
# # import asyncio
# # from dotenv import load_dotenv
# # from openai import OpenAI

# # # --- Constants and Paths ---
# # BASE_PATH = r"C:\Users\hp\OneDrive\codes\Project-hackathon"  # Change if your base folder is different
# # PARSED_TEXT_OUTPUT_FOLDER = os.path.join(BASE_PATH, "Parsed_text")
# # FAISS_INDEX_OUTPUT_FOLDER = os.path.join(BASE_PATH, "faiss_index")
# # AGGREGATED_CHUNKS_FILENAME = 'all_policy_chunks.jsonl'
# # FAISS_INDEX_FILENAME = 'policy_chunks_faiss_index.bin'
# # FAISS_METADATA_FILENAME = 'policy_chunks_metadata.json'
# # EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# # TOP_K_RETRIEVAL = 5

# # os.makedirs(PARSED_TEXT_OUTPUT_FOLDER, exist_ok=True)
# # os.makedirs(FAISS_INDEX_OUTPUT_FOLDER, exist_ok=True)

# # nest_asyncio.apply()

# # # --- FastAPI App ---
# # app = FastAPI(title="Bajaj PDF QnA Pipeline API")

# # # --- Security Setup ---
# # bearer_scheme = HTTPBearer(auto_error=False)
# # # Replace this token with your real hackathon/team token or load from env/config securely
# # VALID_TOKEN = "ssbadscstobc3609e845e387e9f7ac988ea36090473eefbe6dae9cfe880c35c6b67d87a7757"


# # def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
# #     if credentials is None or credentials.scheme.lower() != "bearer":
# #         raise HTTPException(
# #             status_code=status.HTTP_401_UNAUTHORIZED,
# #             detail="Missing or invalid authorization scheme",
# #             headers={"WWW-Authenticate": "Bearer"},
# #         )
# #     token = credentials.credentials
# #     if token != VALID_TOKEN:
# #         raise HTTPException(
# #             status_code=status.HTTP_403_FORBIDDEN,
# #             detail="Invalid token",
# #         )
# #     return token


# # # --- Utility: Cleanup Temporary Files ---
# # def cleanup_temp_files():
# #     try:
# #         if os.path.exists(PARSED_TEXT_OUTPUT_FOLDER):
# #             shutil.rmtree(PARSED_TEXT_OUTPUT_FOLDER)
# #         if os.path.exists(FAISS_INDEX_OUTPUT_FOLDER):
# #             shutil.rmtree(FAISS_INDEX_OUTPUT_FOLDER)
# #     except Exception as e:
# #         print(f"Cleanup error: {e}")


# # # --- PDF Parsing and Chunking ---
# # def parse_and_chunk_pdf(file_path: str) -> List[Dict[str, Any]]:
# #     all_page_texts = []
# #     try:
# #         with open(file_path, "rb") as f:
# #             reader = pypdf.PdfReader(f)
# #             for i, page in enumerate(reader.pages):
# #                 page_text = page.extract_text()
# #                 if page_text:
# #                     all_page_texts.append({"text": page_text, "page_number": i + 1})
# #     except Exception as e:
# #         # Allow raising so calling function can handle
# #         raise RuntimeError(f"PDF reading failed: {e}")

# #     if not all_page_texts:
# #         return []

# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=500,
# #         chunk_overlap=50,
# #         length_function=len,
# #         is_separator_regex=False,
# #     )
# #     processed_chunks = []
# #     for page_info in all_page_texts:
# #         chunks_from_page = text_splitter.split_text(page_info["text"])
# #         for chunk_content in chunks_from_page:
# #             clean_chunk = ' '.join(chunk_content.split()).strip()
# #             if clean_chunk:
# #                 processed_chunks.append({
# #                     "content": clean_chunk,
# #                     "metadata": {
# #                         "page_number": page_info["page_number"],
# #                         "source_file": os.path.basename(file_path),
# #                     }
# #                 })
# #     return processed_chunks


# # def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_dir: str):
# #     os.makedirs(output_dir, exist_ok=True)
# #     out_path = os.path.join(output_dir, AGGREGATED_CHUNKS_FILENAME)
# #     with open(out_path, 'w', encoding='utf-8') as outfile:
# #         for chunk in chunks:
# #             outfile.write(json.dumps(chunk, ensure_ascii=False) + '\n')
# #     return out_path


# # # --- Vectorization Using FAISS ---
# # def vectorize_chunks(jsonl_path: str, output_dir: str):
# #     all_chunks = []
# #     with open(jsonl_path, 'r', encoding='utf-8') as f:
# #         for line in f:
# #             all_chunks.append(json.loads(line))
# #     chunk_contents = [chunk['content'] for chunk in all_chunks]
# #     chunk_metadatas = [
# #         {
# #             "source_file": chunk['metadata']['source_file'],
# #             "page_number": chunk['metadata']['page_number'],
# #             "content": chunk['content']
# #         } for chunk in all_chunks
# #     ]
# #     embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
# #     embeddings = embedding_model.encode(chunk_contents, show_progress_bar=False)
# #     embeddings = np.array(embeddings).astype('float32')
# #     embedding_dim = embeddings.shape[1]
# #     index = faiss.IndexFlatL2(embedding_dim)
# #     index.add(embeddings)

# #     os.makedirs(output_dir, exist_ok=True)
# #     faiss_path = os.path.join(output_dir, FAISS_INDEX_FILENAME)
# #     meta_path = os.path.join(output_dir, FAISS_METADATA_FILENAME)
# #     faiss.write_index(index, faiss_path)
# #     with open(meta_path, 'w', encoding='utf-8') as f:
# #         json.dump(chunk_metadatas, f, ensure_ascii=False, indent=2)
# #     return faiss_path, meta_path


# # # --- Query Helpers ---
# # def clean_query(query: str) -> str:
# #     import re
# #     cleaned_text = ''.join(char for char in query if char.isalnum() or char.isspace()).strip()
# #     return re.sub(r'\s+', ' ', cleaned_text)


# # def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
# #     norms = np.linalg.norm(vecs, axis=1, keepdims=True)
# #     norms[norms == 0] = 1e-12
# #     return vecs / norms


# # async def call_llm_api(prompt_messages: List[Dict[str, str]], json_output: bool = False) -> str:
# #     dotenv_path = os.path.join(BASE_PATH, 'Ai-project', '.env')
# #     if not os.path.exists(dotenv_path):
# #         raise RuntimeError(".env file not found")
# #     load_dotenv(dotenv_path=dotenv_path)
# #     TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
# #     if not TOGETHER_API_KEY:
# #         raise RuntimeError("TOGETHER_API_KEY not set in .env")
# #     client = OpenAI(
# #         api_key=TOGETHER_API_KEY,
# #         base_url="https://api.together.xyz/v1/")
# #     payload = {
# #         "model": "mistralai/Mistral-7B-Instruct-v0.2",
# #         "messages": prompt_messages,
# #         "temperature": 0.2,
# #         "max_tokens": 1024,
# #     }
# #     if json_output:
# #         payload["response_format"] = {"type": "json_object"}
# #     response = client.chat.completions.create(**payload)
# #     return response.choices[0].message.content


# # async def enhance_and_extract_query_with_llm(raw_query: str) -> Dict[str, Any]:
# #     prompt = f"""You are a smart assistant that improves insurance-related queries and extracts important entities.
# # Given:
# # "{raw_query}"
# # Respond with a JSON like:
# # {{
# #     "corrected_and_rephrased_query": "...",
# #     "extracted_entities": {{
# #         "age": "...",
# #         "location": "...",
# #         ...
# #     }}
# # }}
# # Only include non-null entities. Omit any missing or irrelevant ones.
# # """
# #     try:
# #         messages = [{"role": "user", "content": prompt}]
# #         llm_output_str = await call_llm_api(messages, json_output=True)
# #         parsed = json.loads(llm_output_str)
# #         parsed["extracted_entities"] = {
# #             k: v for k, v in parsed.get("extracted_entities", {}).items() if v is not None
# #         }
# #         return parsed
# #     except Exception:
# #         return {
# #             "corrected_and_rephrased_query": clean_query(raw_query),
# #             "extracted_entities": {}
# #         }


# # def construct_rag_prompt(user_query: str, context_chunks: List[Dict[str, Any]], extracted_entities: Dict[str, Any]) -> str:
# #     if not context_chunks:
# #         return f"""You are a helpful assistant. No document context was found. Try to answer this:
# # USER QUERY:
# # {user_query}"""
# #     context_text = "\n\n".join([
# #         f"--- Document: {chunk.get('source_file', 'Unknown')}, Page: {chunk.get('page_number', 'N/A')} ---\n{chunk.get('content') or chunk.get('text', '')}"
# #         for chunk in context_chunks
# #     ])
# #     entity_lines = "\n".join([f"- {k}: {v}" for k, v in extracted_entities.items()]) if extracted_entities else "None"
# #     return f"""You are an expert insurance assistant.
# # USER QUERY:
# # {user_query}
# # EXTRACTED ENTITIES:
# # {entity_lines}
# # DOCUMENT CONTEXT:
# # {context_text}
# # Answer clearly using the document context above. If no answer is found, say: "I could not find the answer in the provided documents."
# # """


# # def retrieve_top_k(query_for_embedding: str, k: int, faiss_index, metadata, embedder) -> List[Dict[str, Any]]:
# #     if faiss_index.ntotal == 0:
# #         return []
# #     query_vector = embedder.encode([query_for_embedding]).astype("float32")
# #     query_vector = normalize_vectors(query_vector)
# #     D, I = faiss_index.search(query_vector, k)
# #     results = []
# #     for idx in I[0]:
# #         if 0 <= idx < len(metadata):
# #             chunk = metadata[idx].copy()
# #             chunk['retrieval_score'] = float(D[0, (I[0] == idx).nonzero()[0][0]])
# #             results.append(chunk)
# #     return results


# # # --- API Route with security ---
# # @app.post("/api/v1/hackrx/run")
# # async def hackrx_run_api(
# #     pdf: UploadFile = File(...),
# #     query: str = Form(...),
# #     token: str = Depends(verify_token)
# # ):
# #     # Cleanup previous
# #     cleanup_temp_files()

# #     # Save PDF file
# #     os.makedirs(PARSED_TEXT_OUTPUT_FOLDER, exist_ok=True)
# #     unique_name = str(uuid.uuid4()) + "_" + pdf.filename
# #     pdf_path = os.path.join(PARSED_TEXT_OUTPUT_FOLDER, unique_name)
# #     with open(pdf_path, "wb") as f:
# #         f.write(await pdf.read())

# #     # Parse and chunk PDF
# #     try:
# #         all_chunks = parse_and_chunk_pdf(pdf_path)
# #     except Exception as e:
# #         cleanup_temp_files()
# #         return {"error": f"Error parsing PDF: {e}"}

# #     if not all_chunks:
# #         cleanup_temp_files()
# #         return {"error": "Parsing failed or PDF contains no extractable text."}

# #     # Save chunks
# #     jsonl_chunks_path = save_chunks_to_jsonl(all_chunks, PARSED_TEXT_OUTPUT_FOLDER)

# #     # Vectorize chunks and save index
# #     try:
# #         faiss_path, meta_path = vectorize_chunks(jsonl_chunks_path, FAISS_INDEX_OUTPUT_FOLDER)
# #     except Exception as e:
# #         cleanup_temp_files()
# #         return {"error": f"Error during vectorization: {e}"}

# #     # Load index and metadata
# #     try:
# #         faiss_index = faiss.read_index(faiss_path)
# #         with open(meta_path, 'r', encoding='utf-8') as f:
# #             metadata = json.load(f)
# #         embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
# #     except Exception as e:
# #         cleanup_temp_files()
# #         return {"error": f"Error loading index/metadata: {e}"}

# #     # Enhance and extract entities from query
# #     try:
# #         processed = await enhance_and_extract_query_with_llm(query)
# #     except Exception:
# #         processed = {
# #             "corrected_and_rephrased_query": clean_query(query),
# #             "extracted_entities": {}
# #         }
# #     query_for_embedding = clean_query(processed["corrected_and_rephrased_query"])
# #     extracted_entities = processed["extracted_entities"]

# #     # Retrieve relevant doc chunks
# #     top_chunks = retrieve_top_k(query_for_embedding, TOP_K_RETRIEVAL, faiss_index, metadata, embedder)

# #     # Construct prompt for LLM
# #     rag_prompt = construct_rag_prompt(query, top_chunks, extracted_entities)

# #     # Call LLM API for answer
# #     try:
# #         answer = await call_llm_api([
# #             {"role": "system", "content": "You are a helpful insurance assistant."},
# #             {"role": "user", "content": rag_prompt}
# #         ], json_output=False)
# #         result = answer.strip()
# #     except Exception as e:
# #         cleanup_temp_files()
# #         return {"error": f"Error while getting answer: {e}"}

# #     # Cleanup temporary files after processing
# #     cleanup_temp_files()

# #     return {"answer": result}

# import os
# import shutil
# import json
# from typing import List, Dict, Any
# from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Security
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# import uuid
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import pypdf
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import nest_asyncio
# import asyncio
# from dotenv import load_dotenv
# from openai import OpenAI

# # --- Constants and Paths ---
# # Use a path relative to the current working directory, which is the standard for cloud deployments
# # This ensures the code works on Render, locally, or any other platform.
# BASE_PATH = os.getcwd()  
# PARSED_TEXT_OUTPUT_FOLDER = os.path.join(BASE_PATH, "Parsed_text")
# FAISS_INDEX_OUTPUT_FOLDER = os.path.join(BASE_PATH, "faiss_index")
# AGGREGATED_CHUNKS_FILENAME = 'all_policy_chunks.jsonl'
# FAISS_INDEX_FILENAME = 'policy_chunks_faiss_index.bin'
# FAISS_METADATA_FILENAME = 'policy_chunks_metadata.json'
# EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# TOP_K_RETRIEVAL = 5

# os.makedirs(PARSED_TEXT_OUTPUT_FOLDER, exist_ok=True)
# os.makedirs(FAISS_INDEX_OUTPUT_FOLDER, exist_ok=True)

# nest_asyncio.apply()

# # --- FastAPI App ---
# app = FastAPI(title="Bajaj PDF QnA Pipeline API")

# # --- Security Setup ---
# bearer_scheme = HTTPBearer(auto_error=False)

# # Load the token from an environment variable for security and portability.
# # It will be set in the Render Dashboard for deployment and can be in a .env file locally.
# VALID_TOKEN ="ssbadscstobc3609e845e387e9f7ac988ea36090473eefbe6dae9cfe880c35c6b67d87a7757"


# def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
#     if credentials is None or credentials.scheme.lower() != "bearer":
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Missing or invalid authorization scheme",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     token = credentials.credentials
#     if token != VALID_TOKEN:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Invalid token",
#         )
#     return token


# # --- Utility: Cleanup Temporary Files ---
# def cleanup_temp_files():
#     try:
#         if os.path.exists(PARSED_TEXT_OUTPUT_FOLDER):
#             shutil.rmtree(PARSED_TEXT_OUTPUT_FOLDER)
#         if os.path.exists(FAISS_INDEX_OUTPUT_FOLDER):
#             shutil.rmtree(FAISS_INDEX_OUTPUT_FOLDER)
#     except Exception as e:
#         print(f"Cleanup error: {e}")


# # --- PDF Parsing and Chunking ---
# def parse_and_chunk_pdf(file_path: str) -> List[Dict[str, Any]]:
#     all_page_texts = []
#     try:
#         with open(file_path, "rb") as f:
#             reader = pypdf.PdfReader(f)
#             for i, page in enumerate(reader.pages):
#                 page_text = page.extract_text()
#                 if page_text:
#                     all_page_texts.append({"text": page_text, "page_number": i + 1})
#     except Exception as e:
#         # Allow raising so calling function can handle
#         raise RuntimeError(f"PDF reading failed: {e}")

#     if not all_page_texts:
#         return []

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     processed_chunks = []
#     for page_info in all_page_texts:
#         chunks_from_page = text_splitter.split_text(page_info["text"])
#         for chunk_content in chunks_from_page:
#             clean_chunk = ' '.join(chunk_content.split()).strip()
#             if clean_chunk:
#                 processed_chunks.append({
#                     "content": clean_chunk,
#                     "metadata": {
#                         "page_number": page_info["page_number"],
#                         "source_file": os.path.basename(file_path),
#                     }
#                 })
#     return processed_chunks


# def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_dir: str):
#     os.makedirs(output_dir, exist_ok=True)
#     out_path = os.path.join(output_dir, AGGREGATED_CHUNKS_FILENAME)
#     with open(out_path, 'w', encoding='utf-8') as outfile:
#         for chunk in chunks:
#             outfile.write(json.dumps(chunk, ensure_ascii=False) + '\n')
#     return out_path


# # --- Vectorization Using FAISS ---
# def vectorize_chunks(jsonl_path: str, output_dir: str):
#     all_chunks = []
#     with open(jsonl_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             all_chunks.append(json.loads(line))
#     chunk_contents = [chunk['content'] for chunk in all_chunks]
#     chunk_metadatas = [
#         {
#             "source_file": chunk['metadata']['source_file'],
#             "page_number": chunk['metadata']['page_number'],
#             "content": chunk['content']
#         } for chunk in all_chunks
#     ]
#     embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
#     embeddings = embedding_model.encode(chunk_contents, show_progress_bar=False)
#     embeddings = np.array(embeddings).astype('float32')
#     embedding_dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(embedding_dim)
#     index.add(embeddings)

#     os.makedirs(output_dir, exist_ok=True)
#     faiss_path = os.path.join(output_dir, FAISS_INDEX_FILENAME)
#     meta_path = os.path.join(output_dir, FAISS_METADATA_FILENAME)
#     faiss.write_index(index, faiss_path)
#     with open(meta_path, 'w', encoding='utf-8') as f:
#         json.dump(chunk_metadatas, f, ensure_ascii=False, indent=2)
#     return faiss_path, meta_path


# # --- Query Helpers ---
# def clean_query(query: str) -> str:
#     import re
#     cleaned_text = ''.join(char for char in query if char.isalnum() or char.isspace()).strip()
#     return re.sub(r'\s+', ' ', cleaned_text)


# def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(vecs, axis=1, keepdims=True)
#     norms[norms == 0] = 1e-12
#     return vecs / norms


# async def call_llm_api(prompt_messages: List[Dict[str, str]], json_output: bool = False) -> str:
#     # Get the directory of the current script.
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     # The .env file should be in the same directory as the main script.
#     dotenv_path = os.path.join(current_dir, '.env')
    
#     # Load environment variables from a local .env file if it exists (for local development).
#     # This call will simply be ignored on Render since the file won't exist.
#     if os.path.exists(dotenv_path):
#         load_dotenv(dotenv_path=dotenv_path)

#     TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
#     if not TOGETHER_API_KEY:
#         # This check is crucial for both local and deployed environments.
#         raise RuntimeError("TOGETHER_API_KEY not set in environment variables.")

#     client = OpenAI(
#         api_key=TOGETHER_API_KEY,
#         base_url="https://api.together.xyz/v1/")
#     payload = {
#         "model": "mistralai/Mistral-7B-Instruct-v0.2",
#         "messages": prompt_messages,
#         "temperature": 0.2,
#         "max_tokens": 1024,
#     }
#     if json_output:
#         payload["response_format"] = {"type": "json_object"}
#     response = client.chat.completions.create(**payload)
#     return response.choices[0].message.content


# async def enhance_and_extract_query_with_llm(raw_query: str) -> Dict[str, Any]:
#     prompt = f"""You are a smart assistant that improves insurance-related queries and extracts important entities.
# Given:
# "{raw_query}"
# Respond with a JSON like:
# {{
#     "corrected_and_rephrased_query": "...",
#     "extracted_entities": {{
#         "age": "...",
#         "location": "...",
#         ...
#     }}
# }}
# Only include non-null entities. Omit any missing or irrelevant ones.
# """
#     try:
#         messages = [{"role": "user", "content": prompt}]
#         llm_output_str = await call_llm_api(messages, json_output=True)
#         parsed = json.loads(llm_output_str)
#         parsed["extracted_entities"] = {
#             k: v for k, v in parsed.get("extracted_entities", {}).items() if v is not None
#         }
#         return parsed
#     except Exception:
#         return {
#             "corrected_and_rephrased_query": clean_query(raw_query),
#             "extracted_entities": {}
#         }


# def construct_rag_prompt(user_query: str, context_chunks: List[Dict[str, Any]], extracted_entities: Dict[str, Any]) -> str:
#     if not context_chunks:
#         return f"""You are a helpful assistant. No document context was found. Try to answer this:
# USER QUERY:
# {user_query}"""
#     context_text = "\n\n".join([
#         f"--- Document: {chunk.get('source_file', 'Unknown')}, Page: {chunk.get('page_number', 'N/A')} ---\n{chunk.get('content') or chunk.get('text', '')}"
#         for chunk in context_chunks
#     ])
#     entity_lines = "\n".join([f"- {k}: {v}" for k, v in extracted_entities.items()]) if extracted_entities else "None"
#     return f"""You are an expert insurance assistant.
# USER QUERY:
# {user_query}
# EXTRACTED ENTITIES:
# {entity_lines}
# DOCUMENT CONTEXT:
# {context_text}
# Answer clearly using the document context above. If no answer is found, say: "I could not find the answer in the provided documents."
# """


# def retrieve_top_k(query_for_embedding: str, k: int, faiss_index, metadata, embedder) -> List[Dict[str, Any]]:
#     if faiss_index.ntotal == 0:
#         return []
#     query_vector = embedder.encode([query_for_embedding]).astype("float32")
#     query_vector = normalize_vectors(query_vector)
#     D, I = faiss_index.search(query_vector, k)
#     results = []
#     for idx in I[0]:
#         if 0 <= idx < len(metadata):
#             chunk = metadata[idx].copy()
#             chunk['retrieval_score'] = float(D[0, (I[0] == idx).nonzero()[0][0]])
#             results.append(chunk)
#     return results


# # --- API Route with security ---
# @app.post("/api/v1/hackrx/run")
# async def hackrx_run_api(
#     pdf: UploadFile = File(...),
#     query: str = Form(...),
#     token: str = Depends(verify_token)
# ):
#     # Cleanup previous
#     cleanup_temp_files()

#     # Save PDF file
#     os.makedirs(PARSED_TEXT_OUTPUT_FOLDER, exist_ok=True)
#     unique_name = str(uuid.uuid4()) + "_" + pdf.filename
#     pdf_path = os.path.join(PARSED_TEXT_OUTPUT_FOLDER, unique_name)
#     with open(pdf_path, "wb") as f:
#         f.write(await pdf.read())

#     # Parse and chunk PDF
#     try:
#         all_chunks = parse_and_chunk_pdf(pdf_path)
#     except Exception as e:
#         cleanup_temp_files()
#         return {"error": f"Error parsing PDF: {e}"}

#     if not all_chunks:
#         cleanup_temp_files()
#         return {"error": "Parsing failed or PDF contains no extractable text."}

#     # Save chunks
#     jsonl_chunks_path = save_chunks_to_jsonl(all_chunks, PARSED_TEXT_OUTPUT_FOLDER)

#     # Vectorize chunks and save index
#     try:
#         faiss_path, meta_path = vectorize_chunks(jsonl_chunks_path, FAISS_INDEX_OUTPUT_FOLDER)
#     except Exception as e:
#         cleanup_temp_files()
#         return {"error": f"Error during vectorization: {e}"}

#     # Load index and metadata
#     try:
#         faiss_index = faiss.read_index(faiss_path)
#         with open(meta_path, 'r', encoding='utf-8') as f:
#             metadata = json.load(f)
#         embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
#     except Exception as e:
#         cleanup_temp_files()
#         return {"error": f"Error loading index/metadata: {e}"}

#     # Enhance and extract entities from query
#     try:
#         processed = await enhance_and_extract_query_with_llm(query)
#     except Exception:
#         processed = {
#             "corrected_and_rephrased_query": clean_query(query),
#             "extracted_entities": {}
#         }
#     query_for_embedding = clean_query(processed["corrected_and_rephrased_query"])
#     extracted_entities = processed["extracted_entities"]

#     # Retrieve relevant doc chunks
#     top_chunks = retrieve_top_k(query_for_embedding, TOP_K_RETRIEVAL, faiss_index, metadata, embedder)

#     # Construct prompt for LLM
#     rag_prompt = construct_rag_prompt(query, top_chunks, extracted_entities)

#     # Call LLM API for answer
#     try:
#         answer = await call_llm_api([
#             {"role": "system", "content": "You are a helpful insurance assistant."},
#             {"role": "user", "content": rag_prompt}
#         ], json_output=False)
#         result = answer.strip()
#     except Exception as e:
#         cleanup_temp_files()
#         return {"error": f"Error while getting answer: {e}"}

#     # Cleanup temporary files after processing
#     cleanup_temp_files()

#     return {"answer": result}

# import os
# import shutil
# import json
# from typing import List, Dict, Any
# from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Security
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# import uuid
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import pypdf
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import nest_asyncio
# from dotenv import load_dotenv
# from openai import OpenAI

# # --- Constants and Paths ---
# BASE_PATH = os.getcwd()  # Use current working directory for portability
# PARSED_TEXT_OUTPUT_FOLDER = os.path.join(BASE_PATH, "Parsed_text")
# FAISS_INDEX_OUTPUT_FOLDER = os.path.join(BASE_PATH, "faiss_index")
# AGGREGATED_CHUNKS_FILENAME = 'all_policy_chunks.jsonl'
# FAISS_INDEX_FILENAME = 'policy_chunks_faiss_index.bin'
# FAISS_METADATA_FILENAME = 'policy_chunks_metadata.json'
# EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# TOP_K_RETRIEVAL = 5

# os.makedirs(PARSED_TEXT_OUTPUT_FOLDER, exist_ok=True)
# os.makedirs(FAISS_INDEX_OUTPUT_FOLDER, exist_ok=True)

# nest_asyncio.apply()

# # --- FastAPI App ---
# app = FastAPI(title="Bajaj PDF QnA Pipeline API")

# # --- Security Setup ---
# bearer_scheme = HTTPBearer(auto_error=False)
# VALID_TOKEN = "ssbadscstobc3609e845e387e9f7ac988ea36090473eefbe6dae9cfe880c35c6b67d87a7757"

# # Global SentenceTransformer model
# embedder = None

# def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
#     if credentials is None or credentials.scheme.lower() != "bearer":
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Missing or invalid authorization scheme",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     token = credentials.credentials
#     if token != VALID_TOKEN:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Invalid token",
#         )
#     return token

# # --- Utility: Cleanup Temporary Files ---
# def cleanup_temp_files():
#     try:
#         if os.path.exists(PARSED_TEXT_OUTPUT_FOLDER):
#             shutil.rmtree(PARSED_TEXT_OUTPUT_FOLDER)
#         if os.path.exists(FAISS_INDEX_OUTPUT_FOLDER):
#             shutil.rmtree(FAISS_INDEX_OUTPUT_FOLDER)
#     except Exception as e:
#         print(f"Cleanup error: {e}")

# # --- PDF Parsing and Chunking ---
# def parse_and_chunk_pdf(file_path: str) -> List[Dict[str, Any]]:
#     all_page_texts = []
#     try:
#         with open(file_path, "rb") as f:
#             reader = pypdf.PdfReader(f)
#             for i, page in enumerate(reader.pages):
#                 page_text = page.extract_text()
#                 if page_text:
#                     all_page_texts.append({"text": page_text, "page_number": i + 1})
#     except Exception as e:
#         raise RuntimeError(f"PDF reading failed: {e}")

#     if not all_page_texts:
#         return []

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     processed_chunks = []
#     for page_info in all_page_texts:
#         chunks_from_page = text_splitter.split_text(page_info["text"])
#         for chunk_content in chunks_from_page:
#             clean_chunk = ' '.join(chunk_content.split()).strip()
#             if clean_chunk:
#                 processed_chunks.append({
#                     "content": clean_chunk,
#                     "metadata": {
#                         "page_number": page_info["page_number"],
#                         "source_file": os.path.basename(file_path),
#                     }
#                 })
#     return processed_chunks

# def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_dir: str):
#     os.makedirs(output_dir, exist_ok=True)
#     out_path = os.path.join(output_dir, AGGREGATED_CHUNKS_FILENAME)
#     with open(out_path, 'w', encoding='utf-8') as outfile:
#         for chunk in chunks:
#             outfile.write(json.dumps(chunk, ensure_ascii=False) + '\n')
#     return out_path

# # --- Vectorization Using FAISS ---
# def vectorize_chunks(jsonl_path: str, output_dir: str):
#     global embedder
#     all_chunks = []
#     with open(jsonl_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             all_chunks.append(json.loads(line))
#     chunk_contents = [chunk['content'] for chunk in all_chunks]
#     chunk_metadatas = [
#         {
#             "source_file": chunk['metadata']['source_file'],
#             "page_number": chunk['metadata']['page_number'],
#             "content": chunk['content']
#         } for chunk in all_chunks
#     ]
#     embeddings = embedder.encode(chunk_contents, show_progress_bar=False)
#     embeddings = np.array(embeddings).astype('float32')
#     embedding_dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(embedding_dim)
#     index.add(embeddings)
#     os.makedirs(output_dir, exist_ok=True)
#     faiss_path = os.path.join(output_dir, FAISS_INDEX_FILENAME)
#     meta_path = os.path.join(output_dir, FAISS_METADATA_FILENAME)
#     faiss.write_index(index, faiss_path)
#     with open(meta_path, 'w', encoding='utf-8') as f:
#         json.dump(chunk_metadatas, f, ensure_ascii=False, indent=2)
#     return faiss_path, meta_path

# # --- Query Helpers ---
# def clean_query(query: str) -> str:
#     import re
#     cleaned_text = ''.join(char for char in query if char.isalnum() or char.isspace()).strip()
#     return re.sub(r'\s+', ' ', cleaned_text)

# def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(vecs, axis=1, keepdims=True)
#     norms[norms == 0] = 1e-12
#     return vecs / norms

# async def call_llm_api(prompt_messages: List[Dict[str, str]], json_output: bool = False) -> str:
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     dotenv_path = os.path.join(current_dir, '.env')
#     if os.path.exists(dotenv_path):
#         load_dotenv(dotenv_path=dotenv_path)

#     TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
#     if not TOGETHER_API_KEY:
#         raise RuntimeError("TOGETHER_API_KEY not set in environment variables.")

#     client = OpenAI(
#         api_key=TOGETHER_API_KEY,
#         base_url="https://api.together.xyz/v1/")
#     payload = {
#         "model": "mistralai/Mistral-7B-Instruct-v0.2",
#         "messages": prompt_messages,
#         "temperature": 0.2,
#         "max_tokens": 1024,
#     }
#     if json_output:
#         payload["response_format"] = {"type": "json_object"}
#     response = client.chat.completions.create(**payload)
#     return response.choices[0].message.content

# async def enhance_and_extract_query_with_llm(raw_query: str) -> Dict[str, Any]:
#     prompt = f"""You are a smart assistant that improves insurance-related queries and extracts important entities.
# Given:
# "{raw_query}"
# Respond with a JSON like:
# {{
#     "corrected_and_rephrased_query": "...",
#     "extracted_entities": {{
#         "age": "...",
#         "location": "...",
#         ...
#     }}
# }}
# Only include non-null entities. Omit any missing or irrelevant ones.
# """
#     try:
#         messages = [{"role": "user", "content": prompt}]
#         llm_output_str = await call_llm_api(messages, json_output=True)
#         parsed = json.loads(llm_output_str)
#         parsed["extracted_entities"] = {
#             k: v for k, v in parsed.get("extracted_entities", {}).items() if v is not None
#         }
#         return parsed
#     except Exception:
#         return {
#             "corrected_and_rephrased_query": clean_query(raw_query),
#             "extracted_entities": {}
#         }

# def construct_rag_prompt(user_query: str, context_chunks: List[Dict[str, Any]], extracted_entities: Dict[str, Any]) -> str:
#     if not context_chunks:
#         return f"""You are a helpful assistant. No document context was found. Try to answer this:
# USER QUERY:
# {user_query}"""
#     context_text = "\n\n".join([
#         f"--- Document: {chunk.get('source_file', 'Unknown')}, Page: {chunk.get('page_number', 'N/A')} ---\n{chunk.get('content') or chunk.get('text', '')}"
#         for chunk in context_chunks
#     ])
#     entity_lines = "\n".join([f"- {k}: {v}" for k, v in extracted_entities.items()]) if extracted_entities else "None"
#     return f"""You are an expert insurance assistant.
# USER QUERY:
# {user_query}
# EXTRACTED ENTITIES:
# {entity_lines}
# DOCUMENT CONTEXT:
# {context_text}
# Answer clearly using the document context above. If no answer is found, say: "I could not find the answer in the provided documents."
# """

# def retrieve_top_k(query_for_embedding: str, k: int, faiss_index, metadata, _) -> List[Dict[str, Any]]:
#     global embedder
#     if faiss_index.ntotal == 0:
#         return []
#     query_vector = embedder.encode([query_for_embedding]).astype("float32")
#     query_vector = normalize_vectors(query_vector)
#     D, I = faiss_index.search(query_vector, k)
#     results = []
#     for idx in I[0]:
#         if 0 <= idx < len(metadata):
#             chunk = metadata[idx].copy()
#             chunk['retrieval_score'] = float(D[0, (I[0] == idx).nonzero()[0][0]])
#             results.append(chunk)
#     return results

# # --- Health Check ---
# @app.get("/health")
# async def health_check():
#     return {"status": "ok"}

# # --- Startup Event ---
# @app.on_event("startup")
# async def startup_event():
#     global embedder
#     embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)  # Preload model

# # --- API Route with Security ---
# @app.post("/api/v1/hackrx/run")
# async def hackrx_run_api(
#     pdf: UploadFile = File(...),
#     query: str = Form(...),
#     token: str = Depends(verify_token)
# ):
#     cleanup_temp_files()
#     os.makedirs(PARSED_TEXT_OUTPUT_FOLDER, exist_ok=True)
#     unique_name = str(uuid.uuid4()) + "_" + pdf.filename
#     pdf_path = os.path.join(PARSED_TEXT_OUTPUT_FOLDER, unique_name)
#     with open(pdf_path, "wb") as f:
#         f.write(await pdf.read())

#     try:
#         all_chunks = parse_and_chunk_pdf(pdf_path)
#     except Exception as e:
#         cleanup_temp_files()
#         return {"error": f"Error parsing PDF: {e}"}

#     if not all_chunks:
#         cleanup_temp_files()
#         return {"error": "Parsing failed or PDF contains no extractable text."}

#     jsonl_chunks_path = save_chunks_to_jsonl(all_chunks, PARSED_TEXT_OUTPUT_FOLDER)

#     try:
#         faiss_path, meta_path = vectorize_chunks(jsonl_chunks_path, FAISS_INDEX_OUTPUT_FOLDER)
#     except Exception as e:
#         cleanup_temp_files()
#         return {"error": f"Error during vectorization: {e}"}

#     try:
#         faiss_index = faiss.read_index(faiss_path)
#         with open(meta_path, 'r', encoding='utf-8') as f:
#             metadata = json.load(f)
#     except Exception as e:
#         cleanup_temp_files()
#         return {"error": f"Error loading index/metadata: {e}"}

#     try:
#         processed = await enhance_and_extract_query_with_llm(query)
#     except Exception:
#         processed = {
#             "corrected_and_rephrased_query": clean_query(query),
#             "extracted_entities": {}
#         }
#     query_for_embedding = clean_query(processed["corrected_and_rephrased_query"])
#     extracted_entities = processed["extracted_entities"]

#     top_chunks = retrieve_top_k(query_for_embedding, TOP_K_RETRIEVAL, faiss_index, metadata, None)

#     rag_prompt = construct_rag_prompt(query, top_chunks, extracted_entities)

#     try:
#         answer = await call_llm_api([
#             {"role": "system", "content": "You are a helpful insurance assistant."},
#             {"role": "user", "content": rag_prompt}
#         ], json_output=False)
#         result = answer.strip()
#     except Exception as e:
#         cleanup_temp_files()
#         return {"error": f"Error while getting answer: {e}"}

#     cleanup_temp_files()
#     return {"answer": result}

# # --- Run Uvicorn ---
# if __name__ == "__main__":
#     port = int(os.getenv("PORT", 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=os.getenv("RENDER") != "true")


# import os
# import shutil
# import json
# from typing import List, Dict, Any
# from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Security
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# import uuid
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import pypdf
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import nest_asyncio
# from dotenv import load_dotenv
# from openai import OpenAI
# import logging
# import uvicorn

# # --- Logging Setup ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # --- Constants and Paths ---
# BASE_PATH = os.getcwd()
# PARSED_TEXT_OUTPUT_FOLDER = os.path.join(BASE_PATH, "Parsed_text")
# FAISS_INDEX_OUTPUT_FOLDER = os.path.join(BASE_PATH, "faiss_index")
# AGGREGATED_CHUNKS_FILENAME = 'all_policy_chunks.jsonl'
# FAISS_INDEX_FILENAME = 'policy_chunks_faiss_index.bin'
# FAISS_METADATA_FILENAME = 'policy_chunks_metadata.json'
# EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # Lightweight model
# TOP_K_RETRIEVAL = 5

# os.makedirs(PARSED_TEXT_OUTPUT_FOLDER, exist_ok=True)
# os.makedirs(FAISS_INDEX_OUTPUT_FOLDER, exist_ok=True)

# nest_asyncio.apply()

# # --- FastAPI App ---
# app = FastAPI(title="Bajaj PDF QnA Pipeline API")

# # --- Security Setup ---
# bearer_scheme = HTTPBearer(auto_error=False)
# VALID_TOKEN = "ssbadscstobc3609e845e387e9f7ac988ea36090473eefbe6dae9cfe880c35c6b67d87a7757"

# # Global SentenceTransformer model
# embedder = None

# def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
#     if credentials is None or credentials.scheme.lower() != "bearer":
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Missing or invalid authorization scheme",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     token = credentials.credentials
#     if token != VALID_TOKEN:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Invalid token",
#         )
#     return token

# # --- Utility: Cleanup Temporary Files ---
# def cleanup_temp_files():
#     try:
#         if os.path.exists(PARSED_TEXT_OUTPUT_FOLDER):
#             shutil.rmtree(PARSED_TEXT_OUTPUT_FOLDER)
#         if os.path.exists(FAISS_INDEX_OUTPUT_FOLDER):
#             shutil.rmtree(FAISS_INDEX_OUTPUT_FOLDER)
#     except Exception as e:
#         logger.error(f"Cleanup error: {e}")

# # --- PDF Parsing and Chunking ---
# def parse_and_chunk_pdf(file_path: str) -> List[Dict[str, Any]]:
#     all_page_texts = []
#     try:
#         with open(file_path, "rb") as f:
#             reader = pypdf.PdfReader(f)
#             for i, page in enumerate(reader.pages):
#                 page_text = page.extract_text()
#                 if page_text:
#                     all_page_texts.append({"text": page_text, "page_number": i + 1})
#     except Exception as e:
#         logger.error(f"PDF reading failed: {e}")
#         raise RuntimeError(f"PDF reading failed: {e}")

#     if not all_page_texts:
#         return []

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     processed_chunks = []
#     for page_info in all_page_texts:
#         chunks_from_page = text_splitter.split_text(page_info["text"])
#         for chunk_content in chunks_from_page:
#             clean_chunk = ' '.join(chunk_content.split()).strip()
#             if clean_chunk:
#                 processed_chunks.append({
#                     "content": clean_chunk,
#                     "metadata": {
#                         "page_number": page_info["page_number"],
#                         "source_file": os.path.basename(file_path),
#                     }
#                 })
#     return processed_chunks

# def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_dir: str):
#     os.makedirs(output_dir, exist_ok=True)
#     out_path = os.path.join(output_dir, AGGREGATED_CHUNKS_FILENAME)
#     with open(out_path, 'w', encoding='utf-8') as outfile:
#         for chunk in chunks:
#             outfile.write(json.dumps(chunk, ensure_ascii=False) + '\n')
#     return out_path

# # --- Vectorization Using FAISS ---
# def vectorize_chunks(jsonl_path: str, output_dir: str):
#     global embedder
#     all_chunks = []
#     with open(jsonl_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             all_chunks.append(json.loads(line))
#     chunk_contents = [chunk['content'] for chunk in all_chunks]
#     chunk_metadatas = [
#         {
#             "source_file": chunk['metadata']['source_file'],
#             "page_number": chunk['metadata']['page_number'],
#             "content": chunk['content']
#         } for chunk in all_chunks
#     ]
#     logger.info("Encoding chunks with SentenceTransformer")
#     embeddings = embedder.encode(chunk_contents, show_progress_bar=False, batch_size=16, device='cpu')
#     embeddings = np.array(embeddings).astype('float32')
#     embedding_dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(embedding_dim)
#     index.add(embeddings)
#     os.makedirs(output_dir, exist_ok=True)
#     faiss_path = os.path.join(output_dir, FAISS_INDEX_FILENAME)
#     meta_path = os.path.join(output_dir, FAISS_METADATA_FILENAME)
#     faiss.write_index(index, faiss_path)
#     with open(meta_path, 'w', encoding='utf-8') as f:
#         json.dump(chunk_metadatas, f, ensure_ascii=False, indent=2)
#     return faiss_path, meta_path

# # --- Query Helpers ---
# def clean_query(query: str) -> str:
#     import re
#     cleaned_text = ''.join(char for char in query if char.isalnum() or char.isspace()).strip()
#     return re.sub(r'\s+', ' ', cleaned_text)

# def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(vecs, axis=1, keepdims=True)
#     norms[norms == 0] = 1e-12
#     return vecs / norms

# async def call_llm_api(prompt_messages: List[Dict[str, str]], json_output: bool = False) -> str:
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     dotenv_path = os.path.join(current_dir, '.env')
#     if os.path.exists(dotenv_path):
#         load_dotenv(dotenv_path=dotenv_path)

#     TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
#     if not TOGETHER_API_KEY:
#         logger.error("TOGETHER_API_KEY not set")
#         raise RuntimeError("TOGETHER_API_KEY not set in environment variables.")

#     client = OpenAI(
#         api_key=TOGETHER_API_KEY,
#         base_url="https://api.together.xyz/v1/")
#     payload = {
#         "model": "mistralai/Mistral-7B-Instruct-v0.2",
#         "messages": prompt_messages,
#         "temperature": 0.2,
#         "max_tokens": 1024,
#     }
#     if json_output:
#         payload["response_format"] = {"type": "json_object"}
#     try:
#         response = client.chat.completions.create(**payload)
#         return response.choices[0].message.content
#     except Exception as e:
#         logger.error(f"LLM API call failed: {e}")
#         raise

# async def enhance_and_extract_query_with_llm(raw_query: str) -> Dict[str, Any]:
#     prompt = f"""You are a smart assistant that improves insurance-related queries and extracts important entities.
# Given:
# "{raw_query}"
# Respond with a JSON like:
# {{
#     "corrected_and_rephrased_query": "...",
#     "extracted_entities": {{
#         "age": "...",
#         "location": "...",
#         ...
#     }}
# }}
# Only include non-null entities. Omit any missing or irrelevant ones.
# """
#     try:
#         messages = [{"role": "user", "content": prompt}]
#         llm_output_str = await call_llm_api(messages, json_output=True)
#         parsed = json.loads(llm_output_str)
#         parsed["extracted_entities"] = {
#             k: v for k, v in parsed.get("extracted_entities", {}).items() if v is not None
#         }
#         return parsed
#     except Exception as e:
#         logger.warning(f"LLM query enhancement failed: {e}")
#         return {
#             "corrected_and_rephrased_query": clean_query(raw_query),
#             "extracted_entities": {}
#         }

# def construct_rag_prompt(user_query: str, context_chunks: List[Dict[str, Any]], extracted_entities: Dict[str, Any]) -> str:
#     if not context_chunks:
#         return f"""You are a helpful assistant. No document context was found. Try to answer this:
# USER QUERY:
# {user_query}"""
#     context_text = "\n\n".join([
#         f"--- Document: {chunk.get('source_file', 'Unknown')}, Page: {chunk.get('page_number', 'N/A')} ---\n{chunk.get('content') or chunk.get('text', '')}"
#         for chunk in context_chunks
#     ])
#     entity_lines = "\n".join([f"- {k}: {v}" for k, v in extracted_entities.items()]) if extracted_entities else "None"
#     return f"""You are an expert insurance assistant.
# USER QUERY:
# {user_query}
# EXTRACTED ENTITIES:
# {entity_lines}
# DOCUMENT CONTEXT:
# {context_text}
# Answer clearly using the document context above. If no answer is found, say: "I could not find the answer in the provided documents."
# """

# def retrieve_top_k(query_for_embedding: str, k: int, faiss_index, metadata, _) -> List[Dict[str, Any]]:
#     global embedder
#     if faiss_index.ntotal == 0:
#         return []
#     query_vector = embedder.encode([query_for_embedding], batch_size=1, device='cpu').astype("float32")
#     query_vector = normalize_vectors(query_vector)
#     D, I = faiss_index.search(query_vector, k)
#     results = []
#     for idx in I[0]:
#         if 0 <= idx < len(metadata):
#             chunk = metadata[idx].copy()
#             chunk['retrieval_score'] = float(D[0, (I[0] == idx).nonzero()[0][0]])
#             results.append(chunk)
#     return results

# # --- Health Check ---
# @app.get("/health")
# async def health_check():
#     logger.info("Health check endpoint called")
#     return {"status": "ok"}

# # --- Startup Event ---
# @app.on_event("startup")
# async def startup_event():
#     global embedder
#     logger.info("Starting SentenceTransformer model loading")
#     try:
#         embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
#         logger.info("SentenceTransformer model loaded successfully")
#     except Exception as e:
#         logger.error(f"Failed to load SentenceTransformer: {e}")
#         raise

# # --- API Route with Security ---
# @app.post("/api/v1/hackrx/run")
# async def hackrx_run_api(
#     pdf: UploadFile = File(...),
#     query: str = Form(...),
#     token: str = Depends(verify_token)
# ):
#     logger.info("Processing /api/v1/hackrx/run request")
#     cleanup_temp_files()
#     os.makedirs(PARSED_TEXT_OUTPUT_FOLDER, exist_ok=True)
#     unique_name = str(uuid.uuid4()) + "_" + pdf.filename
#     pdf_path = os.path.join(PARSED_TEXT_OUTPUT_FOLDER, unique_name)
#     with open(pdf_path, "wb") as f:
#         f.write(await pdf.read())

#     try:
#         all_chunks = parse_and_chunk_pdf(pdf_path)
#     except Exception as e:
#         cleanup_temp_files()
#         logger.error(f"Error parsing PDF: {e}")
#         return {"error": f"Error parsing PDF: {e}"}

#     if not all_chunks:
#         cleanup_temp_files()
#         logger.warning("No extractable text in PDF")
#         return {"error": "Parsing failed or PDF contains no extractable text."}

#     jsonl_chunks_path = save_chunks_to_jsonl(all_chunks, PARSED_TEXT_OUTPUT_FOLDER)

#     try:
#         faiss_path, meta_path = vectorize_chunks(jsonl_chunks_path, FAISS_INDEX_OUTPUT_FOLDER)
#     except Exception as e:
#         cleanup_temp_files()
#         logger.error(f"Error during vectorization: {e}")
#         return {"error": f"Error during vectorization: {e}"}

#     try:
#         faiss_index = faiss.read_index(faiss_path)
#         with open(meta_path, 'r', encoding='utf-8') as f:
#             metadata = json.load(f)
#     except Exception as e:
#         cleanup_temp_files()
#         logger.error(f"Error loading index/metadata: {e}")
#         return {"error": f"Error loading index/metadata: {e}"}

#     try:
#         processed = await enhance_and_extract_query_with_llm(query)
#     except Exception as e:
#         logger.warning(f"Query enhancement failed: {e}")
#         processed = {
#             "corrected_and_rephrased_query": clean_query(query),
#             "extracted_entities": {}
#         }
#     query_for_embedding = clean_query(processed["corrected_and_rephrased_query"])
#     extracted_entities = processed["extracted_entities"]

#     top_chunks = retrieve_top_k(query_for_embedding, TOP_K_RETRIEVAL, faiss_index, metadata, None)

#     rag_prompt = construct_rag_prompt(query, top_chunks, extracted_entities)

#     try:
#         answer = await call_llm_api([
#             {"role": "system", "content": "You are a helpful insurance assistant."},
#             {"role": "user", "content": rag_prompt}
#         ], json_output=False)
#         result = answer.strip()
#     except Exception as e:
#         cleanup_temp_files()
#         logger.error(f"Error while getting answer: {e}")
#         return {"error": f"Error while getting answer: {e}"}

#     cleanup_temp_files()
#     logger.info("Request processed successfully")
#     return {"answer": result}

# # --- Run Uvicorn ---
# if __name__ == "__main__":
#     port = int(os.getenv("PORT", 8000))
#     logger.info(f"Starting Uvicorn on host=0.0.0.0, port={port}")
#     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=os.getenv("RAILWAY_ENVIRONMENT") != "production")






import os
import tempfile
import shutil
import json
import uuid
import faiss
import numpy as np
import re
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

# Load .env explicitly from the same folder as main.py, no matter where uvicorn is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Use OS temp directory to reduce permanent disk usage (works well on cloud hosts)
BASE_PATH = tempfile.gettempdir()
PARSED_TEXT_OUTPUT_FOLDER = os.path.join(BASE_PATH, "Parsed_text")
FAISS_INDEX_OUTPUT_FOLDER = os.path.join(BASE_PATH, "faiss_index")

AGGREGATED_CHUNKS_FILENAME = 'all_policy_chunks.jsonl'
FAISS_INDEX_FILENAME = 'policy_chunks_faiss_index.bin'
FAISS_METADATA_FILENAME = 'policy_chunks_metadata.json'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K_RETRIEVAL = 5

# Ensure folders exist
os.makedirs(PARSED_TEXT_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_OUTPUT_FOLDER, exist_ok=True)

nest_asyncio.apply()

app = FastAPI(title="Bajaj PDF QnA Pipeline API")

bearer_scheme = HTTPBearer(auto_error=False)
VALID_TOKEN = os.getenv(
    "VALID_TOKEN", "ssbakscstobcb3609e845e387e9f7ac988ea36090473eefbe6dae9cfe880c35c6b67d87a7757"
)

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
    try:
        if os.path.exists(PARSED_TEXT_OUTPUT_FOLDER):
            shutil.rmtree(PARSED_TEXT_OUTPUT_FOLDER)
        if os.path.exists(FAISS_INDEX_OUTPUT_FOLDER):
            shutil.rmtree(FAISS_INDEX_OUTPUT_FOLDER)
    except Exception as e:
        print(f"Cleanup error: {e}")

def parse_and_chunk_pdf(file_path: str) -> List[Dict[str, Any]]:
    all_page_texts = []
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                all_page_texts.append({"text": page_text, "page_number": i + 1})

    if not all_page_texts:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len, is_separator_regex=False
    )

    processed_chunks = []
    for page_info in all_page_texts:
        chunks_from_page = text_splitter.split_text(page_info["text"])
        for chunk_content in chunks_from_page:
            clean_chunk = " ".join(chunk_content.split()).strip()
            if clean_chunk:
                processed_chunks.append(
                    {
                        "content": clean_chunk,
                        "metadata": {
                            "page_number": page_info["page_number"],
                            "source_file": os.path.basename(file_path),
                        },
                    }
                )
    return processed_chunks

# Load SentenceTransformer model once globally
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/hf_cache")



def clean_query(query: str) -> str:
    cleaned_text = "".join(char for char in query if char.isalnum() or char.isspace()).strip()
    return re.sub(r"\s+", " ", cleaned_text)

def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vecs / norms

async def call_llm_api(prompt_messages: List[Dict[str, str]], json_output: bool = False) -> str:
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
    response = client.chat.completions.create(**payload)
    return response.choices[0].message.content

async def enhance_and_extract_query_with_llm(raw_query: str) -> Dict[str, Any]:
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
    except Exception:
        return {"corrected_and_rephrased_query": clean_query(raw_query), "extracted_entities": {}}

def construct_rag_prompt(
    user_query: str, context_chunks: List[Dict[str, Any]], extracted_entities: Dict[str, Any]
) -> str:
    if not context_chunks:
        return f"""You are a helpful assistant. No document context was found. Try to answer this:
USER QUERY:
{user_query}"""
    context_text = "\n\n".join(
        [
            f"--- Document: {chunk.get('source_file', 'Unknown')}, Page: {chunk.get('page_number', 'N/A')} ---\n"
            f"{chunk.get('content', '') or chunk.get('text', '')}"
            for chunk in context_chunks
        ]
    )
    entity_lines = (
        "\n".join([f"- {k}: {v}" for k, v in extracted_entities.items()])
        if extracted_entities
        else "None"
    )
    return f"""You are an expert insurance assistant.
USER QUERY:
{user_query}
EXTRACTED ENTITIES:
{entity_lines}
DOCUMENT CONTEXT:
{context_text}
Answer clearly using the document context above. If no answer is found, say: "I could not find the answer in the provided documents."
"""

def retrieve_top_k(
    query_for_embedding: str,
    k: int,
    faiss_index,
    metadata: List[Dict[str, Any]],
    embedder,
) -> List[Dict[str, Any]]:
    if faiss_index.ntotal == 0:
        return []
    query_vector = embedder.encode([query_for_embedding]).astype("float32")
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

@app.post("/api/v1/hackrx/run")
async def hackrx_run_api(
    pdf: UploadFile = File(...),
    query: str = Form(...),
    token: str = Depends(verify_token),
):
    cleanup_temp_files()

    os.makedirs(PARSED_TEXT_OUTPUT_FOLDER, exist_ok=True)
    unique_name = f"{uuid.uuid4()}_{pdf.filename}"
    pdf_path = os.path.join(PARSED_TEXT_OUTPUT_FOLDER, unique_name)
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    try:
        all_chunks = parse_and_chunk_pdf(pdf_path)
    except Exception as e:
        cleanup_temp_files()
        return {"error": f"Error parsing PDF: {e}"}

    if not all_chunks:
        cleanup_temp_files()
        return {"error": "Parsing failed or PDF contains no extractable text."}

    # Vectorize in-memory (no saving intermediate JSONL to disk)
    chunk_contents = [chunk["content"] for chunk in all_chunks]
    chunk_metadatas = [chunk["metadata"] for chunk in all_chunks]

    try:
        embeddings = embedding_model.encode(chunk_contents, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        embedding_dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index.add(embeddings)
    except Exception as e:
        cleanup_temp_files()
        return {"error": f"Error during embedding or FAISS index creation: {e}"}

    try:
        processed = await enhance_and_extract_query_with_llm(query)
    except Exception:
        processed = {
            "corrected_and_rephrased_query": clean_query(query),
            "extracted_entities": {},
        }

    query_for_embedding = clean_query(processed["corrected_and_rephrased_query"])
    extracted_entities = processed["extracted_entities"]

    top_chunks = retrieve_top_k(
        query_for_embedding, TOP_K_RETRIEVAL, faiss_index, chunk_metadatas, embedding_model
    )
    rag_prompt = construct_rag_prompt(query, top_chunks, extracted_entities)

    try:
        answer = await call_llm_api(
            [
                {"role": "system", "content": "You are a helpful insurance assistant."},
                {"role": "user", "content": rag_prompt},
            ],
            json_output=False,
        )
        result = answer.strip()
    except Exception as e:
        cleanup_temp_files()
        return {"error": f"Error while getting answer: {e}"}

    cleanup_temp_files()
    return {"answer": result}
