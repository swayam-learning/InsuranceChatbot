# 🛡️ Insurance Document QnA API

This FastAPI application allows users to upload insurance-related PDF documents and ask natural language questions about their contents. It extracts information from PDFs, semantically enhances queries, retrieves relevant document context using FAISS and embeddings, and generates accurate answers via the Mistral-7B-Instruct model on Together API.

---

## 🚀 Features

- 📄 **PDF Upload & Parsing**  
- ✂️ **Text Chunking using Langchain Splitter**  
- 🧠 **Query Cleaning, Rephrasing, and NER Extraction via LLM**  
- 🔍 **Semantic Search via FAISS and Sentence Transformers**  
- 🤖 **Response Generation with Mistral-7B via Together API**  
- 🔐 **JWT-like Token Authentication (Bearer)**  
- 🧹 **Automatic Cleanup of Temporary Files**

---

## 📦 Tech Stack

- **FastAPI** - Web API Framework  
- **SentenceTransformers** - Semantic Embedding Model (`all-MiniLM-L6-v2`)  
- **FAISS** - Vector similarity search  
- **Together API** - LLM service for rephrasing and answer generation  
- **Langchain** - Text splitter  
- **OpenAI SDK** (used with Together API endpoint)  
- **pypdf** - PDF text extraction  

---

## 🔐 API Authentication

All endpoints are protected via **Bearer Token Authentication**.  
Add the following header in your request:


> Default token (if not set in `.env`):  
> `ssbakscstobcb3609e845e387e9f7ac988ea36090473eefbe6dae9cfe880c35c6b67d87a7757`

---

## 📥 Endpoint

### `POST /hackrx/run`

Upload a PDF and query it using natural language.

#### 🔸 Request

**Form Data:**

- `pdf`: Insurance PDF document  
- `query`: Natural language question  

**Header:**

- `Authorization: Bearer ssbakscstobcb3609e845e387e9f7ac988ea36090473eefbe6dae9cfe880c35c6b67d87a7757`

#### 📄 Example (Postman)

- Method: `POST`  
- URL: `https://bajajfinservhackatho-production.up.railway.app/hackrx/run`  
- Header: `Authorization: Bearer ssbakscstobcb3609e845e387e9f7ac988ea36090473eefbe6dae9cfe880c35c6b67d87a7757 `  
- Body:  
  - `pdf`: Upload your insurance document  
  - `query`: `"What is the minimum age for claim eligibility?"`

#### ✅ Response

```json
{
  "answer": "The minimum age for claim eligibility is 18 years."
}
```
#### Create a .env in your root directory:

TOGETHER_API_KEY=your_together_api_key
