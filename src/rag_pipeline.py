from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from src.config import CHROMA_DB_DIR, HF_EMBEDDING_MODEL_NAME, GROQ_API_KEY, GROQ_API_URL
import os
import requests

def load_vectorstore(persist_directory=CHROMA_DB_DIR):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL_NAME, cache_folder="./hf_cache")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb



def retrieve_docs(query: str, k: int = 3):
    vectordb = load_vectorstore()
    results = vectordb.similarity_search_with_score(query, k=k)

    docs = []
    for doc, score in results:
        docs.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })

    print(f"[retrieve] Retrieved {len(docs)} docs (top {k})")
    return docs


PROMPT_TEMPLATE = """
Use the following context to answer the user question concisely and accurately.

Context:
{context}

Question:
{question}

Answer:
"""


def build_prompt(question: str, docs: list):
    context = "\n\n".join([d["page_content"][:2000] for d in docs])
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return prompt


def call_groq(prompt: str, max_tokens: int = 300, temperature: float = 0.0):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing in .env file")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",   
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(
        f"{GROQ_API_URL}/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        raise RuntimeError(f"Groq API error: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]



def answer_question(question: str, k: int = 3):
    print("[answer] Starting RAG pipeline...")

    docs = retrieve_docs(question, k=k)
    if not docs:
        return {"answer": "No relevant documents found.", "retrieved_docs": []}

    prompt = build_prompt(question, docs)

    try:
        llm_output = call_groq(prompt)
    except Exception as e:
        llm_output = f"(LLM error) {e}"

    return {
        "answer": llm_output,
        "retrieved_docs": docs
    }
