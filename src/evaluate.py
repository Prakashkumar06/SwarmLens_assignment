import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu


def evaluate_rouge(reference: str, candidate: str):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores["rougeL"].fmeasure


def evaluate_bleu(reference: str, candidate: str):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    return sentence_bleu([ref_tokens], cand_tokens)



def evaluate_semantic_similarity(reference: str, candidate: str):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ref_emb = model.encode([reference])
    cand_emb = model.encode([candidate])
    score = cosine_similarity(ref_emb, cand_emb)[0][0]
    return float(score)



import requests
from src.config import GROQ_API_KEY, GROQ_API_URL

def llm_judge(question: str, answer: str, reference: str):
    """
    Ask Groq LLM to score the answer from 1–10 based on correctness and completeness.
    """

    prompt = f"""
You are an evaluator. The user asked a question, and two answers are provided.

Question:
{question}

System-generated answer:
{answer}

Ground-truth / reference answer:
{reference}

Score the system answer on a scale of 1-10 for correctness and completeness.
Return ONLY the number.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",   
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20,
        "temperature": 0
    }

    try:
        response = requests.post(
            f"{GROQ_API_URL}/chat/completions",
            json=payload,
            headers=headers
        )
        out = response.json()
        score = out["choices"][0]["message"]["content"].strip()
        return score
    except Exception as e:
        return f"LLM Judge failed: {e}"


def evaluate_all(question: str, answer: str, reference: str):
    return {
        "rouge_l": evaluate_rouge(reference, answer),
        "bleu": evaluate_bleu(reference, answer),
        "semantic_similarity": evaluate_semantic_similarity(reference, answer),
        "llm_judge_score": llm_judge(question, answer, reference)
    }
