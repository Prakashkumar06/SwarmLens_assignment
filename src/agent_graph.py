from src.rag_pipeline import retrieve_docs, answer_question
from src.config import HF_EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def plan_node(question: str):
    """Decide whether retrieval is needed."""
    print(f"[plan] Question received: {question}")

    lower = question.lower()
    retrieval_needed = any(tok in lower for tok in [
        "what", "explain", "benefits", "define", "how", "why",
        "advantages", "disadvantages"
    ])

    print(f"[plan] retrieval_needed = {retrieval_needed}")
    return {"question": question, "retrieval_needed": retrieval_needed}


def retrieve_node(state: dict):
    if not state.get("retrieval_needed"):
        print("[retrieve] Skipping retrieval per plan.")
        state["retrieved_docs"] = []
        return state
    
    docs = retrieve_docs(state["question"], k=4)
    state["retrieved_docs"] = docs
    return state


def answer_node(state: dict):
    if state.get("retrieved_docs"):
        out = answer_question(state["question"], k=4)
        state["answer"] = out["answer"]
        state["retrieved_docs"] = out["retrieved_docs"]
    else:
        out = answer_question(state["question"], k=0)
        state["answer"] = out["answer"]

    print("[answer] Generated answer (truncated):", str(state["answer"])[:400])
    return state


def reflect_node(state: dict):
    """Evaluate relevance using cosine similarity."""
    print("[reflect] Validating answer relevance...")

    try:
        model = SentenceTransformer(HF_EMBEDDING_MODEL_NAME)

        q_emb = model.encode([state["question"]])
        a_emb = model.encode([state["answer"][:1024]])

        sim = cosine_similarity(q_emb, a_emb)[0][0]
        print(f"[reflect] similarity (question vs answer) = {sim:.3f}")

        state["reflection"] = {
            "similarity": float(sim),
            "pass": sim >= 0.55
        }

    except Exception as e:
        print("[reflect] Reflection failed:", e)
        state["reflection"] = {"similarity": None, "pass": None}

    return state


def run_agent(question: str):
    state = plan_node(question)
    state = retrieve_node(state)
    state = answer_node(state)
    state = reflect_node(state)
    return state
