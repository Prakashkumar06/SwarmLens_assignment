RAG Q&A Agent (LangGraph-Style Workflow)

Workflow: Plan → Retrieve → Answer → Reflect

->>How the Agent Works
1. Plan
- Checks the question
- Decides if document retrieval is needed
2. Retrieve
- Converts question into embeddings
- Searches ChromaDB
- Returns top relevant chunks
3. Answer
- Sends question + retrieved context to Groq LLM
- Generates a contextual answer
4. Reflect
- Computes semantic similarity between question and answer
- Ensures relevance and reduces hallucination

Streamlit App
Run the UI:
python -m streamlit run src/ui_app.py

Features:
- Ask questions
- View AI answer
- See reflection score
- Inspect retrieved document chunks

Evaluation
Run:
python -m src.run_evaluation
Includes:
- ROUGE-L
- BLEU
- Semantic similarity
- LLM-as-a-Judge


Project Structure
src/
agent_graph.py ->> Plan → Retrieve → Answer → Reflect
rag_pipeline.py ->> Retrieval + LLM
embed_store.py ->> Load/Create embeddings
data_loader.py ->> Read PDFs/TXT
evaluate.py ->> Evaluation metrics
run_evaluation.py ->> Evaluation runner
ui_app.py ->> Streamlit UI


Challenges Faced
- LangChain package split requiring updated imports
- Streamlit path issues fixed using sys.path.append
- Groq endpoint mismatch corrected
- Embedding rebuild time solved by pre-building Chroma DB
- Chroma deprecation warnings resolved using langchain-chroma