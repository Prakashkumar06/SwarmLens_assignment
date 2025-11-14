# src/ui_app.py

import os
import sys
import streamlit as st

# --- Make sure Python can find src/ ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.agent_graph import run_agent


# --- Streamlit UI Setup ---
st.set_page_config(page_title="RAG Agent (LangGraph Style)", layout="centered")
st.title("RAG Q & A Agent — Streamlit Demo")


# ======================
# Main Q&A Section
# ======================
st.markdown("### Ask a question related to the knowledge base")

question = st.text_input("Your Question:", "")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner(" Running AI Agent (plan → retrieve → answer → reflect)..."):
            try:
                result = run_agent(question)

                # --- Answer ---
                st.subheader(" AI Answer")
                st.write(result.get("answer"))

                # --- Reflection ---
                st.subheader("Reflection / Validation")
                st.json(result.get("reflection"))

                # --- Retrieved Documents ---
                st.subheader("Retrieved Documents (Top matches)")
                retrieved = result.get("retrieved_docs") or []

                for i, d in enumerate(retrieved):
                    st.markdown(
                        f"**{i+1}. Source:** `{d['metadata'].get('source')}`, "
                        f"**Chunk:** {d['metadata'].get('chunk')}`, "
                        f"**Score:** {d.get('score')}"
                    )
                    st.write(
                        d["page_content"][:500]
                        + ("..." if len(d["page_content"]) > 500 else "")
                    )
                    st.markdown("---")

            except Exception as e:
                st.error(f"Agent run failed: {e}")
