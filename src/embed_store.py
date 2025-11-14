# import os
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from src.data_loader import load_text_files, chunk_text
# from src.config import HUGGINGFACEHUB_API_TOKEN, HF_EMBEDDING_MODEL_NAME, CHROMA_DB_DIR

# def create_chroma_from_data(data_dir: str, persist_directory: str = CHROMA_DB_DIR):
#     print("[embed_store] Loading documents...")
#     docs = load_text_files(data_dir)

#     texts = []
#     metadatas = []

#     for d in docs:
#         chunks = chunk_text(d["text"], chunk_size=500, overlap=80)
#         for i, c in enumerate(chunks):
#             texts.append(c)
#             metadatas.append({"source": d["source"], "chunk": i})

#     print(f"[embed_store] Preparing embeddings with model {HF_EMBEDDING_MODEL_NAME}...")
    
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN or ""

#     embeddings = HuggingFaceEmbeddings(
#         model_name=HF_EMBEDDING_MODEL_NAME,
#         cache_folder="./hf_cache"
#     )

#     print("[embed_store] Creating/Updating ChromaDB...")

#     vectordb = Chroma.from_texts(
#         texts=texts,
#         embedding=embeddings,
#         metadatas=metadatas,
#         persist_directory=persist_directory
#     )

#     vectordb.persist()

#     print("[embed_store] Finished. Chroma DB stored at:", persist_directory)
#     return vectordb


# if __name__ == "__main__":
#     data_path = os.path.join(os.getcwd(), "data")
#     create_chroma_from_data(data_path)



#  new code that don't regenerate emb 

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.data_loader import load_text_files, chunk_text
from src.config import HUGGINGFACEHUB_API_TOKEN, HF_EMBEDDING_MODEL_NAME, CHROMA_DB_DIR


def create_chroma_from_data(data_dir: str, persist_directory: str = CHROMA_DB_DIR):
    # If DB already exists → load instead of rebuild
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        print(f"[embed_store] Existing Chroma DB found at {persist_directory}. Loading...")
        embeddings = HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL_NAME,
            cache_folder="./hf_cache"
        )
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    print("[embed_store] Creating new Chroma DB...")

    docs = load_text_files(data_dir)

    texts, metadatas = [], []
    for d in docs:
        chunks = chunk_text(d["text"], chunk_size=500, overlap=80)
        for i, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({"source": d["source"], "chunk": i})

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN or ""
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL_NAME,
        cache_folder="./hf_cache"
    )

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )

    vectordb.persist()
    print("[embed_store] Chroma DB created at:", persist_directory)

    return vectordb


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data")
    create_chroma_from_data(data_path)
