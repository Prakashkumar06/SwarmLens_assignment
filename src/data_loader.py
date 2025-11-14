import os
from pathlib import Path
from typing import List
from pypdf import PdfReader

def load_text_files(data_dir: str) -> List[dict]:
    """
    Load text and pdf files from data_dir and return list of dicts:
    [{'source': filename, 'text': text}, ...]
    """
    data_dir = Path(data_dir)
    docs = []
    for p in data_dir.iterdir():
        if p.suffix.lower() in [".txt", ".md"]:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({"source": str(p.name), "text": text})
        elif p.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(p))
                pages = []
                for page in reader.pages:
                    pages.append(page.extract_text() or "")
                text = "\n".join(pages)
                docs.append({"source": str(p.name), "text": text})
            except Exception as e:
                print(f"[data_loader] Failed to read {p.name}: {e}")
    return docs

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    """Simple chunker by words"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks
