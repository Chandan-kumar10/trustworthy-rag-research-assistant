import json
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHUNKS_FILE, CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL


def build_index(batch_size: int = 64):
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Rebuild from scratch each time
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        col.add(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c["meta"] for c in batch],
        )
        print(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print("Index ready.")


if __name__ == "__main__":
    build_index()