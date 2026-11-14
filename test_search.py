import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, QUERY_PREFIX

model = SentenceTransformer(EMBED_MODEL)
col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)

query = "How does reranking improve RAG performance?"
emb = model.encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
res = col.query(query_embeddings=emb, n_results=5)

for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"\n{meta['title']} | {meta['section']} | p.{meta['page']} | dist={dist:.3f}")
    print(doc[:250], "...")