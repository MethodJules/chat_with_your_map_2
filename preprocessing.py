import pandas as pd
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack import Pipeline


# Load the csv file
df = pd.read_csv("./data/csv.csv", sep=";", encoding="latin-1")

print(df.head)

df["searchable_text"] = df.apply(
  lambda row: f"{row['Name']} {row['Feature Type'] if pd.notna(row['Feature Type']) else ''}",
  axis=1)

# Keep it simple for prototype
documents = []
for _, row in df.iterrows():
  documents.append({
    "id": str(row["ID"]),
    "content": str(row["searchable_text"]),
    "name": row["Name"],
    "url": row["URL"],
  })

print(f"Prepared {len(documents)} documents")


# Initialize dcoument store
document_store = InMemoryDocumentStore()

# Create haystack documents
haystack_docs = [
  Document(
    content=doc["content"],
    meta={
      "layer_id": doc["id"],
      "name": doc["name"],
      "url": doc["url"],
    }
  ) for doc in documents
]

# Embed documents
doc_embedder = SentenceTransformersDocumentEmbedder(
  model="intfloat/multilingual-e5-small" # TODO: Change for embedding with IONOS?
)

doc_embedder.warm_up()

print("Embedding documents... (this takes a few minutes)")
docs_with_embeddings = doc_embedder.run(haystack_docs)
document_store.write_documents(docs_with_embeddings["documents"])

print(f"✅ {document_store.count_documents()} documents embedded and stored")

# Build query pipeline
query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(
  model="intfloat/multilingual-e5-small"
))

query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(
  document_store=document_store,
  top_k=10
))

query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

def get_layer_candidates(entity: str, top_k: int = 10):
  """Simple retrieval function"""
  result = query_pipeline.run({
    "text_embedder": {"text": entity},
    "retriever": {"top_k": top_k}
  })

  print(result)

  candidates = []
  for doc in result["retriever"]["documents"]:
    print(f"DEBUG: Available meta keys: {doc.meta.keys()}")
    candidates.append({
      "id": doc.meta["layer_id"],
      "name": doc.meta["name"],
      "score": doc.score,
      "url": doc.meta["url"]
    })

  return candidates


### TEST ###
print("===TESTTING===")
test_queries = ["Fahrradstationen", "S-Bahn", "Parkhaus"]
for query in test_queries:
    print(f"\nQuery: {query}")
    results = get_layer_candidates(query, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name']} (ID: {r['id']}, score: {r['score']:.3f})")
