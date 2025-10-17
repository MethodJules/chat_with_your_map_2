import pandas as pd
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack import Pipeline
from query_expansion import expand_query
import pickle

def load_data():
  # Load the csv file
  df = pd.read_csv("./data/csv.csv", sep=";", encoding="latin-1")
  print(df.head)
  return df

def filter_data(df):
  # Filter: Only keep OAF Layers
  df_oaf = df[df["Type"] == "OAF"].copy()

  print(f"Total layers {len(df)}")
  print(f"OAF Layers only: {len(df_oaf)}")
  print(f"Reduction: {len(df) - len(df_oaf)} layers filtered out")
  return df_oaf

def prepare_documents(df_oaf):
  df_oaf["searchable_text"] = df_oaf.apply(
    lambda row: f"{row['Name']} {row['Feature Type'] if pd.notna(row['Feature Type']) else ''}",
    axis=1)

  # Keep it simple for prototype
  documents = []
  for _, row in df_oaf.iterrows():
    documents.append({
      "id": str(row["ID"]),
      "content": str(row["searchable_text"]),
      "name": row["Name"],
      "url": row["URL"],
    })

  print(f"Prepared {len(documents)} documents")
  return documents

def create_document_store(documents):
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
  return document_store

def create_query_pipeline(document_store):
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
  return query_pipeline

######## Processing Step 1 ##################
# Load data
data = load_data()
# Filter data so we have only OAF Types
data_oaf = filter_data(data)
# Prepare documents
documents = prepare_documents(df_oaf=data_oaf)
# Create document store
document_store = create_document_store(documents=documents)
#with open("document_store.pkl", "wb") as f:
#   pickle.dump(document_store, f)
# Create pipeline
query_pipeline = create_query_pipeline(document_store=document_store)
#with open("query_pipeline.pkl", "wb") as f:
#   pickle.dump(query_pipeline, f)






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


def get_layer_candidates_v2(entity: str, top_k: int = 10):
    """Enhanced retrieval with query expansion"""

    # Expand the query
    search_terms = expand_query(entity)
    print(f"Searching for '{entity}' using terms {search_terms}")

    all_results = []
    seen_ids = set()

    # Search for each expand term
    for term in search_terms:
        result = query_pipeline.run({
            "text_embedder": {"text": term},
            "retriever": {"top_k": 5}
        })

        # Collect unique results
        for doc in result["retriever"]["documents"]:
            doc_id = doc.meta["layer_id"]
            if doc_id not in seen_ids:
              seen_ids.add(doc_id)
              all_results.append({
                "id": doc_id,
                "name": doc.meta["name"],
                "score": doc.score,
                "url": doc.meta["url"],
                "matched_term": term
              })

    # Sort by score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]




# Test the improvement
print("\n" + "=" * 60)
print("TESTING QUERY EXPANSION")
print("=" * 60)

test_queries = ["Fahrradstationen", "Bahnhaltestellen", "Parkhäuser"]

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    print("-" * 60)

    # Compare old vs new
    print("WITHOUT expansion:")
    results_old = get_layer_candidates(query, top_k=3)
    for i, r in enumerate(results_old, 1):
        print(f"  {i}. {r['name']} [{r['id']}] (score: {r['score']:.3f})")

    print("\nWITH expansion:")
    results_new = get_layer_candidates_v2(query, top_k=3)
    for i, r in enumerate(results_new, 1):
        print(
            f"  {i}. {r['name']} [{r['id']}] (via: {r['matched_term']}, score: {r['score']:.3f})"
        )
