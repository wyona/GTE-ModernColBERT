import torch
from pylate import indexes, models, retrieve

# Gerät automatisch wählen: CUDA (NVIDIA), MPS (Apple Silicon) oder CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model_name_or_path = "lightonai/GTE-ModernColBERT-v1"
print(f"Load model '{model_name_or_path}' (device={device}) from Hugging Face...")
model = models.ColBERT(
    model_name_or_path=model_name_or_path,
    device=device,
)

# Sample data / documents similar to the LIMIT dataset
documents = [
    {"id": "doc_1", "text": "Geneva Durben mag Quokkas und Äpfel."},
    {"id": "doc_2", "text": "Dorathea Bastress mag Bananen und Kirschen."},
    {"id": "doc_3", "text": "Armand Schweda mag Quokkas und Birnen."},
    {"id": "doc_4", "text": "Flor Lemaire mag Äpfel und Orangen."},
]
texts_by_id = {doc["id"]: doc["text"] for doc in documents}

print("Generate embeddings, split each text into a sequence of token-vectors ...")
corpus_embeddings = model.encode(
    [doc["text"] for doc in documents],
    batch_size=2,
    is_query=False,
)
print(f"Number of indexed documents: {len(corpus_embeddings)}")
for i, emb in enumerate(corpus_embeddings):
    number_of_token_vectors = emb.shape[0]
    dimensions = emb.shape[1]
    print(
        f" -> Document {i + 1} contains {number_of_token_vectors} Token-Vectors (each {dimensions}-dimensional).")

# Create search index (HNSW-Index via Voyager)
print(f"Initialize search index and add embeddings ...")
index = indexes.Voyager(
    index_folder="indexes",
    index_name="my_searchindex",
    override=True,
)
index.add_documents(
    documents_ids=[doc["id"] for doc in documents],
    documents_embeddings=corpus_embeddings,
)
retriever = retrieve.ColBERT(index=index)

# Run search query
query = "Wer mag Äpfel?"
print(f"\nExecute search query: '{query}'")

query_embeddings = model.encode([query], is_query=True)

# Search for top-2 best results
results = retriever.retrieve(queries_embeddings=query_embeddings, k=2)

# Show search results
for hit in results[0]:
    print(f"- [Score: {hit['score']:.4f}] {texts_by_id[hit['id']]}")
