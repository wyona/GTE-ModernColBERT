import argparse
import os
from pathlib import Path

import torch
import uvicorn
from datasets import load_dataset
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from pylate import indexes, models, retrieve


HUGGINGFACE_CACHE_ROOT = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).resolve()


class EmbeddingsRequest(BaseModel):
    text: str | None = Field(default=None, description="Single input text to embed. Use either text or texts.")
    texts: list[str] | None = Field(default=None, description="Multiple input texts to embed.")
    is_query: bool = Field(default=False, description="Whether the texts are queries or documents.")

    @property
    def resolved_texts(self):
        if self.texts is not None:
            items = self.texts
        elif self.text is not None:
            items = [self.text]
        else:
            items = []
        if not items:
            raise HTTPException(status_code=400, detail="Either 'text' or 'texts' must be provided.")
        return items


app = FastAPI(title="GTE-ModernColBERT Embedding API")
MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        model_name_or_path = "lightonai/GTE-ModernColBERT-v1"
        MODEL = models.ColBERT(
            model_name_or_path=model_name_or_path,
            device=get_device(),
        )
    return MODEL


def serialize_embeddings(raw_embeddings):
    if hasattr(raw_embeddings, "tolist"):
        return raw_embeddings.tolist()
    if isinstance(raw_embeddings, list):
        return [serialize_embeddings(item) for item in raw_embeddings]
    return raw_embeddings


@app.get("/health")
def health():
    return {"status": "ok", "model": "lightonai/GTE-ModernColBERT-v1"}


@app.post("/v1/embed/documents")
def embed_documents(payload: EmbeddingsRequest):
    payload.is_query = False
    return _embed_texts(payload)


@app.post("/v1/embed/queries")
def embed_queries(payload: EmbeddingsRequest):
    payload.is_query = True
    return _embed_texts(payload)


@app.post("/v1/embeddings")
def embed_generic(payload: EmbeddingsRequest):
    return _embed_texts(payload)


def _embed_texts(payload: EmbeddingsRequest):
    texts = payload.resolved_texts
    embeddings = get_model().encode(
        texts,
        is_query=payload.is_query,
        batch_size=32,
        convert_to_numpy=True,
    )
    flattened = [serialize_embeddings(item) for item in embeddings]
    return {
        "is_query": payload.is_query,
        "count": len(flattened),
        "embeddings": flattened,
    }


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_mock_data():
    documents = [
        {"id": "doc_1", "text": "Geneva Durben mag Quokkas und Äpfel."},
        {"id": "doc_2", "text": "Dorathea Bastress mag Bananen und Kirschen."},
        {"id": "doc_3", "text": "Armand Schweda mag Quokkas und Birnen."},
        {"id": "doc_4", "text": "Flor Lemaire mag Äpfel und Orangen."},
        {"id": "doc_5", "text": "Michael mag Mangos und Datteln."},
        {"id": "doc_6", "text": "Katerina mag Früchte, die auf Palmen wachsen."},
        {"id": "doc_7", "text": "Michael wurde am 16. Februar 1969 geboren."},
        {"id": "doc_8", "text": "Michael wurde in St. Gallen geboren."},
        {"id": "doc_9", "text": "Michael lebt in Zürich."},
        {"id": "doc_10", "text": "Michael wurde an einem Sonntag in St. Gallen geboren."},
    ]
    texts_by_id = {doc["id"]: doc["text"] for doc in documents}

    query_dataset = [
        {"_id": "query_0", "text": "Wer mag Äpfel?"},
        {"_id": "query_1", "text": "Wer mag Datteln?"},
        {"_id": "query_2", "text": "Wo wurde Michael geboren?"},
        #{"_id": "query_2", "text": "Where wurde Michael geboren"},
        {"_id": "query_3", "text": "Where was Michael born?"},
    ]
    relevant_documents_by_query = {
        "query_0": ("doc_1", "doc_4"),
        "query_1": ("doc_5", "doc_6"),
        "query_2": ("doc_8", "doc_10"), # For some "strange" reasons doc_8 is not ranked top, but retrieved top-5: ['doc_7', 'doc_8', 'doc_10', 'doc_5', 'doc_9']
        "query_3": ("doc_8", "doc_10"),
    }

    return documents, texts_by_id, query_dataset, relevant_documents_by_query


def load_limit_small_data():
    corpus_dataset = load_dataset("orionweller/LIMIT-small", "corpus", split="corpus")
    query_dataset = load_dataset("orionweller/LIMIT-small", "queries", split="queries")
    qrels_dataset = load_dataset("orionweller/LIMIT-small", split="test")

    documents = [
        {"id": row["_id"], "text": row["text"]}
        for row in corpus_dataset
    ]
    texts_by_id = {doc["id"]: doc["text"] for doc in documents}

    relevant_documents_by_query = {}
    for row in qrels_dataset:
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]
        if row["score"] > 0:
            if query_id not in relevant_documents_by_query:
                relevant_documents_by_query[query_id] = []
            if corpus_id not in relevant_documents_by_query[query_id]:
                relevant_documents_by_query[query_id].append(corpus_id)

    return documents, texts_by_id, query_dataset, relevant_documents_by_query


def resolve_model_local_path(model_name_or_path):
    if model_name_or_path.startswith("./") or model_name_or_path.startswith("/"):
        return str(Path(model_name_or_path).resolve())

    try:
        local_dir = snapshot_download(repo_id=model_name_or_path, cache_dir=str(HUGGINGFACE_CACHE_ROOT))
        return local_dir
    except Exception:
        return str(HUGGINGFACE_CACHE_ROOT / "hub" / "models--" / model_name_or_path.replace('/', '--'))


def build_retriever(model, documents, index_name="limit_small_searchindex"):
    print(f"Generate embeddings for {len(documents)} documents, split each text into token-vectors...")
    corpus_embeddings = model.encode(
        [doc["text"] for doc in documents],
        batch_size=32,
        is_query=False,
    )

    print(f"Number of indexed documents: {len(corpus_embeddings)}")
    for i, emb in enumerate(corpus_embeddings):
        number_of_token_vectors = emb.shape[0]
        dimensions = emb.shape[1]
        print(
            f" -> Document {i + 1} contains {number_of_token_vectors} Token-Vectors (each {dimensions}-dimensional)."
        )

    index_folder = "indexes"
    index_path = Path(index_folder) / index_name
    print(f"Initialize search index at '{index_path.resolve()}' and add embeddings...")
    index = indexes.Voyager(
        index_folder=index_folder,
        index_name=index_name,
        override=True,
    )
    index.add_documents(
        documents_ids=[doc["id"] for doc in documents],
        documents_embeddings=corpus_embeddings,
    )
    print(f"Index creation completed for '{index_name}' at '{index_path.resolve()}'")
    return retrieve.ColBERT(index=index)


def evaluate(model, retriever, query_dataset, relevant_documents_by_query, corpus_name):
    print(f"\nEvaluate Recall@2, Recall@5 and MRR for all {len(query_dataset)} queries against the '{corpus_name}' corpus")
    recall_at_2_values = []
    recall_at_5_values = []
    mrr_values = []

    for query_row in query_dataset:
        query_id = query_row["_id"]
        query_text = query_row["text"]
        query_embeddings = model.encode([query_text], is_query=True)
        results = retriever.retrieve(queries_embeddings=query_embeddings, k=5)

        retrieved_ids = [hit["id"] for hit in results[0]]
        relevant_ids = relevant_documents_by_query.get(query_id, [])
        if isinstance(relevant_ids, set):
            relevant_ids = list(dict.fromkeys(relevant_ids))
        else:
            relevant_ids = list(relevant_ids)
        relevant_id_set = set(relevant_ids)

        if relevant_ids:
            recall_at_2 = len(set(retrieved_ids[:2]) & relevant_id_set) / len(relevant_id_set)
            recall_at_5 = len(set(retrieved_ids[:5]) & relevant_id_set) / len(relevant_id_set)

            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(retrieved_ids[:5], start=1):
                if doc_id in relevant_id_set:
                    reciprocal_rank = 1.0 / rank
                    break
        else:
            recall_at_2 = 0.0
            recall_at_5 = 0.0
            reciprocal_rank = 0.0

        recall_at_2_values.append(recall_at_2)
        recall_at_5_values.append(recall_at_5)
        mrr_values.append(reciprocal_rank)

        if len(recall_at_2_values) <= 3:
            print(f"Query '{query_id}': '{query_text}'")
            print(f" -> retrieved top-5: {retrieved_ids}")
            print(f" -> relevant: {relevant_ids}")
            print(f" -> Recall@2: {recall_at_2:.4f}, Recall@5: {recall_at_5:.4f}, MRR: {reciprocal_rank:.4f}")

    avg_recall_at_2 = sum(recall_at_2_values) / len(recall_at_2_values) if recall_at_2_values else 0.0
    avg_recall_at_5 = sum(recall_at_5_values) / len(recall_at_5_values) if recall_at_5_values else 0.0
    avg_mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0

    print(f"\nAverage Recall@2: {avg_recall_at_2:.4f}")
    print(f"Average Recall@5: {avg_recall_at_5:.4f}")
    print(f"Average MRR: {avg_mrr:.4f}")

    return {
        "recall@2": avg_recall_at_2,
        "recall@5": avg_recall_at_5,
        "mrr": avg_mrr,
    }


def decode_token_ids(tokenizer, token_ids):
    decoded = []
    for token_id in token_ids:
        token_text = tokenizer.decode([int(token_id)]).strip()
        if not token_text:
            continue
        if token_text.startswith("[") and token_text.endswith("]"):
            continue
        decoded.append(token_text)
    return decoded


def log_token_similarity_matrix(model, query_text, document_text):
    query_output = model.encode([query_text], is_query=True, output_value=None, convert_to_numpy=False)
    document_output = model.encode([document_text], is_query=False, output_value=None, convert_to_numpy=False)

    query_ids = query_output["input_ids"][0]
    document_ids = document_output["input_ids"][0]
    query_mask = query_output["masks"][0]
    document_mask = document_output["masks"][0]

    query_ids = query_ids[query_mask]
    document_ids = document_ids[document_mask]

    query_token_text = decode_token_ids(model.tokenizer, query_ids)
    document_token_text = decode_token_ids(model.tokenizer, document_ids)

    if not query_token_text or not document_token_text:
        print("No non-special tokens found for similarity matrix logging.")
        return

    query_embeddings = query_output["token_embeddings"][0][query_mask]
    document_embeddings = document_output["token_embeddings"][0][document_mask]
    token_similarity = query_embeddings @ document_embeddings.T

    print(f"\nContextualized token-level similarity matrix for query '{query_text}' and document '{document_text}'")
    print(f"Query tokens: {query_token_text}")
    print(f"Doc tokens  : {document_token_text}")

    header = " " * 12 + " | " + " ".join(f"{token:<12}" for token in document_token_text)
    print(header)
    for i, query_token in enumerate(query_token_text):
        row = " ".join(f"{token_similarity[i, j]:>7.3f}" for j in range(token_similarity.shape[1]))
        print(f"{query_token:<12} | {row}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ColBERT retrieval on mock data or LIMIT-small, or serve the embedding sidecar API.")
    parser.add_argument(
        "--dataset",
        choices=["mock", "limit_small"],
        default="mock",
        help="Dataset to evaluate: 'mock' or 'limit_small'.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional limit for the number of queries to evaluate (useful for quick tests).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI sidecar endpoint instead of running the local evaluation pipeline.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for the FastAPI server when --serve is enabled.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the FastAPI server when --serve is enabled.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.serve:
        uvicorn.run(app, host=args.host, port=args.port)
        return

    dataset_name = args.dataset
    max_queries = args.max_queries

    device = get_device()
    model_name_or_path = "lightonai/GTE-ModernColBERT-v1"
    #model_name_or_path = "VAGOsolutions/SauerkrautLM-Multi-ModernColBERT"

    model_local_path = resolve_model_local_path(model_name_or_path)
    print(f"\n\nModel source: '{model_name_or_path}'")
    print(f"Resolved local model path: '{model_local_path}'")
    print(f"HF cache root: '{HUGGINGFACE_CACHE_ROOT}'")

    model = models.ColBERT(
        model_name_or_path=model_name_or_path,
        device=device,
    )

    print(f"Model '{model_name_or_path}' loaded and ready for indexing/searching.")

    if dataset_name == "mock":
        documents, texts_by_id, query_dataset, relevant_documents_by_query = load_mock_data()
        corpus_name = "Mock Dataset"
    else:
        documents, texts_by_id, query_dataset, relevant_documents_by_query = load_limit_small_data()
        corpus_name = "LIMIT-small"

    if max_queries is not None:
        query_dataset = query_dataset[:max_queries]
        print(f"Only evaluating the first {len(query_dataset)} queries (requested via --max-queries={max_queries}).")

    # TODO: Add index name
    retriever = build_retriever(model, documents)

    example_query = query_dataset[0]["text"]
    example_query_embeddings = model.encode([example_query], is_query=True)
    example_results = retriever.retrieve(queries_embeddings=example_query_embeddings, k=3)
    print(f"\nExample query: '{example_query}'")
    for hit in example_results[0]:
        print(f"- [Score: {hit['score']:.4f}] {hit['id']} -> {texts_by_id[hit['id']][:120]}")

    if dataset_name == "mock":
        query_id = "query_2"
        #query_id = "query_3"
        query_text = next(row["text"] for row in query_dataset if row["_id"] == query_id)
        for doc_id in ["doc_7", "doc_8"]:
            print(f"\n--- Token similarity matrix for {query_id} vs {doc_id} ---")
            log_token_similarity_matrix(model, query_text, texts_by_id[doc_id])

    evaluate(model, retriever, query_dataset, relevant_documents_by_query, corpus_name)


if __name__ == "__main__":
    main()
