import argparse
import torch
from datasets import load_dataset
from pylate import indexes, models, retrieve


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
    ]
    relevant_documents_by_query = {
        "query_0": {"doc_1", "doc_4"},
        "query_1": {"doc_5", "doc_6"},
        "query_2": {"doc_8", "doc_10"},
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
            relevant_documents_by_query.setdefault(query_id, set()).add(corpus_id)

    return documents, texts_by_id, query_dataset, relevant_documents_by_query


def build_retriever(model, documents):
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

    print("Initialize search index and add embeddings...")
    index = indexes.Voyager(
        index_folder="indexes",
        index_name="limit_small_searchindex",
        override=True,
    )
    index.add_documents(
        documents_ids=[doc["id"] for doc in documents],
        documents_embeddings=corpus_embeddings,
    )
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
        relevant_ids = relevant_documents_by_query.get(query_id, set())

        if relevant_ids:
            recall_at_2 = len(set(retrieved_ids[:2]) & relevant_ids) / len(relevant_ids)
            recall_at_5 = len(set(retrieved_ids[:5]) & relevant_ids) / len(relevant_ids)

            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(retrieved_ids[:5], start=1):
                if doc_id in relevant_ids:
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
            print(f" -> relevant: {sorted(relevant_ids)}")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ColBERT retrieval on mock data or LIMIT-small.")
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
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset
    max_queries = args.max_queries

    device = get_device()
    model_name_or_path = "lightonai/GTE-ModernColBERT-v1"
    print(f"Load model '{model_name_or_path}' (device={device}) from Hugging Face...")
    model = models.ColBERT(
        model_name_or_path=model_name_or_path,
        device=device,
    )

    if dataset_name == "mock":
        documents, texts_by_id, query_dataset, relevant_documents_by_query = load_mock_data()
        corpus_name = "Mock Dataset"
    else:
        documents, texts_by_id, query_dataset, relevant_documents_by_query = load_limit_small_data()
        corpus_name = "LIMIT-small"

    if max_queries is not None:
        query_dataset = query_dataset[:max_queries]
        print(f"Only evaluating the first {len(query_dataset)} queries (requested via --max-queries={max_queries}).")

    retriever = build_retriever(model, documents)

    example_query = query_dataset[0]["text"]
    example_query_embeddings = model.encode([example_query], is_query=True)
    example_results = retriever.retrieve(queries_embeddings=example_query_embeddings, k=3)
    print(f"\nExample query: '{example_query}'")
    for hit in example_results[0]:
        print(f"- [Score: {hit['score']:.4f}] {hit['id']} -> {texts_by_id[hit['id']][:120]}")

    evaluate(model, retriever, query_dataset, relevant_documents_by_query, corpus_name)


if __name__ == "__main__":
    main()
