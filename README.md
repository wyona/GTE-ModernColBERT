# GTE-ModernColBERT

## Quick start

* python3 -m venv .venv
* source .venv/bin/activate
* pip3 install -r requirements.txt
* python3 src/main.py --dataset mock
* python3 src/main.py --dataset limit_small
* python3 src/main.py --serve --host 127.0.0.1 --port 8000
  * GET http://127.0.0.1:8000/health
  * POST http://127.0.0.1:8000/v1/embed/documents
    * curl -sS -X POST http://127.0.0.1:8000/v1/embed/documents -H 'Content-Type: application/json' -d '{"text":"Michael mag Äpfel"}'
  * POST http://127.0.0.1:8000/v1/embed/queries
    * curl -sS -X POST http://127.0.0.1:8000/v1/embed/queries -H 'Content-Type: application/json' -d '{"text":"Wer mag Äpfel?"}'
  * POST http://127.0.0.1:8000//v1/embeddings
    * curl -sS -X POST http://127.0.0.1:8000/v1/embeddings -H 'Content-Type: application/json' -d '{"text":"Michael mag Äpfel","is_query":true}'

Please note, that the script picks CUDA, then Apple Silicon's GPU (MPS), then CPU.

## About

* models.ColBERT loads the model (lightonai/GTE-ModernColBERT-v1) and encodes text
* indexes.Voyager stores the vectors
* retrieve.ColBERT runs the search

## LIMIT-small benchmark

* Compare results using GTE-ModernColBERT for LIMIT-Small with https://openreview.net/forum?id=k9CzIvzfaA&noteId=fTsViv8muf
  * According to a comment the authors receive: recall@2: 0.835
  * With our implementation we receive: recall@2: 0.9975
