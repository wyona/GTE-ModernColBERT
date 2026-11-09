# GTE-ModernColBERT

## Quick start

* python3 -m venv .venv
* source .venv/bin/activate
* pip3 install -r requirements.txt
* python3 src/main.py

Please note, that the script picks CUDA, then Apple Silicon's GPU (MPS), then CPU.

## About

* models.ColBERT loads the model (lightonai/GTE-ModernColBERT-v1) and encodes text
* indexes.Voyager stores the vectors
* retrieve.ColBERT runs the search

## TODOs

* Compare results of LIMIT-Small with https://openreview.net/forum?id=k9CzIvzfaA&noteId=fTsViv8muf
