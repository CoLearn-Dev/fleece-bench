## Installation

### Install fleece-worker
```
pip install -e .
```

## Run benchmarking
```
CUDA_VISIBLE_DEVICES=0 python bench.py -m llama-2-7b-chat-slice > results7b.json
```
```
CUDA_VISIBLE_DEVICES=0 python bench.py -m llama-2-70b-chat-slice > results70b.json
```