# LLM-Final-Project
This repo contain the codes of implementing HippoRAG and KG2RAG for persian PQUAD and PersianMHQA datasets.
## Baseline Methods
### Simple Semantic RAG
Using models below for evaluation:

* Embedding: Qwen3-Embedding-0.6B
* Generation: gemini-flash-2.5
#### PersianMHQA Results

| Method     | EM   | P    | R    | F1   | Acc  |
|------------|------|------|------|------|------|
| Retrieval  | 0.02 | 0.72 | 0.45 | 0.52 |  -   |
| LLM        | 0.13 | 0.17 | 0.21 | 0.18 | 0.43 |
