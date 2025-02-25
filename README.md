# Retrieval-Augmented Generation (RAG) for Knowledge-Intensive NLP Tasks

This project implements a state-of-the-art RAG pipeline for answering complex, knowledge-intensive questions. It combines dense retrieval with generative language models to retrieve relevant documents and synthesize accurate, context-aware answers.

## Features
- **Dense Retrieval with FAISS**: Efficient document retrieval using dense embeddings and FAISS for similarity search.
- **Generative Model Fine-tuning**: Fine-tuning of generative models (e.g., LLaMA, FLAN-T5) using LoRA and QLoRA for domain-specific tasks.
- **Hybrid Retrieval**: Combines dense retrieval with BM25 for improved recall and precision.
- **Evaluation Framework**: Comprehensive evaluation using BLEU, ROUGE, and Exact Match (EM) on benchmark datasets like Natural Questions (NQ) and TriviaQA.
- **Real-time Deployment**: Scalable deployment using FastAPI and Docker for real-time inference.
- **Knowledge Graph Integration**: Optional integration with knowledge graphs for enhanced context understanding.
- **Custom CUDA Kernels**: Optimized inference with custom CUDA kernels for faster retrieval and generation.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Training
```bash
./scripts/train.sh
```

### Evaluation
```bash
./scripts/evaluate.sh
```

### Deployment
```bash
./scripts/deploy.sh
```

## Project Structure
```
rag-llm/
├── README.md
├── LICENSE
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   ├── deploy.sh
├── src/
│   ├── data/
│   │   ├── dataloader.py
│   │   ├── preprocess.py
│   ├── models/
│   │   ├── retrieval.py
│   │   ├── generation.py
│   │   ├── hybrid_retrieval.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   ├── api/
│   │   ├── app.py
│   │   ├── schemas.py
│   ├── utils/
│   │   ├── faiss_utils.py
│   │   ├── cuda_kernels.cu
│   │   ├── logger.py
├── notebooks/
│   ├── retrieval_benchmark.ipynb
│   ├── generation_finetuning.ipynb
│   ├── hybrid_retrieval_demo.ipynb
├── tests/
│   ├── test_retrieval.py
│   ├── test_generation.py
│   ├── test_hybrid_retrieval.py
├── configs/
│   ├── train_config.yaml
│   ├── eval_config.yaml
│   ├── deploy_config.yaml
├── models/
│   ├── fine_tuned/
│   ├── pretrained/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
├── logs/
│   ├── training.log
│   ├── evaluation.log
```

## License
This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.