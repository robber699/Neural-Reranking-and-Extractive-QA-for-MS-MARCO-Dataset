# Neural Reranking and Extractive QA for MS MARCO Dataset

## Overview
This project focuses on improving retrieval quality using neural reranking and extractive QA models. We start by aggregating raw judgements into robust labels, then implement neural network reranking models, and finally analyze pre-trained extractive QA models. The dataset used is a subset of the MS MARCO retrieval dataset, and we utilize PyTorch, AllenNLP, and HuggingFace Transformers for the tasks.

## Goals
1. **Test Collection Preparation **
   - Aggregate raw judgements to usable labels.
   - Implement and analyze advanced aggregation methods beyond the baseline.

2. **Neural Re-Ranking **
   - Develop KNRM and TK neural models for re-ranking.
   - Train and evaluate these models using various datasets.
   - Compare results with baseline methods.

3. **Extractive QA **
   - Use HuggingFace Transformers to run extractive QA on top-1 re-ranking results.
   - Evaluate QA performance on the MS MARCO and FiRA datasets.

## Setup and Requirements
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/neural-reranking-extractive-qa.git
   cd neural-reranking-extractive-qa
